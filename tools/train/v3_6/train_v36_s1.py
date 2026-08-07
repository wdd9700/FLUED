"""FLUED v3.6 S1.0: three-task trainer with fully separated decoder/backbone roles.

Tasks (spec section 20, user-designed):
1. direct (codec fidelity): decoder restores the AS-ENCODED byte sequence from
   the raw readout -- masked input means the target contains MASK_ID at masked
   positions. Pure translation, no inference.
2. backbone completion: readout -> backbone -> new matrix -> decoder restores
   the CLEAN text. Masked completion is ONLY scored on this path, so the
   backbone has an irreplaceable role (no more idling).
3. backbone prediction: ``--predict-mode decode`` (v36.4+ default) decodes
   backbone_out[i] into chunk i+1's bytes through the FROZEN decoder
   (functional_call with detached params — gradients reach the backbone only,
   the decoder and shared byte table stay a fixed public ruler), plus a weak
   latent style anchor (MSE to decoder_in(content[i+1]).detach(), weight
   ``--predict-latent-weight``). Since v2.1 the predict branch consumes
   ``content.detach()``: predict CE trains the backbone only and can no longer
   deform the encoder/KDA compressed representation (E31: full-gradient v2.0
   collapsed the KDA state and destroyed the codec tasks).
   ``--predict-mode latent`` keeps the old pure-MSE behaviour for historical runs.

Metric redefinition (S1.0+): direct fidelity excludes nothing but MASK_ID
positions are reported separately (trivially correct); masked acc is backbone
only; prediction reported as predict_cos + sampled byte-level decode accuracy.

Mask modes (--mask-mode):
- ``byte_span``: legacy random 1-8B spans (kept for historical comparability).
- ``mixed`` (v36.3 default): 40% of the byte budget masks whole UTF-8 chars
  (1-3 chars per span), 60% masks whole BPE words against the 128k reference
  tokenizer. Whole-char masking trains single-char understanding; whole-word
  masking tests semantic inference without letting the architecture collapse
  into a fancy BPE (word share is kept below half).

Protocol: from-scratch ablation (only S0 four prefixes loaded), same
data/seed/eval as canonical.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID  # noqa: E402
from tools.train.v3_3.train_v33 import (  # noqa: E402
    _append_jsonl,
    _cosine_with_warmup,
    _safe_acc,
    build_optimizer,
    make_byte_mask,
    make_dataloaders,
    make_targets,
)
from tools.train.v3_6.train_v36 import _ce, build_model  # noqa: E402

# ---------------------------------------------------------------------------
# Mixed char/BPE-word masking (v36.3, T3)
# ---------------------------------------------------------------------------

# Module-level so the non-serializable tokenizer never lands in vars(args)
# (resolved_config.json / summary.json dump the full args namespace).
_BPE_TOKENIZER = None
# Dedicated CPU generator: evaluate() re-seeds it so the eval mask is
# deterministic regardless of dataset-RNG consumption order.
_MASK_GENERATOR = torch.Generator()


def _load_bpe_tokenizer(path: str) -> None:
    global _BPE_TOKENIZER
    from tokenizers import Tokenizer

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    _BPE_TOKENIZER = Tokenizer.from_file(str(resolved))


def _utf8_char_spans(bs: bytes) -> list:
    """Whole-UTF-8-char byte spans: continuation bytes (0b10xxxxxx) never start a char."""
    starts = [i for i, b in enumerate(bs) if (b & 0xC0) != 0x80]
    starts.append(len(bs))
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def _bpe_word_spans(bs: bytes, tokenizer) -> list:
    """Whole-BPE-word byte spans against the reference tokenizer.

    Returns [] when the window is not valid UTF-8 (streaming windows can cut
    mid-char at the edges); the caller then folds the word budget into chars.
    """
    try:
        text = bs.decode("utf-8")
    except UnicodeDecodeError:
        return []
    byte_offsets = [0]
    for ch in text:
        byte_offsets.append(byte_offsets[-1] + len(ch.encode("utf-8")))
    spans = []
    max_char = len(byte_offsets) - 1
    for c0, c1 in tokenizer.encode(text, add_special_tokens=False).offsets:
        # guard: offsets must stay inside the char->byte map (a normalizer
        # changing text length would break this mapping; ours has none)
        if c1 > c0 and 0 <= c0 and c1 <= max_char:
            spans.append((byte_offsets[c0], byte_offsets[c1]))
    return spans


def make_mixed_mask(
    clean: torch.Tensor,
    mask_prob: float,
    char_frac: float = 0.4,
    char_span_max: int = 3,
    tokenizer=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """40/60 mixed mask on CPU token ids (PAD-offset encoding).

    ``char_frac`` of the masked-byte budget goes to whole UTF-8 chars
    (1-``char_span_max`` chars per span), the rest to whole BPE words.
    Spans never overlap; positions beyond the valid length stay unmasked.
    """
    bsz, _seq_len = clean.shape
    mask = torch.zeros_like(clean, dtype=torch.bool)
    if mask_prob <= 0:
        return mask
    for b in range(bsz):
        ids = clean[b][clean[b].ne(PAD_ID)].tolist()
        if not ids:
            continue
        bs = bytes(i - BYTE_OFFSET for i in ids)
        n = len(bs)
        budget = float(mask_prob) * n
        word_budget = (1.0 - float(char_frac)) * budget
        char_budget = float(char_frac) * budget
        m = torch.zeros(n, dtype=torch.bool)
        word_spans = _bpe_word_spans(bs, tokenizer) if tokenizer is not None else []
        if word_spans:
            acc = 0.0
            for i in torch.randperm(len(word_spans), generator=generator).tolist():
                if acc >= word_budget:
                    break
                s, e = word_spans[i]
                if m[s:e].any():
                    continue
                m[s:e] = True
                acc += e - s
        else:
            char_budget += word_budget
        char_spans = _utf8_char_spans(bs)
        acc = 0.0
        trials = 0
        max_trials = max(8 * len(char_spans), 64)
        while acc < char_budget and trials < max_trials:
            trials += 1
            ci = int(torch.randint(len(char_spans), (1,), generator=generator))
            ln = int(torch.randint(1, int(char_span_max) + 1, (1,), generator=generator))
            s = char_spans[ci][0]
            e = char_spans[min(ci + ln, len(char_spans)) - 1][1]
            if m[s:e].any():
                continue
            m[s:e] = True
            acc += e - s
        mask[b, :n] = m
    return mask


def s1_forward(model, source: torch.Tensor, args) -> dict:
    byte_states, confidence, valid, _ = model._encode(source)
    hard_cut, utf8_cont, cut_overflow = model._cuts(source, confidence, valid)
    chunks = model.chunk_builder(byte_states, valid, hard_cut, confidence)
    chunks = model.bridge(chunks, byte_states, confidence, valid, utf8_cont)
    memory = model.summarizer(chunks.span_embeddings, chunks.token_mask)
    gates = model.write_head(memory)
    package, state_norm = model.state_machine(gates, chunks.chunk_mask)
    if not model.config.per_chunk_readout:
        raise ValueError("S1.0 requires per_chunk_readout=True")
    content = package.mean(dim=2)  # (B, C, d_pack) — readout of S_i per chunk
    n_chunks = chunks.chunk_mask.size(1)
    pos = model.chunk_pos.weight.unsqueeze(0)[:, :n_chunks]
    cond_direct = model.decoder_in(content) + pos
    if getattr(model.config, "backbone_readout", "per_chunk") == "final":
        # k=1 backbone interface (user-ruled 2026-08-06): the backbone consumes
        # ONLY the final state's readout (carries 0..C history through the
        # recurrence). Per-chunk readouts stay decoder-side, which is what
        # per-chunk conditioning was always for (task density, E23).
        last = chunks.chunk_mask.long().sum(dim=1).clamp(min=1) - 1
        ar = torch.arange(content.size(0), device=content.device)
        backbone_out = model.backbone(content[ar, last].unsqueeze(1))
    else:
        backbone_out = model.backbone(content)
    cond_backbone = backbone_out + pos  # (B,1,d) broadcasts over chunks in final mode
    logits_direct = model.decoder(cond_direct, chunks.token_mask)
    logits_backbone = model.decoder(cond_backbone, chunks.token_mask)
    return {
        "logits_direct": logits_direct,
        "logits_backbone": logits_backbone,
        "content": content,
        "backbone_out": backbone_out,
        "chunks": chunks,
        "cut_overflow": cut_overflow,
        "state_norm": state_norm,
        "boundary_confidence_mean": confidence[valid].mean() if valid.any() else confidence.mean(),
        "hard_cut_fraction": hard_cut.float()[valid].mean() if valid.any() else hard_cut.float().mean(),
    }


def _acc_tensor(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sync-free accuracy as a GPU scalar tensor (0.0 on empty mask).

    Training-loop metrics stay on device and are only synced at log time;
    calling ``.item()``/``float()`` per step (~19 metrics) serializes the
    CPU/GPU pipeline and costs real throughput at small batch sizes.
    """
    m = mask.float()
    return (pred.eq(target).float() * m).sum() / m.sum().clamp(min=1.0)


def _frozen_decoder_logits(model, cond: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Decoder forward with every parameter detached.

    Gradients flow into ``cond`` (the backbone path) but never into the
    decoder or the shared byte table: in predict-mode=decode the decoder is
    a fixed public ruler that the backbone must learn to write for.
    """
    from torch.func import functional_call

    params = {k: v.detach() for k, v in model.decoder.named_parameters()}
    buffers = {k: v for k, v in model.decoder.named_buffers()}
    # Checkpointing is incompatible with functional_call's param substitution
    # (backward recompute would see the live params) -- disable it for this call.
    model.decoder._ckpt_enabled = False
    try:
        return functional_call(model.decoder, (params, buffers), (cond, token_mask))
    finally:
        model.decoder._ckpt_enabled = True


def step_model(model, batch, args, device, train: bool):
    clean = batch[0].to(device)
    valid = clean.ne(PAD_ID)
    if getattr(args, "mask_mode", "byte_span") == "mixed":
        byte_mask = make_mixed_mask(
            clean.cpu(),
            args.mask_prob,
            char_frac=args.mask_char_frac,
            char_span_max=args.mask_char_span_max,
            tokenizer=_BPE_TOKENIZER,
            generator=_MASK_GENERATOR,
        ).to(device)
    else:
        byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    source = clean.masked_fill(byte_mask, MASK_ID)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        out = s1_forward(model, source, args)
    chunks = out["chunks"]
    targets, slot_mask, masked_slot = make_targets(
        clean, byte_mask, chunks.chunk_ids, chunks.offsets, args.max_chunks, args.max_span
    )
    # as-encoded targets: the masked source itself (MASK_ID at masked slots)
    encoded_targets, encoded_slot_mask, _ = make_targets(
        source, torch.zeros_like(byte_mask), chunks.chunk_ids, chunks.offsets, args.max_chunks, args.max_span
    )
    unmasked_slot = slot_mask & ~masked_slot

    direct_loss = _ce(out["logits_direct"], encoded_targets, encoded_slot_mask)
    completion_loss = _ce(out["logits_backbone"], targets, slot_mask)

    if getattr(args, "backbone_readout", "per_chunk") == "final":
        # k=1: predict the LAST real chunk from the penultimate readout -- the
        # only in-window pair whose input still carries the full preceding
        # context through the state (chunk-level autoregression form).
        last = chunks.chunk_mask.long().sum(dim=1).clamp(min=2) - 1
        ar = torch.arange(out["content"].size(0), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            pred = model.backbone(out["content"][ar, last - 1].unsqueeze(1)).float().squeeze(1)
        with torch.no_grad():
            tgt = model.decoder_in(out["content"].float())[ar, last].detach()
        se = (pred - tgt).square().mean(dim=-1).mean()
        with torch.no_grad():
            predict_cos = F.cosine_similarity(pred, tgt, dim=-1).mean()
        pair_mask = chunks.chunk_mask.new_ones((1,))  # placeholder, unused in final mode
    else:
        backbone_out = out["backbone_out"].float()
        pair_mask = (chunks.chunk_mask[:, :-1] & chunks.chunk_mask[:, 1:]).float()
        pred = backbone_out[:, :-1]
        with torch.no_grad():
            tgt = model.decoder_in(out["content"].float())[:, 1:].detach()
        se = ((pred - tgt).square().mean(dim=-1) * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
        with torch.no_grad():
            cos = F.cosine_similarity(pred, tgt.float(), dim=-1)
            predict_cos = (cos * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)

    predict_mode = getattr(args, "predict_mode", "latent")
    predict_ce = torch.zeros((), device=device)
    predict_byte_acc = torch.zeros((), device=device)
    if predict_mode == "decode":
        # frozen-decoder supervision, v2.1: the predict branch consumes
        # content.detach() -- predict CE trains the BACKBONE ONLY. s13 (E31)
        # showed that letting this gradient reach the encoder/KDA side collapses
        # the state (norm 1.7->0.3) and destroys the codec tasks: the compressed
        # representation is owned by direct/completion, not by the predictor.
        pos = model.chunk_pos.weight.unsqueeze(0)[:, : chunks.chunk_mask.size(1)].float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            backbone_pred = model.backbone(out["content"].detach())
        cond_pred = backbone_pred.float()[:, :-1] + pos[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits_pred = _frozen_decoder_logits(model, cond_pred, chunks.token_mask[:, 1:])
        pred_w = slot_mask[:, 1:] & pair_mask.unsqueeze(-1).bool()
        predict_ce = _ce(logits_pred, targets[:, 1:], pred_w)
        predict_byte_acc = _acc_tensor(logits_pred.argmax(dim=-1), targets[:, 1:], pred_w)
        predict_loss = predict_ce + args.predict_latent_weight * se
    else:
        predict_loss = se

    loss = (
        args.task1_loss_weight * direct_loss
        + args.task2_loss_weight * completion_loss
        + args.predict_weight * predict_loss
    )
    # Metrics stay on device (detached scalars); the training loop accumulates
    # them and only syncs to host at log time -- per-step .item() calls on
    # ~19 metrics were serializing the pipeline.
    metrics = {
        "loss": loss.detach(),
        "direct_loss": direct_loss.detach(),
        "completion_loss": completion_loss.detach(),
        "predict_loss": predict_loss.detach(),
        "predict_ce": predict_ce.detach(),
        "predict_byte_acc": predict_byte_acc.detach(),
        "direct_acc": _acc_tensor(out["logits_direct"].argmax(dim=-1), encoded_targets, encoded_slot_mask),
        "backbone_acc": _acc_tensor(out["logits_backbone"].argmax(dim=-1), targets, slot_mask),
        "backbone_masked_acc": _acc_tensor(out["logits_backbone"].argmax(dim=-1), targets, masked_slot),
        "backbone_unmasked_acc": _acc_tensor(out["logits_backbone"].argmax(dim=-1), targets, unmasked_slot),
        "predict_cos": predict_cos.detach(),
        "mask_rate": (byte_mask.float().sum() / valid.float().sum().clamp(min=1.0)).detach(),
        "truncated_tokens": chunks.pack_info["truncated_tokens"].float().sum().detach(),
        "cut_capacity_overflow": out["cut_overflow"].float().sum().detach(),
        "chunks_per_sample": chunks.chunk_mask.float().sum(dim=1).mean().detach(),
        "hard_cut_fraction": out["hard_cut_fraction"].detach(),
        "state_norm": out["state_norm"].detach(),
        "boundary_confidence_mean": out["boundary_confidence_mean"].detach(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, eval_loader, args, device):
    model.eval()
    torch.manual_seed(args.eval_mask_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_mask_seed)
    _MASK_GENERATOR.manual_seed(args.eval_mask_seed)
    rows = []
    for i, batch in enumerate(eval_loader):
        if i >= args.max_eval_batches:
            break
        _, metrics = step_model(model, batch, args, device, train=False)
        rows.append({k: float(v) for k, v in metrics.items()})
    model.train()
    merged = {}
    for key in rows[0]:
        merged[f"eval_{key}"] = sum(r[key] for r in rows) / len(rows)
    merged["eval_backbone_ppl"] = float(math.exp(min(merged["eval_completion_loss"], 20.0)))
    merged["eval_direct_ppl"] = float(math.exp(min(merged["eval_direct_loss"], 20.0)))
    return merged


def main() -> None:
    from tools.train.v3_6.train_v36 import build_parser

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    pre_args, _ = pre.parse_known_args()
    parser = build_parser()
    parser.add_argument("--predict-weight", type=float, default=1.0)
    parser.add_argument("--predict-mode", choices=["latent", "decode"], default="latent")
    parser.add_argument("--predict-latent-weight", type=float, default=0.1)
    parser.add_argument("--mask-mode", choices=["byte_span", "mixed"], default="mixed")
    parser.add_argument("--mask-char-frac", type=float, default=0.4)
    parser.add_argument("--mask-char-span-max", type=int, default=3)
    parser.add_argument(
        "--bpe-tokenizer-path",
        type=str,
        default="checkpoints/bpe_tokenizer_128k_v4/tokenizer.json",
    )
    if pre_args.config:
        parser.set_defaults(**json.loads(Path(pre_args.config).read_text(encoding="utf-8")))
    args = parser.parse_args()
    if not args.per_chunk_readout:
        raise SystemExit("S1.0 requires --per-chunk-readout")
    if getattr(args, "backbone_readout", "per_chunk") == "final" and args.predict_mode == "decode":
        raise SystemExit(
            "--backbone-readout final supports --predict-mode latent only "
            "(decode CE is the E34-poisoned loss form)"
        )
    if args.mask_mode == "mixed":
        _load_bpe_tokenizer(args.bpe_tokenizer_path)
    _MASK_GENERATOR.manual_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model = build_model(args).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        current = model.state_dict()
        compatible = {
            k: v for k, v in payload["model"].items() if k in current and current[k].shape == v.shape
        }
        init_prefixes = [p.strip() for p in args.init_prefixes.split(",") if p.strip()]
        if init_prefixes:
            compatible = {k: v for k, v in compatible.items() if any(k.startswith(p) for p in init_prefixes)}
        model.load_state_dict(compatible, strict=False)
        print(f"[s1] init: loaded={len(compatible)} skipped={len(payload['model']) - len(compatible)} prefixes={init_prefixes or 'ALL'}", flush=True)
    frozen = [p.strip() for p in args.freeze_prefixes.split(",") if p.strip()]
    if frozen:
        n = 0
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in frozen):
                param.requires_grad_(False)
                n += 1
        print(f"[s1] frozen params={n} prefixes={frozen}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer(args, iter(trainable))
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    train_loader, eval_loader = make_dataloaders(args)

    start_step = 0
    latest = out_dir / "latest.pt"
    if args.resume and latest.exists():
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if args.save_optimizer and "optimizer" in payload:
            opt.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        if start_step > 0:
            sched.step(start_step)
        print(f"[s1] resumed from step {start_step}", flush=True)

    log_path = out_dir / "train_log.jsonl"
    model.train()
    t0 = time.time()
    nan_skips = 0
    step = start_step
    pending: dict[str, torch.Tensor] = {}
    pending_n = 0
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            loss, metrics = step_model(model, batch, args, device, train=True)
            if not bool(torch.isfinite(loss)):
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            for k, v in metrics.items():
                pending[k] = pending[k] + v if k in pending else v.clone()
            pending_n += 1
            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad": float(grad.item()),
                    "steps_per_sec": step / max(time.time() - t0, 1e-9),
                    "nan_skips": nan_skips,
                    **{k: float((v / pending_n).item()) for k, v in pending.items()},
                }
                _append_jsonl(log_path, row)
                pending = {}
                pending_n = 0
            if step % args.ckpt_every == 0:
                payload = {"step": step, "model": model.state_dict(), "args": vars(args)}
                if args.save_optimizer:
                    payload["optimizer"] = opt.state_dict()
                torch.save(payload, latest)
            if step % args.milestone_every == 0:
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, out_dir / f"step_{step:06d}.pt")

    payload = {"step": step, "model": model.state_dict(), "args": vars(args)}
    if args.save_optimizer:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, latest)

    eval_stats = evaluate(model, eval_loader, args, device)
    summary = {
        "run_id": args.run_id,
        "steps": step,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": time.time() - t0,
        "steps_per_sec": step / max(time.time() - t0, 1e-9),
        "nan_skips": nan_skips,
        "args": vars(args),
        **eval_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
