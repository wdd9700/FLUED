"""FLUED v3.6 S1.0: three-task trainer with fully separated decoder/backbone roles.

Tasks (spec section 20, user-designed):
1. direct (codec fidelity): decoder restores the AS-ENCODED byte sequence from
   the raw readout -- masked input means the target contains MASK_ID at masked
   positions. Pure translation, no inference.
2. backbone completion: readout -> backbone -> new matrix -> decoder restores
   the CLEAN text. Masked completion is ONLY scored on this path, so the
   backbone has an irreplaceable role (no more idling).
3. backbone prediction (fast latent path): MSE(backbone_out[i],
   content[i+1].detach()) -- next-chunk latent prediction with stop-gradient
   on the target. Bare run: no reweighting/crutches.

Metric redefinition (S1.0+): direct fidelity excludes nothing but MASK_ID
positions are reported separately (trivially correct); masked acc is backbone
only; prediction reported as predict_cos + sampled byte-level decode accuracy.

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

from flued.data import MASK_ID, PAD_ID  # noqa: E402
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
    backbone_out = model.backbone(content)
    n_chunks = chunks.chunk_mask.size(1)
    pos = model.chunk_pos.weight.unsqueeze(0)[:, :n_chunks]
    cond_direct = model.decoder_in(content) + pos
    cond_backbone = backbone_out + pos
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


def step_model(model, batch, args, device, train: bool):
    clean = batch[0].to(device)
    valid = clean.ne(PAD_ID)
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

    backbone_out = out["backbone_out"].float()
    pair_mask = (chunks.chunk_mask[:, :-1] & chunks.chunk_mask[:, 1:]).float()
    pred = backbone_out[:, :-1]
    with torch.no_grad():
        tgt = model.decoder_in(out["content"].float())[:, 1:].detach()
    se = ((pred - tgt).square().mean(dim=-1) * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
    predict_loss = se
    with torch.no_grad():
        cos = F.cosine_similarity(pred, tgt.float(), dim=-1)
        predict_cos = ((cos * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)).item()

    loss = (
        args.task1_loss_weight * direct_loss
        + args.task2_loss_weight * completion_loss
        + args.predict_weight * predict_loss
    )
    metrics = {
        "loss": float(loss.item()),
        "direct_loss": float(direct_loss.item()),
        "completion_loss": float(completion_loss.item()),
        "predict_loss": float(predict_loss.item()),
        "direct_acc": _safe_acc(out["logits_direct"].argmax(dim=-1), encoded_targets, encoded_slot_mask),
        "backbone_acc": _safe_acc(out["logits_backbone"].argmax(dim=-1), targets, slot_mask),
        "backbone_masked_acc": _safe_acc(out["logits_backbone"].argmax(dim=-1), targets, masked_slot),
        "backbone_unmasked_acc": _safe_acc(out["logits_backbone"].argmax(dim=-1), targets, unmasked_slot),
        "predict_cos": predict_cos,
        "truncated_tokens": float(chunks.pack_info["truncated_tokens"].float().sum().item()),
        "cut_capacity_overflow": float(out["cut_overflow"].float().sum().item()),
        "chunks_per_sample": float(chunks.chunk_mask.float().sum(dim=1).mean().item()),
        "hard_cut_fraction": float(out["hard_cut_fraction"].item()),
        "state_norm": float(out["state_norm"].item()),
        "boundary_confidence_mean": float(out["boundary_confidence_mean"].item()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, eval_loader, args, device):
    model.eval()
    torch.manual_seed(args.eval_mask_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_mask_seed)
    rows = []
    for i, batch in enumerate(eval_loader):
        if i >= args.max_eval_batches:
            break
        _, metrics = step_model(model, batch, args, device, train=False)
        rows.append(metrics)
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
    if pre_args.config:
        parser.set_defaults(**json.loads(Path(pre_args.config).read_text(encoding="utf-8")))
    args = parser.parse_args()
    if not args.per_chunk_readout:
        raise SystemExit("S1.0 requires --per-chunk-readout")

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
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            loss, metrics = step_model(model, batch, args, device, train=True)
            if not torch.isfinite(loss):
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
            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad": float(grad.item()),
                    "steps_per_sec": step / max(time.time() - t0, 1e-9),
                    "nan_skips": nan_skips,
                    **metrics,
                }
                _append_jsonl(log_path, row)
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
