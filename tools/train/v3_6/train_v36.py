"""FLUED v3.6 trainer: whole-prompt single-readout codec training.

Two mask-native tasks share the global span decoder: task 1 decodes from the
readout package directly (precise restore), task 2 routes the package through
the backbone (full restore); the strict masked-source protocol (5%, span 1-8,
mask before encode) is native to both. Boundary is dynamic by default
(tau_cut over segmentor confidence with UTF-8 continuation guard and
capacity-safe cuts); canonical runs take over a S0-pretrained
byte_lookup/encoder/segmentor via --init-checkpoint + --freeze-prefixes.

Guards per run: truncated_tokens == 0, NaN skip counter, pre-clip grad norm.
Artifacts per run dir: resolved_config.json, train_log.jsonl, latest.pt,
step_XXXX.pt milestones, summary.json.
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

from tools.train.v3_3.train_v33 import (  # noqa: E402
    _append_jsonl,
    _cosine_with_warmup,
    _safe_acc,
    build_optimizer,
    make_byte_mask,
    make_dataloaders,
    make_targets,
)
from flued.data import MASK_ID, PAD_ID  # noqa: E402
from flued.v36 import FLUEDV36, V36Config  # noqa: E402


def build_model(args: Namespace) -> FLUEDV36:
    return FLUEDV36(
        V36Config(
            d_byte=args.d_byte,
            encoder_layers=args.encoder_layers,
            segmentor_layers=args.segmentor_layers,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
            d_mem=args.d_mem,
            summarizer_slots=args.summarizer_slots,
            summarizer_hidden=args.summarizer_hidden,
            kda_heads=args.kda_heads,
            kda_head_k=args.kda_head_k,
            kda_head_v=args.kda_head_v,
            write_hidden=args.write_hidden,
            kda_tau_max=args.kda_tau_max,
            readout_queries=args.readout_queries,
            d_pack=args.d_pack,
            d_backbone=args.d_backbone,
            backbone_layers=args.backbone_layers,
            backbone_nhead=args.backbone_nhead,
            backbone_ffn=args.backbone_ffn,
            decoder_hidden=args.decoder_hidden,
            decoder_layers=args.decoder_layers,
            max_chunks=args.max_chunks,
            max_span=args.max_span,
            bytes_per_chunk=args.bytes_per_chunk,
            tau_cut=args.tau_cut,
            tau_trans=args.tau_trans,
            boundary_mode=args.boundary_mode,
            boundary_temperature=args.boundary_temperature,
            boundary_bridge_gradient_scale=args.boundary_bridge_gradient_scale,
            max_positions=args.max_positions,
            per_chunk_readout=args.per_chunk_readout,
        )
    )


def _ce(logits: torch.Tensor, targets: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
    picked_logits = logits[select]
    picked_targets = targets[select]
    if picked_targets.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(picked_logits.float(), picked_targets.clamp(min=0, max=257), ignore_index=PAD_ID)


def step_model(model, batch, args, device, train: bool):
    clean = batch[0].to(device)
    valid = clean.ne(PAD_ID)
    byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    source = clean.masked_fill(byte_mask, MASK_ID)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        out = model(source)
    targets, slot_mask, masked_slot = make_targets(
        clean, byte_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span
    )
    direct_loss = _ce(out.logits_direct, targets, slot_mask)
    backbone_loss = _ce(out.logits_backbone, targets, slot_mask)
    loss = args.task1_loss_weight * direct_loss + args.task2_loss_weight * backbone_loss
    unmasked_slot = slot_mask & ~masked_slot
    metrics = {
        "loss": float(loss.item()),
        "direct_loss": float(direct_loss.item()),
        "backbone_loss": float(backbone_loss.item()),
        "direct_acc": _safe_acc(out.logits_direct.argmax(dim=-1), targets, slot_mask),
        "backbone_acc": _safe_acc(out.logits_backbone.argmax(dim=-1), targets, slot_mask),
        "backbone_masked_acc": _safe_acc(out.logits_backbone.argmax(dim=-1), targets, masked_slot),
        "backbone_unmasked_acc": _safe_acc(out.logits_backbone.argmax(dim=-1), targets, unmasked_slot),
        "direct_masked_acc": _safe_acc(out.logits_direct.argmax(dim=-1), targets, masked_slot),
        "truncated_tokens": float(out.chunks.pack_info["truncated_tokens"].float().sum().item()),
        "cut_capacity_overflow": float(out.aux["cut_capacity_overflow"]),
        "chunks_per_sample": float(out.chunks.chunk_mask.float().sum(dim=1).mean().item()),
        "hard_cut_fraction": float(out.aux["hard_cut_fraction"].item()),
        "state_norm": float(out.state_norm.item()),
        "boundary_confidence_mean": float(out.aux["boundary_confidence_mean"].item()),
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
    merged["eval_backbone_ppl"] = float(math.exp(min(merged["eval_backbone_loss"], 20.0)))
    merged["eval_direct_ppl"] = float(math.exp(min(merged["eval_direct_loss"], 20.0)))
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--run-id", default="v36_v0")
    parser.add_argument("--out-dir", default="checkpoints/v36_v0")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--streaming-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-samples-per-worker", type=int, default=50000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--max-lines", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-mask-seed", type=int, default=1042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-byte", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--segmentor-layers", type=int, default=9)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1152)
    parser.add_argument("--d-mem", type=int, default=512)
    parser.add_argument("--summarizer-slots", type=int, default=4)
    parser.add_argument("--summarizer-hidden", type=int, default=1024)
    parser.add_argument("--kda-heads", type=int, default=4)
    parser.add_argument("--kda-head-k", type=int, default=128)
    parser.add_argument("--kda-head-v", type=int, default=256)
    parser.add_argument("--write-hidden", type=int, default=1024)
    parser.add_argument("--kda-tau-max", type=float, default=256.0)
    parser.add_argument("--readout-queries", type=int, default=1)
    parser.add_argument("--d-pack", type=int, default=1536)
    parser.add_argument("--d-backbone", type=int, default=384)
    parser.add_argument("--backbone-layers", type=int, default=3)
    parser.add_argument("--backbone-nhead", type=int, default=8)
    parser.add_argument("--backbone-ffn", type=int, default=1024)
    parser.add_argument("--decoder-hidden", type=int, default=1024)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--max-span", type=int, default=64)
    parser.add_argument("--bytes-per-chunk", type=int, default=16)
    parser.add_argument("--tau-cut", type=float, default=0.94)
    parser.add_argument("--tau-trans", type=float, default=0.75)
    parser.add_argument("--boundary-mode", choices=["dynamic", "uniform"], default="dynamic")
    parser.add_argument("--boundary-temperature", type=float, default=0.15)
    parser.add_argument("--boundary-bridge-gradient-scale", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=64)
    parser.add_argument("--per-chunk-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--task1-loss-weight", type=float, default=1.0)
    parser.add_argument("--task2-loss-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="fused_adamw")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--milestone-every", type=int, default=3000)
    parser.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--init-prefixes", default="", help="comma-separated; only load matching prefixes from init checkpoint (rest stays freshly initialized)")
    parser.add_argument("--freeze-prefixes", default="")
    return parser


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    pre_args, _ = pre.parse_known_args()
    parser = build_parser()
    if pre_args.config:
        parser.set_defaults(**json.loads(Path(pre_args.config).read_text(encoding="utf-8")))
    args = parser.parse_args()

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
            compatible = {
                k: v for k, v in compatible.items() if any(k.startswith(p) for p in init_prefixes)
            }
        model.load_state_dict(compatible, strict=False)
        print(
            f"[v36] init from {args.init_checkpoint}: loaded={len(compatible)} skipped={len(payload['model']) - len(compatible)} prefixes={init_prefixes or 'ALL'}",
            flush=True,
        )
    frozen = [p.strip() for p in args.freeze_prefixes.split(",") if p.strip()]
    if frozen:
        n_frozen = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen):
                param.requires_grad_(False)
                n_frozen += 1
        print(f"[v36] frozen params={n_frozen} prefixes={frozen}", flush=True)
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
        print(f"[v36] resumed from step {start_step}", flush=True)

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
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        else:
            continue
        break

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
