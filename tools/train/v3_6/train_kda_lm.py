"""FLUED v3.6 R0 trainer: byte-level KDA-LM hybrid 3:1 (spec sections 4/7).

Protocol mirrors the H-Net reproduction trainer and canonical v3.6 arms:
same streaming loader (512B windows, stride=window), 20K steps, batch 8,
lr 2e-4 cosine + 200 warmup, bf16 autocast, same held-out eval stream.
Metric: next-byte CE / BPB (CE / ln 2) -- native LM metric, directly
comparable to the H-Net reproduction anchor (0.653 BPB); the v3.6 masked-
infilling PPL is a different (bidirectional) task, compared with caveat.

R0 verdict (pre-registered, spec section 7): same-params parity with the
FLUED full stack => v3.6 closes; a win requires quality AND segment-timed
speed. Two ratio arms: A wide-shallow (d512/L12/ffn1792, 48.21M) and
B narrow-deep (d448/L16/ffn1536, ~47.3M), FLUED stack = 47.2M.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import (  # noqa: E402
    _append_jsonl,
    _cosine_with_warmup,
    build_optimizer,
    make_dataloaders,
)
from flued.data import PAD_ID  # noqa: E402
from flued.v36.kda_lm import KDALM, KDALMConfig  # noqa: E402


def step_model(model, batch, args, device):
    ids = batch[0].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    valid = targets.ne(PAD_ID)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1).clamp(min=0),
        reduction="none",
        ignore_index=PAD_ID,
    ).reshape_as(targets)
    mean_ce = (ce * valid).sum() / valid.sum().clamp(min=1)
    # Metrics stay on device; synced only at log time (see train_v36_s1).
    acc = ((logits.argmax(dim=-1) == targets).float() * valid).sum() / valid.sum().clamp(min=1)
    metrics = {
        "loss": mean_ce.detach(),
        "bpb": (mean_ce / 0.6931471805599453).detach(),
        "byte_acc": acc.detach(),
    }
    return mean_ce, metrics


@torch.no_grad()
def evaluate(model, eval_loader, args, device):
    model.eval()
    ces, accs, n = [], [], 0
    for i, batch in enumerate(eval_loader):
        if i >= args.max_eval_batches:
            break
        _, metrics = step_model(model, batch, args, device)
        ces.append(float(metrics["loss"]))
        accs.append(float(metrics["byte_acc"]))
        n += 1
    model.train()
    ce = sum(ces) / max(n, 1)
    return {
        "eval_ce": ce,
        "eval_bpb": ce / 0.6931471805599453,
        "eval_ppl": float(math.exp(min(ce, 20.0))),
        "eval_byte_acc": sum(accs) / max(n, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="kda_lm_r0")
    parser.add_argument("--out-dir", default="checkpoints/kda_lm_r0")
    parser.add_argument("--data-path", default="")
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--kda-head-dim", type=int, default=128)
    parser.add_argument("--attn-nhead", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1792)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model = KDALM(
        KDALMConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            kda_head_dim=args.kda_head_dim,
            attn_nhead=args.attn_nhead,
            ffn_dim=args.ffn_dim,
        )
    ).to(device)
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
        print(f"[kda_lm] resumed from step {start_step}", flush=True)

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
            loss, metrics = step_model(model, batch, args, device)
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
