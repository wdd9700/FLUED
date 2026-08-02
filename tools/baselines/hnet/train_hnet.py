"""H-Net reproduction trainer: same-scale baseline vs FLUED v3.6 (next-byte LM).

Trains HNetRepro on the same corpus/loader protocol as v3.6 arms (512B windows,
stride=window, 20K steps, lr 2e-4, bf16). Loss = next-byte CE + ratio loss.
Eval: next-byte BPB/PPL on held-out stream (native LM metric; v3.6's masked-
infilling PPL is a different, bidirectional task — compared with caveat).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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
from flued.hnet_repro import HNetRepro, HNetReproConfig  # noqa: E402


from tools.train.v3_3.train_v33 import (  # noqa: E402
    _append_jsonl,
    _cosine_with_warmup,
    build_optimizer,
    make_byte_mask,
    make_dataloaders,
)
from flued.data import PAD_ID  # noqa: E402
from flued.hnet_repro import HNetRepro, HNetReproConfig  # noqa: E402

MASK_ID = 257


def step_model(model, batch, args, device):
    ids = batch[0].to(device)
    valid = ids.ne(PAD_ID)
    if args.mode == "dit":
        byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
        source = ids.masked_fill(byte_mask, MASK_ID)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            out = model(source)
        logits = out["logits"]
        ce = F.cross_entropy(logits.reshape(-1, 258).float(), ids.reshape(-1), reduction="none")
        ce = (ce * valid.reshape(-1).float()).sum() / valid.sum().clamp(min=1)
        loss = ce + args.ratio_weight * out["ratio_loss"]
        pred = logits.argmax(dim=-1)
        acc = ((pred == ids) & valid).float().sum() / valid.sum().clamp(min=1)
        masked_acc = ((pred == ids) & byte_mask).float().sum() / byte_mask.sum().clamp(min=1)
        metrics = {
            "loss": float(loss.item()),
            "recon_ce": float(ce.item()),
            "recon_acc": float(acc.item()),
            "masked_acc": float(masked_acc.item()),
            "boundary_rate": float(out["boundary_rate"].item()),
            "chunks_per_sample": float(out["chunks_per_sample"].item()),
            "ratio_loss": float(out["ratio_loss"].item()),
        }
        return loss, metrics
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        out = model(ids)
    logits = out["logits"][:, :-1]
    targets = ids[:, 1:]
    mask = valid[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, 258).float(), targets.reshape(-1), reduction="none")
    ce = (ce * mask.reshape(-1).float()).sum() / mask.sum().clamp(min=1)
    loss = ce + args.ratio_weight * out["ratio_loss"]
    pred = logits.argmax(dim=-1)
    acc = ((pred == targets) & mask).float().sum() / mask.sum().clamp(min=1)
    metrics = {
        "loss": float(loss.item()),
        "next_byte_ce": float(ce.item()),
        "next_byte_acc": float(acc.item()),
        "boundary_rate": float(out["boundary_rate"].item()),
        "chunks_per_sample": float(out["chunks_per_sample"].item()),
        "ratio_loss": float(out["ratio_loss"].item()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, eval_loader, args, device):
    model.eval()
    if args.mode == "dit":
        torch.manual_seed(args.eval_mask_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_mask_seed)
    rows = []
    for i, batch in enumerate(eval_loader):
        if i >= args.max_eval_batches:
            break
        _, metrics = step_model(model, batch, args, device)
        rows.append(metrics)
    model.train()
    merged = {}
    for key in rows[0]:
        merged[f"eval_{key}"] = sum(r[key] for r in rows) / len(rows)
    if args.mode == "dit":
        merged["eval_recon_ppl"] = float(math.exp(min(merged["eval_recon_ce"], 20.0)))
    else:
        ce = merged["eval_next_byte_ce"]
        merged["eval_next_byte_ppl"] = float(math.exp(min(ce, 20.0)))
        merged["eval_bpb"] = ce / math.log(2.0)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--run-id", default="hnet_repro")
    parser.add_argument("--out-dir", default="checkpoints/hnet_repro")
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
    parser.add_argument("--mode", choices=["ar", "dit"], default="ar")
    parser.add_argument("--decoder-skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--main-layers", type=int, default=9)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--ratio-target", type=float, default=0.2)
    parser.add_argument("--ratio-weight", type=float, default=0.03)
    parser.add_argument("--max-chunks", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="fused_adamw")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    args = parser.parse_args()
    if args.config:
        for k, v in json.loads(Path(args.config).read_text(encoding="utf-8")).items():
            setattr(args, k, v)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model = HNetRepro(
        HNetReproConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
            encoder_layers=args.encoder_layers,
            main_layers=args.main_layers,
            decoder_layers=args.decoder_layers,
            ratio_target=args.ratio_target,
            ratio_weight=args.ratio_weight,
            max_chunks=args.max_chunks,
            causal=(args.mode == "ar"),
            decoder_skip=args.decoder_skip,
        )
    ).to(device)
    print(f"[hnet] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = build_optimizer(args, model.parameters())
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    train_loader, eval_loader = make_dataloaders(args)
    log_path = out_dir / "train_log.jsonl"
    model.train()
    t0 = time.time()
    step = 0
    nan_skips = 0
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            loss, metrics = step_model(model, batch, args, device)
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
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, out_dir / "latest.pt")
        else:
            continue
        break
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, out_dir / "latest.pt")
    eval_stats = evaluate(model, eval_loader, args, device)
    summary = {
        "run_id": args.run_id,
        "steps": step,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": time.time() - t0,
        "nan_skips": nan_skips,
        **eval_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
