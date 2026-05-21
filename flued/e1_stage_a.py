"""
FLUED E1 Stage A Runner — DSC reconstruction validation.

Goal
----
Verify that FLUEDAutoencoder can compress a byte sequence into dynamic
semantic units and reconstruct the original input with high accuracy.

    x  →  DSC(x)  →  DSC⁻¹(Z, spans)  →  x̂
    loss = CrossEntropy(x̂, x)

E1 does NOT test generation or downstream tasks.  It only tests whether
the tied-weight inverse can faithfully reconstruct the encoder's input.

Usage
-----
    # Quick sanity check (CPU, no data file needed)
    python -m flued.e1_stage_a --preset smoke_cpu

    # Small GPU run
    python -m flued.e1_stage_a --preset small_gpu --data-path corpus.txt

    # RTX 5080 16 GB — large model with AMP + gradient accumulation
    python -m flued.e1_stage_a \\
        --preset class300m_16gb \\
        --data-path corpus.txt \\
        --grad-accum-steps 16 \\
        --amp --amp-dtype bf16

Pass / fail criteria
--------------------
    --target-accuracy   (default 0.99)  — per-token reconstruction accuracy
    --min-compression   (default 0.125) — m/n must be ≥ this
    --max-compression   (default 0.5)   — m/n must be ≤ this

Exit code: 0 = PASS, 1 = FAIL.
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("flued.e1")


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict] = {
    "smoke_cpu": {
        "d_model": 64,
        "nhead": 4,
        "dim_feedforward": 128,
        "num_layers": 2,
        "max_seq_len": 32,
        "dropout": 0.0,
        "batch_size": 4,
        "max_steps": 200,
        "lr": 3e-4,
        "warmup_steps": 20,
        "grad_accum_steps": 1,
        "seq_len": 32,
        "stride": 16,
        "device": "cpu",
        "amp": False,
    },
    "small_gpu": {
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 1024,
        "num_layers": 4,
        "max_seq_len": 256,
        "dropout": 0.0,
        "batch_size": 16,
        "max_steps": 2000,
        "lr": 1e-4,
        "warmup_steps": 200,
        "grad_accum_steps": 1,
        "seq_len": 128,
        "stride": 64,
        "device": "cuda",
        "amp": False,
    },
    "class300m_16gb": {
        "d_model": 1024,
        "nhead": 16,
        "dim_feedforward": 4096,
        "num_layers": 24,
        "max_seq_len": 512,
        "dropout": 0.0,
        "batch_size": 1,
        "max_steps": 10000,
        "lr": 3e-5,
        "warmup_steps": 500,
        "grad_accum_steps": 16,
        "seq_len": 512,
        "stride": 256,
        "device": "cuda",
        "amp": True,
        "amp_dtype": "bf16",
    },
}


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Reconstruction accuracy (PAD-aware)
# ---------------------------------------------------------------------------

def reconstruction_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-token accuracy ignoring PAD (id=0)."""
    preds = logits.argmax(dim=-1)
    mask = targets != 0
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / total


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_e1(args: argparse.Namespace) -> bool:
    """Execute E1 training loop.  Returns True if criteria are met."""
    from flued.data import ByteReconstructionDataset, safe_train_eval_split
    from flued.model import FLUEDAutoencoder

    # --- Resolve device ---
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # --- Build model ---
    model = FLUEDAutoencoder(
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        boundary_threshold=args.boundary_threshold,
        target_compression=args.target_compression,
        compression_weight=args.compression_weight,
    ).to(device)

    n_params = model.count_parameters()
    logger.info(
        "E1 model: d=%d  layers=%d  params=%s  device=%s",
        args.d_model, args.num_layers, f"{n_params:,}", device
    )

    # --- Dataset ---
    texts: Optional[List[str]] = None
    if args.data_path:
        with open(args.data_path, encoding="utf-8") as fh:
            texts = fh.readlines()
        logger.info("Loaded %d lines from %s", len(texts), args.data_path)

    dataset = ByteReconstructionDataset(
        texts=texts,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.1, seed=42)
    logger.info("Dataset: %d train / %d eval chunks", len(train_ds), len(eval_ds))

    def _make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=len(ds) > args.batch_size,
            pin_memory=(device_str == "cuda"),
        )

    train_loader = _make_loader(train_ds, shuffle=True)
    eval_loader = _make_loader(eval_ds, shuffle=False)

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = _cosine_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # --- AMP ---
    use_amp = args.amp and device_str == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "bf16") == "bf16" else torch.float16
    # Use the current torch.amp.GradScaler API (torch.cuda.amp.GradScaler is deprecated)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # --- Training loop ---
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_recon_loss = 0.0
    running_comp_loss = 0.0
    running_acc = 0.0
    running_soft_mon = 0.0
    running_hard_mon = 0.0
    running_num_units = 0.0
    running_bp_mean = 0.0
    running_bp_std = 0.0
    running_grad_norm = 0.0

    global_step = 0
    accum_steps = 0

    optimizer.zero_grad()

    while global_step < args.max_steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)

        src = src.to(device)
        # E1: tgt == src (reconstruction)

        with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
            logits, metrics = model(src)
            recon_loss = criterion(logits.view(-1, logits.size(-1)), src.view(-1))
            comp_loss = metrics["compression_loss"]
            loss = (recon_loss + comp_loss) / args.grad_accum_steps

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_steps += 1
        running_loss += (recon_loss + comp_loss).item()
        running_recon_loss += recon_loss.item()
        running_comp_loss += comp_loss.item()
        running_acc += reconstruction_accuracy(logits.detach(), src)
        running_soft_mon += metrics["soft_m_over_n"].item()
        running_hard_mon += float(metrics["hard_m_over_n"])
        running_num_units += float(metrics["num_units"])
        _bp = metrics["boundary_probs"].detach()
        running_bp_mean += _bp.mean().item()
        running_bp_std += _bp.std().item()

        if accum_steps < args.grad_accum_steps:
            continue

        # Gradient step
        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp and amp_dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        # Capture boundary_head grad_norm before zeroing (P0 gate metric)
        if model.boundary_head.weight.grad is not None:
            running_grad_norm += model.boundary_head.weight.grad.norm().item()
        optimizer.zero_grad()
        accum_steps = 0
        global_step += 1

        if global_step % 50 == 0:
            log_n = 50 * args.grad_accum_steps  # micro-step count
            log_g = 50  # gradient-step count
            logger.info(
                "step=%5d  loss=%.4f  recon=%.4f  comp=%.4f  recon_acc=%.4f"
                "  soft_m/n=%.3f  hard_m/n=%.3f  units=%.1f"
                "  bp_mean=%.3f  bp_std=%.3f  bhead_gnorm=%.4f  lr=%.2e",
                global_step,
                running_loss / log_n,
                running_recon_loss / log_n,
                running_comp_loss / log_n,
                running_acc / log_n,
                running_soft_mon / log_n,
                running_hard_mon / log_n,
                running_num_units / log_n,
                running_bp_mean / log_n,
                running_bp_std / log_n,
                running_grad_norm / log_g,
                scheduler.get_last_lr()[0],
            )
            running_loss = running_recon_loss = running_comp_loss = 0.0
            running_acc = running_soft_mon = running_hard_mon = 0.0
            running_num_units = running_bp_mean = running_bp_std = running_grad_norm = 0.0

    # --- Evaluation ---
    model.eval()
    total_acc = 0.0
    total_mon = 0.0
    n_eval = 0
    with torch.no_grad():
        for src, tgt in eval_loader:
            src = src.to(device)
            with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
                logits, metrics = model(src)
            total_acc += reconstruction_accuracy(logits, src)
            total_mon += metrics["m_over_n"]
            n_eval += 1

    final_acc = total_acc / max(1, n_eval)
    final_mon = total_mon / max(1, n_eval)

    logger.info(
        "E1 eval — reconstruction_accuracy=%.4f  m/n=%.4f", final_acc, final_mon
    )

    # --- Pass / fail ---
    acc_ok = final_acc >= args.target_accuracy
    mon_ok = args.min_compression <= final_mon <= args.max_compression

    if acc_ok and mon_ok:
        logger.info(
            "PASS: E1 criteria met  (acc=%.4f ≥ %.4f, %.3f ≤ m/n=%.4f ≤ %.3f)",
            final_acc, args.target_accuracy,
            args.min_compression, final_mon, args.max_compression,
        )
    else:
        reasons = []
        if not acc_ok:
            reasons.append(f"acc={final_acc:.4f} < {args.target_accuracy}")
        if not mon_ok:
            reasons.append(f"m/n={final_mon:.4f} not in [{args.min_compression}, {args.max_compression}]")
        logger.warning("FAIL: E1 criteria not met — %s", "; ".join(reasons))

    # Optional JSON output
    if args.output_json:
        result = {
            "reconstruction_accuracy": final_acc,
            "m_over_n": final_mon,
            "pass": bool(acc_ok and mon_ok),
            "steps": args.max_steps,
        }
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Results written to %s", args.output_json)

    return acc_ok and mon_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUED E1 Stage A — reconstruction validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Apply a named preset (values can be overridden by other flags)",
    )

    # Model
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--target-compression", type=float, default=0.3)
    parser.add_argument("--compression-weight", type=float, default=0.1)

    # Training
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--device", default=None)

    # Data
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

    # AMP
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")

    # Evaluation criteria
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--min-compression", type=float, default=0.125)
    parser.add_argument("--max-compression", type=float, default=0.5)

    # Output
    parser.add_argument("--output-json", default=None)

    args = parser.parse_args()

    # Apply preset defaults first
    defaults = PRESETS.get(args.preset or "smoke_cpu", PRESETS["smoke_cpu"]).copy()

    for key, val in defaults.items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    # Final fallbacks
    for attr, val in [
        ("d_model", 64), ("nhead", 4), ("dim_feedforward", 128),
        ("num_layers", 2), ("max_seq_len", 64), ("dropout", 0.0),
        ("batch_size", 4), ("max_steps", 200), ("lr", 3e-4),
        ("warmup_steps", 20), ("grad_accum_steps", 1),
        ("device", "cpu"), ("seq_len", 32), ("stride", 16),
        ("amp", False),
    ]:
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    return args


def main() -> None:
    args = _parse_args()
    passed = run_e1(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
