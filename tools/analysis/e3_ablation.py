"""
FLUED E3 Ablation Framework.

Systematically tests the contribution of each boundary loss term by
training controlled variants from the same base checkpoint and comparing
reconstruction quality, compression ratio, and boundary properties.

Ablation Dimensions
-------------------
  A. Loss term removal:
     - full:       all 5 terms (baseline)
     - no_type:    λ_type = 0
     - no_utf8:    λ_utf8 = 0
     - no_entropy: λ_entropy = 0
     - no_var:     λ_var = 0
     - pure_recon: all boundary terms off (baseline lower bound)

  B. Hyperparameter sweep:
     - compression_weight ∈ {0.05, 0.1, 0.2, 0.5, 1.0}
     - target_compression ∈ {0.15, 0.30, 0.50}
     - entropy_theta sweep for BLT

  C. Architecture comparison:
     - FLUED vs BLT (entropy) vs BLT (fixed) vs BPE

Usage
-----
    # Run all ablations from a checkpoint
    python e3_ablation.py --base-ckpt checkpoints/e1_step31000.pt \\
        --preset quick --ablations no_type,no_utf8,no_entropy,no_var,pure_recon

    # Sweep compression weight
    python e3_ablation.py --base-ckpt checkpoints/e1_step31000.pt \\
        --preset quick --sweep compression_weight --values 0.05,0.1,0.2,0.5,1.0

    # Full comparison: FLUED vs BLT vs BPE
    python e3_ablation.py --mode compare_architectures --preset 300m \\
        --data-path corpus.txt --max-lines 50000

Output
------
    results/e3_ablation.json  — per-run metrics
    results/e3_ablation.csv   — summary table
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("e3.ablation")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Single ablation run configuration."""
    name: str
    description: str = ""
    # FLUED boundary loss toggles
    lambda_var: float = 0.5
    lambda_entropy: float = 0.05
    lambda_utf8: float = 0.02
    lambda_type: float = 0.05
    compression_weight: float = 0.1
    target_compression: float = 0.30
    # Architecture
    model_type: str = "flued"  # flued | blt_entropy | blt_fixed | bpe

@dataclass
class AblationResult:
    """Results from a single ablation run."""
    name: str
    model_type: str
    steps: int = 0
    recon_acc: float = 0.0
    recon_loss: float = 0.0
    hard_mn: float = 0.0
    soft_mn: float = 0.0
    bp_mean: float = 0.0
    bp_std: float = 0.0
    cjk_bp: float = 0.0
    utf8_bp: float = 0.0
    ascii_bp: float = 0.0
    op_bp: float = 0.0
    digit_bp: float = 0.0
    num_params: int = 0
    duration_s: float = 0.0
    status: str = "ok"


# ---------------------------------------------------------------------------
# Presets (quick smoke vs full 300M)
# ---------------------------------------------------------------------------

ABLATION_PRESETS = {
    "quick": {
        "d_model": 128, "nhead": 4, "dim_feedforward": 512,
        "num_layers": 4, "max_seq_len": 64, "dropout": 0.0,
        "batch_size": 8, "train_steps": 200, "lr": 1e-4,
        "warmup_steps": 20, "grad_accum_steps": 1,
        "seq_len": 64, "stride": 32, "device": "cuda",
        "amp": False,
    },
    "medium": {
        "d_model": 256, "nhead": 8, "dim_feedforward": 1024,
        "num_layers": 8, "max_seq_len": 256, "dropout": 0.0,
        "batch_size": 16, "train_steps": 1000, "lr": 5e-5,
        "warmup_steps": 100, "grad_accum_steps": 2,
        "seq_len": 128, "stride": 64, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
    "300m": {
        "d_model": 1024, "nhead": 16, "dim_feedforward": 4096,
        "num_layers": 24, "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 1, "train_steps": 5000, "lr": 3e-5,
        "warmup_steps": 200, "grad_accum_steps": 16,
        "seq_len": 512, "stride": 256, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup(optimizer, warmup: int, total: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def recon_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != 0
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return ((preds == targets) & mask).sum().item() / total


# ---------------------------------------------------------------------------
# Ablation Runners
# ---------------------------------------------------------------------------

def _build_flued(ab_cfg: AblationConfig, p: Dict) -> nn.Module:
    from flued.model import FLUEDAutoencoder
    return FLUEDAutoencoder(
        d_model=p["d_model"], nhead=p["nhead"], dim_feedforward=p["dim_feedforward"],
        num_layers=p["num_layers"], max_seq_len=p["max_seq_len"], dropout=p["dropout"],
        lambda_var=ab_cfg.lambda_var, lambda_entropy=ab_cfg.lambda_entropy,
        lambda_utf8=ab_cfg.lambda_utf8, lambda_type=ab_cfg.lambda_type,
        compression_weight=ab_cfg.compression_weight,
        target_compression=ab_cfg.target_compression,
    )


def _build_blt(ab_cfg: AblationConfig, p: Dict) -> nn.Module:
    from blt_baseline.model import BLTAutoencoder
    patch_mode = "entropy" if ab_cfg.model_type == "blt_entropy" else "fixed"
    return BLTAutoencoder(
        vocab_size=257, d_model=p["d_model"], nhead=p["nhead"],
        dim_feedforward=p["dim_feedforward"],
        num_encoder_layers=p["num_layers"] // 2,
        num_decoder_layers=p["num_layers"] // 2,
        max_seq_len=p["max_seq_len"], dropout=p["dropout"],
        patch_mode=patch_mode, entropy_theta=0.5, fixed_patch_size=4,
    )


def run_single_ablation(
    ab_cfg: AblationConfig,
    preset: Dict,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    base_ckpt: Optional[str] = None,
) -> AblationResult:
    """Train one ablation variant and return results."""
    device = torch.device(preset.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    t0 = time.time()

    # Build model
    if ab_cfg.model_type == "flued":
        model = _build_flued(ab_cfg, preset).to(device)
    elif ab_cfg.model_type in ("blt_entropy", "blt_fixed"):
        model = _build_blt(ab_cfg, preset).to(device)
    else:
        return AblationResult(name=ab_cfg.name, model_type=ab_cfg.model_type,
                              status=f"unknown model_type: {ab_cfg.model_type}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load base checkpoint if provided (for FLUED variants)
    if base_ckpt and os.path.exists(base_ckpt) and ab_cfg.model_type == "flued":
        ckpt = torch.load(base_ckpt, map_location=device)
        # Only load shared params; boundary_head may differ
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in ckpt["model"].items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
        logger.info("  Loaded %d/%d params from base checkpoint", len(pretrained), len(model_dict))

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=preset["lr"], weight_decay=1e-2)
    scheduler = _cosine_with_warmup(optimizer, preset["warmup_steps"], preset["train_steps"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    use_amp = preset.get("amp", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if preset.get("amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    accum = preset.get("grad_accum_steps", 1)

    # Training
    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad()
    accum_steps = 0
    total_steps = preset["train_steps"]

    for step in range(total_steps):
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)
        src = src.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            result = model(src)
            logits = result[0]
            aux = result[1]
            if isinstance(aux, dict):
                aux_loss = aux.get("compression_loss", torch.tensor(0.0, device=device))
            else:
                aux_loss = aux
            loss = (criterion(logits.view(-1, logits.size(-1)), src.view(-1)) + aux_loss) / accum

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_steps += 1
        if accum_steps < accum:
            continue

        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp and amp_dtype == torch.float16:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                optimizer.zero_grad()
                accum_steps = 0
                continue
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        accum_steps = 0

        if (step + 1) % max(1, total_steps // 4) == 0:
            logger.info("  %s step %d/%d", ab_cfg.name, step + 1, total_steps)

    # Evaluation
    model.eval()
    total_acc = 0.0
    total_recon_loss = 0.0
    total_hard_mn = 0.0
    total_soft_mn = 0.0
    bp_mean_sum = 0.0
    bp_std_sum = 0.0
    type_bp: Dict[str, float] = {"cjk": 0.0, "utf8_cont": 0.0, "ascii": 0.0, "op": 0.0, "digit": 0.0}
    n_eval = 0

    with torch.no_grad():
        for src, tgt in eval_loader:
            if n_eval >= 20:
                break
            src = src.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                result = model(src)
            logits = result[0]
            aux = result[1]

            total_acc += recon_acc(logits, src)
            total_recon_loss += criterion(logits.view(-1, logits.size(-1)), src.view(-1)).item()

            if isinstance(aux, dict):
                total_hard_mn += float(aux.get("hard_m_over_n", 0))
                total_soft_mn += float(aux.get("soft_m_over_n", 0))
                bp = aux.get("boundary_probs")
                if bp is not None:
                    bp_mean_sum += bp.mean().item()
                    bp_std_sum += bp.std().item()
                for k in type_bp:
                    v = aux.get(f"{k}_bp_mean", float("nan"))
                    if v == v:
                        type_bp[k] += v
            n_eval += 1

    duration = time.time() - t0
    n = max(1, n_eval)

    return AblationResult(
        name=ab_cfg.name, model_type=ab_cfg.model_type, steps=total_steps,
        recon_acc=total_acc / n, recon_loss=total_recon_loss / n,
        hard_mn=total_hard_mn / n, soft_mn=total_soft_mn / n,
        bp_mean=bp_mean_sum / n, bp_std=bp_std_sum / n,
        cjk_bp=type_bp["cjk"] / n, utf8_bp=type_bp["utf8_cont"] / n,
        ascii_bp=type_bp["ascii"] / n, op_bp=type_bp["op"] / n,
        digit_bp=type_bp["digit"] / n, num_params=n_params, duration_s=duration,
    )


# ---------------------------------------------------------------------------
# Ablation Plan Builders
# ---------------------------------------------------------------------------

def build_loss_ablations() -> List[AblationConfig]:
    """Build list of single-term removal ablations."""
    base = AblationConfig(name="full", description="All 5 boundary loss terms")
    return [
        base,
        AblationConfig(name="no_type", description="Remove type-conditional MSE",
                       lambda_var=0.5, lambda_entropy=0.05, lambda_utf8=0.02, lambda_type=0.0),
        AblationConfig(name="no_utf8", description="Remove UTF-8 continuation penalty",
                       lambda_var=0.5, lambda_entropy=0.05, lambda_utf8=0.0, lambda_type=0.05),
        AblationConfig(name="no_entropy", description="Remove binary entropy polarization",
                       lambda_var=0.5, lambda_entropy=0.0, lambda_utf8=0.02, lambda_type=0.05),
        AblationConfig(name="no_var", description="Remove variance bonus",
                       lambda_var=0.0, lambda_entropy=0.05, lambda_utf8=0.02, lambda_type=0.05),
        AblationConfig(name="pure_recon", description="No boundary loss at all",
                       lambda_var=0.0, lambda_entropy=0.0, lambda_utf8=0.0, lambda_type=0.0,
                       compression_weight=0.0),
    ]


def build_compression_sweep(values: List[float]) -> List[AblationConfig]:
    return [
        AblationConfig(name=f"comp_w={w}", description=f"compression_weight={w}",
                       compression_weight=w)
        for w in values
    ]


def build_target_sweep(values: List[float]) -> List[AblationConfig]:
    return [
        AblationConfig(name=f"target={t}", description=f"target_compression={t}",
                       target_compression=t)
        for t in values
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablations(args: argparse.Namespace) -> List[AblationResult]:
    from flued.data import ByteReconstructionDataset, safe_train_eval_split

    preset = ABLATION_PRESETS.get(args.preset, ABLATION_PRESETS["quick"])

    # Data
    texts: Optional[List[str]] = None
    if args.data_path:
        max_lines = getattr(args, "max_lines", None)
        with open(args.data_path, encoding="utf-8") as fh:
            if max_lines:
                texts = [line.rstrip("\n") for i, line in enumerate(fh) if i < max_lines]
            else:
                texts = [line.rstrip("\n") for line in fh]
        logger.info("Loaded %d lines", len(texts))

    dataset = ByteReconstructionDataset(texts=texts, seq_len=preset["seq_len"], stride=preset["stride"])
    train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.1, seed=42)

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=preset["batch_size"], shuffle=shuffle,
                          drop_last=len(ds) > preset["batch_size"],
                          pin_memory=(preset.get("device", "cuda") == "cuda"))

    train_loader = _loader(train_ds, True)
    eval_loader = _loader(eval_ds, False)

    # Build ablation list
    ablations: List[AblationConfig] = []
    if args.ablations:
        name_map = {a.name: a for a in build_loss_ablations()}
        for name in args.ablations.split(","):
            name = name.strip()
            if name in name_map:
                ablations.append(name_map[name])
            else:
                logger.warning("Unknown ablation: %s", name)

    if args.sweep:
        values = [float(v) for v in args.values.split(",")]
        if args.sweep == "compression_weight":
            ablations.extend(build_compression_sweep(values))
        elif args.sweep == "target_compression":
            ablations.extend(build_target_sweep(values))

    if not ablations:
        ablations = build_loss_ablations()

    logger.info("Running %d ablation(s)", len(ablations))

    # Run
    results = []
    for ab_cfg in ablations:
        logger.info("=== %s: %s ===", ab_cfg.name, ab_cfg.description)
        result = run_single_ablation(ab_cfg, preset, train_loader, eval_loader, args.base_ckpt)
        results.append(result)
        logger.info(
            "  acc=%.4f  hard_mn=%.3f  bp_std=%.3f  cjk=%.3f  utf8=%.3f  ascii=%.3f  %ds",
            result.recon_acc, result.hard_mn, result.bp_std,
            result.cjk_bp, result.utf8_bp, result.ascii_bp, int(result.duration_s),
        )

    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="FLUED E3 Ablation Framework")
    parser.add_argument("--preset", choices=list(ABLATION_PRESETS), default="quick")
    parser.add_argument("--base-ckpt", default=None, help="Base checkpoint for FLUED ablations")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-lines", type=int, default=None)

    # Ablation selection
    parser.add_argument("--ablations", default=None,
                        help="Comma-separated: full,no_type,no_utf8,no_entropy,no_var,pure_recon")
    parser.add_argument("--sweep", default=None,
                        choices=["compression_weight", "target_compression"])
    parser.add_argument("--values", default="0.05,0.1,0.2,0.5,1.0",
                        help="Comma-separated sweep values")

    # Output
    parser.add_argument("--output-json", default="results/e3_ablation.json")
    parser.add_argument("--output-csv", default="results/e3_ablation.csv")
    return parser.parse_args()


def _write_results(results: List[AblationResult], json_path: str, csv_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    # JSON
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([{
            "name": r.name, "model_type": r.model_type, "steps": r.steps,
            "recon_acc": r.recon_acc, "recon_loss": r.recon_loss,
            "hard_mn": r.hard_mn, "soft_mn": r.soft_mn,
            "bp_mean": r.bp_mean, "bp_std": r.bp_std,
            "cjk_bp": r.cjk_bp, "utf8_bp": r.utf8_bp, "ascii_bp": r.ascii_bp,
            "op_bp": r.op_bp, "digit_bp": r.digit_bp,
            "num_params": r.num_params, "duration_s": r.duration_s,
            "status": r.status,
        } for r in results], fh, indent=2)
    logger.info("JSON → %s", json_path)

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "model_type", "recon_acc", "hard_mn", "soft_mn",
                          "bp_std", "cjk_bp", "utf8_bp", "ascii_bp", "op_bp", "digit_bp",
                          "num_params", "duration_s", "status"])
        for r in results:
            writer.writerow([r.name, r.model_type, f"{r.recon_acc:.4f}",
                              f"{r.hard_mn:.4f}", f"{r.soft_mn:.4f}",
                              f"{r.bp_std:.4f}", f"{r.cjk_bp:.4f}",
                              f"{r.utf8_bp:.4f}", f"{r.ascii_bp:.4f}",
                              f"{r.op_bp:.4f}", f"{r.digit_bp:.4f}",
                              r.num_params, f"{r.duration_s:.1f}", r.status])
    logger.info("CSV → %s", csv_path)


if __name__ == "__main__":
    args = _parse_args()
    results = run_ablations(args)
    _write_results(results, args.output_json, args.output_csv)

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Ablation':<20} {'Acc':>8} {'hard_mn':>8} {'bp_std':>8} {'cjk':>8} {'utf8':>8} {'ascii':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r.name:<20} {r.recon_acc:>8.4f} {r.hard_mn:>8.4f} {r.bp_std:>8.4f} "
              f"{r.cjk_bp:>8.4f} {r.utf8_bp:>8.4f} {r.ascii_bp:>8.4f}")
    print("=" * 90)
