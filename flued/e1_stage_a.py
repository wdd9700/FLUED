"""
FLUED E1 Stage A Runner — DSC denoising reconstruction validation.

Goal
----
Verify that FLUEDAutoencoder can compress a clean or corrupted byte sequence
into dynamic semantic units and reconstruct the clean input.

    clean x -> corrupt spans with MASK_ID -> DSC(x_corrupt) -> DSC⁻¹(Z, spans) -> x̂
    loss = CrossEntropy(x̂, x_clean) + boundary losses + optional latent consistency

E1 does NOT test generation or downstream tasks.  It tests whether the
tied-weight inverse can recover clean bytes while the encoder is forced to
use context under span corruption.

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
    --max-compression   (default 1.0)   — m/n must be ≤ this

Exit code: 0 = PASS, 1 = FAIL.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
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


def corrupt_byte_inputs(
    src: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_id: int,
    corrupt_rate: float,
    span_mask_prob: float,
    span_min: int,
    span_max: int,
) -> torch.Tensor:
    corrupted = src.clone()
    if corrupt_rate <= 0:
        return corrupted

    bsz, _ = src.shape
    span_min = max(1, span_min)
    span_max = max(span_min, span_max)
    for b in range(bsz):
        valid_len = int(valid_mask[b].sum().item())
        if valid_len <= 0:
            continue
        budget = max(1, int(valid_len * corrupt_rate))
        while budget > 0:
            if torch.rand((), device=src.device).item() < span_mask_prob:
                span_len = int(torch.randint(span_min, span_max + 1, (), device=src.device).item())
            else:
                span_len = 1
            span_len = min(span_len, valid_len, budget)
            start = int(torch.randint(0, max(valid_len - span_len, 0) + 1, (), device=src.device).item())
            corrupted[b, start : start + span_len] = mask_id
            budget -= span_len
    return corrupted


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
    # L20 48 GB — larger batch to leverage extra VRAM
    "class300m_48gb": {
        "d_model": 1024,
        "nhead": 16,
        "dim_feedforward": 4096,
        "num_layers": 24,
        "max_seq_len": 512,
        "dropout": 0.0,
        "batch_size": 8,
        "max_steps": 50000,
        "lr": 3e-5,
        "warmup_steps": 500,
        "grad_accum_steps": 4,
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
    from flued.data import ByteReconstructionDataset, StreamingReconstructionDataset, safe_train_eval_split
    from flued.model import FLUEDAutoencoder, MASK_ID

    # --- Reproducibility ---
    seed: Optional[int] = getattr(args, "seed", None)
    if seed is not None:
        import random
        import numpy as np
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        logger.info("Random seed set to %d (torch + random + numpy)", seed)

    # --- Log file (persist even without shell tee) ---
    ckpt_dir: str = getattr(args, "ckpt_dir", "checkpoints")
    log_name = f"e1_{getattr(args, 'preset', 'custom')}.log"
    log_path = os.path.join(ckpt_dir, log_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("Log file: %s", log_path)

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
        swiglu_hidden=args.swiglu_hidden,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        assignment_window=args.assignment_window,
        dropout=args.dropout,
        boundary_threshold=args.boundary_threshold,
        boundary_temperature=args.boundary_temperature,
        target_compression=args.target_compression,
        compression_weight=args.compression_weight,
        min_boundary_units=args.min_boundary_units,
        lambda_var=args.lambda_var,
        lambda_entropy=args.lambda_entropy,
        lambda_utf8=args.lambda_utf8,
        lambda_cjk=args.lambda_cjk,
        cjk_target=args.cjk_target,
        lambda_type=args.lambda_type,
    ).to(device)

    n_params = model.count_parameters()
    logger.info(
        "E1 model: d=%d  layers=%d  params=%s  device=%s",
        args.d_model, args.num_layers, f"{n_params:,}", device
    )

    # --- Dataset ---
    texts: Optional[List[str]] = None
    streaming = False
    if args.data_path:
        max_lines: Optional[int] = getattr(args, "max_lines", None)
        file_size = os.path.getsize(args.data_path)
        if max_lines is None and file_size > 256 * 1024 * 1024:
            # File > 256 MB and no line cap → use mmap streaming
            streaming = True
            logger.info(
                "Using STREAMING mode (file=%.1f GB, no max-lines). "
                "Random mmap chunks, no full-file load.",
                file_size / 1e9,
            )
        else:
            with open(args.data_path, encoding="utf-8") as fh:
                if max_lines is not None:
                    texts = []
                    for line in fh:
                        texts.append(line.rstrip("\n"))
                        if len(texts) >= max_lines:
                            break
                else:
                    texts = [line.rstrip("\n") for line in fh]
            logger.info(
                "Loaded %d lines from %s%s",
                len(texts), args.data_path,
                f" (capped at {max_lines})" if max_lines is not None else "",
            )

    if streaming:
        dataset = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=2500,
            seed=args.seed if args.seed is not None else 42,
        )
        train_ds, eval_ds = dataset, dataset  # IterableDataset; split handled differently
        total_samples = dataset.samples_per_worker * 4  # 4 workers
        logger.info(
            "Dataset: ~%d train / ~%d eval chunks (streaming, random mmap)",
            int(total_samples * 0.9), int(total_samples * 0.1),
        )
    else:
        dataset = ByteReconstructionDataset(
            texts=texts,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.1, seed=42)
        logger.info("Dataset: %d train / %d eval chunks", len(train_ds), len(eval_ds))

    def _make_loader(ds, shuffle: bool) -> DataLoader:
        kwargs = dict(
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=(device_str == "cuda"),
            num_workers=4,
            persistent_workers=True,
        )
        if streaming:
            # IterableDataset: shuffle is done by random sampling; can't drop_last by len()
            kwargs["shuffle"] = False
        else:
            kwargs["drop_last"] = len(ds) > args.batch_size  # type: ignore[arg-type]
        return DataLoader(ds, **kwargs)

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

    # --- Checkpoint helpers ---
    ckpt_dir: str = getattr(args, "ckpt_dir", "checkpoints")
    ckpt_every: int = getattr(args, "ckpt_every", 500)

    def _save_ckpt(step: int) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        state = {
            "global_step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "amp_dtype": getattr(args, "amp_dtype", "bf16"),
            "model_config": {
                "vocab_size": model.vocab_size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "dim_feedforward": args.dim_feedforward,
                "swiglu_hidden": args.swiglu_hidden,
                "num_layers": args.num_layers,
                "max_seq_len": args.max_seq_len,
                "assignment_window": args.assignment_window,
                "target_compression": args.target_compression,
                "compression_weight": args.compression_weight,
                "min_boundary_units": args.min_boundary_units,
            },
        }
        # Atomic save: write to temp first, then rename (avoids corruption on
        # disk-full or I/O errors that have crashed training twice at ckpt writes).
        latest_path = os.path.join(ckpt_dir, "e1_latest.pt")
        step_path = os.path.join(ckpt_dir, f"e1_step{step:05d}.pt")
        for path in (latest_path, step_path):
            tmp = path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, path)  # atomic on same filesystem
        logger.info("Checkpoint saved → %s + %s", latest_path, step_path)

    # --- Resume from checkpoint ---
    global_step = 0
    resume_path: Optional[str] = getattr(args, "resume", None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp and amp_dtype == torch.float16:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt["global_step"]
        logger.info("Resumed from %s at step %d", resume_path, global_step)

    # --- Training loop ---
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_recon_loss = 0.0
    running_comp_loss = 0.0
    running_latent_loss = 0.0
    running_denoise = 0.0
    running_acc = 0.0
    running_soft_mon = 0.0
    running_hard_mon = 0.0
    running_num_units = 0.0
    running_bp_mean = 0.0
    running_bp_std = 0.0
    running_grad_norm = 0.0
    running_hmn55 = 0.0  # hard_m/n @ threshold 0.55
    running_hmn60 = 0.0  # hard_m/n @ threshold 0.60
    running_hmn65 = 0.0  # hard_m/n @ threshold 0.65
    skipped_steps = 0   # fp16 GradScaler overflow skips
    running_micro_count = 0  # actual micro-batch count (excludes overflow skips)
    # Per-type boundary prob accumulators (sum, valid-step count)
    _type_keys = ("utf8_cont", "ascii", "cjk", "op", "digit")
    _type_sums: Dict[str, float] = dict.fromkeys(_type_keys, 0.0)
    _type_cnts: Dict[str, int]   = dict.fromkeys(_type_keys, 0)

    accum_steps = 0

    optimizer.zero_grad()

    # Entropy warmup: ramp lambda_entropy from 0 to full over N steps.
    # This prevents early entropy gradient from locking CJK bp above 0.5
    # before the type / utf8 losses have established correct differentiation.
    entropy_warmup_steps: int = getattr(args, "entropy_warmup_steps", 0)
    base_lambda_entropy: float = args.lambda_entropy

    while global_step < args.max_steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)

        src = src.to(device)
        # E1: tgt == src (reconstruction)
        clean_src = src
        pad_mask = clean_src == 0
        valid_mask = ~pad_mask
        use_denoise = torch.rand((), device=device).item() < args.denoise_prob
        model_src = (
            corrupt_byte_inputs(
                clean_src,
                valid_mask,
                mask_id=MASK_ID,
                corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob,
                span_min=args.span_min,
                span_max=args.span_max,
            )
            if use_denoise
            else clean_src
        )

        # --- Entropy warmup: ramp lambda_entropy ---
        if entropy_warmup_steps > 0 and global_step < entropy_warmup_steps:
            model.lambda_entropy = base_lambda_entropy * (global_step / entropy_warmup_steps)
        else:
            model.lambda_entropy = base_lambda_entropy

        with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
            logits, metrics = model(
                model_src,
                src_key_padding_mask=pad_mask,
                boundary_src=clean_src,
                skip_hard=True,
            )
            recon_loss = criterion(logits.view(-1, logits.size(-1)), clean_src.view(-1))
            comp_loss = metrics["compression_loss"]
            latent_loss = logits.new_zeros(())
            if args.latent_consistency_weight > 0 and use_denoise and valid_mask.any():
                with torch.no_grad():
                    clean_expanded, _ = model.encode(
                        clean_src,
                        pad_mask,
                        boundary_src=clean_src,
                        skip_hard=True,
                    )
                latent_loss = torch.nn.functional.mse_loss(metrics["expanded"][valid_mask], clean_expanded[valid_mask])
            total_loss = recon_loss + comp_loss + args.latent_consistency_weight * latent_loss
            loss = total_loss / args.grad_accum_steps

        if getattr(args, "debug_nan", False):
            bp = metrics["boundary_probs"].detach()
            if torch.isnan(bp).any():
                raise FloatingPointError(
                    f"[step {global_step}] boundary_probs contains NaN"
                )
            if torch.isnan(logits).any():
                raise FloatingPointError(
                    f"[step {global_step}] logits contains NaN"
                )
            if torch.isnan(loss):
                raise FloatingPointError(
                    f"[step {global_step}] loss is NaN "
                    f"(recon={recon_loss.item():.4f}, comp={comp_loss.item():.4f}, latent={latent_loss.item():.4f})"
                )

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_steps += 1
        running_loss += total_loss.item()
        running_recon_loss += recon_loss.item()
        running_comp_loss += comp_loss.item()
        running_latent_loss += latent_loss.item()
        running_denoise += float(use_denoise)
        running_acc += reconstruction_accuracy(logits.detach(), clean_src)
        running_soft_mon += metrics["soft_m_over_n"].item()
        running_hard_mon += float(metrics["hard_m_over_n"])
        running_num_units += float(metrics["num_units"])
        _bp = metrics["boundary_probs"].detach()
        running_bp_mean += _bp.mean().item()
        running_bp_std += _bp.std().item()
        _sweep = metrics.get("hard_mn_sweep", {})
        running_hmn55 += _sweep.get(0.55, 0.0)
        running_hmn60 += _sweep.get(0.60, 0.0)
        running_hmn65 += _sweep.get(0.65, 0.0)
        for _tk in _type_keys:
            _v = metrics.get(f"{_tk}_bp_mean", float("nan"))
            if _v == _v:
                _type_sums[_tk] += _v
                _type_cnts[_tk] += 1
        running_micro_count += 1  # actual micro-batch count (excludes overflow skips)

        if accum_steps < args.grad_accum_steps:
            continue

        # Gradient step
        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp and amp_dtype == torch.float16:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_steps += 1
                # Undo this micro-batch's metrics (they won't be counted)
                running_loss -= total_loss.item()
                running_recon_loss -= recon_loss.item()
                running_comp_loss -= comp_loss.item()
                running_latent_loss -= latent_loss.item()
                running_denoise -= float(use_denoise)
                running_acc -= reconstruction_accuracy(logits.detach(), clean_src)
                running_soft_mon -= metrics["soft_m_over_n"].item()
                running_hard_mon -= float(metrics["hard_m_over_n"])
                running_num_units -= float(metrics["num_units"])
                running_bp_mean -= _bp.mean().item()
                running_bp_std -= _bp.std().item()
                running_hmn55 -= _sweep.get(0.55, 0.0)
                running_hmn60 -= _sweep.get(0.60, 0.0)
                running_hmn65 -= _sweep.get(0.65, 0.0)
                for _tk in _type_keys:
                    _v = metrics.get(f"{_tk}_bp_mean", float("nan"))
                    if _v == _v:
                        _type_sums[_tk] -= _v
                        _type_cnts[_tk] -= 1
                running_micro_count -= 1
                optimizer.zero_grad()
                accum_steps = 0
                continue
        else:
            optimizer.step()
        scheduler.step()
        # Capture boundary_head grad_norm before zeroing (P0 gate metric)
        if model.boundary_head.weight.grad is not None:
            running_grad_norm += model.boundary_head.weight.grad.norm().item()
        optimizer.zero_grad()
        accum_steps = 0
        global_step += 1

        if global_step % 100 == 0:
            log_n = max(1, running_micro_count)  # actual micro-batch count
            log_g = max(1, running_micro_count // args.grad_accum_steps)  # effective steps
            scaler_scale = scaler.get_scale() if (use_amp and amp_dtype == torch.float16) else 0.0
            logger.info(
                "step=%5d  loss=%.4f  recon=%.4f  comp=%.4f  latent=%.4f  recon_acc=%.4f"
                "  soft_m/n=%.3f  hard_m/n=%.3f  units=%.1f"
                "  bp_mean=%.3f  bp_std=%.3f  bhead_gnorm=%.4f  lr=%.2e"
                "  denoise=%.2f  skip=%d  scale=%.0f"
                "  h@55=%.3f  h@60=%.3f  h@65=%.3f",
                global_step,
                running_loss / log_n,
                running_recon_loss / log_n,
                running_comp_loss / log_n,
                running_latent_loss / log_n,
                running_acc / log_n,
                running_soft_mon / log_n,
                running_hard_mon / log_n,
                running_num_units / log_n,
                running_bp_mean / log_n,
                running_bp_std / log_n,
                running_grad_norm / log_g,
                scheduler.get_last_lr()[0],
                running_denoise / log_n,
                skipped_steps,
                scaler_scale,
                running_hmn55 / log_n,
                running_hmn60 / log_n,
                running_hmn65 / log_n,
            )
            running_loss = running_recon_loss = running_comp_loss = running_latent_loss = running_denoise = 0.0
            running_acc = running_soft_mon = running_hard_mon = 0.0
            running_num_units = running_bp_mean = running_bp_std = running_grad_norm = 0.0
            running_hmn55 = running_hmn60 = running_hmn65 = 0.0
            running_micro_count = 0
            skipped_steps = 0
            # Second log line: per-type boundary prob means
            def _fmt(k: str) -> str:
                n = _type_cnts[k]
                return f"{_type_sums[k]/n:.3f}" if n > 0 else "N/A"
            logger.info(
                "step=%5d  [type_bp]  utf8=%-7s  ascii=%-7s  cjk=%-7s  op=%-7s  digit=%-7s",
                global_step,
                _fmt("utf8_cont"), _fmt("ascii"), _fmt("cjk"), _fmt("op"), _fmt("digit"),
            )
            for _tk in _type_keys:
                _type_sums[_tk] = 0.0
                _type_cnts[_tk] = 0

        if ckpt_every > 0 and global_step % ckpt_every == 0:
            _save_ckpt(global_step)

    # --- Evaluation ---
    model.eval()
    total_acc = 0.0
    total_mon = 0.0
    n_eval = 0
    max_eval_batches: Optional[int] = getattr(args, "max_eval_batches", None)
    with torch.no_grad():
        for src, tgt in eval_loader:
            if max_eval_batches is not None and n_eval >= max_eval_batches:
                break
            src = src.to(device)
            with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
                logits, metrics = model(src, skip_hard=True)
            total_acc += reconstruction_accuracy(logits, src)
            total_mon += float(metrics["m_over_n"])
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
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
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
    parser.add_argument("--swiglu-hidden", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--assignment-window", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-temperature", type=float, default=1.0)
    parser.add_argument("--target-compression", type=float, default=0.3)
    parser.add_argument("--compression-weight", type=float, default=0.1)
    parser.add_argument("--min-boundary-units", type=float, default=1.0)
    parser.add_argument("--lambda-var",          type=float, default=0.0,
                        help="variance bonus weight (encourage bp spread)")
    parser.add_argument("--lambda-entropy",      type=float, default=0.0,
                        help="binary entropy weight (polarize bp toward 0/1)")
    parser.add_argument("--lambda-utf8",         type=float, default=0.0,
                        help="UTF-8 continuation penalty weight")
    parser.add_argument("--lambda-cjk",          type=float, default=0.0,
                        help="CJK-specific boundary BCE prior weight (target=cjk-target)")
    parser.add_argument("--cjk-target",          type=float, default=0.16,
                        help="Target boundary probability for CJK lead bytes")
    parser.add_argument("--lambda-type",         type=float, default=0.0,
                        help="Type-conditional BCE prior weight. When >0, averages BCE("
                             "p[type_mask], target_type) over the 5 byte types "
                             "(cjk_lead, utf8_cont, alpha, digit, operator). "
                             "Replaces the long-claimed but never-implemented "
                             "type_prior; targets motivated by initial cjk=0.058 run.")
    parser.add_argument(
        "--entropy-warmup-steps", type=int, default=0,
        help="Linearly ramp lambda_entropy from 0→full over N steps. "
             "Gives type/utf8 losses time to establish correct byte-type "
             "differentiation before entropy polarises (prevents CJK bp "
             "from being locked above 0.5 by early entropy gradient)."
    )
    parser.add_argument(
        "--no-boundary-reg", action="store_true",
        help="Ablation convenience: zero out ALL boundary regularization "
             "(lambda_var, lambda_entropy, lambda_utf8, lambda_type). "
             "Equivalent to --lambda-var 0 --lambda-entropy 0 --lambda-utf8 0 --lambda-type 0."
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--device", default=None)

    # Data
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--max-lines", type=int, default=None,
        help="Maximum number of lines to load from data-path (useful for large corpora).",
    )
    parser.add_argument(
        "--max-eval-batches", type=int, default=None,
        help="Cap eval pass at this many batches (speeds up CPU eval on large datasets).",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--denoise-prob", type=float, default=0.7)
    parser.add_argument("--corrupt-rate", type=float, default=0.15)
    parser.add_argument("--span-mask-prob", type=float, default=0.7)
    parser.add_argument("--span-min", type=int, default=1)
    parser.add_argument("--span-max", type=int, default=8)
    parser.add_argument("--latent-consistency-weight", type=float, default=0.03)

    # AMP
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")

    # Evaluation criteria
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--min-compression", type=float, default=0.125)
    parser.add_argument("--max-compression", type=float, default=1.0)

    # Checkpoint
    parser.add_argument(
        "--ckpt-dir", default="checkpoints",
        help="Directory for checkpoint files.",
    )
    parser.add_argument(
        "--ckpt-every", type=int, default=2500,
        help="Save a checkpoint every N gradient steps (0 = disabled).",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a .pt checkpoint to resume training from.",
    )

    # Debug
    parser.add_argument(
        "--debug-nan", action="store_true", default=False,
        help="Raise FloatingPointError immediately when NaN is detected in key tensors."
             " Useful for locating the first NaN step; disables for production.",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (torch + random + numpy).",
    )

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
        ("swiglu_hidden", None),
        ("num_layers", 2), ("max_seq_len", 64), ("assignment_window", 128), ("dropout", 0.0),
        ("batch_size", 4), ("max_steps", 200), ("lr", 3e-4),
        ("warmup_steps", 20), ("grad_accum_steps", 1),
        ("device", "cpu"), ("seq_len", 32), ("stride", 16),
        ("amp", False),
    ]:
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    # --- Ablation convenience ---
    if getattr(args, "no_boundary_reg", False):
        args.lambda_var = 0.0
        args.lambda_entropy = 0.0
        args.lambda_utf8 = 0.0
        args.lambda_type = 0.0
        logger = logging.getLogger("flued.e1")
        logger.info("Ablation: --no-boundary-reg → all boundary lambdas = 0")

    return args


def main() -> None:
    args = _parse_args()
    passed = run_e1(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
