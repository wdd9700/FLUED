"""
BLT Stage 2 — Train Global Transformer + Decoder with Frozen ByteLM.

Faithful reproduction of Pagnoni et al. (2024):
  Stage 1: Pre-train small ByteLM  ->  python train_blt_stage1.py
  Stage 2: Freeze ByteLM, train Global TF + Decoder  ->  python train_blt.py

Usage
-----
    python train_blt.py --preset 300m_frozen \
        --local-lm-ckpt 'checkpoints/bytel m_latest.pt' \
        --data-path corpus.txt --max-lines 50000
"""

import argparse
import logging
import math
import os
import sys
import time as _time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("blt.stage2")

from blt_baseline.model import BLTAutoencoder, ByteLanguageModel


# ===========================================================================
# Presets
# ===========================================================================

PRESETS: Dict[str, Dict] = {
    "smoke": {
        "d_model": 64, "nhead": 4, "dim_feedforward": 128,
        "global_layers": 1, "decoder_layers": 1, "local_lm_d_model": 64,
        "max_seq_len": 32, "dropout": 0.0,
        "batch_size": 4, "max_steps": 100, "lr": 3e-4,
        "warmup_steps": 10, "grad_accum_steps": 1,
        "seq_len": 32, "stride": 16, "device": "cpu", "amp": False,
    },
    "300m_frozen": {
        "d_model": 1024, "nhead": 16, "dim_feedforward": 4096,
        "global_layers": 12, "decoder_layers": 12, "local_lm_d_model": 512,
        "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 1, "max_steps": 40000, "lr": 3e-5,
        "warmup_steps": 500, "grad_accum_steps": 16,
        "seq_len": 512, "stride": 256, "device": "cuda",
        "amp": True, "amp_dtype": "fp16", "entropy_theta": 3.5,
    },
    "300m_joint": {
        "d_model": 1024, "nhead": 16, "dim_feedforward": 4096,
        "global_layers": 10, "decoder_layers": 12, "local_lm_d_model": 1024,
        "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 1, "max_steps": 40000, "lr": 3e-5,
        "warmup_steps": 500, "grad_accum_steps": 16,
        "seq_len": 512, "stride": 256, "device": "cuda",
        "amp": True, "amp_dtype": "fp16", "entropy_theta": 3.5,
    },
}


# ===========================================================================
# Helpers
# ===========================================================================

def cosine_schedule(optimizer, warmup: int, total: int):
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def recon_acc(logits, targets):
    preds = logits.argmax(dim=-1)
    mask = targets != 0
    total = mask.sum().item()
    if total == 0: return 0.0
    return ((preds == targets) & mask).sum().item() / total


# ===========================================================================
# Data
# ===========================================================================

def load_texts(data_path, max_lines=None):
    with open(data_path, encoding="utf-8") as fh:
        if max_lines:
            texts = []
            for i, line in enumerate(fh):
                if i >= max_lines: break
                line = line.rstrip("\n")
                if line.strip(): texts.append(line)
            return texts
        return [line.rstrip("\n") for line in fh if line.strip()]


class ByteChunkDataset(torch.utils.data.Dataset):
    def __init__(self, texts, seq_len=512, stride=256):
        self.seq_len = seq_len
        all_bytes = []
        for text in texts:
            if not text: continue
            all_bytes.extend(b + 1 for b in text.encode("utf-8"))
            all_bytes.append(0)
        if not all_bytes: all_bytes = [0]
        self.data = torch.tensor(all_bytes, dtype=torch.long)
        self.chunks = []
        for start in range(0, max(1, len(self.data) - seq_len + 1), max(1, stride)):
            chunk = self.data[start:start + seq_len]
            if chunk.numel() < seq_len:
                chunk = torch.cat([chunk, torch.zeros(seq_len - chunk.numel(), dtype=torch.long)])
            self.chunks.append(chunk)
        if not self.chunks: self.chunks = [torch.zeros(seq_len, dtype=torch.long)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return c, c.clone()


def split_dataset(dataset, eval_frac=0.1, seed=42):
    n = len(dataset)
    n_eval = max(1, int(n * eval_frac))
    n_train = max(1, n - n_eval)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()
    return (torch.utils.data.Subset(dataset, sorted(idx[:n_train])),
            torch.utils.data.Subset(dataset, sorted(idx[n_train:])))


# ===========================================================================
# Training
# ===========================================================================

def train(args):
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # --- Load pre-trained ByteLM ---
    local_lm = None
    local_lm_ckpt = getattr(args, "local_lm_ckpt", None)
    if local_lm_ckpt and os.path.exists(local_lm_ckpt):
        ckpt = torch.load(local_lm_ckpt, map_location="cpu")
        cfg = ckpt.get("config", {})
        local_lm = ByteLanguageModel(
            vocab_size=257,
            d_model=cfg.get("d_model", args.local_lm_d_model),
            nhead=cfg.get("nhead", 8),
            dim_feedforward=cfg.get("dim_feedforward", 2048),
            num_layers=cfg.get("num_layers", 4),
            max_len=args.max_seq_len, dropout=cfg.get("dropout", 0.0),
        )
        local_lm.load_state_dict(ckpt["model"])
        logger.info("Loaded frozen ByteLM: d=%d layers=%d step=%d",
                     local_lm.d_model, cfg.get("num_layers", "?"), ckpt["global_step"])
    else:
        logger.info("No ByteLM checkpoint — joint training mode")

    # --- Build BLT Autoencoder ---
    model = BLTAutoencoder(
        vocab_size=257, d_model=args.d_model, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        global_layers=args.global_layers, decoder_layers=args.decoder_layers,
        local_lm=local_lm, local_lm_d_model=args.local_lm_d_model,
        patch_mode="entropy", entropy_theta=getattr(args, "entropy_theta", 3.5),
        max_seq_len=args.max_seq_len, dropout=args.dropout,
    ).to(device)

    if local_lm is not None:
        model.freeze_local_lm()
        logger.info("Local ByteLM FROZEN")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Model: d_global=%d d_local=%d global_l=%d dec_l=%d trainable=%s total=%s",
                args.d_model, args.local_lm_d_model, args.global_layers, args.decoder_layers,
                f"{n_trainable:,}", f"{n_total:,}")

    # --- Data ---
    texts = load_texts(args.data_path, getattr(args, "max_lines", None)) if args.data_path else ["hello"]
    logger.info("Loaded %d lines", len(texts))
    dataset = ByteChunkDataset(texts, seq_len=args.seq_len, stride=args.stride)
    train_ds, eval_ds = split_dataset(dataset, eval_frac=0.1, seed=42)
    logger.info("Dataset: %d train / %d eval (seq_len=%d)", len(train_ds), len(eval_ds), args.seq_len)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          drop_last=len(ds) > args.batch_size,
                          pin_memory=(device_str == "cuda"))
    train_loader = make_loader(train_ds, True)
    eval_loader = make_loader(eval_ds, False)

    # --- Optimizer ---
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-2)
    scheduler = cosine_schedule(optimizer, args.warmup_steps, args.max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # --- AMP ---
    use_amp = args.amp and device_str == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    accum = args.grad_accum_steps

    # --- Checkpoint ---
    ckpt_dir = getattr(args, "ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_every = getattr(args, "ckpt_every", 500)

    def save_ckpt(step):
        state = {"global_step": step, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict()}
        latest = os.path.join(ckpt_dir, "blt_latest.pt")
        tmp = latest + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, latest)
        if getattr(args, "save_step_ckpts", False):
            torch.save(state, os.path.join(ckpt_dir, f"blt_step{step:05d}.pt"))
        logger.info("Checkpoint -> step=%d latest=%s", step, latest)

    # --- Resume ---
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp and ckpt["scaler"]: scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt["global_step"]
        if local_lm is not None: model.freeze_local_lm()
        logger.info("Resumed from %s at step %d", args.resume, global_step)

    # --- Training Loop ---
    model.train()
    if local_lm is not None: model.local_lm.eval()
    train_iter = iter(train_loader)
    running_loss, running_acc, running_patches = 0.0, 0.0, 0.0
    skipped, grad_step = 0, 0
    optimizer.zero_grad()
    t_start = _time.time()
    log_every = getattr(args, "log_interval", 50)

    while global_step < args.max_steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)
        src = src.to(device)

        with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
            logits, metrics = model(src)
            loss = criterion(logits.view(-1, 257), src.view(-1)) / accum

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_step += 1
        running_loss += loss.item() * accum
        running_acc += recon_acc(logits.detach(), src)
        running_patches += metrics.get("avg_num_patches", 0)

        if grad_step < accum: continue

        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp and amp_dtype == torch.float16:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1; optimizer.zero_grad(); grad_step = 0; continue
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        grad_step = 0
        global_step += 1

        if global_step % log_every == 0:
            n = log_every * accum
            elapsed = _time.time() - t_start
            s = scaler.get_scale() if (use_amp and amp_dtype == torch.float16) else 0.0
            logger.info(
                "step=%5d  loss=%.4f  acc=%.4f  avg_patches=%.1f  lr=%.2e  skip=%d  scale=%.0f  %.1f step/min",
                global_step, running_loss / n, running_acc / n, running_patches / n,
                scheduler.get_last_lr()[0], skipped, s,
                global_step / max(1, elapsed) * 60,
            )
            running_loss = running_acc = running_patches = 0.0
            skipped = 0

        if ckpt_every > 0 and global_step % ckpt_every == 0:
            save_ckpt(global_step)

    save_ckpt(global_step)

    # --- Eval ---
    model.eval()
    total_acc, total_patches, n_eval = 0.0, 0.0, 0
    max_eval = getattr(args, "max_eval_batches", 50)
    with torch.no_grad():
        for src, tgt in eval_loader:
            if n_eval >= max_eval: break
            src = src.to(device)
            with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
                logits, metrics = model(src)
            total_acc += recon_acc(logits, src)
            total_patches += metrics.get("avg_num_patches", 0)
            n_eval += 1
    logger.info("Eval: acc=%.4f  avg_patches=%.1f  steps=%d  time=%.1f min",
                total_acc / max(1, n_eval), total_patches / max(1, n_eval),
                global_step, (_time.time() - t_start) / 60)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="BLT Stage 2")
    parser.add_argument("--preset", choices=list(PRESETS), default="300m_frozen")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--global-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--local-lm-d-model", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--local-lm-ckpt", default=None)
    parser.add_argument("--entropy-theta", type=float, default=3.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--save-step-ckpts", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()
    defaults = PRESETS.get(args.preset, PRESETS["300m_frozen"]).copy()
    for k, v in defaults.items():
        attr = k.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, v)
    return args


if __name__ == "__main__":
    train(parse_args())
