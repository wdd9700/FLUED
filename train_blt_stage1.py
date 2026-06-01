"""
BLT Stage 1 — Pre-train Byte Language Model.

Trains a small autoregressive byte-level LM that will later be frozen
and used for entropy-based patching in the BLT autoencoder (Stage 2).

Usage
-----
    python train_blt_stage1.py --preset blt_small \
        --data-path corpus.txt --max-lines 50000

    # Resume
    python train_blt_stage1.py --preset blt_small \
        --data-path corpus.txt --resume checkpoints/bytel m_step05000.pt

Output
------
    checkpoints/bytel m_step*.pt  — Stage 1 checkpoints
    checkpoints/bytel m_latest.pt — Most recent checkpoint (used by Stage 2)
"""

import argparse
import logging
import math
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("blt.stage1")

from blt_baseline.model import ByteLanguageModel
from flued.data import StreamingReconstructionDataset


# ===========================================================================
# Presets
# ===========================================================================

PRESETS: Dict[str, Dict] = {
    "blt_tiny": {
        "d_model": 128, "nhead": 4, "dim_feedforward": 512,
        "num_layers": 2, "max_seq_len": 256, "dropout": 0.0,
        "batch_size": 32, "max_steps": 500, "lr": 3e-4,
        "warmup_steps": 50, "grad_accum_steps": 1,
        "seq_len": 128, "stride": 64, "device": "cpu",
        "amp": False,
    },
    "blt_small": {
        # ~25M params — fast pre-training, good for quick iteration
        "d_model": 256, "nhead": 4, "dim_feedforward": 1024,
        "num_layers": 4, "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 16, "max_steps": 5000, "lr": 2e-4,
        "warmup_steps": 200, "grad_accum_steps": 1,
        "seq_len": 256, "stride": 128, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
    "blt_300m_local": {
        # ~85M param local LM — matches the scale used in BLT paper
        # d=512, 4 layers — produces high-quality entropy estimates
        "d_model": 512, "nhead": 8, "dim_feedforward": 2048,
        "num_layers": 4, "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 4, "max_steps": 20000, "lr": 1e-4,
        "warmup_steps": 500, "grad_accum_steps": 4,
        "seq_len": 512, "stride": 256, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
    "blt_300m_streaming": {
        # ~12.9M param ByteLM — streaming mmap over 22 GB corpus
        # 100K steps, full data diversity, no OOM risk
        "d_model": 512, "nhead": 8, "dim_feedforward": 2048,
        "num_layers": 4, "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 4, "max_steps": 100000, "lr": 1e-4,
        "warmup_steps": 1000, "grad_accum_steps": 4,
        "seq_len": 512, "stride": 0, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
    "blt_100m": {
        # ~100M param ByteLM — matches BLT paper §4.2
        # "100M parameters, 14 layers, hidden dimensionality 512"
        # dim_feedforward=6144 to hit ~100M at d=512/14 layers
        "d_model": 512, "nhead": 8, "dim_feedforward": 6144,
        "num_layers": 14, "max_seq_len": 512, "dropout": 0.0,
        "batch_size": 2, "max_steps": 100000, "lr": 1e-4,
        "warmup_steps": 1000, "grad_accum_steps": 8,
        "seq_len": 512, "stride": 0, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
}


# ===========================================================================
# Dataset
# ===========================================================================

class ByteLMDataset(torch.utils.data.Dataset):
    """Byte-level language modeling dataset.

    Chunks raw bytes (PAD-offset encoded) into fixed-length sequences.
    Target is input shifted by 1 (next-byte prediction).
    """

    def __init__(self, texts, seq_len: int = 256, stride: int = 128):
        self.seq_len = seq_len

        # Concatenate all bytes with EOS-like terminator
        all_bytes = []
        for text in texts:
            if not text:
                continue
            b = list(text.encode("utf-8"))
            all_bytes.extend(b + 1 for b in b)  # PAD-offset
            all_bytes.append(0)  # zero-byte as separator

        if not all_bytes:
            all_bytes = [0]

        self.data = torch.tensor(all_bytes, dtype=torch.long)
        self.chunks = []
        for start in range(0, max(1, len(self.data) - seq_len), max(1, stride)):
            chunk = self.data[start:start + seq_len + 1]  # +1 for target
            if chunk.numel() < seq_len + 1:
                pad = torch.zeros(seq_len + 1 - chunk.numel(), dtype=torch.long)
                chunk = torch.cat([chunk, pad])
            self.chunks.append(chunk)

        if not self.chunks:
            self.chunks = [torch.zeros(seq_len + 1, dtype=torch.long)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:self.seq_len], chunk[1:self.seq_len + 1]  # (src, tgt)


def load_texts(data_path: str, max_lines: Optional[int] = None):
    with open(data_path, encoding="utf-8") as fh:
        if max_lines:
            texts = []
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                line = line.rstrip("\n")
                if line.strip():
                    texts.append(line)
            return texts
        return [line.rstrip("\n") for line in fh if line.strip()]


def train_eval_split(dataset, eval_fraction=0.1, seed=42):
    n = len(dataset)
    n_eval = max(1, int(n * eval_fraction))
    n_train = max(1, n - n_eval)
    gen = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n, generator=gen).tolist()
    return (torch.utils.data.Subset(dataset, sorted(shuffled[:n_train])),
            torch.utils.data.Subset(dataset, sorted(shuffled[n_train:])))


# ===========================================================================
# Helpers
# ===========================================================================

def cosine_schedule(optimizer, warmup: int, total: int):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def lm_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != 0
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return ((preds == targets) & mask).sum().item() / total


# ===========================================================================
# Training
# ===========================================================================

def train(args):
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # Model
    model = ByteLanguageModel(
        vocab_size=257, d_model=args.d_model, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, num_layers=args.num_layers,
        max_len=args.max_seq_len, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("ByteLM: d=%d layers=%d params=%s", args.d_model, args.num_layers, f"{n_params:,}")

    # Data — streaming mmap over full 22 GB corpus (no RAM OOM)
    train_ds = StreamingReconstructionDataset(
        args.data_path, seq_len=args.seq_len,
        samples_per_worker=5000, seed=42,
    )
    eval_ds = StreamingReconstructionDataset(
        args.data_path, seq_len=args.seq_len,
        samples_per_worker=500, seed=999,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=4, pin_memory=(device_str == "cuda"),
        # IterableDataset: shuffle=False (randomness built-in via mmap seek)
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size,
        num_workers=2, pin_memory=(device_str == "cuda"),
    )
    logger.info("Streaming dataset: mmap over %s → infinite random byte chunks", args.data_path)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = cosine_schedule(optimizer, args.warmup_steps, args.max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # AMP
    use_amp = args.amp and device_str == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    accum = args.grad_accum_steps

    # Checkpoint
    ckpt_dir = getattr(args, "ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_ckpt(step):
        state = {"global_step": step, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": {"d_model": args.d_model, "nhead": args.nhead,
                            "dim_feedforward": args.dim_feedforward, "num_layers": args.num_layers,
                            "dropout": args.dropout}}
        torch.save(state, os.path.join(ckpt_dir, f"bytel m_step{step:05d}.pt"))
        torch.save(state, os.path.join(ckpt_dir, "bytel m_latest.pt"))
        logger.info("Checkpoint saved → step=%d", step)

    # Resume
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt["global_step"]
        logger.info("Resumed from %s at step %d", args.resume, global_step)

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_acc = 0.0
    running_entropy = 0.0
    skipped = 0
    grad_step = 0
    optimizer.zero_grad()
    t_start = time.time()

    log_every = getattr(args, "log_interval", 50)
    ckpt_every = getattr(args, "ckpt_every", 1000)

    while global_step < args.max_steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)
        src, tgt = src.to(device), tgt.to(device)

        with torch.autocast(device_type=device_str, dtype=amp_dtype, enabled=use_amp):
            hidden, logits = model(src)
            loss = criterion(logits.view(-1, 257), tgt.view(-1)) / accum

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_step += 1
        running_loss += loss.item() * accum
        running_acc += lm_accuracy(logits.detach(), tgt)

        with torch.no_grad():
            # Force FP32 for entropy computation to avoid FP16 underflow
            _logits = logits.detach().float()
            probs = torch.softmax(_logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
            running_entropy += ent[tgt != 0].mean().item()

        if grad_step < accum:
            continue

        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp and amp_dtype == torch.float16:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
                optimizer.zero_grad()
                grad_step = 0
                continue
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        grad_step = 0
        global_step += 1

        if global_step % log_every == 0:
            n = log_every * accum
            elapsed = time.time() - t_start
            logger.info(
                "step=%5d  loss=%.4f  acc=%.4f  entropy=%.3f  lr=%.2e  skip=%d  %.1f step/min",
                global_step, running_loss / n, running_acc / n, running_entropy / n,
                scheduler.get_last_lr()[0], skipped,
                global_step / max(1, elapsed) * 60,
            )
            running_loss = running_acc = running_entropy = 0.0
            skipped = 0

        if ckpt_every > 0 and global_step % ckpt_every == 0:
            save_ckpt(global_step)

    # Final
    save_ckpt(global_step)
    logger.info("Stage 1 complete: %d steps, %.1f min", global_step, (time.time() - t_start) / 60)

    # Quick eval
    model.eval()
    total_acc = 0.0
    total_entropy = 0.0
    n_eval = 0
    with torch.no_grad():
        for src, tgt in eval_loader:
            if n_eval >= 50:
                break
            src, tgt = src.to(device), tgt.to(device)
            _, logits = model(src)
            total_acc += lm_accuracy(logits, tgt)
            probs = torch.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(-1)
            total_entropy += ent[tgt != 0].mean().item()
            n_eval += 1
    logger.info("Eval: acc=%.4f  entropy=%.3f", total_acc / max(1, n_eval), total_entropy / max(1, n_eval))


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="BLT Stage 1 — Pre-train Byte LM")
    parser.add_argument("--preset", choices=list(PRESETS), default="blt_small")

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

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
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()
    defaults = PRESETS.get(args.preset, PRESETS["blt_small"]).copy()
    for k, v in defaults.items():
        attr = k.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, v)

    for attr in ["d_model", "nhead", "dim_feedforward", "num_layers", "max_seq_len",
                 "dropout", "batch_size", "max_steps", "lr", "warmup_steps",
                 "grad_accum_steps", "device", "seq_len", "stride"]:
        if getattr(args, attr, None) is None:
            setattr(args, attr, PRESETS["blt_small"][attr])

    return args


if __name__ == "__main__":
    train(parse_args())
