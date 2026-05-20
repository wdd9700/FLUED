"""
Unified trainer for FLUED Stage A autoencoder experiments.

Supports all three model types (flued, bpe, blt) through the same
Trainer class. Select the model via ModelConfig.model_type.

Usage
-----
    python -m flued.train --model-type flued --size small --max-steps 1000
    python -m flued.train --model-type bpe   --size small
    python -m flued.train --model-type blt   --size small
"""

import logging
import math
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from flued.config import ModelConfig, TrainConfig
from flued.data import (
    STUB_CORPUS,
    BPETextDataset,
    ByteTextDataset,
    SimpleBPE,
    get_dataloader,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("flued.train")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set Python / PyTorch random seeds for experiment reproducibility.

    Sets cudnn.deterministic=True and cudnn.benchmark=False to guarantee
    bit-exact reproducibility across runs.  This trades some GPU throughput
    for determinism; for maximum throughput, call this only during evaluation
    or when exact reproducibility is required.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN kernels — may be ~10-20% slower than benchmark mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(model_cfg: ModelConfig) -> nn.Module:
    """Instantiate the correct model from a ModelConfig."""
    if model_cfg.model_type == "flued":
        from flued.model import FLUEDAutoencoder

        return FLUEDAutoencoder(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            dim_feedforward=model_cfg.dim_feedforward,
            num_encoder_layers=model_cfg.num_encoder_layers,
            num_decoder_layers=model_cfg.num_decoder_layers,
            max_seq_len=model_cfg.max_seq_len,
            dropout=model_cfg.dropout,
            shallow_layers=model_cfg.shallow_layers,
            gate_entropy_weight=model_cfg.gate_entropy_weight,
        )

    elif model_cfg.model_type == "bpe":
        from bpe_baseline.model import BPETransformerAutoencoder

        return BPETransformerAutoencoder(
            vocab_size=model_cfg.bpe_vocab_size,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            dim_feedforward=model_cfg.dim_feedforward,
            num_encoder_layers=model_cfg.num_encoder_layers,
            num_decoder_layers=model_cfg.num_decoder_layers,
            max_seq_len=model_cfg.max_seq_len,
            dropout=model_cfg.dropout,
        )

    elif model_cfg.model_type == "blt":
        from blt_baseline.model import BLTAutoencoder

        return BLTAutoencoder(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            dim_feedforward=model_cfg.dim_feedforward,
            num_encoder_layers=model_cfg.num_encoder_layers,
            num_decoder_layers=model_cfg.num_decoder_layers,
            max_seq_len=model_cfg.max_seq_len,
            dropout=model_cfg.dropout,
            local_layers=model_cfg.local_layers,
            patch_size=model_cfg.patch_size,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_cfg.model_type!r}")


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------


def build_dataset(
    model_cfg: ModelConfig, train_cfg: TrainConfig
) -> Tuple[torch.utils.data.Dataset, Optional[SimpleBPE]]:
    """Build the appropriate dataset (and BPE tokenizer if needed).

    Returns:
        (dataset, bpe) — bpe is None for flued/blt model types.
    """
    texts = None
    if train_cfg.data_path:
        with open(train_cfg.data_path, encoding="utf-8") as fh:
            texts = fh.readlines()
        logger.info("Loaded %d lines from %s", len(texts), train_cfg.data_path)

    if model_cfg.model_type == "bpe":
        corpus = texts if texts is not None else STUB_CORPUS
        bpe = SimpleBPE(vocab_size=model_cfg.bpe_vocab_size)
        logger.info("Training BPE on %d documents …", len(corpus))
        bpe.train(corpus)
        logger.info("BPE vocab size after training: %d", bpe.current_vocab_size)
        dataset = BPETextDataset(
            bpe=bpe,
            texts=texts,
            seq_len=model_cfg.max_seq_len,
        )
        return dataset, bpe
    else:
        dataset = ByteTextDataset(
            texts=texts,
            seq_len=model_cfg.max_seq_len,
        )
        return dataset, None


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------


def cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup, clipped at min_lr_ratio."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_reconstruction_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> float:
    """Token-level reconstruction accuracy (ignores padding id=0).

    Args:
        logits:  [B, T, vocab_size]
        targets: [B, T]

    Returns:
        Float in [0, 1].
    """
    preds = logits.argmax(dim=-1)           # [B, T]
    mask = targets != 0                     # ignore padding
    correct = (preds == targets) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_step(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Run a short evaluation loop and return averaged metrics.

    Args:
        model:       the autoencoder (any of the three types)
        dataloader:  evaluation DataLoader
        device:      torch device
        max_batches: cap on how many batches to evaluate

    Returns:
        dict with keys "loss" and "reconstruction_accuracy".
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for i, (src, tgt) in enumerate(dataloader):
        if i >= max_batches:
            break
        src, tgt = src.to(device), tgt.to(device)
        logits, aux_loss = model(src, tgt)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1)) + aux_loss
        total_loss += loss.item()
        total_acc += compute_reconstruction_accuracy(logits, tgt)
        n += 1

    model.train()
    return {
        "loss": total_loss / max(1, n),
        "reconstruction_accuracy": total_acc / max(1, n),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Unified trainer for FLUED Stage A autoencoder experiments.

    Works identically for flued, bpe, and blt model types.
    Training uses teacher-forced reconstruction with cross-entropy loss,
    plus any model-specific auxiliary losses (e.g. SGL entropy for FLUED).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        train_cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.cfg = train_cfg
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.max_steps,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        os.makedirs(train_cfg.output_dir, exist_ok=True)
        self.global_step = 0
        self.best_eval_loss = float("inf")

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Main training loop — runs until max_steps is reached."""
        self.model.train()
        train_iter = iter(self.train_loader)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Training  model=%s  params=%s  device=%s",
            self.model.__class__.__name__,
            f"{n_params:,}",
            self.device,
        )

        running_loss = 0.0
        running_acc = 0.0

        while self.global_step < self.cfg.max_steps:
            # Cycle through the DataLoader indefinitely
            try:
                src, tgt = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                src, tgt = next(train_iter)

            src, tgt = src.to(self.device), tgt.to(self.device)

            self.optimizer.zero_grad()
            logits, aux_loss = self.model(src, tgt)

            recon_loss = self.criterion(
                logits.view(-1, logits.size(-1)), tgt.view(-1)
            )
            loss = recon_loss + aux_loss
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            running_acc += compute_reconstruction_accuracy(logits.detach(), tgt)
            self.global_step += 1

            # --- Periodic logging ---
            if self.global_step % self.cfg.log_interval == 0:
                avg_loss = running_loss / self.cfg.log_interval
                avg_acc = running_acc / self.cfg.log_interval
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "step=%6d  loss=%.4f  recon_acc=%.4f  lr=%.2e",
                    self.global_step, avg_loss, avg_acc, lr,
                )
                running_loss = 0.0
                running_acc = 0.0

            # --- Periodic evaluation ---
            if self.global_step % self.cfg.eval_interval == 0:
                metrics = eval_step(self.model, self.eval_loader, self.device)
                logger.info(
                    "EVAL  step=%6d  loss=%.4f  recon_acc=%.4f",
                    self.global_step,
                    metrics["loss"],
                    metrics["reconstruction_accuracy"],
                )
                if metrics["loss"] < self.best_eval_loss:
                    self.best_eval_loss = metrics["loss"]
                    self._save_checkpoint("best.pt")

            # --- Periodic checkpoint ---
            if self.global_step % self.cfg.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step:06d}.pt")

        logger.info(
            "Training complete.  Best eval loss: %.4f", self.best_eval_loss
        )

    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        path = os.path.join(self.cfg.output_dir, name)
        torch.save(
            {
                "step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_eval_loss": self.best_eval_loss,
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse config, build everything, and start training."""
    from flued.config import parse_args

    model_cfg, train_cfg = parse_args()
    set_seed(train_cfg.seed)

    device = torch.device(
        train_cfg.device if torch.cuda.is_available() else "cpu"
    )
    if train_cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")

    model = build_model(model_cfg)
    dataset, _ = build_dataset(model_cfg, train_cfg)

    # 90 / 10 train / eval split
    n_eval = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_eval
    train_ds, eval_ds = random_split(
        dataset,
        [n_train, n_eval],
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )

    train_loader = get_dataloader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    eval_loader = get_dataloader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False)

    trainer = Trainer(model, train_loader, eval_loader, train_cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
