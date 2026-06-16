"""Shared training utilities for local FLUED experiments."""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flued.config import ModelConfig, TrainConfig
from flued.data import (
    STUB_CORPUS,
    BPETextDataset,
    ByteReconstructionDataset,
    ByteTextDataset,
    SimpleBPE,
    get_dataloader,
    safe_train_eval_split,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("flued.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_cfg: ModelConfig) -> nn.Module:
    if model_cfg.model_type == "flued":
        from flued.model import FLUEDAutoencoder

        return FLUEDAutoencoder(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            nhead=model_cfg.nhead,
            dim_feedforward=model_cfg.dim_feedforward,
            swiglu_hidden=model_cfg.swiglu_hidden,
            num_layers=model_cfg.num_layers,
            max_seq_len=model_cfg.max_seq_len,
            assignment_window=model_cfg.assignment_window,
            dropout=model_cfg.dropout,
            boundary_threshold=model_cfg.boundary_threshold,
            target_compression=model_cfg.target_compression,
            compression_weight=model_cfg.compression_weight,
            min_boundary_units=model_cfg.min_boundary_units,
        )

    elif model_cfg.model_type == "bpe":
        from bpe_baseline.model import BPETransformerAutoencoder

        return BPETransformerAutoencoder(
            vocab_size=model_cfg.token_vocab_size,
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
            patch_mode="entropy",
            entropy_theta=3.5,
        )

    raise ValueError(f"Unknown model_type: {model_cfg.model_type!r}")


def build_dataset(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Tuple[torch.utils.data.Dataset, Optional[SimpleBPE]]:
    texts = None
    if train_cfg.data_path:
        with open(train_cfg.data_path, encoding="utf-8") as fh:
            texts = [line.rstrip("\n") for line in fh if line.strip()]
        if not texts:
            raise ValueError(f"No non-empty lines found in data file: {train_cfg.data_path}")

    if model_cfg.model_type == "bpe":
        corpus = texts if texts is not None else STUB_CORPUS
        bpe = SimpleBPE(vocab_size=model_cfg.token_vocab_size)
        bpe.train(corpus)
        return BPETextDataset(bpe=bpe, texts=texts, seq_len=model_cfg.max_seq_len), bpe

    return ByteReconstructionDataset(texts=texts, seq_len=model_cfg.max_seq_len), None


def cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_reconstruction_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != 0
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return ((preds == targets) & mask).sum().item() / total


@torch.no_grad()
def eval_step(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 10) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    total_acc = 0.0
    total_m_over_n = 0.0
    n = 0

    for i, (src, tgt) in enumerate(dataloader):
        if i >= max_batches:
            break
        src, tgt = src.to(device), tgt.to(device)
        result = model(src, tgt)
        logits = result[0]
        aux = result[1]
        if isinstance(aux, dict):
            aux_loss = aux.get("compression_loss", torch.tensor(0.0, device=device))
            total_m_over_n += aux.get("m_over_n", 0.0)
        else:
            aux_loss = aux
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1)) + aux_loss
        total_loss += loss.item()
        total_acc += compute_reconstruction_accuracy(logits, tgt)
        n += 1

    model.train()
    return {
        "loss": total_loss / max(1, n),
        "reconstruction_accuracy": total_acc / max(1, n),
        "m_over_n": total_m_over_n / max(1, n),
    }


class Trainer:
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

        self.optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.scheduler = cosine_schedule_with_warmup(self.optimizer, train_cfg.warmup_steps, train_cfg.max_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        amp_enabled = train_cfg.amp and device.type == "cuda"
        if train_cfg.amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        elif train_cfg.amp_dtype == "fp16":
            self.amp_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported amp_dtype: {train_cfg.amp_dtype}. Expected 'bf16' or 'fp16'.")
        self.autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=amp_enabled))
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and self.amp_dtype == torch.float16)

        os.makedirs(train_cfg.output_dir, exist_ok=True)
        self.global_step = 0
        self.best_eval_loss = float("inf")

    def train(self) -> None:
        self.model.train()
        train_iter = iter(self.train_loader)
        running_loss = 0.0
        running_acc = 0.0

        while self.global_step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_acc = 0.0

            for _ in range(self.cfg.grad_accum_steps):
                try:
                    src, tgt = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    src, tgt = next(train_iter)

                src, tgt = src.to(self.device), tgt.to(self.device)

                with self.autocast_ctx():
                    result = self.model(src, tgt)
                    logits = result[0]
                    aux = result[1]
                    if isinstance(aux, dict):
                        aux_loss = aux.get("compression_loss", torch.tensor(0.0, device=self.device))
                    else:
                        aux_loss = aux
                    loss = self.criterion(logits.view(-1, logits.size(-1)), tgt.view(-1)) + aux_loss
                    loss = loss / self.cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                step_loss += loss.item()
                step_acc += compute_reconstruction_accuracy(logits.detach(), tgt) / self.cfg.grad_accum_steps

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1
            running_loss += step_loss
            running_acc += step_acc

            if self.global_step % self.cfg.log_interval == 0:
                logger.info(
                    "step=%6d loss=%.4f recon_acc=%.4f lr=%.2e",
                    self.global_step,
                    running_loss / self.cfg.log_interval,
                    running_acc / self.cfg.log_interval,
                    self.scheduler.get_last_lr()[0],
                )
                running_loss = 0.0
                running_acc = 0.0

            if self.global_step % self.cfg.eval_interval == 0:
                metrics = eval_step(self.model, self.eval_loader, self.device)
                logger.info(
                    "EVAL step=%6d loss=%.4f recon_acc=%.4f m_over_n=%.4f",
                    self.global_step,
                    metrics["loss"],
                    metrics["reconstruction_accuracy"],
                    metrics["m_over_n"],
                )
                if metrics["loss"] < self.best_eval_loss:
                    self.best_eval_loss = metrics["loss"]
                    self._save_checkpoint("best.pt")

            if self.global_step % self.cfg.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step:06d}.pt")

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


def main() -> None:
    from flued.config import parse_args

    model_cfg, train_cfg = parse_args()
    set_seed(train_cfg.seed)

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    if train_cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")

    model = build_model(model_cfg)
    dataset, _ = build_dataset(model_cfg, train_cfg)

    # 90 / 10 train / eval split (robust to tiny datasets)
    train_ds, eval_ds = safe_train_eval_split(
        dataset, eval_fraction=0.1, seed=train_cfg.seed
    )

    train_loader = get_dataloader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    eval_loader = get_dataloader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, train_cfg=train_cfg, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
