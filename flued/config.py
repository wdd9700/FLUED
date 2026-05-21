"""Configuration objects and CLI parsing for local FLUED experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


SIZE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "d_model": 128,
        "nhead": 4,
        "dim_feedforward": 1024,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        # FLUED v0.4 uses a single num_layers (tied enc+dec weights)
        "num_layers": 4,
    },
    "small": {
        "d_model": 256,
        "nhead": 8,
        "dim_feedforward": 2048,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "num_layers": 8,
    },
    "300M": {
        # ~300M params — full Stage A experiment (fits in 16 GB VRAM with batch 8–16)
        # BPE/BLT: 12 enc + 12 dec layers × ~12.6M each ≈ 302M
        # FLUED v0.4: 24 tied layers × ~12.6M each ≈ 302M (shared enc/dec weights)
        "d_model": 1024,
        "nhead": 16,
        "dim_feedforward": 4096,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "num_layers": 24,
    },
}


@dataclass
class ModelConfig:
    model_type: str = "flued"
    size: str = "small"

    d_model: int = 256
    nhead: int = 8
    dim_feedforward: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout: float = 0.0          # E1 default: 0.0 (tied inverse is dropout-sensitive)
    max_seq_len: int = 512

    # Byte-level vocabulary — v0.4 PAD-offset encoding: PAD=0, byte b → b+1
    vocab_size: int = 257

    # ------- FLUED v0.4-specific -------
    # num_layers is the single depth parameter (encoder and decoder share weights)
    num_layers: int = 4
    # Hard boundary threshold for span extraction (inference / metrics)
    boundary_threshold: float = 0.5
    # Soft boundary density target (compression_loss drives boundary_head training)
    target_compression: float = 0.3
    # Weight on the compression loss term
    compression_weight: float = 0.1

    # ------- Legacy FLUED fields (ignored by FLUEDAutoencoder v0.4) -------
    shallow_layers: int = 2
    bridge_decay: float = 0.1
    gate_entropy_weight: float = 0.0

    # token baseline vocab (sentencepiece or simple bpe)
    token_vocab_size: int = 8192

    # BLT legacy knobs (kept for compatibility)
    local_layers: int = 2
    patch_size: int = 4

    def apply_size(self) -> "ModelConfig":
        preset = SIZE_CONFIGS.get(self.size)
        if preset:
            for key, value in preset.items():
                setattr(self, key, value)
        return self


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 4
    max_steps: int = 500
    lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    amp: bool = False
    amp_dtype: str = "bf16"  # bf16|fp16

    log_interval: int = 20
    eval_interval: int = 100
    save_interval: int = 500

    output_dir: str = "checkpoints"
    data_path: Optional[str] = None
    device: str = "cuda"
    num_workers: int = 0


def parse_args() -> Tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(
        description="FLUED local trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", default="flued", choices=["flued", "bpe", "blt"])
    parser.add_argument("--size", default="small", choices=list(SIZE_CONFIGS.keys()))

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token-vocab-size", type=int, default=8192)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")

    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")

    model_cfg = ModelConfig(
        model_type=args.model_type,
        size=args.size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        token_vocab_size=args.token_vocab_size,
    ).apply_size()

    overrides = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "num_layers": args.num_layers,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(model_cfg, key, value)

    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
        num_workers=args.num_workers,
    )
    return model_cfg, train_cfg
