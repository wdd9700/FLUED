"""
Configuration for FLUED Stage A autoencoder experiments.

Supports three model types:
  - "flued"  : FLUED Dynamic Semantic Compiler autoencoder
  - "bpe"    : 64k BPE-Transformer autoencoder baseline
  - "blt"    : Byte Latent Transformer-style autoencoder baseline

Size presets target ~300M parameters on a class-A Transformer backbone.
"""

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

# Each preset specifies the Transformer backbone dimensions.
# "300M" targets ~300M total parameters (encoder + decoder combined).
SIZE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small": {
        # ~3M params — fast smoke tests
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 1024,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        # FLUED v0.4 uses a single num_layers (tied enc+dec weights)
        "num_layers": 4,
    },
    "medium": {
        # ~50M params — development/ablation
        "d_model": 512,
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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Which model to train: "flued", "bpe", or "blt"
    model_type: str = "flued"

    # Size preset applied first; individual fields can override afterwards
    size: str = "small"

    # Core Transformer dimensions
    d_model: int = 256
    nhead: int = 4
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

    # ------- BPE-specific -------
    # Target BPE vocabulary size (64k as per problem statement)
    bpe_vocab_size: int = 65536

    # ------- BLT-specific -------
    # Number of local (byte-level) encoder/decoder layers
    local_layers: int = 2
    # Bytes per patch (fixed-size stub; entropy-based patching is noted)
    patch_size: int = 4

    def apply_size(self) -> "ModelConfig":
        """Apply the size preset in-place, then return self."""
        if self.size in SIZE_CONFIGS:
            for k, v in SIZE_CONFIGS[self.size].items():
                setattr(self, k, v)
        return self


@dataclass
class TrainConfig:
    """Training and experiment configuration."""

    # Reproducibility seed (sets Python, NumPy, and PyTorch seeds)
    seed: int = 42

    batch_size: int = 8
    max_steps: int = 5000
    lr: float = 1e-4
    weight_decay: float = 1e-2

    # Linear warmup then cosine decay
    warmup_steps: int = 500

    # Gradient clipping max norm
    grad_clip: float = 1.0

    # Logging and checkpoint intervals (in training steps)
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000

    output_dir: str = "checkpoints"

    # Path to a plain-text corpus file (one document per line).
    # If None, the built-in stub corpus is used.
    data_path: Optional[str] = None

    # "cuda" or "cpu"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def parse_args():
    """Parse CLI arguments and return (ModelConfig, TrainConfig)."""
    parser = argparse.ArgumentParser(
        description="FLUED Stage A Autoencoder Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model selection ---
    parser.add_argument(
        "--model-type",
        default="flued",
        choices=["flued", "bpe", "blt"],
        help="Which model architecture to train",
    )
    parser.add_argument(
        "--size",
        default="small",
        choices=list(SIZE_CONFIGS.keys()),
        help="Model size preset (applied before per-field overrides)",
    )

    # --- Architecture overrides (applied after --size preset) ---
    parser.add_argument("--d-model", type=int, default=None, help="Transformer hidden dim")
    parser.add_argument("--nhead", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--num-decoder-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--bpe-vocab-size", type=int, default=65536)
    parser.add_argument("--shallow-layers", type=int, default=None,
                        help="FLUED: number of shallow DSC encoder layers")
    parser.add_argument("--local-layers", type=int, default=None,
                        help="BLT: number of local byte-level encoder/decoder layers")
    parser.add_argument("--patch-size", type=int, default=4,
                        help="BLT: bytes per patch")

    # --- Training ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--data-path", default=None,
                        help="Path to plain-text corpus (one doc per line)")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Build ModelConfig: apply size preset, then per-field overrides
    model_cfg = ModelConfig(
        model_type=args.model_type,
        size=args.size,
        max_seq_len=args.max_seq_len,
        bpe_vocab_size=args.bpe_vocab_size,
        patch_size=args.patch_size,
    )
    model_cfg.apply_size()

    # Apply any explicit per-field overrides
    overrides = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "shallow_layers": args.shallow_layers,
        "local_layers": args.local_layers,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(model_cfg, key, val)

    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
    )

    return model_cfg, train_cfg
