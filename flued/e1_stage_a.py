"""Stage A (E1) local runner for FLUED v0.4 strict reconstruction."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Tuple

import torch
from torch.utils.data import random_split

from flued.config import ModelConfig, TrainConfig
from flued.data import ByteReconstructionDataset, get_dataloader
from flued.train import Trainer, build_model, eval_step, set_seed

logger = logging.getLogger("flued.e1_stage_a")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


E1_PRESETS = {
    "smoke_cpu": {
        "model": dict(size="smoke", max_seq_len=96),
        "train": dict(batch_size=2, max_steps=20, grad_accum_steps=1, device="cpu", amp=False),
    },
    "small_gpu": {
        "model": dict(size="small", max_seq_len=256),
        "train": dict(batch_size=2, max_steps=400, grad_accum_steps=8, device="cuda", amp=True, amp_dtype="bf16"),
    },
    "class300m_16gb": {
        "model": dict(size="300m", max_seq_len=256),
        "train": dict(batch_size=1, max_steps=1000, grad_accum_steps=16, device="cuda", amp=True, amp_dtype="bf16"),
    },
}


def build_e1_configs(preset: str = "smoke_cpu") -> Tuple[ModelConfig, TrainConfig]:
    if preset not in E1_PRESETS:
        raise ValueError(f"Unknown E1 preset: {preset}")
    model_cfg = ModelConfig(model_type="flued", **E1_PRESETS[preset]["model"]).apply_size()
    train_cfg = TrainConfig(**E1_PRESETS[preset]["train"])
    return model_cfg, train_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUED E1 Stage A reconstruction runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=list(E1_PRESETS.keys()), default="smoke_cpu")
    parser.add_argument("--data-path", default=None, help="Optional UTF-8 text file")
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--min-compression", type=float, default=0.125)
    parser.add_argument("--max-compression", type=float, default=0.5)
    parser.add_argument("--output-dir", default="checkpoints/e1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_cfg, train_cfg = build_e1_configs(args.preset)

    train_cfg = replace(
        train_cfg,
        seed=args.seed,
        output_dir=args.output_dir,
        data_path=args.data_path,
        max_steps=args.max_steps or train_cfg.max_steps,
        batch_size=args.batch_size or train_cfg.batch_size,
        grad_accum_steps=args.grad_accum_steps or train_cfg.grad_accum_steps,
        device=args.device or train_cfg.device,
        amp=args.amp or train_cfg.amp,
        amp_dtype=args.amp_dtype or train_cfg.amp_dtype,
    )

    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    if train_cfg.device == "cuda" and device.type == "cpu":
        logger.warning("CUDA requested but not available, using CPU.")

    dataset = ByteReconstructionDataset(file_path=train_cfg.data_path, seq_len=model_cfg.max_seq_len)
    n_eval = max(1, len(dataset) // 10)
    n_train = max(1, len(dataset) - n_eval)
    train_ds, eval_ds = random_split(dataset, [n_train, n_eval], generator=torch.Generator().manual_seed(train_cfg.seed))

    train_loader = get_dataloader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    eval_loader = get_dataloader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = build_model(model_cfg)
    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, train_cfg=train_cfg, device=device)
    trainer.train()

    metrics = eval_step(trainer.model, eval_loader, device, max_batches=20)
    recon_acc = metrics["reconstruction_accuracy"]
    m_over_n = metrics.get("m_over_n")
    if m_over_n is None:
        print("FAIL: E1 criteria unavailable because model did not produce m_over_n.")
        return 1
    logger.info("Final E1 metrics: recon_acc=%.4f m_over_n=%.4f", recon_acc, m_over_n)

    passed = (
        recon_acc >= args.target_accuracy
        and args.min_compression <= m_over_n <= args.max_compression
    )
    if passed:
        print("PASS: E1 criteria met")
        return 0

    print(
        "FAIL: E1 criteria not met "
        f"(acc={recon_acc:.4f}, target={args.target_accuracy:.4f}, "
        f"m/n={m_over_n:.4f}, expected=[{args.min_compression:.3f}, {args.max_compression:.3f}])"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
