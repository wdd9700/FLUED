"""Measure whether a trained v3.4 model uses the semantic content of memory."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import sys
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import LatentInfillBackbone, make_dataloaders  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import build_model, evaluate  # noqa: E402


INTERVENTIONS = ("normal", "zero", "shuffle_chunk", "stale_batch")


def _transform_memory(memory: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "normal":
        return memory
    if mode == "zero":
        return torch.zeros_like(memory)
    if mode == "shuffle_chunk":
        return memory.roll(shifts=1, dims=1)
    if mode == "stale_batch":
        if memory.size(0) < 2:
            # The shared evaluator also runs a one-sample order probe after the
            # batched metrics. Leave that auxiliary sample unchanged; the
            # stale-memory result itself is measured on batch_size >= 2.
            return memory
        return memory.roll(shifts=1, dims=0)
    raise ValueError(f"unknown intervention: {mode}")


def _install_intervention(model, mode: str) -> Callable:
    original = model.memory_read.forward

    def forward(readout, memory, chunk_mask, **kwargs):
        return original(readout, _transform_memory(memory, mode), chunk_mask, **kwargs)

    model.memory_read.forward = forward
    return original


def _markdown(rows: dict[str, dict[str, float]], checkpoint: Path) -> str:
    keys = (
        "identity_acc",
        "completion_mask_acc",
        "completion_ppl",
        "actual_backbone_units_per_byte",
        "memory_residual_ratio",
    )
    normal = rows["normal"]
    lines = [
        "# FLUED v3.4 memory 内容干预",
        "",
        f"检查点：`{checkpoint}`",
        "",
        "所有模式使用同一模型、输入、掩码和评估种子，只改变送入 interpreter 的 memory 内容。",
        "",
        "| 模式 | 重建准确率 | 补全准确率 | 补全困惑度 | 实际 latent/byte | memory 残差比 | 困惑度变化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, row in rows.items():
        delta = row["completion_ppl"] - normal["completion_ppl"]
        lines.append(
            f"| {mode} | {row[keys[0]]:.4f} | {row[keys[1]]:.4f} | "
            f"{row[keys[2]]:.3f} | {row[keys[3]]:.4f} | {row[keys[4]]:.4f} | {delta:+.3f} |"
        )
    lines.extend(
        [
            "",
            "判定规则：normal 必须稳定优于 zero、shuffle_chunk 和 stale_batch，才能证明当前 memory 内容有正向作用。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-eval-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    checkpoint = Path(args_cli.checkpoint)
    config_path = Path(args_cli.config) if args_cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["device"] = args_cli.device
    config["max_eval_batches"] = args_cli.max_eval_batches
    config["num_workers"] = 0
    args = Namespace(**config)
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    runtime_boundary_state = payload.get("runtime_boundary_state", {})
    if runtime_boundary_state:
        model.config.boundary_mode = runtime_boundary_state.get(
            "mode", model.config.boundary_mode
        )
        model.coding_rate_selector.mode = runtime_boundary_state.get(
            "coding_rate_mode", model.coding_rate_selector.mode
        )
        model.config.boundary_blend_alpha = float(
            runtime_boundary_state.get("blend_alpha", model.config.boundary_blend_alpha)
        )
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    ).to(device)
    backbone.load_state_dict(payload["backbone"], strict=True)
    _, eval_loader = make_dataloaders(args)

    rows: dict[str, dict[str, float]] = {}
    original = model.memory_read.forward
    for mode in INTERVENTIONS:
        model.memory_read.forward = original
        _install_intervention(model, mode)
        metrics = evaluate(model, backbone, eval_loader, args, device)
        rows[mode] = metrics
        print(
            f"[{mode}] identity={metrics['identity_acc']:.4f} "
            f"completion={metrics['completion_mask_acc']:.4f} "
            f"ppl={metrics['completion_ppl']:.3f}",
            flush=True,
        )
    model.memory_read.forward = original

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "runtime_boundary_state": {
            "mode": model.config.boundary_mode,
            "coding_rate_mode": model.coding_rate_selector.mode,
            "blend_alpha": float(model.config.boundary_blend_alpha),
        },
        "max_eval_batches": args_cli.max_eval_batches,
        "interventions": rows,
    }
    (out_dir / "memory_interventions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "memory_interventions.md").write_text(
        _markdown(rows, checkpoint), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
