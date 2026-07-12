"""Evaluate one v3.4 checkpoint across inference-time emit thresholds.

Weights and evaluation batches stay fixed. Only the hard emit threshold changes,
exposing the reconstruction/completion/compute Pareto curve without retraining.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import LatentInfillBackbone, make_dataloaders  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    apply_boundary_curriculum,
    build_model,
    evaluate,
)


def _resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_thresholds(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("thresholds must contain values in [0, 1]")
    return values


def _restore_boundary_state(model, args: Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("runtime_boundary_state")
    if state:
        model.config.boundary_mode = str(state["mode"])
        model.config.coding_rate_mode = str(state["coding_rate_mode"])
        model.config.boundary_blend_alpha = float(state.get("blend_alpha", 1.0))
        model.coding_rate_selector.mode = model.config.coding_rate_mode
        source = "checkpoint"
    else:
        apply_boundary_curriculum(model, args, int(payload.get("step", 0)))
        source = "curriculum_step_fallback"
    return {
        "mode": model.config.boundary_mode,
        "coding_rate_mode": model.coding_rate_selector.mode,
        "blend_alpha": float(model.config.boundary_blend_alpha),
        "restored_from": source,
    }


def run(cli: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    checkpoint = _resolve_checkpoint(cli.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = dict(payload["args"])
    if cli.data_path:
        saved_args["data_path"] = cli.data_path
    saved_args["device"] = cli.device
    saved_args["max_eval_batches"] = cli.max_eval_batches
    saved_args["num_workers"] = 0
    saved_args.setdefault("emit_threshold", 0.5)
    args = Namespace(**saved_args)

    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    unexpected = [name for name in unexpected if name != "logic_transition_prior"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    ).to(device)
    backbone.load_state_dict(payload["backbone"])
    boundary_state = _restore_boundary_state(model, args, payload)

    rows = []
    for threshold in _parse_thresholds(cli.thresholds):
        random.seed(cli.eval_seed)
        torch.manual_seed(cli.eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cli.eval_seed)
        model.emit_controller.threshold = threshold
        _, eval_loader = make_dataloaders(args)
        stats = evaluate(model, backbone, eval_loader, args, device)
        row = {
            "emit_threshold": threshold,
            "identity_acc": stats["identity_acc"],
            "identity_loss": stats["identity_loss"],
            "completion_mask_acc": stats["completion_mask_acc"],
            "completion_masked_loss": stats["completion_masked_loss"],
            "completion_ppl": math.exp(min(20.0, stats["completion_masked_loss"])),
            "completion_preserve_acc": stats["completion_preserve_acc"],
            "actual_backbone_units_per_byte": stats["actual_backbone_units_per_byte"],
            "soft_readout_units_per_byte": stats["soft_readout_units_per_byte"],
            "chunks_per_byte": stats["chunks_per_byte"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "use_memory": bool(args.use_memory),
        "eval_seed": cli.eval_seed,
        "max_eval_batches": cli.max_eval_batches,
        "boundary_state": boundary_state,
        "rows": rows,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
