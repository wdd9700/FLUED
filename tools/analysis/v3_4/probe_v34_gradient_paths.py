"""Audit main-task gradient paths at v3.4 curriculum milestones."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import re
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import LatentInfillBackbone, make_dataloaders  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    apply_boundary_curriculum,
    build_model,
    step_model,
)


AUXILIARY_WEIGHTS = (
    "boundary_loss_weight",
    "boundary_continuation_loss_weight",
    "boundary_punctuation_loss_weight",
    "boundary_neutral_loss_weight",
    "boundary_rate_alignment_weight",
    "boundary_rate_calibration_weight",
    "boundary_rate_density_weight",
    "boundary_rate_margin_weight",
    "boundary_rate_dual_augmented_weight",
    "boundary_budget_augmented_weight",
    "coding_rate_loss_weight",
    "memory_usage_loss_weight",
    "emit_value_loss_weight",
    "ar_delta_loss_weight",
)


def _grad_stats(module) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    tensors = 0
    elements = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        squared += float(grad.square().sum().item())
        maximum = max(maximum, float(grad.abs().max().item()))
        tensors += 1
        elements += grad.numel()
    return {
        "l2": squared ** 0.5,
        "max_abs": maximum,
        "tensors": tensors,
        "elements": elements,
    }


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    rows = []
    for checkpoint_text in cli.checkpoint:
        checkpoint = Path(checkpoint_text)
        config_path = checkpoint.with_name("resolved_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update({"device": str(device), "batch_size": cli.batch_size, "num_workers": 0})
        for name in AUXILIARY_WEIGHTS:
            config[name] = 0.0
        config["boundary_rate_dual_lr"] = 0.0
        config["boundary_dual_lr"] = 0.0
        args = Namespace(**config)
        step = _checkpoint_step(checkpoint)

        torch.manual_seed(args.eval_mask_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_mask_seed)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model(args).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.boundary_rate_dual.zero_()
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
        apply_boundary_curriculum(model, args, step)
        train_loader, _ = make_dataloaders(args)
        batch = next(iter(train_loader))

        model.train()
        backbone.train()
        model.zero_grad(set_to_none=True)
        backbone.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, _ = step_model(
                model,
                backbone,
                batch,
                args,
                device,
                collect_metrics=False,
                global_step=-1,
            )
        loss.backward()
        row = {
            "checkpoint": str(checkpoint),
            "step": step,
            "decoder_mode": args.decoder_mode,
            "boundary_mode": model.config.boundary_mode,
            "boundary_blend_alpha": model.config.boundary_blend_alpha,
            "loss": float(loss.item()),
            "gradients": {
                "segmentor_blocks": _grad_stats(model.segmentor_blocks),
                "segmentor_head": _grad_stats(model.segmentor_head),
                "coding_rate_selector": _grad_stats(model.coding_rate_selector),
                "readout_pool": _grad_stats(model.readout_pool),
                "interpreter_blocks": _grad_stats(model.interpreter_blocks),
                "decoder": _grad_stats(model.decoder),
                "backbone": _grad_stats(backbone),
            },
        }
        rows.append(row)
        print(
            f"[{checkpoint.parent.name} {step}] mode={model.config.boundary_mode} "
            f"head={row['gradients']['segmentor_head']['l2']:.3e} "
            f"segmentor={row['gradients']['segmentor_blocks']['l2']:.3e} "
            f"decoder={row['gradients']['decoder']['l2']:.3e}",
            flush=True,
        )
        del model, backbone, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"main_task_only": True, "runs": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
