"""Offline V0 audit for Counterfactual Byte-Interface Utility (CBIU).

This tool deliberately does not train the model. It evaluates paired, frozen
interventions with the same bytes and masks, then reports three risks in
bits/target-byte:

* clean reconstruction;
* strict masked-byte completion;
* preservation of visible bytes in affected chunks.

The first implementation covers interventions that can be executed without
approximating a new segmentation: readout removal, memory content/skip, and
small-AR skip. Boundary merge requires rebuilding chunks and is intentionally
left for the next protocol revision.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterator

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID  # noqa: E402
from tools.train.v3_3.train_v33 import (  # noqa: E402
    LatentInfillBackbone,
    make_byte_mask,
    make_dataloaders,
    make_targets,
    masked_readouts_from_slots,
)
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    _ce,
    _run_completion,
    build_model,
)


LN2 = math.log(2.0)
RISK_NAMES = ("reconstruction_bpb", "completion_bpb", "preservation_bpb")


@dataclass
class WeightedRisk:
    loss_sum: float = 0.0
    targets: float = 0.0

    def add(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        weight = mask.to(values.dtype)
        self.loss_sum += float((values * weight).sum().item())
        self.targets += float(weight.sum().item())

    def bits(self) -> float:
        return self.loss_sum / max(self.targets, 1.0) / LN2


@dataclass
class ModeTotals:
    reconstruction: WeightedRisk = field(default_factory=WeightedRisk)
    completion: WeightedRisk = field(default_factory=WeightedRisk)
    preservation: WeightedRisk = field(default_factory=WeightedRisk)
    emitted_readouts: float = 0.0
    valid_bytes: float = 0.0
    emit_probability_sum: dict[int, float] = field(default_factory=dict)
    emit_probability_count: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        result = {
            "reconstruction_bpb": self.reconstruction.bits(),
            "completion_bpb": self.completion.bits(),
            "preservation_bpb": self.preservation.bits(),
            "actual_readouts_per_byte": self.emitted_readouts / max(self.valid_bytes, 1.0),
        }
        for slot, total in self.emit_probability_sum.items():
            result[f"emit_probability_slot_{slot}"] = total / max(
                self.emit_probability_count.get(slot, 0.0), 1.0
            )
        return result


def normalize_risks(
    row: dict[str, float],
    rich: dict[str, float],
    null: dict[str, float],
    epsilon: float = 1.0e-6,
) -> tuple[dict[str, float | None], list[str]]:
    """Normalize risks against frozen rich/null anchors without hiding bad anchors."""

    normalized: dict[str, float | None] = {}
    invalid: list[str] = []
    for key in RISK_NAMES:
        gap = null[key] - rich[key]
        if gap <= epsilon:
            normalized[key] = None
            invalid.append(key)
        else:
            normalized[key] = (row[key] - rich[key]) / gap
    return normalized, invalid


def robust_risk(normalized: dict[str, float | None]) -> float | None:
    values = [value for value in normalized.values() if value is not None]
    return max(values) if len(values) == len(RISK_NAMES) else None


def score_to_probability(score: float) -> float:
    """Map the signed FLUED confidence convention to a calibrated probability."""

    return min(max(0.5 * (score + 1.0), 0.0), 1.0)


def _select_readouts(out, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_mask = out.chunks.chunk_mask
    if mode == "rich_all_readouts":
        active = chunk_mask.unsqueeze(-1).expand_as(out.emit_hard)
        return out.readout_candidates, active
    if mode == "null_fallback_only":
        active = torch.zeros_like(out.emit_hard)
        active[..., 0] = chunk_mask
        return out.readout_candidates * active.unsqueeze(-1), active

    readout = out.readout_z
    active = out.emit_hard.clone()
    if mode.startswith("drop_emit_slot_"):
        slot = int(mode.rsplit("_", 1)[-1])
        if not 0 < slot < readout.size(2):
            raise ValueError(f"invalid extra readout slot {slot} for R={readout.size(2)}")
        readout = readout.clone()
        readout[:, :, slot] = 0.0
        active[:, :, slot] = False
    elif mode != "policy":
        raise ValueError(f"unknown readout mode: {mode}")
    return readout, active


def _strict_affected_readouts(
    masked_slot: torch.Tensor,
    chunk_mask: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    affected_chunks = masked_slot.any(dim=-1) & chunk_mask
    affected = masked_readouts_from_slots(masked_slot, chunk_mask, active.size(-1)) & active
    # Every affected chunk has the mandatory fallback as a writable location.
    affected = affected.clone()
    affected[..., 0] |= affected_chunks
    return affected & active


@contextmanager
def _model_intervention(model, mode: str) -> Iterator[None]:
    mode = mode.removesuffix("_fixed_emit")
    original_memory_forward = model.memory_read.forward
    original_use_memory = bool(model.config.use_memory)
    original_use_ar = bool(model.config.use_ar)

    if mode in {"memory_zero", "memory_stale_batch"}:
        transform = (
            (lambda memory: torch.zeros_like(memory))
            if mode == "memory_zero"
            else (lambda memory: memory.roll(shifts=1, dims=0) if memory.size(0) > 1 else memory)
        )

        def forward(readout, memory, chunk_mask, **kwargs):
            return original_memory_forward(readout, transform(memory), chunk_mask, **kwargs)

        model.memory_read.forward = forward
    elif mode == "memory_skip_execution":
        model.config.use_memory = False
    elif mode == "small_ar_skip_execution":
        model.config.use_ar = False
    elif mode not in {
        "policy",
        "rich_all_readouts",
        "null_fallback_only",
    } and not mode.startswith("drop_emit_slot_"):
        raise ValueError(f"unknown intervention: {mode}")

    try:
        yield
    finally:
        model.memory_read.forward = original_memory_forward
        model.config.use_memory = original_use_memory
        model.config.use_ar = original_use_ar


@torch.no_grad()
def _run_mode_batch(
    model,
    backbone,
    clean: torch.Tensor,
    byte_mask: torch.Tensor,
    args: Namespace,
    mode: str,
    totals: ModeTotals,
    tracked_slots: list[int],
    active_override: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    valid = clean.ne(PAD_ID)
    source = clean.masked_fill(byte_mask, MASK_ID)
    readout_mode = mode if mode.startswith("drop_emit_slot_") else (
        mode if mode in {"rich_all_readouts", "null_fallback_only"} else "policy"
    )

    with _model_intervention(model, mode):
        clean_out = model(clean)
        clean_z, clean_active = _select_readouts(clean_out, readout_mode)
        if active_override is not None:
            clean_active = active_override[0]
            clean_z = clean_out.readout_candidates * clean_active.unsqueeze(-1)
        clean_logits = model.decode(clean_z, clean_out.chunks.chunk_mask, clean_active)
        zero_mask = torch.zeros_like(byte_mask)
        clean_targets, clean_slot_mask, _ = make_targets(
            clean,
            zero_mask,
            clean_out.chunks.chunk_ids,
            clean_out.chunks.offsets,
            args.max_chunks,
            args.max_span,
        )
        totals.reconstruction.add(_ce(clean_logits, clean_targets), clean_slot_mask)

        masked_out = model(source)
        masked_z, masked_active = _select_readouts(masked_out, readout_mode)
        if active_override is not None:
            masked_active = active_override[1]
            masked_z = masked_out.readout_candidates * masked_active.unsqueeze(-1)
        observed_targets, observed_slot_mask, _ = make_targets(
            source,
            zero_mask,
            masked_out.chunks.chunk_ids,
            masked_out.chunks.offsets,
            args.max_chunks,
            args.max_span,
        )
        clean_targets, target_slot_mask, masked_slot = make_targets(
            clean,
            byte_mask,
            masked_out.chunks.chunk_ids,
            masked_out.chunks.offsets,
            args.max_chunks,
            args.max_span,
        )
        del observed_targets
        target_slot_mask &= observed_slot_mask
        affected = _strict_affected_readouts(
            masked_slot,
            masked_out.chunks.chunk_mask,
            masked_active,
        )
        completed_logits, _ = _run_completion(
            model,
            backbone,
            masked_z,
            masked_active,
            affected,
            masked_out.chunks.chunk_mask,
        )
        completed_ce = _ce(completed_logits, clean_targets)
        totals.completion.add(completed_ce, masked_slot & target_slot_mask)
        affected_chunks = masked_slot.any(dim=-1) & masked_out.chunks.chunk_mask
        preserve_slot = target_slot_mask & affected_chunks.unsqueeze(-1) & ~masked_slot
        totals.preservation.add(completed_ce, preserve_slot)

    totals.emitted_readouts += float(masked_active.float().sum().item())
    totals.valid_bytes += float(valid.float().sum().item())
    if mode == "policy":
        active_chunks = clean_out.chunks.chunk_mask
        for slot in tracked_slots:
            if slot >= clean_out.emit_soft.size(-1):
                continue
            totals.emit_probability_sum[slot] = totals.emit_probability_sum.get(slot, 0.0) + float(
                clean_out.emit_soft[:, :, slot][active_chunks].sum().item()
            )
            totals.emit_probability_count[slot] = totals.emit_probability_count.get(slot, 0.0) + float(
                active_chunks.sum().item()
            )


@torch.no_grad()
def _baseline_active_masks(
    model,
    clean: torch.Tensor,
    byte_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = clean.masked_fill(byte_mask, MASK_ID)
    clean_active = model(clean).emit_hard.detach().clone()
    masked_active = model(source).emit_hard.detach().clone()
    return clean_active, masked_active


def _restore_runtime_state(model, payload: dict) -> dict:
    state = payload.get("runtime_boundary_state", {})
    if state:
        model.config.boundary_mode = state.get("mode", model.config.boundary_mode)
        model.coding_rate_selector.mode = state.get(
            "coding_rate_mode", model.coding_rate_selector.mode
        )
        model.config.boundary_blend_alpha = float(
            state.get("blend_alpha", model.config.boundary_blend_alpha)
        )
    return {
        "mode": model.config.boundary_mode,
        "coding_rate_mode": model.coding_rate_selector.mode,
        "blend_alpha": float(model.config.boundary_blend_alpha),
    }


def _calibration(rows: dict[str, dict[str, float]], policy: dict[str, float], slots: list[int]) -> dict:
    examples: list[dict[str, float | int | bool]] = []
    for slot in slots:
        key = f"drop_emit_slot_{slot}"
        if key not in rows or rows[key].get("quality_utility_keep") is None:
            continue
        probability = float(policy.get(f"emit_probability_slot_{slot}", 0.0))
        utility = float(rows[key]["quality_utility_keep"])
        label = utility > 0.0
        examples.append(
            {
                "slot": slot,
                "predicted_probability": probability,
                "quality_utility_keep": utility,
                "useful": label,
                "brier": (probability - float(label)) ** 2,
            }
        )
    if not examples:
        return {"examples": [], "brier": None, "sign_accuracy_at_0_5": None}
    return {
        "examples": examples,
        "brier": sum(float(row["brier"]) for row in examples) / len(examples),
        "sign_accuracy_at_0_5": sum(
            (float(row["predicted_probability"]) >= 0.5) == bool(row["useful"])
            for row in examples
        ) / len(examples),
        "warning": "slot-level pilot only; boundary and per-chunk calibration remain pending",
    }


def _markdown(result: dict) -> str:
    rows = result["modes"]
    lines = [
        "# FLUED v3.4 CBIU V0 离线探针",
        "",
        f"检查点：`{result['checkpoint']}`",
        "",
        "本轮只报告可严格执行的 readout、memory 和 small-AR 干预。Boundary merge 尚未纳入。",
        "",
        "| 模式 | clean 重建 BPB | mask 补全 BPB | 可见保持 BPB | 实际 readout/byte | rho | 保留质量效用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        rho = "-" if row.get("rho") is None else f"{row['rho']:.4f}"
        utility = "-" if row.get("quality_utility_keep") is None else f"{row['quality_utility_keep']:+.4f}"
        lines.append(
            f"| {name} | {row['reconstruction_bpb']:.4f} | {row['completion_bpb']:.4f} | "
            f"{row['preservation_bpb']:.4f} | {row['actual_readouts_per_byte']:.4f} | {rho} | {utility} |"
        )
    lines.extend(
        [
            "",
            "`quality_utility_keep = rho(intervention) - rho(policy)`：正值表示被移除的组件有益。",
            "memory zero/stale 不节省计算，只测内容依赖；当前只对 emit 记录可验证的 readout 成本差。",
            "",
            "## Anchor 检查",
            "",
            f"无效 anchor 维度：`{result['invalid_anchor_dimensions']}`",
            "",
            "## Emit 槽位校准",
            "",
            f"Brier：`{result['emit_calibration']['brier']}`；符号准确率：`{result['emit_calibration']['sign_accuracy_at_0_5']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-eval-batches", type=int, default=4)
    parser.add_argument("--emit-slots", default="1,4,8,12,15")
    parser.add_argument("--compute-shadow-price", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["device"] = cli.device
    config["max_eval_batches"] = cli.max_eval_batches
    config["num_workers"] = 0
    args = Namespace(**config)
    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    slots = sorted({int(value) for value in cli.emit_slots.split(",") if value.strip()})

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    runtime_state = _restore_runtime_state(model, payload)
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
    model.eval()
    backbone.eval()
    _, eval_loader = make_dataloaders(args)

    modes = ["rich_all_readouts", "null_fallback_only", "policy"]
    if args.use_memory:
        modes.extend(
            [
                "memory_zero",
                "memory_stale_batch",
                "memory_skip_execution",
                "memory_zero_fixed_emit",
                "memory_stale_batch_fixed_emit",
                "memory_skip_execution_fixed_emit",
            ]
        )
    if args.use_ar:
        modes.extend(["small_ar_skip_execution", "small_ar_skip_execution_fixed_emit"])
    modes.extend(f"drop_emit_slot_{slot}" for slot in slots if 0 < slot < args.readout_vectors)
    totals = {mode: ModeTotals() for mode in modes}

    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(args.eval_mask_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_mask_seed)
        for batch_index, batch in enumerate(eval_loader):
            if batch_index >= cli.max_eval_batches:
                break
            clean = batch[0].to(device, non_blocking=device.type == "cuda")
            valid = clean.ne(PAD_ID)
            byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
            baseline_active = _baseline_active_masks(model, clean, byte_mask)
            for mode in modes:
                _run_mode_batch(
                    model,
                    backbone,
                    clean,
                    byte_mask,
                    args,
                    mode,
                    totals[mode],
                    slots,
                    active_override=baseline_active if mode.endswith("_fixed_emit") else None,
                )
            print(f"[CBIU V0] batch {batch_index + 1}/{cli.max_eval_batches}", flush=True)

    rows = {mode: value.as_dict() for mode, value in totals.items()}
    rich = rows["rich_all_readouts"]
    null = rows["null_fallback_only"]
    invalid_dimensions: set[str] = set()
    for row in rows.values():
        normalized, invalid = normalize_risks(row, rich, null)
        row["normalized_risks"] = normalized
        row["rho"] = robust_risk(normalized)
        invalid_dimensions.update(invalid)

    baseline = rows["policy"]
    for name, row in rows.items():
        if name in {"rich_all_readouts", "null_fallback_only", "policy"} or row["rho"] is None or baseline["rho"] is None:
            row["quality_utility_keep"] = None
            row["readout_cost_delta"] = row["actual_readouts_per_byte"] - baseline["actual_readouts_per_byte"]
            row["net_utility_keep"] = None
            continue
        quality_utility = row["rho"] - baseline["rho"]
        readout_cost_delta = row["actual_readouts_per_byte"] - baseline["actual_readouts_per_byte"]
        row["quality_utility_keep"] = quality_utility
        row["readout_cost_delta"] = readout_cost_delta
        row["net_utility_keep"] = (
            quality_utility + cli.compute_shadow_price * readout_cost_delta
            if name.startswith("drop_emit_slot_")
            else None
        )

    result = {
        "protocol": "CBIU_V0_OFFLINE_20260716",
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "runtime_boundary_state": runtime_state,
        "max_eval_batches": cli.max_eval_batches,
        "emit_slots": slots,
        "compute_shadow_price_per_readout_per_byte": cli.compute_shadow_price,
        "cost_coverage": {
            "emit": "actual compact readout count proxy",
            "memory": "quality only; profiler cost pending",
            "small_ar": "quality only; profiler cost pending",
            "boundary": "not implemented; strict merge rebuild pending",
        },
        "invalid_anchor_dimensions": sorted(invalid_dimensions),
        "modes": rows,
        "emit_calibration": _calibration(rows, baseline, slots),
    }

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cbiu_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "cbiu_v0.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result["emit_calibration"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
