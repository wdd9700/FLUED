"""Counterfactual Byte-Interface Utility training primitives.

The model-specific interventions live in the v3.4 trainer.  This module keeps
the protocol state and the rate-distortion-style utility math independent and
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


CBIU_PROTOCOL_VERSION = "CBIU_V1_ONLINE_20260717"
RISK_NAMES = ("reconstruction_bpb", "completion_bpb", "preservation_bpb")


def _anchor_vector(report: dict[str, Any], mode: str) -> list[float]:
    row = report["modes"][mode]
    return [float(row[name]) for name in RISK_NAMES]


@dataclass
class CBIUState:
    protocol_version: str
    anchor_rich: torch.Tensor
    anchor_null: torch.Tensor
    anchor_file: str
    anchor_checkpoint: str
    compute_budget: float
    compute_dual: torch.Tensor
    update_count: int = 0

    @classmethod
    def from_anchor_file(
        cls,
        anchor_file: str,
        compute_budget: float,
        device: torch.device,
    ) -> "CBIUState":
        path = Path(anchor_file).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        rich = torch.tensor(_anchor_vector(report, "rich_all_readouts"), device=device)
        null = torch.tensor(_anchor_vector(report, "null_fallback_only"), device=device)
        if not torch.all(null > rich):
            bad = [
                name
                for name, rich_value, null_value in zip(RISK_NAMES, rich.tolist(), null.tolist())
                if null_value <= rich_value
            ]
            raise ValueError(f"CBIU anchors do not separate rich/null for: {bad}")
        return cls(
            protocol_version=CBIU_PROTOCOL_VERSION,
            anchor_rich=rich,
            anchor_null=null,
            anchor_file=str(path),
            anchor_checkpoint=str(report.get("checkpoint", "")),
            compute_budget=float(compute_budget),
            compute_dual=torch.zeros((), device=device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "anchor_rich": self.anchor_rich.detach().cpu(),
            "anchor_null": self.anchor_null.detach().cpu(),
            "anchor_file": self.anchor_file,
            "anchor_checkpoint": self.anchor_checkpoint,
            "compute_budget": self.compute_budget,
            "compute_dual": self.compute_dual.detach().cpu(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, payload: dict[str, Any], device: torch.device) -> None:
        if payload.get("protocol_version") != self.protocol_version:
            raise RuntimeError(
                "CBIU protocol mismatch: "
                f"checkpoint={payload.get('protocol_version')!r}, current={self.protocol_version!r}"
            )
        saved_rich = payload["anchor_rich"].to(device=device, dtype=self.anchor_rich.dtype)
        saved_null = payload["anchor_null"].to(device=device, dtype=self.anchor_null.dtype)
        if not torch.equal(saved_rich, self.anchor_rich) or not torch.equal(saved_null, self.anchor_null):
            raise RuntimeError("CBIU anchor mismatch while resuming; refusing to reset utility scale")
        saved_budget = float(payload["compute_budget"])
        if abs(saved_budget - self.compute_budget) > 1.0e-12:
            raise RuntimeError(
                f"CBIU compute budget mismatch: checkpoint={saved_budget}, current={self.compute_budget}"
            )
        self.compute_dual.copy_(payload["compute_dual"].to(device))
        self.update_count = int(payload.get("update_count", 0))


def normalize_cbiu_risks(
    risks: torch.Tensor,
    rich: torch.Tensor,
    null: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize [..., 3] BPB risks against frozen rich/null anchors."""

    gap = null - rich
    if bool((gap <= epsilon).any()):
        raise ValueError("CBIU anchor gap must be positive in every risk dimension")
    return (risks - rich) / gap


def robust_cbiu_risk(normalized_risks: torch.Tensor) -> torch.Tensor:
    """Worst normalized task risk; lower is better."""

    return normalized_risks.max(dim=-1).values


def cbiu_keep_utility(
    on_risks: torch.Tensor,
    off_risks: torch.Tensor,
    on_cost: torch.Tensor,
    off_cost: torch.Tensor,
    state: CBIUState,
    augmented_weight: float,
) -> dict[str, torch.Tensor]:
    """Return positive utility when retaining an emit action is preferable."""

    on_normalized = normalize_cbiu_risks(on_risks, state.anchor_rich, state.anchor_null)
    off_normalized = normalize_cbiu_risks(off_risks, state.anchor_rich, state.anchor_null)
    rho_on = robust_cbiu_risk(on_normalized)
    rho_off = robust_cbiu_risk(off_normalized)
    budget = max(float(state.compute_budget), 1.0e-8)
    on_violation = torch.relu(on_cost / budget - 1.0)
    off_violation = torch.relu(off_cost / budget - 1.0)
    dual = state.compute_dual.detach()
    objective_on = rho_on + dual * on_violation + 0.5 * float(augmented_weight) * on_violation.square()
    objective_off = rho_off + dual * off_violation + 0.5 * float(augmented_weight) * off_violation.square()
    return {
        "on_normalized": on_normalized,
        "off_normalized": off_normalized,
        "rho_on": rho_on,
        "rho_off": rho_off,
        "quality_utility": rho_off - rho_on,
        "cost_utility": (objective_off - rho_off) - (objective_on - rho_on),
        "net_utility": objective_off - objective_on,
        "on_violation": on_violation,
        "off_violation": off_violation,
    }


@torch.no_grad()
def update_cbiu_dual(
    state: CBIUState,
    actual_readouts_per_byte: torch.Tensor,
    dual_lr: float,
    dual_max: float,
) -> torch.Tensor:
    """Projected dual ascent on the batch-average hard execution budget."""

    budget = max(float(state.compute_budget), 1.0e-8)
    constraint = actual_readouts_per_byte.detach().float().mean() / budget - 1.0
    state.compute_dual.add_(float(dual_lr) * constraint).clamp_(min=0.0, max=float(dual_max))
    state.update_count += 1
    return constraint
