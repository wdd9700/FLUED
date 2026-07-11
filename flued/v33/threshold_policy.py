from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ThresholdPolicyOutput:
    confidence: torch.Tensor
    hard_cut: torch.Tensor
    soft_transition: torch.Tensor
    force_continue: torch.Tensor
    aux: dict


class DualThresholdPolicy(nn.Module):
    """Convert signed confidence to executable chunk decisions."""

    def __init__(self, tau_cut: float = 0.90, tau_trans: float = 0.75, tau_keep: float = 0.65) -> None:
        super().__init__()
        self.tau_cut = float(tau_cut)
        self.tau_trans = float(tau_trans)
        self.tau_keep = float(tau_keep)

    def forward(self, confidence: torch.Tensor, valid: torch.Tensor) -> ThresholdPolicyOutput:
        hard_cut = confidence.gt(self.tau_cut) & valid
        soft_transition = confidence.gt(self.tau_trans) & confidence.le(self.tau_cut) & valid
        force_continue = confidence.lt(-self.tau_keep) & valid
        hard_cut = hard_cut & ~force_continue
        if valid.ndim == 2:
            first = valid.float().argmax(dim=1)
            hard_cut = hard_cut.clone()
            hard_cut[torch.arange(valid.size(0), device=valid.device), first] = valid.any(dim=1)
        return ThresholdPolicyOutput(
            confidence=confidence,
            hard_cut=hard_cut,
            soft_transition=soft_transition,
            force_continue=force_continue,
            aux={"tau_cut": self.tau_cut, "tau_trans": self.tau_trans, "tau_keep": self.tau_keep},
        )
