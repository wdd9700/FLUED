from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SegmentorOutput:
    confidence: torch.Tensor
    logits: torch.Tensor
    aux: dict


class SignedBoundarySegmentor(nn.Module):
    """Lightweight signed boundary field.

    The output is not a cut probability.  Positive values indicate cut pressure;
    negative values indicate continuation pressure.
    """

    def __init__(self, d_model: int, hidden: int | None = None, temperature: float = 1.0) -> None:
        super().__init__()
        hidden = int(hidden or d_model)
        self.temperature = float(temperature)
        self.local = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.logit = nn.Linear(hidden, 1)

    def forward(self, byte_features: torch.Tensor, valid: torch.Tensor | None = None) -> SegmentorOutput:
        h = self.local(byte_features)
        logits = self.logit(h).squeeze(-1)
        confidence = torch.tanh(logits / max(self.temperature, 1.0e-6))
        if valid is not None:
            confidence = confidence.masked_fill(~valid, 0.0)
            logits = logits.masked_fill(~valid, 0.0)
        return SegmentorOutput(confidence=confidence, logits=logits, aux={"temperature": self.temperature})
