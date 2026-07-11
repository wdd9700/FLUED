from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CodingRateSelection:
    marginal_rate: torch.Tensor
    hard_cut: torch.Tensor
    soft_cut: torch.Tensor
    target_chunks: torch.Tensor


class MarginalCodingRateSelector(nn.Module):
    """ByteFlow-style marginal coding rate with fixed/bucketed Top-K.

    The exact path computes all prefix log-determinants in parallel in a small
    projected feature space. The l2 path is the low-cost screening ablation.
    """

    def __init__(
        self,
        dim: int,
        rate_dim: int = 16,
        epsilon: float = 1.0,
        temperature: float = 0.15,
        mode: str = "exact",
    ) -> None:
        super().__init__()
        if mode not in {"exact", "l2"}:
            raise ValueError("coding-rate mode must be exact or l2")
        self.rate_dim = int(rate_dim)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, self.rate_dim, bias=False)

    def _marginal_rate(self, features: torch.Tensor) -> torch.Tensor:
        device_type = features.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            z = self.proj(self.norm(features).float()).float()
            if self.mode == "l2":
                return 0.5 * torch.log1p(z.square().sum(dim=-1) / (self.epsilon**2))
            outer = z.unsqueeze(-1) * z.unsqueeze(-2)
            prefix = outer.cumsum(dim=1)
            alpha = self.rate_dim / (self.epsilon**2)
            eye = torch.eye(self.rate_dim, device=z.device, dtype=torch.float32)
            matrices = eye.view(1, 1, self.rate_dim, self.rate_dim) + alpha * prefix
            sign, logabsdet = torch.linalg.slogdet(matrices)
            rate = 0.5 * torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))
            previous = torch.cat([torch.zeros_like(rate[:, :1]), rate[:, :-1]], dim=1)
            return rate - previous

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        forbidden: torch.Tensor,
        max_chunks: int,
        fixed_chunks: int = 0,
        bytes_per_chunk: int = 16,
    ) -> CodingRateSelection:
        marginal = self._marginal_rate(features)
        eligible = valid & ~forbidden
        first = valid.float().argmax(dim=1)
        first_mask = torch.zeros_like(valid)
        first_mask[torch.arange(valid.size(0), device=valid.device), first] = valid.any(dim=1)
        eligible = eligible | first_mask

        valid_len = valid.sum(dim=1)
        if fixed_chunks > 0:
            target = torch.full_like(valid_len, int(fixed_chunks))
        else:
            target = torch.div(valid_len + bytes_per_chunk - 1, bytes_per_chunk, rounding_mode="floor")
        eligible_count = eligible.sum(dim=1)
        target = target.clamp(min=1, max=int(max_chunks)).minimum(eligible_count.clamp(min=1))

        score = marginal.masked_fill(~eligible, torch.finfo(marginal.dtype).min)
        order = torch.argsort(score, dim=1, descending=True)
        rank = torch.empty_like(order)
        positions = torch.arange(score.size(1), device=score.device).view(1, -1).expand_as(order)
        rank.scatter_(1, order, positions)
        hard = rank.lt(target.unsqueeze(1)) & eligible
        hard = hard | first_mask

        sorted_score = torch.gather(score, 1, order)
        kth = torch.gather(sorted_score, 1, (target - 1).unsqueeze(1)).detach()
        soft = torch.sigmoid((score - kth) / self.temperature) * eligible.float()
        soft = torch.where(first_mask, torch.ones_like(soft), soft)
        return CodingRateSelection(marginal, hard, soft, target)


@dataclass
class EmitDecision:
    logits: torch.Tensor
    soft: torch.Tensor
    hard: torch.Tensor
    straight_through: torch.Tensor


class ReadoutEmitController(nn.Module):
    """One fallback readout plus value-gated optional readouts."""

    def __init__(self, dim: int, initial_extra_probability: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        probability = min(max(float(initial_extra_probability), 1.0e-4), 1.0 - 1.0e-4)
        bias = torch.logit(torch.tensor(probability)).item()
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, candidates: torch.Tensor, chunk_mask: torch.Tensor) -> EmitDecision:
        # Controller losses specialize the controller rather than distorting a
        # candidate representation merely to change its emit score.
        logits = self.head(self.norm(candidates.detach())).squeeze(-1)
        soft = torch.sigmoid(logits) * chunk_mask.unsqueeze(-1).to(logits.dtype)
        hard = soft.ge(0.5) & chunk_mask.unsqueeze(-1)
        if hard.size(-1):
            hard = hard.clone()
            hard[..., 0] = chunk_mask
            soft = soft.clone()
            soft[..., 0] = chunk_mask.to(soft.dtype)
        straight_through = soft + (hard.to(soft.dtype) - soft).detach()
        return EmitDecision(logits, soft, hard, straight_through)
