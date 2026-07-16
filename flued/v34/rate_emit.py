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
    """Marginal coding-rate scores and historical fixed-budget Top-K.

    The exact path computes all prefix log-determinants in parallel in a small
    projected feature space. ``diag`` is a parallel diagonal-covariance
    approximation to the same prefix log-determinant. ``l2`` is retained only
    to reproduce historical v3.4 checkpoints; it is pointwise energy, not a
    marginal coding rate.
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
        if mode not in {"exact", "diag", "l2"}:
            raise ValueError("coding-rate mode must be exact, diag, or l2")
        self.rate_dim = int(rate_dim)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, self.rate_dim, bias=False)

    def _marginal_rate(self, features: torch.Tensor) -> torch.Tensor:
        device_type = features.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            normalized = self.norm(features).float()
            if self.mode == "diag":
                # Diagonal approximation of
                # 0.5*logdet(I + alpha*sum_i z_i z_i^T). Unlike the historical
                # pointwise L2 score, differencing prefix rates measures the
                # incremental coding contribution at each position and remains
                # fully parallel through cumsum.
                prefix_energy = normalized.square().cumsum(dim=1)
                alpha = 1.0 / (self.epsilon**2)
                prefix_rate = 0.5 * torch.log1p(alpha * prefix_energy).mean(dim=-1)
                previous = torch.cat(
                    [torch.zeros_like(prefix_rate[:, :1]), prefix_rate[:, :-1]], dim=1
                )
                return prefix_rate - previous
            z = self.proj(normalized).float()
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

    def marginal_rate(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-position scores without applying a chunk budget."""
        return self._marginal_rate(features)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        forbidden: torch.Tensor,
        max_chunks: int,
        fixed_chunks: int = 0,
        bytes_per_chunk: int = 16,
        anchor_score: torch.Tensor | None = None,
        blend_alpha: float = 1.0,
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

        eligible_extra = eligible & ~first_mask
        extra_target = (target - 1).clamp(min=0)
        score = marginal
        alpha = 1.0
        if anchor_score is not None:
            alpha = min(max(float(blend_alpha), 0.0), 1.0)

            def normalize(value: torch.Tensor) -> torch.Tensor:
                weight = eligible_extra.to(value.dtype)
                count = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
                mean = (value * weight).sum(dim=1, keepdim=True) / count
                variance = ((value - mean).square() * weight).sum(dim=1, keepdim=True) / count
                return (value - mean) / variance.sqrt().clamp(min=1.0e-5)

            score = alpha * normalize(marginal) + (1.0 - alpha) * normalize(anchor_score.to(marginal.dtype))
        score = score.masked_fill(~eligible_extra, torch.finfo(marginal.dtype).min)
        order = torch.argsort(score, dim=1, descending=True)
        rank = torch.empty_like(order)
        positions = torch.arange(score.size(1), device=score.device).view(1, -1).expand_as(order)
        rank.scatter_(1, order, positions)
        hard = rank.lt(extra_target.unsqueeze(1)) & eligible_extra
        hard = hard | first_mask

        sorted_score = torch.gather(score, 1, order)
        kth_index = (extra_target - 1).clamp(min=0)
        kth = torch.gather(sorted_score, 1, kth_index.unsqueeze(1)).detach()
        soft = torch.sigmoid((score - kth) / self.temperature) * eligible_extra.float()
        soft = soft * extra_target.gt(0).unsqueeze(1).to(soft.dtype)
        if anchor_score is not None:
            anchor_soft = anchor_score.to(soft.dtype).clamp(min=0.0, max=1.0) * eligible_extra.float()
            soft = (1.0 - alpha) * anchor_soft + alpha * soft
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

    def __init__(self, dim: int, initial_extra_probability: float = 0.1, threshold: float = 0.5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        self.threshold = float(threshold)
        probability = min(max(float(initial_extra_probability), 1.0e-4), 1.0 - 1.0e-4)
        bias = torch.logit(torch.tensor(probability)).item()
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, candidates: torch.Tensor, chunk_mask: torch.Tensor) -> EmitDecision:
        # Controller losses specialize the controller rather than distorting a
        # candidate representation merely to change its emit score.
        logits = self.head(self.norm(candidates.detach())).squeeze(-1)
        soft = torch.sigmoid(logits) * chunk_mask.unsqueeze(-1).to(logits.dtype)
        hard = soft.ge(self.threshold) & chunk_mask.unsqueeze(-1)
        if hard.size(-1):
            hard = hard.clone()
            hard[..., 0] = chunk_mask
            soft = soft.clone()
            soft[..., 0] = chunk_mask.to(soft.dtype)
        straight_through = soft + (hard.to(soft.dtype) - soft).detach()
        return EmitDecision(logits, soft, hard, straight_through)
