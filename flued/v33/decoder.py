from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from flued.data import BYTE_VOCAB_SIZE


class SharedParameterDecoder(nn.Module):
    """Latent-to-byte span decoder for v3.3 interface tests."""

    def __init__(
        self,
        d_z: int,
        hidden: int,
        max_span: int,
        vocab_size: int = BYTE_VOCAB_SIZE,
        byte_lookup: nn.Module | None = None,
        d_model: int | None = None,
    ) -> None:
        super().__init__()
        self.max_span = int(max_span)
        self.vocab_size = int(vocab_size)
        self.byte_lookup = byte_lookup
        self.length_head = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.max_span),
        )
        self.slot_embed = nn.Embedding(self.max_span, d_z)
        self.slot_decoder = nn.Sequential(
            nn.LayerNorm(d_z),
            nn.Linear(d_z, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        if byte_lookup is None:
            self.byte_head = nn.Linear(hidden, vocab_size)
            self.byte_feature_head = None
            self.logit_scale = None
        else:
            if d_model is None:
                raise ValueError("d_model is required when byte_lookup is tied")
            self.byte_head = None
            self.byte_feature_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(d_model)))
            self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        z_content: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
        readout_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = torch.arange(self.max_span, device=z_content.device)
        slot_embed = self.slot_embed(slots).to(z_content.dtype)
        if z_content.ndim == 3:
            readout_z = z_content.unsqueeze(2)
            gate = torch.ones((*z_content.shape[:2], 1), dtype=z_content.dtype, device=z_content.device)
        elif z_content.ndim == 4:
            readout_z = z_content
            if readout_gate is None:
                gate = torch.ones(z_content.shape[:-1], dtype=z_content.dtype, device=z_content.device)
            else:
                gate = readout_gate.to(z_content.dtype)
        else:
            raise ValueError("z_content must be [B,C,D] or [B,C,R,D]")

        scores = torch.einsum("sd,bcrd->bcsr", slot_embed, readout_z) / max(float(readout_z.size(-1)), 1.0) ** 0.5
        scores = scores + gate.clamp(min=1.0e-6).log().unsqueeze(2)
        attn = torch.softmax(scores.float(), dim=-1).to(readout_z.dtype)
        context = torch.einsum("bcsr,bcrd->bcsd", attn, readout_z)
        slot_h = context + slot_embed.view(1, 1, self.max_span, -1)
        byte_h = self.slot_decoder(slot_h)
        if self.byte_lookup is None:
            byte_logits = self.byte_head(byte_h)
        else:
            vocab_ids = torch.arange(self.vocab_size, device=z_content.device)
            vocab_h = self.byte_lookup(vocab_ids).to(dtype=byte_h.dtype)
            byte_features = self.byte_feature_head(byte_h)
            byte_logits = self.logit_scale.clamp(min=1.0, max=100.0) * torch.matmul(
                F.normalize(byte_features, dim=-1),
                F.normalize(vocab_h, dim=-1).transpose(0, 1),
            )
        denom = gate.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        z_summary = readout_z.sum(dim=2) / denom
        length_logits = self.length_head(z_summary)
        if chunk_mask is not None:
            byte_logits = byte_logits.masked_fill(~chunk_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            length_logits = length_logits.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)
        return byte_logits, length_logits
