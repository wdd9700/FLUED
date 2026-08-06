"""FLUED v3.6 R0 Occam baseline: byte-level KDA-LM hybrid 3:1 (spec section 4).

Decoder-only language model over the RAW byte stream -- no segmentation, no
summarizer, no write head, no readout: the KDA state consumes bytes directly.
KDA layers and full-attention transformer layers interleave 3:1 (Moonshot's
published best ratio, user-ruled). The parameter budget matches the FLUED
v3.6 full stack (~47M) so the R0 verdict is a same-params comparison
(spec section 7: parity => v3.6 closes; the win condition requires better
quality AND better segment-timed speed).

Training/eval protocol mirrors canonical (same streaming corpus, 512-byte
windows, batch 8, 20K steps, lr 2e-4 cosine, same eval lines) so BPB numbers
land on the same scale as the H-Net reproduction anchor (0.653).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from flued.data import PAD_ID

try:
    from fla.layers.kda import KimiDeltaAttention as _KDA
except Exception:  # fla is optional (tests / non-GPU environments)
    _KDA = None


@dataclass
class KDALMConfig:
    """Arm A (wide-shallow) totals 48.21M vs FLUED v3.6 full stack 47.2M;
    the second ratio arm (spec section 4) is d=448/L=16/ffn=1536 (~47.3M)."""

    vocab_size: int = 258  # PAD-offset byte table (PAD=0, byte b -> b+1, MASK=257)
    d_model: int = 512
    n_layers: int = 12  # 3:1 pattern -> 9 KDA + 3 full attention
    kda_head_dim: int = 128
    attn_nhead: int = 8
    ffn_dim: int = 1792
    max_seq: int = 512
    tie_embeddings: bool = True


class _Block(nn.Module):
    """Pre-norm residual block: mixer (KDA or causal attention) + gated MLP."""

    def __init__(self, mixer: nn.Module, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_in = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.ff_out = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        if attn_mask is None:  # KDA layer
            h, *_ = self.mixer(h)
        else:
            h, *_ = self.mixer(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        gate, value = self.ff_in(self.norm2(x)).chunk(2, dim=-1)
        return x + self.ff_out(F.silu(gate) * value)


class KDALM(nn.Module):
    """Byte-level decoder-only KDA/transformer 3:1 hybrid (R0 Occam baseline)."""

    def __init__(self, config: KDALMConfig | None = None) -> None:
        super().__init__()
        self.config = config or KDALMConfig()
        c = self.config
        if _KDA is None:
            raise RuntimeError("KDALM requires the fla package (KimiDeltaAttention)")
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        blocks = []
        for i in range(c.n_layers):
            if i % 4 == 3:  # every 4th layer is full causal attention (3:1)
                mixer = nn.MultiheadAttention(
                    c.d_model, c.attn_nhead, dropout=0.0, batch_first=True, bias=False
                )
                blocks.append(_Block(mixer, c.d_model, c.ffn_dim))
            else:
                mixer = _KDA(
                    hidden_size=c.d_model,
                    head_dim=c.kda_head_dim,
                    num_heads=max(1, c.d_model // c.kda_head_dim),
                    mode="chunk",
                    layer_idx=i,
                )
                blocks.append(_Block(mixer, c.d_model, c.ffn_dim))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) PAD-offset ids -> next-byte logits (B, T, vocab)."""
        causal = torch.triu(
            torch.full((token_ids.size(1), token_ids.size(1)), float("-inf"), device=token_ids.device),
            diagonal=1,
        )
        x = self.embed(token_ids)
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask=causal if i % 4 == 3 else None)
        return self.lm_head(self.norm(x))

    def loss_bpb(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-byte CE (mean over valid positions) and BPB = CE / ln 2.

        Positions with PAD target are excluded; the prediction at position t
        targets byte t+1 (shift by one).
        """
        logits = self.forward(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        valid = targets.ne(PAD_ID)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1).clamp(min=0),
            reduction="none",
            ignore_index=PAD_ID,
        )
        ce = ce.reshape_as(targets)
        mean_ce = (ce * valid).sum() / valid.sum().clamp(min=1)
        return mean_ce, mean_ce / 0.6931471805599453
