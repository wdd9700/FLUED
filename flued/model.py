"""FLUED v0.4 minimal model.

ONE idea: semantic units are dynamically compiled in prefill and decode is an
inverse/tied-weight expansion process.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TiedTransformerBlock(nn.Module):
    """Single residual block used in both encode and inverse decode."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward_block(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(y)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

    def inverse_block(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Approximate inverse using the same tied parameters.
        y, _ = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=key_padding_mask, need_weights=False)
        x = x - self.drop(y)
        x = x - self.drop(self.ffn(self.ln2(x)))
        return x


class FLUEDAutoencoder(nn.Module):
    """FLUED v0.4 dynamic semantic-unit autoencoder."""

    def __init__(
        self,
        vocab_size: int = 257,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        boundary_threshold: float = 0.5,
        target_compression: float = 0.3,
        **_: Any,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.boundary_threshold = boundary_threshold
        self.target_compression = target_compression

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_seq_len * 2, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TiedTransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.boundary_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def _encode_hidden(self, src: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embedding(src))
        for block in self.blocks:
            h = block.forward_block(h, key_padding_mask=padding_mask)
        return h

    def _boundary_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(hidden)
        delta[:, 1:] = hidden[:, 1:] - hidden[:, :-1]
        return self.boundary_head(delta).squeeze(-1)

    @staticmethod
    def _hard_spans(boundaries: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[List[List[Tuple[int, int]]], torch.Tensor]:
        # returns spans and span_id map [B,T]
        bsz, seq_len = boundaries.shape
        span_ids = torch.zeros_like(boundaries, dtype=torch.long)
        all_spans: List[List[Tuple[int, int]]] = []
        for b in range(bsz):
            spans: List[Tuple[int, int]] = []
            start = None
            sid = -1
            for t in range(seq_len):
                if not valid_mask[b, t]:
                    continue
                if start is None:
                    start = t
                    sid += 1
                elif boundaries[b, t]:
                    spans.append((start, t))
                    start = t
                    sid += 1
                span_ids[b, t] = sid
            if start is not None:
                last = int(valid_mask[b].sum().item())
                spans.append((start, last))
            all_spans.append(spans)
        return all_spans, span_ids

    def _compile_semantic_units(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        probs = torch.sigmoid(scores)
        hard = probs > self.boundary_threshold
        hard[:, 0] = True
        hard = hard & valid_mask

        spans, span_ids = self._hard_spans(hard, valid_mask)
        bsz, seq_len, dim = hidden.shape
        num_units = [max(1, len(s)) for s in spans]
        m_max = max(num_units)
        z = hidden.new_zeros((bsz, m_max, dim))

        for b in range(bsz):
            if not spans[b]:
                continue
            for sid, (start, end) in enumerate(spans[b]):
                z[b, sid] = hidden[b, start:end].mean(dim=0)

        expanded = hidden.new_zeros((bsz, seq_len, dim))
        for b in range(bsz):
            for t in range(seq_len):
                if valid_mask[b, t]:
                    expanded[b, t] = z[b, span_ids[b, t]]

        lengths = valid_mask.sum(dim=1).clamp(min=1)
        m_over_n = torch.tensor(num_units, device=hidden.device, dtype=torch.float32) / lengths.float()
        compression_loss = (m_over_n - self.target_compression).pow(2).mean()

        metrics: Dict[str, Any] = {
            "m_over_n": m_over_n,
            "num_units": num_units,
            "spans": spans,
            "span_lengths": [[e - s for s, e in item] for item in spans],
            "compression_loss": compression_loss,
        }
        return z, expanded, metrics

    def _inverse_decode(self, expanded: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = expanded
        for block in reversed(self.blocks):
            x = block.inverse_block(x, key_padding_mask=padding_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Stage A strict reconstruction: tgt is intentionally ignored.
        _ = tgt
        padding_mask = src.eq(0)
        valid_mask = ~padding_mask

        hidden = self._encode_hidden(src, padding_mask)
        boundary_scores = self._boundary_scores(hidden)
        z, expanded, metrics = self._compile_semantic_units(hidden, boundary_scores, valid_mask)

        inv_hidden = self._inverse_decode(expanded, padding_mask)
        logits = F.linear(inv_hidden, self.embedding.weight)

        metrics["boundary_scores"] = boundary_scores
        metrics["num_units_tensor"] = torch.tensor(metrics["num_units"], device=src.device)
        metrics["z"] = z
        return logits, metrics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
