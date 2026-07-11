"""Train a small FLUED-v3.2 language-codec prototype.

This file covers the staged v3.2 prototype.  Stage 1 replaces the 256-way byte
embedding lookup with a 16x16 factorized byte coordinate seed.  Stage 2 keeps
boundary memory-free and adds causal past-memory retrieval to the interpreter:

  bytes -> weak boundary proposal -> shared segment representation
        -> causal summary slots written as append-only internal memory
        -> top-k retrieval from previous chunk memories
        -> readout latent sequence (external backbone interface)
        -> latent+length decoder -> byte span reconstruction

Boundary remains memory-free, backbone sees only readout latent, and decoder
does not read encoder memory.  The non-causal byte encoder switch is kept only
for leakage/ablation checks; mainline Stage 2 evidence must use the default
causal byte encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import (
    BYTE_OFFSET,
    BYTE_VOCAB_SIZE,
    ByteReconstructionDataset,
    MASK_ID,
    PAD_ID,
    STUB_CORPUS,
    StreamingReconstructionDataset,
)
from tools.analysis.train_v3_commit_controller_small import (
    _append_jsonl,
    _cosine_with_warmup,
    _load_texts,
)


PUNCT_OR_SPACE = set(b" \t\r\n.,;:!?()[]{}<>\"'`~@#$%^&*-+=_/\\|")


def _raw_byte(token_id: int) -> int:
    return int(token_id) - BYTE_OFFSET


def _is_utf8_continuation(token_id: int) -> bool:
    if token_id <= PAD_ID or token_id >= MASK_ID:
        return False
    b = _raw_byte(token_id)
    return 0x80 <= b <= 0xBF


def _utf8_codepoint_len_from_start(token_id: int) -> int:
    if token_id <= PAD_ID or token_id >= MASK_ID:
        return 1
    b = _raw_byte(token_id)
    if b < 0x80:
        return 1
    if 0xC2 <= b <= 0xDF:
        return 2
    if 0xE0 <= b <= 0xEF:
        return 3
    if 0xF0 <= b <= 0xF4:
        return 4
    return 1


def weak_boundary_starts(
    src: torch.Tensor,
    valid: torch.Tensor,
    min_span: int,
    max_span: int,
) -> torch.Tensor:
    """Weak segment-start labels.

    The first valid byte starts a segment. Later starts are allowed only at
    UTF-8 codepoint boundaries and are encouraged after punctuation/space or
    when max_span is reached.
    """

    bsz, seq_len = src.shape
    starts = torch.zeros_like(valid, dtype=torch.bool)
    min_span = max(1, int(min_span))
    max_span = max(min_span, int(max_span))

    for b in range(bsz):
        span_len = 0
        for t in range(seq_len):
            if not bool(valid[b, t]):
                continue
            tok = int(src[b, t].item())
            if span_len == 0:
                starts[b, t] = True
                span_len = 1
                continue

            prev_tok = int(src[b, t - 1].item()) if t > 0 else PAD_ID
            can_start = not _is_utf8_continuation(tok)
            prev_raw = _raw_byte(prev_tok) if BYTE_OFFSET <= prev_tok < MASK_ID else -1
            cur_raw = _raw_byte(tok) if BYTE_OFFSET <= tok < MASK_ID else -1
            weak_break = prev_raw in PUNCT_OR_SPACE or cur_raw in PUNCT_OR_SPACE
            # If the current byte starts a multi-byte codepoint that would not
            # fit, start a new segment before it. This keeps decoder targets
            # within max_span without cutting UTF-8 codepoints.
            forced_break = span_len + _utf8_codepoint_len_from_start(tok) > max_span

            if can_start and span_len >= min_span and (weak_break or forced_break):
                starts[b, t] = True
                span_len = 1
            else:
                span_len += 1
    return starts


def complete_utf8_edge_valid(src: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Drop incomplete UTF-8 codepoints at chunk edges.

    Byte windows can start or end in the middle of a multi-byte codepoint. The
    codec target should only contain complete byte spans, so edge fragments are
    masked out before weak segmentation. Internal UTF-8 validity is still
    enforced by boundary placement.
    """

    out = valid.clone()
    bsz, seq_len = src.shape
    for b in range(bsz):
        idx = out[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        first = int(idx[0].item())
        last = int(idx[-1].item())

        while first <= last and _is_utf8_continuation(int(src[b, first].item())):
            out[b, first] = False
            first += 1
        if first > last:
            continue

        start = last
        while start > first and _is_utf8_continuation(int(src[b, start].item())):
            start -= 1
        expected = _utf8_codepoint_len_from_start(int(src[b, start].item()))
        if start + expected - 1 > last:
            out[b, start : last + 1] = False
    return out


def build_segments(
    src: torch.Tensor,
    valid: torch.Tensor,
    starts: torch.Tensor,
    max_units: int,
    max_span: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build segment ids and padded byte-span targets.

    Returns:
      seg_ids: [B,T], -1 for invalid/truncated bytes.
      seg_targets: [B,U,S], PAD_ID beyond length.
      seg_lengths: [B,U], 0 for inactive units.
      seg_mask: [B,U].
    """

    bsz, seq_len = src.shape
    device = src.device
    seg_ids = torch.full((bsz, seq_len), -1, dtype=torch.long, device=device)
    seg_targets = torch.full((bsz, max_units, max_span), PAD_ID, dtype=torch.long, device=device)
    seg_lengths = torch.zeros((bsz, max_units), dtype=torch.long, device=device)
    seg_mask = torch.zeros((bsz, max_units), dtype=torch.bool, device=device)

    for b in range(bsz):
        unit = -1
        slot = 0
        for t in range(seq_len):
            if not bool(valid[b, t]):
                continue
            if bool(starts[b, t]) or unit < 0:
                unit += 1
                slot = 0
            if unit >= max_units:
                break
            seg_ids[b, t] = unit
            if slot < max_span:
                seg_targets[b, unit, slot] = src[b, t]
                seg_lengths[b, unit] = slot + 1
                seg_mask[b, unit] = True
            slot += 1
    return seg_ids, seg_targets, seg_lengths, seg_mask


def segment_mean_pool(h: torch.Tensor, seg_ids: torch.Tensor, seg_mask: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, hidden = h.shape
    max_units = seg_mask.size(1)
    valid = seg_ids >= 0
    idx = seg_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden)
    pooled = h.new_zeros((bsz, max_units, hidden))
    pooled.scatter_add_(1, idx, h * valid.unsqueeze(-1).to(h.dtype))
    counts = h.new_zeros((bsz, max_units, 1))
    counts.scatter_add_(1, seg_ids.clamp(min=0).unsqueeze(-1), valid.unsqueeze(-1).to(h.dtype))
    pooled = pooled / counts.clamp(min=1.0)
    return pooled * seg_mask.unsqueeze(-1).to(h.dtype)


def segment_edge_pool(h: torch.Tensor, seg_ids: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return first and last token states for each segment.

    Mean pooling alone is weak for longer spans because it blurs local order.
    First/last states add cheap boundary and direction information while the
    exported interface remains one readout latent per segment.
    """

    bsz, seq_len, hidden = h.shape
    max_units = seg_mask.size(1)
    valid = seg_ids >= 0
    idx = seg_ids.clamp(min=0)
    pos = torch.arange(seq_len, device=h.device).view(1, seq_len).expand(bsz, -1)

    first_src = torch.where(valid, pos, torch.full_like(pos, seq_len))
    first_pos = torch.full((bsz, max_units), seq_len, dtype=torch.long, device=h.device)
    first_pos.scatter_reduce_(1, idx, first_src, reduce="amin", include_self=True)

    last_src = torch.where(valid, pos, torch.full_like(pos, -1))
    last_pos = torch.full((bsz, max_units), -1, dtype=torch.long, device=h.device)
    last_pos.scatter_reduce_(1, idx, last_src, reduce="amax", include_self=True)

    batch = torch.arange(bsz, device=h.device).view(bsz, 1).expand(-1, max_units)
    first = h[batch, first_pos.clamp(min=0, max=max(seq_len - 1, 0))]
    last = h[batch, last_pos.clamp(min=0, max=max(seq_len - 1, 0))]
    mask = seg_mask.unsqueeze(-1).to(h.dtype)
    return first * mask, last * mask


class CodecCollator:
    """Create segment targets on CPU, preferably inside DataLoader workers.

    The weak boundary builder and span packer intentionally use simple Python
    loops because they are rule-based prototype code. Running them after
    ``src.to("cuda")`` would force thousands of tiny GPU synchronizations per
    batch, so the training path keeps this work on CPU and only transfers the
    finished tensors.
    """

    def __init__(self, min_span: int, max_span: int, max_units: int) -> None:
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        self.max_units = int(max_units)

    def __call__(self, batch):
        src = torch.stack([item[0] for item in batch], dim=0).long()
        valid = complete_utf8_edge_valid(src, src.ne(PAD_ID))
        starts = weak_boundary_starts(src, valid, self.min_span, self.max_span)
        max_units = min(self.max_units, src.size(1))
        seg_ids, targets, lengths, seg_mask = build_segments(src, valid, starts, max_units, self.max_span)
        return src, starts, seg_ids, targets, lengths, seg_mask


def move_codec_batch(batch, device: torch.device):
    src, starts, seg_ids, targets, lengths, seg_mask = batch
    non_blocking = device.type == "cuda"
    return (
        src.to(device, non_blocking=non_blocking),
        starts.to(device, non_blocking=non_blocking),
        seg_ids.to(device, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
        lengths.to(device, non_blocking=non_blocking),
        seg_mask.to(device, non_blocking=non_blocking),
    )


class ResidualFF(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
        )
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        x = x + self.gate(y) * self.ff(y)
        return x * mask.unsqueeze(-1).to(x.dtype)


class FactorizedByteSeed(nn.Module):
    """16x16 factorized byte-coordinate seed.

    This is deliberately not a full 256-entry byte lookup.  Raw byte ids are
    decomposed into high/low nibbles, then mixed with coarse byte-type features.
    Semantic content must still come from byte streams and context.
    """

    TYPE_PAD = 0
    TYPE_MASK = 1
    TYPE_ASCII_LETTER = 2
    TYPE_ASCII_DIGIT = 3
    TYPE_ASCII_SPACE = 4
    TYPE_ASCII_PUNCT = 5
    TYPE_UTF8_CONT = 6
    TYPE_UTF8_START2 = 7
    TYPE_UTF8_START3 = 8
    TYPE_UTF8_START4 = 9
    TYPE_OTHER = 10
    NUM_TYPES = 11

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.row_embed = nn.Embedding(16, d_model)
        self.col_embed = nn.Embedding(16, d_model)
        self.type_embed = nn.Embedding(self.NUM_TYPES, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _byte_types(raw: torch.Tensor, valid_byte: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        types = torch.full_like(raw, FactorizedByteSeed.TYPE_OTHER)
        types = torch.where(token_ids.eq(PAD_ID), torch.full_like(types, FactorizedByteSeed.TYPE_PAD), types)
        types = torch.where(token_ids.eq(MASK_ID), torch.full_like(types, FactorizedByteSeed.TYPE_MASK), types)

        ascii_letter = ((65 <= raw) & (raw <= 90)) | ((97 <= raw) & (raw <= 122))
        ascii_digit = (48 <= raw) & (raw <= 57)
        ascii_space = (raw == 9) | (raw == 10) | (raw == 13) | (raw == 32)
        ascii_printable = (32 <= raw) & (raw <= 126)
        ascii_punct = ascii_printable & ~(ascii_letter | ascii_digit | ascii_space)
        utf8_cont = (0x80 <= raw) & (raw <= 0xBF)
        utf8_start2 = (0xC2 <= raw) & (raw <= 0xDF)
        utf8_start3 = (0xE0 <= raw) & (raw <= 0xEF)
        utf8_start4 = (0xF0 <= raw) & (raw <= 0xF4)

        types = torch.where(valid_byte & ascii_letter, torch.full_like(types, FactorizedByteSeed.TYPE_ASCII_LETTER), types)
        types = torch.where(valid_byte & ascii_digit, torch.full_like(types, FactorizedByteSeed.TYPE_ASCII_DIGIT), types)
        types = torch.where(valid_byte & ascii_space, torch.full_like(types, FactorizedByteSeed.TYPE_ASCII_SPACE), types)
        types = torch.where(valid_byte & ascii_punct, torch.full_like(types, FactorizedByteSeed.TYPE_ASCII_PUNCT), types)
        types = torch.where(valid_byte & utf8_cont, torch.full_like(types, FactorizedByteSeed.TYPE_UTF8_CONT), types)
        types = torch.where(valid_byte & utf8_start2, torch.full_like(types, FactorizedByteSeed.TYPE_UTF8_START2), types)
        types = torch.where(valid_byte & utf8_start3, torch.full_like(types, FactorizedByteSeed.TYPE_UTF8_START3), types)
        types = torch.where(valid_byte & utf8_start4, torch.full_like(types, FactorizedByteSeed.TYPE_UTF8_START4), types)
        return types

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp(min=PAD_ID, max=MASK_ID)
        valid_byte = (BYTE_OFFSET <= token_ids) & (token_ids < MASK_ID)
        raw = (token_ids - BYTE_OFFSET).clamp(min=0, max=255)
        hi = torch.where(valid_byte, raw >> 4, torch.zeros_like(raw))
        lo = torch.where(valid_byte, raw & 15, torch.zeros_like(raw))
        types = self._byte_types(raw, valid_byte, token_ids)
        seed = self.row_embed(hi) + self.col_embed(lo) + self.type_embed(types)
        return self.norm(seed)


class V32LanguageCodec2M(nn.Module):
    """Small codec-only FLUED-v3.2 prototype."""

    def __init__(
        self,
        vocab_size: int = BYTE_VOCAB_SIZE,
        d_model: int = 192,
        hidden: int = 192,
        nhead: int = 4,
        encoder_layers: int = 2,
        ffn_dim: int = 768,
        max_span: int = 16,
        refine_steps: int = 1,
        dropout: float = 0.0,
        pool_mode: str = "mean",
        memory_slots_per_chunk: int = 2,
        memory_topk: int = 4,
        memory_retrieval_mode: str = "topk",
        causal_byte_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.max_span = int(max_span)
        self.refine_steps = int(refine_steps)
        self.pool_mode = str(pool_mode)
        self.memory_slots_per_chunk = max(0, int(memory_slots_per_chunk))
        self.memory_topk = max(1, int(memory_topk))
        self.memory_retrieval_mode = str(memory_retrieval_mode)
        self.causal_byte_encoder = bool(causal_byte_encoder)
        if self.pool_mode not in {"mean", "mean_first_last"}:
            raise ValueError(f"unsupported pool_mode: {self.pool_mode}")
        if self.memory_retrieval_mode not in {"topk", "random"}:
            raise ValueError(f"unsupported memory_retrieval_mode: {self.memory_retrieval_mode}")
        segment_in_dim = hidden * 3 if self.pool_mode == "mean_first_last" else hidden
        self.byte_seed = FactorizedByteSeed(d_model)
        self.input_proj = nn.Linear(d_model, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.segment_proj = nn.Sequential(
            nn.LayerNorm(segment_in_dim),
            nn.Linear(segment_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.summary_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        if self.memory_slots_per_chunk > 0:
            self.memory_slot_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.memory_slots_per_chunk * hidden),
            )
            self.memory_key = nn.Linear(hidden, hidden, bias=False)
            self.memory_value = nn.Linear(hidden, hidden, bias=False)
            self.memory_query = nn.Linear(hidden, hidden, bias=False)
        else:
            self.memory_slot_head = None
            self.memory_key = None
            self.memory_value = None
            self.memory_query = None
        self.readout_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.refiners = nn.ModuleList([ResidualFF(hidden, dropout=dropout) for _ in range(max(0, refine_steps))])
        self.length_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, max_span),
        )
        self.slot_embed = nn.Embedding(max_span, hidden)
        self.slot_decoder = ResidualFF(hidden, dropout=dropout)
        self.byte_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, vocab_size),
        )

    def retrieve_past_memory(
        self,
        segment: torch.Tensor,
        memory_slots: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Top-k sparse retrieval over strictly previous chunk memory.

        ``memory_slots`` are produced for every chunk, but chunk ``i`` can only
        retrieve slots from chunks ``< i``.  This keeps summary append-only and
        prevents current chunk memory from leaking into current interpretation.
        """

        bsz, units, hidden = segment.shape
        if self.memory_slots_per_chunk <= 0 or memory_slots.numel() == 0:
            zero = segment.new_zeros(segment.shape)
            metrics = {
                "retrieval_weights": segment.new_zeros((bsz, units, 0)),
                "retrieval_indices": torch.empty((bsz, units, 0), dtype=torch.long, device=segment.device),
                "retrieval_valid": torch.empty((bsz, units, 0), dtype=torch.bool, device=segment.device),
                "retrieval_entropy": segment.new_zeros(()),
                "retrieval_valid_frac": segment.new_zeros(()),
                "retrieval_no_past_frac": segment.new_zeros(()),
                "retrieval_active_units": segment.new_zeros(()),
                "retrieval_past_only_violation_count": segment.new_zeros(()),
                "retrieval_max_selected_unit_delta": segment.new_full((), -1.0),
            }
            return zero, metrics

        slots_per_chunk = self.memory_slots_per_chunk
        flat_slots = memory_slots.reshape(bsz, units * slots_per_chunk, hidden)
        keys = self.memory_key(flat_slots)
        values = self.memory_value(flat_slots)
        query = self.memory_query(segment)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(float(hidden))

        slot_units = torch.arange(units, device=segment.device).repeat_interleave(slots_per_chunk)
        current_units = torch.arange(units, device=segment.device)
        past_mask = slot_units.view(1, 1, -1) < current_units.view(1, -1, 1)
        slot_mask = seg_mask.repeat_interleave(slots_per_chunk, dim=1).view(bsz, 1, -1)
        valid_mem = past_mask & slot_mask & seg_mask.view(bsz, units, 1)

        if self.memory_retrieval_mode == "random":
            masked_scores = torch.rand_like(scores).masked_fill(~valid_mem, -1.0e9)
        else:
            masked_scores = scores.masked_fill(~valid_mem, -1.0e9)
        k = min(self.memory_topk, masked_scores.size(-1))
        top_scores, top_idx = torch.topk(masked_scores, k=k, dim=-1)
        top_valid = torch.gather(valid_mem, 2, top_idx)
        has_valid = top_valid.any(dim=-1, keepdim=True)
        if self.memory_retrieval_mode == "random":
            weights = top_valid.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            top_scores = top_scores.masked_fill(~top_valid, -1.0e9)
            weights = torch.softmax(top_scores, dim=-1)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights))

        gather_values = values.unsqueeze(1).expand(-1, units, -1, -1)
        gathered = torch.gather(gather_values, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, hidden))
        context = (weights.unsqueeze(-1) * gathered).sum(dim=2)
        context = context * seg_mask.unsqueeze(-1).to(context.dtype)

        active = seg_mask & has_valid.squeeze(-1)
        safe_weights = weights.clamp(min=1.0e-8)
        entropy_per_unit = -(weights * safe_weights.log()).sum(dim=-1)
        retrieval_entropy = entropy_per_unit[active].mean() if active.any() else segment.new_zeros(())
        retrieval_valid_frac = top_valid.float().sum() / top_valid.numel() if top_valid.numel() else segment.new_zeros(())
        active_units = seg_mask.float().sum()
        no_past_frac = ((seg_mask & ~has_valid.squeeze(-1)).float().sum() / active_units.clamp(min=1.0)) if active_units.item() else segment.new_zeros(())
        selected_units = top_idx // slots_per_chunk
        query_units = torch.arange(units, device=segment.device).view(1, units, 1)
        selected_delta = selected_units - query_units
        violations = (selected_delta >= 0) & top_valid
        violation_count = violations.float().sum()
        valid_delta = selected_delta[top_valid]
        max_delta = valid_delta.float().max() if valid_delta.numel() else segment.new_full((), -1.0)
        metrics = {
            "retrieval_weights": weights,
            "retrieval_indices": top_idx,
            "retrieval_valid": top_valid,
            "retrieval_entropy": retrieval_entropy,
            "retrieval_valid_frac": retrieval_valid_frac,
            "retrieval_no_past_frac": no_past_frac,
            "retrieval_active_units": active_units,
            "retrieval_past_only_violation_count": violation_count,
            "retrieval_max_selected_unit_delta": max_delta,
        }
        return context, metrics

    @staticmethod
    def causal_summary_memory(summary: torch.Tensor, seg_mask: torch.Tensor) -> torch.Tensor:
        p = seg_mask.unsqueeze(-1).to(summary.dtype)
        numer = torch.cumsum(summary * p, dim=1)
        denom = torch.cumsum(p, dim=1).clamp(min=1.0)
        hist = numer / denom
        zero = hist.new_zeros(hist.size(0), 1, hist.size(-1))
        return torch.cat([zero, hist[:, :-1]], dim=1)

    def forward(
        self,
        src: torch.Tensor,
        valid: torch.Tensor,
        seg_ids: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        emb = self.byte_seed(src)
        h = self.input_proj(emb)
        if self.causal_byte_encoder:
            seq_len = h.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            causal_mask = None
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=~valid)
        h = h * valid.unsqueeze(-1).to(h.dtype)

        pooled = segment_mean_pool(h, seg_ids, seg_mask)
        if self.pool_mode == "mean_first_last":
            first, last = segment_edge_pool(h, seg_ids, seg_mask)
            pooled = torch.cat([pooled, first, last], dim=-1)
        segment = self.segment_proj(pooled) * seg_mask.unsqueeze(-1).to(h.dtype)
        summary = self.summary_head(segment) * seg_mask.unsqueeze(-1).to(h.dtype)
        if self.memory_slots_per_chunk > 0:
            memory_slots = self.memory_slot_head(summary).view(
                summary.size(0),
                summary.size(1),
                self.memory_slots_per_chunk,
                summary.size(2),
            )
            memory_slots = memory_slots * seg_mask.unsqueeze(-1).unsqueeze(-1).to(memory_slots.dtype)
        else:
            memory_slots = summary.new_zeros((summary.size(0), summary.size(1), 0, summary.size(2)))
        memory, retrieval_metrics = self.retrieve_past_memory(segment, memory_slots, seg_mask)
        readout = self.readout_head(torch.cat([segment, memory], dim=-1))
        readout = readout * seg_mask.unsqueeze(-1).to(h.dtype)
        for block in self.refiners:
            readout = block(readout, seg_mask)

        length_logits = self.length_head(readout)
        slots = torch.arange(self.max_span, device=src.device)
        slot_h = readout.unsqueeze(2) + self.slot_embed(slots).view(1, 1, self.max_span, -1)
        slot_h = slot_h.view(-1, self.max_span, readout.size(-1))
        slot_mask = seg_mask.unsqueeze(-1).expand(-1, -1, self.max_span).reshape(-1, self.max_span)
        slot_h = self.slot_decoder(slot_h, slot_mask)
        slot_h = slot_h.view(readout.size(0), readout.size(1), self.max_span, readout.size(-1))
        byte_logits = self.byte_head(slot_h)
        boundary_logits = self.boundary_head(h).squeeze(-1)
        active_summary = summary[seg_mask]
        active_slots = memory_slots[seg_mask] if memory_slots.numel() else memory_slots.new_zeros((0, summary.size(-1)))
        active_memory = memory[seg_mask]
        first_unit_memory = memory[:, 0] if memory.size(1) > 0 else memory.new_zeros((memory.size(0), memory.size(-1)))
        metrics = {
            "h": h,
            "segment": segment,
            "readout": readout,
            "summary": summary,
            "memory": memory,
            "memory_slots": memory_slots,
            "summary_norm": active_summary.norm(dim=-1).mean() if active_summary.numel() else summary.new_zeros(()),
            "memory_slot_norm": active_slots.norm(dim=-1).mean() if active_slots.numel() else summary.new_zeros(()),
            "memory_context_norm": active_memory.norm(dim=-1).mean() if active_memory.numel() else summary.new_zeros(()),
            "first_unit_memory_norm": first_unit_memory.norm(dim=-1).mean() if first_unit_memory.numel() else summary.new_zeros(()),
            **retrieval_metrics,
            "boundary_logits": boundary_logits,
        }
        return byte_logits, length_logits, metrics


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return x.new_zeros(())
    return x[mask].float().mean()


@torch.no_grad()
def evaluate(model: V32LanguageCodec2M, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, List[float]] = {
        "loss": [],
        "recon_acc": [],
        "length_acc": [],
        "boundary_acc": [],
        "units_per_byte": [],
        "readout_units_per_byte": [],
        "memory_slots_per_byte": [],
        "retrieval_entropy": [],
        "retrieval_valid_frac": [],
        "retrieval_no_past_frac": [],
        "retrieval_active_units": [],
        "retrieval_past_only_violation_count": [],
        "retrieval_max_selected_unit_delta": [],
        "memory_context_norm": [],
        "memory_slot_norm": [],
        "summary_norm": [],
        "first_unit_memory_norm": [],
    }
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        src, starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
        valid = src.ne(PAD_ID)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            byte_logits, length_logits, metrics = model(src, valid, seg_ids, seg_mask)
            slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
            recon_loss = F.cross_entropy(
                byte_logits.float().view(-1, byte_logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_ID,
            )
            length_target = (lengths.clamp(min=1, max=args.max_span) - 1).clamp(min=0)
            length_loss = F.cross_entropy(length_logits[seg_mask].float(), length_target[seg_mask]) if seg_mask.any() else recon_loss.new_zeros(())
            boundary_loss = F.binary_cross_entropy_with_logits(metrics["boundary_logits"][valid].float(), starts[valid].float()) if valid.any() else recon_loss.new_zeros(())
            loss = recon_loss + args.length_loss_weight * length_loss + args.boundary_loss_weight * boundary_loss

        pred = byte_logits.argmax(dim=-1)
        recon_acc = (pred[slot_mask] == targets[slot_mask]).float().mean().item() if slot_mask.any() else 0.0
        len_acc = (length_logits.argmax(dim=-1)[seg_mask] == length_target[seg_mask]).float().mean().item() if seg_mask.any() else 0.0
        bpred = metrics["boundary_logits"].sigmoid().ge(0.5)
        bacc = (bpred[valid] == starts[valid]).float().mean().item() if valid.any() else 0.0
        units = seg_mask.float().sum().item()
        bytes_n = valid.float().sum().item()
        memory_slots = metrics.get("memory_slots")
        if memory_slots is not None:
            memory_slot_count = float(seg_mask.float().sum().item() * int(getattr(model, "memory_slots_per_chunk", 0)))
        else:
            memory_slot_count = 0.0
        totals["loss"].append(float(loss.item()))
        totals["recon_acc"].append(float(recon_acc))
        totals["length_acc"].append(float(len_acc))
        totals["boundary_acc"].append(float(bacc))
        totals["units_per_byte"].append(float(units / max(bytes_n, 1.0)))
        totals["readout_units_per_byte"].append(float(units / max(bytes_n, 1.0)))
        totals["memory_slots_per_byte"].append(float(memory_slot_count / max(bytes_n, 1.0)))
        totals["retrieval_entropy"].append(float(metrics.get("retrieval_entropy", torch.zeros((), device=device)).float().item()))
        totals["retrieval_valid_frac"].append(float(metrics.get("retrieval_valid_frac", torch.zeros((), device=device)).float().item()))
        totals["retrieval_no_past_frac"].append(float(metrics.get("retrieval_no_past_frac", torch.zeros((), device=device)).float().item()))
        totals["retrieval_active_units"].append(float(metrics.get("retrieval_active_units", torch.zeros((), device=device)).float().item()))
        totals["retrieval_past_only_violation_count"].append(float(metrics.get("retrieval_past_only_violation_count", torch.zeros((), device=device)).float().item()))
        totals["retrieval_max_selected_unit_delta"].append(float(metrics.get("retrieval_max_selected_unit_delta", torch.full((), -1.0, device=device)).float().item()))
        totals["memory_context_norm"].append(float(metrics.get("memory_context_norm", torch.zeros((), device=device)).float().item()))
        totals["memory_slot_norm"].append(float(metrics.get("memory_slot_norm", torch.zeros((), device=device)).float().item()))
        totals["summary_norm"].append(float(metrics.get("summary_norm", torch.zeros((), device=device)).float().item()))
        totals["first_unit_memory_norm"].append(float(metrics.get("first_unit_memory_norm", torch.zeros((), device=device)).float().item()))
    return {k: sum(v) / max(len(v), 1) for k, v in totals.items()}


def run(args: argparse.Namespace) -> Dict[str, float]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.streaming_train:
        train_ds = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=args.stream_samples_per_worker,
            seed=args.seed,
        )
        if args.streaming_eval and args.data_path:
            eval_ds = StreamingReconstructionDataset(
                file_path=args.data_path,
                seq_len=args.seq_len,
                samples_per_worker=max(args.batch_size * args.max_eval_batches, 1024),
                seed=args.seed + 9999,
            )
        else:
            eval_texts = _load_texts(args.data_path, args.eval_max_lines) if args.data_path else STUB_CORPUS
            eval_ds = ByteReconstructionDataset(texts=eval_texts, seq_len=args.seq_len, stride=args.stride)
        shuffle = False
    else:
        texts = _load_texts(args.data_path, args.max_lines) if args.data_path else STUB_CORPUS * 64
        ds = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
        train_ds = ds
        eval_ds = ds
        shuffle = True

    collate_fn = CodecCollator(args.min_span, args.max_span, args.max_units)
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = V32LanguageCodec2M(
        d_model=args.d_model,
        hidden=args.hidden,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        ffn_dim=args.ffn_dim,
        max_span=args.max_span,
        refine_steps=args.refine_steps,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
        memory_slots_per_chunk=args.memory_slots_per_chunk,
        memory_topk=args.memory_topk,
        memory_retrieval_mode=args.memory_retrieval_mode,
        causal_byte_encoder=args.causal_byte_encoder,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"

    step = 0
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = 0
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            src, starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
            valid = src.ne(PAD_ID)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                byte_logits, length_logits, metrics = model(src, valid, seg_ids, seg_mask)
                slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
                recon_loss = F.cross_entropy(
                    byte_logits.float().view(-1, byte_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=PAD_ID,
                )
                length_target = (lengths.clamp(min=1, max=args.max_span) - 1).clamp(min=0)
                length_loss = F.cross_entropy(length_logits[seg_mask].float(), length_target[seg_mask]) if seg_mask.any() else recon_loss.new_zeros(())
                boundary_loss = F.binary_cross_entropy_with_logits(metrics["boundary_logits"][valid].float(), starts[valid].float()) if valid.any() else recon_loss.new_zeros(())
                loss = recon_loss + args.length_loss_weight * length_loss + args.boundary_loss_weight * boundary_loss
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                now = time.perf_counter()
                elapsed = now - train_start_time
                recent_elapsed = now - last_log_time
                recent_steps = max(step - last_log_step, 1)
                total_steps_per_sec = (step + 1) / max(elapsed, 1e-9)
                recent_steps_per_sec = recent_steps / max(recent_elapsed, 1e-9)
                if device.type == "cuda":
                    max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                else:
                    max_mem_mb = 0.0
                pred = byte_logits.argmax(dim=-1)
                recon_acc = (pred[slot_mask] == targets[slot_mask]).float().mean().item() if slot_mask.any() else 0.0
                len_acc = (length_logits.argmax(dim=-1)[seg_mask] == length_target[seg_mask]).float().mean().item() if seg_mask.any() else 0.0
                bpred = metrics["boundary_logits"].sigmoid().ge(0.5)
                bacc = (bpred[valid] == starts[valid]).float().mean().item() if valid.any() else 0.0
                units = seg_mask.float().sum().item()
                bytes_n = valid.float().sum().item()
                memory_slots_per_byte = float(units * args.memory_slots_per_chunk / max(bytes_n, 1.0))
                retrieval_entropy = float(metrics.get("retrieval_entropy", loss.new_zeros(())).float().item())
                retrieval_valid_frac = float(metrics.get("retrieval_valid_frac", loss.new_zeros(())).float().item())
                retrieval_no_past_frac = float(metrics.get("retrieval_no_past_frac", loss.new_zeros(())).float().item())
                retrieval_active_units = float(metrics.get("retrieval_active_units", loss.new_zeros(())).float().item())
                retrieval_violation_count = float(metrics.get("retrieval_past_only_violation_count", loss.new_zeros(())).float().item())
                retrieval_max_delta = float(metrics.get("retrieval_max_selected_unit_delta", loss.new_full((), -1.0)).float().item())
                memory_context_norm = float(metrics.get("memory_context_norm", loss.new_zeros(())).float().item())
                memory_slot_norm = float(metrics.get("memory_slot_norm", loss.new_zeros(())).float().item())
                summary_norm = float(metrics.get("summary_norm", loss.new_zeros(())).float().item())
                first_unit_memory_norm = float(metrics.get("first_unit_memory_norm", loss.new_zeros(())).float().item())
                row = {
                    "step": step,
                    "memory_enabled": bool(args.memory_slots_per_chunk > 0),
                    "memory_path": "past_slot_retrieval" if args.memory_slots_per_chunk > 0 else "none",
                    "memory_retrieval_mode": str(args.memory_retrieval_mode),
                    "summary_causal": bool(args.causal_byte_encoder),
                    "byte_encoder_causal": bool(args.causal_byte_encoder),
                    "memory_read_scope": "past_only",
                    "boundary_reads_memory": False,
                    "interpreter_reads_memory": bool(args.memory_slots_per_chunk > 0),
                    "decoder_reads_memory": False,
                    "loss": float(loss.item()),
                    "recon_loss": float(recon_loss.item()),
                    "length_loss": float(length_loss.item()),
                    "boundary_loss": float(boundary_loss.item()),
                    "recon_acc": float(recon_acc),
                    "length_acc": float(len_acc),
                    "boundary_acc": float(bacc),
                    "units_per_byte": float(units / max(bytes_n, 1.0)),
                    "readout_units_per_byte": float(units / max(bytes_n, 1.0)),
                    "memory_slots_per_byte": memory_slots_per_byte,
                    "retrieval_topk": int(args.memory_topk),
                    "retrieval_entropy": retrieval_entropy,
                    "retrieval_valid_frac": retrieval_valid_frac,
                    "retrieval_no_past_frac": retrieval_no_past_frac,
                    "retrieval_active_units": retrieval_active_units,
                    "retrieval_past_only_violation_count": retrieval_violation_count,
                    "retrieval_max_selected_unit_delta": retrieval_max_delta,
                    "memory_context_norm": memory_context_norm,
                    "memory_slot_norm": memory_slot_norm,
                    "summary_norm": summary_norm,
                    "first_unit_memory_norm": first_unit_memory_norm,
                    "grad": float(grad.item() if hasattr(grad, "item") else grad),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "elapsed_sec": float(elapsed),
                    "steps_per_sec": float(total_steps_per_sec),
                    "recent_steps_per_sec": float(recent_steps_per_sec),
                    "samples_per_sec": float(total_steps_per_sec * args.batch_size),
                    "recent_samples_per_sec": float(recent_steps_per_sec * args.batch_size),
                    "bytes_per_sec": float(total_steps_per_sec * args.batch_size * args.seq_len),
                    "max_memory_allocated_mb": float(max_mem_mb),
                }
                _append_jsonl(log_path, row)
                print(
                    f"step={step} loss={row['loss']:.4f} recon={row['recon_acc']:.3f} "
                    f"len={row['length_acc']:.3f} boundary={row['boundary_acc']:.3f} "
                    f"u/b={row['units_per_byte']:.3f} samples/s={row['recent_samples_per_sec']:.0f} "
                    f"m/b={row['memory_slots_per_byte']:.3f} retrH={row['retrieval_entropy']:.3f} "
                    f"mem={row['max_memory_allocated_mb']:.0f}MB",
                    flush=True,
                )
                last_log_time = now
                last_log_step = step

            if step > 0 and step % args.ckpt_every == 0:
                payload = {"model": model.state_dict(), "args": vars(args), "step": step, "params": params}
                torch.save(payload, out_dir / f"step{step}.pt")
                torch.save(payload, out_dir / "latest.pt")
            step += 1

    eval_stats = evaluate(model, eval_loader, args, device)
    elapsed_sec = time.perf_counter() - train_start_time
    if args.memory_slots_per_chunk > 0:
        if args.memory_retrieval_mode == "random":
            model_version = "v3.2-stage3-random-past-memory-interpreter"
        else:
            model_version = "v3.2-stage2-causal-past-memory-interpreter" if args.causal_byte_encoder else "v3.2-stage2-noncausal-memory-interpreter"
    else:
        model_version = "v3.2-stage1-factorized-byte-seed"
    result = {"params": params, "steps": step, "model_version": model_version, **{f"eval_{k}": v for k, v in eval_stats.items()}}
    result["memory_enabled"] = bool(args.memory_slots_per_chunk > 0)
    result["memory_path"] = "past_slot_retrieval" if args.memory_slots_per_chunk > 0 else "none"
    result["memory_retrieval_mode"] = str(args.memory_retrieval_mode)
    result["summary_causal"] = bool(args.causal_byte_encoder)
    result["byte_encoder_causal"] = bool(args.causal_byte_encoder)
    result["memory_read_scope"] = "past_only"
    result["boundary_reads_memory"] = False
    result["interpreter_reads_memory"] = bool(args.memory_slots_per_chunk > 0)
    result["decoder_reads_memory"] = False
    result["eval_mode"] = "streaming" if args.streaming_train and args.streaming_eval else "fixed_text"
    result["train_elapsed_sec"] = elapsed_sec
    result["train_steps_per_sec"] = step / max(elapsed_sec, 1e-9)
    result["train_samples_per_sec"] = (step * args.batch_size) / max(elapsed_sec, 1e-9)
    result["train_bytes_per_sec"] = (step * args.batch_size * args.seq_len) / max(elapsed_sec, 1e-9)
    if device.type == "cuda":
        result["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step, "summary": result}, out_dir / "latest.pt")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FLUED-v3.2 language codec prototype")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=3000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--refine-steps", type=int, default=1)
    parser.add_argument("--pool-mode", choices=["mean", "mean_first_last"], default="mean")
    parser.add_argument("--memory-slots-per-chunk", type=int, default=0)
    parser.add_argument("--memory-topk", type=int, default=4)
    parser.add_argument("--memory-retrieval-mode", choices=["topk", "random"], default="topk")
    parser.set_defaults(causal_byte_encoder=True)
    parser.add_argument("--causal-byte-encoder", dest="causal_byte_encoder", action="store_true")
    parser.add_argument(
        "--no-causal-byte-encoder",
        dest="causal_byte_encoder",
        action="store_false",
        help="Diagnostic leakage ablation only; invalid for mainline v3.2 Stage 2 evidence.",
    )
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=128)
    parser.add_argument("--length-loss-weight", type=float, default=0.25)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
