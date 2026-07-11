"""Train / evaluate the FLUED v3.3 byte-to-latent interface.

This is intentionally small and runnable.  It is not a SOTA training recipe;
it is the public entry point for ablations around the v3.3 architecture:

* signed segmentor and dual-threshold chunking
* no-memory mainline or prompt-local no-self memory branch
* strict masked-source reconstruction
* optional small latent backbone for decoder-leakage / interface-utility tests
* vector-rate, boundary, length, and reconstruction losses
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
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import (  # noqa: E402
    BYTE_VOCAB_SIZE,
    MASK_ID,
    PAD_ID,
    ShardedStreamingReconstructionDataset,
    STUB_CORPUS,
    ByteReconstructionDataset,
    StreamingReconstructionDataset,
)
from flued.v33 import FLUEDV33, FLUEDV33Config  # noqa: E402


def _append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / max(len(vals), 1)


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def _cosine_with_warmup(opt: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def build_optimizer(args: argparse.Namespace, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "fused_adamw":
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except RuntimeError as exc:
            print(f"fused AdamW unavailable, falling back to foreach AdamW: {exc}", flush=True)
            return torch.optim.AdamW(params, foreach=True, **kwargs)
    if args.optimizer == "foreach_adamw":
        return torch.optim.AdamW(params, foreach=True, **kwargs)
    return torch.optim.AdamW(params, **kwargs)


def _load_texts(path: str, max_lines: int) -> List[str]:
    if not path:
        return STUB_CORPUS * 64
    rows: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_lines > 0 and i >= max_lines:
                break
            rows.append(line.rstrip("\n"))
    return rows or STUB_CORPUS * 64


class LatentInfillBackbone(nn.Module):
    """Small external backbone used to test latent-interface utility."""

    def __init__(self, d_z: int, hidden: int, layers: int, nhead: int, ffn_dim: int, max_chunks: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_z, hidden)
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.pos = nn.Embedding(max_chunks, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d_z))

    def forward(
        self,
        z: torch.Tensor,
        active: torch.Tensor,
        masked_units: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        units = z.size(1)
        h = self.in_proj(z)
        h = torch.where(masked_units.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        if position_ids is None:
            position_ids = torch.arange(units, device=z.device).view(1, units)
        h = h + self.pos(position_ids.clamp(max=self.pos.num_embeddings - 1))
        h = self.encoder(h, src_key_padding_mask=~active)
        return self.out(h) * active.unsqueeze(-1).to(h.dtype)


def make_byte_mask(valid: torch.Tensor, mask_prob: float, span_min: int, span_max: int) -> torch.Tensor:
    if mask_prob <= 0:
        return torch.zeros_like(valid, dtype=torch.bool)
    _bsz, seq_len = valid.shape
    span_min = max(1, int(span_min))
    span_max = max(span_min, int(span_max))
    avg_span = 0.5 * float(span_min + span_max)
    start_prob = min(max(float(mask_prob) / max(avg_span, 1.0), 0.0), 1.0)
    starts = (torch.rand(valid.shape, device=valid.device) < start_prob) & valid
    lengths = torch.randint(span_min, span_max + 1, valid.shape, device=valid.device)
    mask = torch.zeros_like(valid, dtype=torch.bool)
    # Vectorized span expansion.  This preserves the denoising semantics while
    # avoiding hundreds of per-sample .item() synchronizations on CUDA.
    for offset in range(span_max):
        active = starts & lengths.gt(offset)
        if offset == 0:
            shifted = active
        else:
            shifted = torch.zeros_like(active)
            shifted[:, offset:] = active[:, : seq_len - offset]
        mask |= shifted
    return mask & valid


def make_targets(clean: torch.Tensor, byte_mask: torch.Tensor, chunk_ids: torch.Tensor, offsets: torch.Tensor, max_chunks: int, max_span: int):
    bsz, seq_len = clean.shape
    targets = torch.full((bsz, max_chunks, max_span), PAD_ID, dtype=torch.long, device=clean.device)
    slot_mask = torch.zeros((bsz, max_chunks, max_span), dtype=torch.bool, device=clean.device)
    masked_slot = torch.zeros_like(slot_mask)
    valid = chunk_ids.ge(0) & chunk_ids.lt(max_chunks) & offsets.ge(0) & offsets.lt(max_span) & clean.ne(PAD_ID)
    b_idx, t_idx = valid.nonzero(as_tuple=True)
    c_idx = chunk_ids[b_idx, t_idx]
    s_idx = offsets[b_idx, t_idx]
    targets[b_idx, c_idx, s_idx] = clean[b_idx, t_idx]
    slot_mask[b_idx, c_idx, s_idx] = True
    masked_slot[b_idx, c_idx, s_idx] = byte_mask[b_idx, t_idx]
    return targets, slot_mask, masked_slot


def masked_readouts_from_slots(masked_slot: torch.Tensor, chunk_mask: torch.Tensor, readouts: int) -> torch.Tensor:
    if readouts <= 1:
        return (masked_slot.any(dim=-1, keepdim=True) & chunk_mask.unsqueeze(-1))
    max_span = masked_slot.size(-1)
    offsets = torch.arange(max_span, device=masked_slot.device)
    # Slot 0 is the always-visible fallback.  Extra slots cover contiguous
    # byte-offset bands, so local byte masks do not erase the whole chunk.
    extra_slots = 1 + torch.div(offsets * (readouts - 1), max(max_span, 1), rounding_mode="floor")
    slot_map = F.one_hot(extra_slots.clamp(max=readouts - 1), num_classes=readouts).to(torch.bool)
    masked = (masked_slot.unsqueeze(-1) & slot_map.view(1, 1, max_span, readouts)).any(dim=2)
    masked[..., 0] = False
    return masked & chunk_mask.unsqueeze(-1)


def slots_from_readout_mask(slot_mask: torch.Tensor, readout_mask: torch.Tensor) -> torch.Tensor:
    readouts = readout_mask.size(-1)
    if readouts <= 1:
        return slot_mask & readout_mask.squeeze(-1).unsqueeze(-1)
    max_span = slot_mask.size(-1)
    offsets = torch.arange(max_span, device=slot_mask.device)
    extra_slots = 1 + torch.div(offsets * (readouts - 1), max(max_span, 1), rounding_mode="floor")
    slot_map = F.one_hot(extra_slots.clamp(max=readouts - 1), num_classes=readouts).to(torch.bool)
    covered = (readout_mask.unsqueeze(2) & slot_map.view(1, 1, max_span, readouts)).any(dim=-1)
    return slot_mask & covered


def compute_readout_active_mask(
    chunk_mask: torch.Tensor,
    readout_emit: torch.Tensor,
    masked_readouts: torch.Tensor,
    mode: str,
    threshold: float,
) -> torch.Tensor:
    fallback = torch.zeros_like(readout_emit, dtype=torch.bool)
    fallback[..., 0] = chunk_mask
    if mode == "all":
        return chunk_mask.unsqueeze(-1).expand_as(readout_emit)
    emitted = readout_emit.ge(float(threshold)) & chunk_mask.unsqueeze(-1)
    emitted = emitted | fallback
    if mode == "emitted":
        return emitted
    if mode == "masked_or_emitted":
        return emitted | (masked_readouts & chunk_mask.unsqueeze(-1))
    raise ValueError(f"unknown emit_compute_mode: {mode}")


def boundary_prior_losses(
    confidence: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    tau_cut: float,
    tau_trans: float,
    tau_keep: float,
    collect_metrics: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    raw = (token_ids - 1).clamp(min=0, max=255)
    utf8_cont = raw.ge(0x80) & raw.le(0xBF) & valid
    whitespace = ((raw == 10) | (raw == 13) | (raw == 32) | (raw == 9)) & valid
    punct = (
        (raw.ge(0x21) & raw.le(0x2F))
        | (raw.ge(0x3A) & raw.le(0x40))
        | (raw.ge(0x5B) & raw.le(0x60))
        | (raw.ge(0x7B) & raw.le(0x7E))
    ) & valid
    first = torch.zeros_like(valid)
    first_idx = valid.float().argmax(dim=1)
    first[torch.arange(valid.size(0), device=valid.device), first_idx] = valid.any(dim=1)

    punct_mask = (whitespace | punct) & ~utf8_cont
    neutral_mask = valid & ~utf8_cont & ~punct_mask & ~first
    conf_f = confidence.float()
    cont_w = utf8_cont.to(conf_f.dtype)
    punct_w = punct_mask.to(conf_f.dtype)
    neutral_w = neutral_mask.to(conf_f.dtype)
    continue_target = -min(0.95, float(tau_keep) + 0.10)
    punct_target = min(0.95, 0.5 * (float(tau_trans) + float(tau_cut)))
    cont_loss = (F.smooth_l1_loss(conf_f, torch.full_like(conf_f, continue_target), reduction="none") * cont_w).sum() / cont_w.sum().clamp(min=1.0)
    punct_loss = (F.smooth_l1_loss(conf_f, torch.full_like(conf_f, punct_target), reduction="none") * punct_w).sum() / punct_w.sum().clamp(min=1.0)
    neutral_mean = (conf_f * neutral_w).sum() / neutral_w.sum().clamp(min=1.0)
    neutral_loss = neutral_mean.pow(2)
    loss = cont_loss + punct_loss + neutral_loss
    stats = {}
    if collect_metrics:
        cont_count = cont_w.sum()
        punct_count = punct_w.sum()
        stats = {
            "boundary_cont_loss": float(cont_loss.item()),
            "boundary_punct_loss": float(punct_loss.item()),
            "boundary_neutral_mean": float(neutral_mean.item()),
            "boundary_cont_mean": float(((conf_f * cont_w).sum() / cont_count.clamp(min=1.0)).item()),
            "boundary_punct_mean": float(((conf_f * punct_w).sum() / punct_count.clamp(min=1.0)).item()),
            "boundary_continue_target": float(continue_target),
            "boundary_punct_target": float(punct_target),
        }
    return loss, stats


def token_values_from_slots(slot_values: torch.Tensor, chunk_ids: torch.Tensor, offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = chunk_ids.shape
    token_values = slot_values.new_zeros((bsz, seq_len))
    token_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=slot_values.device)
    max_chunks, max_span = slot_values.size(1), slot_values.size(2)
    valid = chunk_ids.ge(0) & chunk_ids.lt(max_chunks) & offsets.ge(0) & offsets.lt(max_span)
    b_idx, t_idx = valid.nonzero(as_tuple=True)
    c_idx = chunk_ids[b_idx, t_idx]
    s_idx = offsets[b_idx, t_idx]
    token_values[b_idx, t_idx] = slot_values[b_idx, c_idx, s_idx]
    token_mask[b_idx, t_idx] = True
    return token_values, token_mask


def plastic_boundary_credit_loss(
    confidence: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    token_credit: torch.Tensor,
    credit_mask: torch.Tensor,
    collect_metrics: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Backward-only credit assignment for the signed boundary field.

    Forward segmentation still uses hard dual-threshold chunks.  This term only
    asks neutral byte-boundary confidence to distribute around zero according to
    detached downstream difficulty, so reconstruction/backbone losses can shape
    the confidence field without being passed as interpreter features.
    """
    raw = (token_ids - 1).clamp(min=0, max=255)
    utf8_cont = raw.ge(0x80) & raw.le(0xBF) & valid
    whitespace = ((raw == 10) | (raw == 13) | (raw == 32) | (raw == 9)) & valid
    punct = (
        (raw.ge(0x21) & raw.le(0x2F))
        | (raw.ge(0x3A) & raw.le(0x40))
        | (raw.ge(0x5B) & raw.le(0x60))
        | (raw.ge(0x7B) & raw.le(0x7E))
    ) & valid
    first = torch.zeros_like(valid)
    first_idx = valid.float().argmax(dim=1)
    first[torch.arange(valid.size(0), device=valid.device), first_idx] = valid.any(dim=1)
    candidate = valid & credit_mask & ~utf8_cont & ~whitespace & ~punct & ~first

    candidate_w = candidate.to(confidence.dtype)
    count = candidate_w.sum().clamp(min=1.0)
    values = token_credit.detach().float()
    mean = (values * candidate_w.float()).sum() / count.float()
    centered = values - mean
    scale = ((centered.square() * candidate_w.float()).sum() / count.float()).sqrt().clamp(min=1.0e-4)
    target = 0.35 * torch.tanh(centered / scale)
    per_token = F.smooth_l1_loss(confidence.float(), target, reduction="none")
    loss = ((per_token * candidate_w.float()).sum() / count.float()).to(confidence.dtype)
    stats = {}
    if collect_metrics:
        active = candidate_w.float().sum()
        target_mean = (target * candidate_w.float()).sum() / active.clamp(min=1.0)
        target_std = (((target - target_mean).square() * candidate_w.float()).sum() / active.clamp(min=1.0)).sqrt()
        stats = {
            "boundary_credit_loss": float(loss.item()),
            "boundary_credit_target_mean": float(target_mean.item()),
            "boundary_credit_target_std": float(target_std.item()),
            "boundary_credit_active": float(active.item()),
        }
    return loss, stats


def _active_flat(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim == mask.ndim + 1:
        return values[mask]
    if values.ndim == 4 and mask.ndim == 3:
        return values[mask]
    raise ValueError("values must have one trailing feature dimension")


def _flatten_readout(z: torch.Tensor, gate: torch.Tensor, chunk_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z.ndim == 3:
        units = z.size(1)
        pos = torch.arange(units, device=z.device).view(1, units).expand(z.size(0), units)
        return z, chunk_mask, pos
    if z.ndim != 4:
        raise ValueError("readout z must be [B,C,D] or [B,C,R,D]")
    bsz, chunks, readouts, dim = z.shape
    flat_z = z.reshape(bsz, chunks * readouts, dim)
    # Slot 0 is the hard fallback.  Extra slots stay in the sequence so their
    # gates receive reconstruction/backbone gradients through the readout value.
    flat_active = chunk_mask.unsqueeze(-1).expand_as(gate).reshape(bsz, chunks * readouts)
    flat_pos = torch.arange(chunks * readouts, device=z.device).view(1, chunks * readouts).expand(bsz, chunks * readouts)
    return flat_z, flat_active, flat_pos


def _compact_active_readout(
    flat_z: torch.Tensor,
    flat_active: torch.Tensor,
    flat_masked: torch.Tensor,
    flat_pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, units, dim = flat_z.shape
    if bsz == 1:
        active = flat_active[0]
        compact_z = flat_z[:, active, :]
        compact_masked = flat_masked[:, active]
        compact_pos = flat_pos[:, active]
        compact_active = torch.ones(compact_z.shape[:2], dtype=torch.bool, device=flat_z.device)
        return compact_z, compact_active, compact_masked, compact_pos

    counts = flat_active.sum(dim=1)
    max_units = int(counts.max().item())
    order_base = torch.arange(units, device=flat_z.device).view(1, units).expand(bsz, units)
    sort_key = torch.where(flat_active, order_base, order_base + units)
    order = torch.argsort(sort_key, dim=1)[:, :max_units]
    gather_z = order.unsqueeze(-1).expand(-1, -1, dim)
    compact_z = torch.gather(flat_z, 1, gather_z)
    compact_active = torch.gather(flat_active, 1, order)
    compact_masked = torch.gather(flat_masked, 1, order) & compact_active
    compact_pos = torch.gather(flat_pos, 1, order)
    compact_z = compact_z * compact_active.unsqueeze(-1).to(compact_z.dtype)
    return compact_z, compact_active, compact_masked, compact_pos


def _scatter_compact_readout(pred_compact: torch.Tensor, compact_pos: torch.Tensor, units: int) -> torch.Tensor:
    bsz, _compact_units, dim = pred_compact.shape
    pred_flat = pred_compact.new_zeros((bsz, units, dim))
    pred_flat.scatter_(1, compact_pos.unsqueeze(-1).expand(-1, -1, dim), pred_compact)
    return pred_flat


def coding_rate_loss(z_content: torch.Tensor, chunk_mask: torch.Tensor, collect_metrics: bool = True) -> Tuple[torch.Tensor, float]:
    z = _active_flat(z_content, chunk_mask)
    if z.size(0) < 2:
        return z_content.new_zeros(()), 0.0
    z = F.normalize(z.float(), dim=-1)
    gram = torch.matmul(z, z.transpose(0, 1)).float() / max(float(z.size(0)), 1.0)
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    sign, logabsdet = torch.linalg.slogdet(eye + gram)
    rate = torch.where(sign > 0, logabsdet / max(float(z.size(0)), 1.0), gram.new_zeros(()))
    return -rate.to(z_content.dtype), float(rate.item()) if collect_metrics else 0.0


def range_loss(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return F.relu(float(low) - value).pow(2) + F.relu(value - float(high)).pow(2)


def memory_regularization_losses(out, collect_metrics: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    zero = out.z_content.new_zeros(())
    m_write = out.interpreter.m_write
    if m_write is None:
        return zero, zero, {
            "memory_vector_overlap": 0.0,
            "memory_gate_mean": 0.0,
            "memory_has_context_frac": 0.0,
        } if collect_metrics else {}

    z = out.z_content
    chunk_mask = out.chunks.chunk_mask
    m_avg = m_write.mean(dim=2)
    active = chunk_mask & (m_avg.norm(dim=-1) > 0) & (z.norm(dim=-1) > 0)
    shared_dim = min(z.size(-1), m_avg.size(-1))
    active_w = active.to(z.dtype)
    sim = F.cosine_similarity(z[..., :shared_dim].float(), m_avg[..., :shared_dim].float(), dim=-1).pow(2)
    vector_overlap = (sim * active_w.float()).sum() / active_w.float().sum().clamp(min=1.0)
    vector_overlap_loss = range_loss(vector_overlap, 0.10, 0.30).to(z.dtype)

    gate = out.aux.get("memory_gate")
    has_context = out.aux.get("memory_has_context")
    read_entropy = out.aux.get("memory_read_entropy")
    read_norm = out.aux.get("memory_read_norm")
    self_allowed = out.aux.get("memory_self_allowed_count")
    visible_slots = out.aux.get("memory_visible_slots")
    if gate is not None and has_context is not None:
        gate_active = chunk_mask & has_context
        gate_w = gate_active.to(gate.dtype)
        gate_mean = (gate.float() * gate_w.float()).sum() / gate_w.float().sum().clamp(min=1.0)
        gate_loss = range_loss(gate_mean, 0.20, 0.50).to(z.dtype)
        chunk_w = chunk_mask.to(gate.dtype)
        has_context_frac = (has_context.float() * chunk_w.float()).sum() / chunk_w.float().sum().clamp(min=1.0)
        read_entropy_mean = (
            (read_entropy.float() * gate_w.float()).sum() / gate_w.float().sum().clamp(min=1.0)
            if read_entropy is not None
            else zero.float()
        )
        read_norm_mean = (
            (read_norm.float() * gate_w.float()).sum() / gate_w.float().sum().clamp(min=1.0)
            if read_norm is not None
            else zero.float()
        )
        self_allowed_mean = (
            (self_allowed.float() * chunk_w.float()).sum() / chunk_w.float().sum().clamp(min=1.0)
            if self_allowed is not None
            else zero.float()
        )
        visible_slots_mean = (
            (visible_slots.float() * chunk_w.float()).sum() / chunk_w.float().sum().clamp(min=1.0)
            if visible_slots is not None
            else zero.float()
        )
    else:
        gate_mean = zero.float()
        gate_loss = zero
        has_context_frac = zero.float()
        read_entropy_mean = zero.float()
        read_norm_mean = zero.float()
        self_allowed_mean = zero.float()
        visible_slots_mean = zero.float()

    stats = {}
    if collect_metrics:
        stats = {
            "memory_vector_overlap": float(vector_overlap.item()),
            "memory_gate_mean": float(gate_mean.item()),
            "memory_has_context_frac": float(has_context_frac.item()),
            "memory_read_entropy_mean": float(read_entropy_mean.item()),
            "memory_read_norm_mean": float(read_norm_mean.item()),
            "memory_self_allowed_mean": float(self_allowed_mean.item()),
            "memory_visible_slots_mean": float(visible_slots_mean.item()),
        }
    return vector_overlap_loss, gate_loss, stats


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    if args.streaming_train and (args.data_manifest or args.data_path):
        stream_cls = ShardedStreamingReconstructionDataset if args.data_manifest else StreamingReconstructionDataset
        stream_key = "manifest_path" if args.data_manifest else "file_path"
        stream_value = args.data_manifest or args.data_path
        train_ds = stream_cls(
            **{stream_key: stream_value},
            seq_len=args.seq_len,
            samples_per_worker=args.stream_samples_per_worker,
            seed=args.seed,
        )
        eval_ds = stream_cls(
            **{stream_key: stream_value},
            seq_len=args.seq_len,
            samples_per_worker=max(args.batch_size * args.max_eval_batches, 512),
            seed=args.seed + 9999,
        ) if args.streaming_eval else ByteReconstructionDataset(
            texts=_load_texts(args.data_path, args.eval_max_lines),
            seq_len=args.seq_len,
            stride=args.stride,
        )
        shuffle = False
    else:
        texts = _load_texts(args.data_path, args.max_lines)
        train_ds = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
        eval_ds = ByteReconstructionDataset(texts=texts[: max(1, args.eval_max_lines)], seq_len=args.seq_len, stride=args.stride)
        shuffle = True
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
        drop_last=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train, eval_loader


def _avg_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


def build_model(args: argparse.Namespace) -> FLUEDV33:
    config = FLUEDV33Config(
        d_model=args.d_model,
        d_z=args.d_z,
        d_mem=args.d_mem,
        hidden=args.hidden,
        max_chunks=args.max_chunks,
        max_span=args.max_span,
        use_memory=args.use_memory,
        memory_rank=args.memory_rank,
        memory_top_k=args.memory_top_k,
        memory_build_mode=args.memory_build_mode,
        memory_visibility=args.memory_visibility,
        max_readout_vectors=args.max_readout_vectors,
        chunk_mixer=args.chunk_mixer,
        tau_cut=args.tau_cut,
        tau_trans=args.tau_trans,
        tau_keep=args.tau_keep,
    )
    return FLUEDV33(config)


def step_model(
    model: FLUEDV33,
    backbone: LatentInfillBackbone | None,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    collect_metrics: bool = True,
):
    clean = batch[0].to(device, non_blocking=device.type == "cuda")
    valid = clean.ne(PAD_ID)
    valid_bytes = valid.float().sum().clamp(min=1.0)
    byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    src = clean.masked_fill(byte_mask, MASK_ID) if args.strict_masked_source else clean
    out = model(src)
    targets, slot_mask, masked_slot = make_targets(clean, byte_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span)
    visible_slot = slot_mask & (~masked_slot)
    ce = F.cross_entropy(
        out.byte_logits.float().reshape(-1, BYTE_VOCAB_SIZE),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(targets)
    masked_w = masked_slot.to(ce.dtype)
    visible_w = visible_slot.to(ce.dtype)
    masked_loss = (ce * masked_w).sum() / masked_w.sum().clamp(min=1.0)
    visible_loss = (ce * visible_w).sum() / visible_w.sum().clamp(min=1.0)
    recon_loss = args.masked_loss_weight * masked_loss + args.visible_loss_weight * visible_loss
    masked_chunks = masked_slot.any(dim=-1) & out.chunks.chunk_mask
    masked_chunk_fraction = masked_chunks.float().sum() / out.chunks.chunk_mask.float().sum().clamp(min=1.0)
    masked_readouts = masked_readouts_from_slots(masked_slot, out.chunks.chunk_mask, out.readout_z.size(2))
    compute_readouts = compute_readout_active_mask(
        out.chunks.chunk_mask,
        out.readout_emit,
        masked_readouts,
        args.emit_compute_mode,
        args.emit_threshold,
    )
    supervised_backbone_readouts = masked_readouts & compute_readouts
    backbone_slot_mask = masked_slot & slots_from_readout_mask(slot_mask, supervised_backbone_readouts)
    masked_readout_fraction = masked_readouts.float().sum() / out.chunks.chunk_mask.unsqueeze(-1).expand_as(masked_readouts).float().sum().clamp(min=1.0)

    chunk_lengths = out.chunks.lengths.clamp(min=1, max=args.max_span)
    length_target = (chunk_lengths - 1).clamp(min=0)
    length_ce = F.cross_entropy(
        out.length_logits.float().reshape(-1, args.max_span),
        length_target.reshape(-1),
        reduction="none",
    ).view_as(length_target)
    chunk_w = out.chunks.chunk_mask.to(length_ce.dtype)
    length_loss = (length_ce * chunk_w).sum() / chunk_w.sum().clamp(min=1.0)

    boundary_loss, boundary_stats = boundary_prior_losses(
        out.segmentor.confidence,
        src,
        valid,
        args.tau_cut,
        args.tau_trans,
        args.tau_keep,
        collect_metrics=collect_metrics,
    )

    rate = out.aux["readout_units_per_byte"]
    if args.rate_loss == "upper":
        rate_loss = F.relu(rate - args.target_rate).pow(2)
    else:
        rate_loss = (rate - args.target_rate).pow(2)
    readout_active = out.chunks.chunk_mask.unsqueeze(-1).expand_as(out.readout_gate)
    coding_loss, coding_rate = coding_rate_loss(out.readout_z, readout_active, collect_metrics=collect_metrics)
    memory_vector_overlap_loss, memory_gate_loss, memory_stats = memory_regularization_losses(out, collect_metrics=collect_metrics)

    backbone_loss = recon_loss.new_zeros(())
    backbone_mask_acc = 0.0
    leakage_gap = 0.0
    backbone_ppl = 1.0
    backbone_units = 0.0
    backbone_active_units = 0.0
    bb_ce = None
    if backbone is not None:
        z_in = out.readout_z.detach() if args.detach_backbone_input else out.readout_z
        flat_z, _flat_active_all, flat_pos = _flatten_readout(z_in, out.readout_gate, out.chunks.chunk_mask)
        readouts = out.readout_z.size(2)
        flat_active = compute_readouts.reshape(compute_readouts.size(0), -1)
        masked_units = supervised_backbone_readouts.reshape(supervised_backbone_readouts.size(0), -1)
        if args.active_only_backbone:
            compact_z, compact_active, compact_masked, compact_pos = _compact_active_readout(flat_z, flat_active, masked_units, flat_pos)
            pred_compact = backbone(compact_z, compact_active, compact_masked, position_ids=compact_pos)
            pred_flat = _scatter_compact_readout(pred_compact, compact_pos, flat_z.size(1))
            backbone_units = float(compact_z.size(0) * compact_z.size(1))
            if collect_metrics:
                backbone_active_units = float(compact_active.float().sum().item())
        else:
            pred_flat = backbone(flat_z, flat_active, masked_units, position_ids=flat_pos)
            backbone_units = float(flat_z.size(0) * flat_z.size(1))
            if collect_metrics:
                backbone_active_units = float(flat_active.float().sum().item())
        pred_z = pred_flat.view_as(out.readout_z)
        keep_z = out.readout_z.detach() if args.detach_backbone_keep else out.readout_z
        mixed_z = torch.where(supervised_backbone_readouts.unsqueeze(-1), pred_z, keep_z)
        bb_logits, _bb_len = model.decoder(mixed_z, out.chunks.chunk_mask, readout_gate=out.readout_gate)
        bb_ce = F.cross_entropy(
            bb_logits.float().reshape(-1, BYTE_VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(targets)
        backbone_w = backbone_slot_mask.to(bb_ce.dtype)
        backbone_loss = (bb_ce * backbone_w).sum() / backbone_w.sum().clamp(min=1.0)
        if collect_metrics:
            backbone_mask_acc = _safe_acc(bb_logits.argmax(dim=-1), targets, backbone_slot_mask)
            leakage_gap = backbone_mask_acc - _safe_acc(out.byte_logits.argmax(dim=-1), targets, backbone_slot_mask)
            backbone_ppl = float(torch.exp(backbone_loss.detach().float().clamp(max=20.0)).item())

    credit_slots = ce.detach()
    if bb_ce is not None:
        credit_slots = credit_slots + args.boundary_credit_backbone_weight * bb_ce.detach()
    token_credit, token_credit_mask = token_values_from_slots(credit_slots, out.chunks.chunk_ids, out.chunks.offsets)
    boundary_credit_loss, boundary_credit_stats = plastic_boundary_credit_loss(
        out.segmentor.confidence,
        src,
        valid,
        token_credit,
        token_credit_mask,
        collect_metrics=collect_metrics,
    )

    loss = (
        recon_loss
        + args.length_loss_weight * length_loss
        + args.boundary_loss_weight * boundary_loss
        + args.boundary_credit_loss_weight * boundary_credit_loss
        + args.rate_loss_weight * rate_loss
        + args.coding_rate_loss_weight * coding_loss
        + args.memory_vector_overlap_loss_weight * memory_vector_overlap_loss
        + args.memory_gate_loss_weight * memory_gate_loss
        + args.backbone_loss_weight * backbone_loss
    )

    metrics: Dict[str, float] = {}
    if collect_metrics:
        with torch.no_grad():
            pred = out.byte_logits.argmax(dim=-1)
            metrics = {
                "loss": float(loss.item()),
                "recon_loss": float(recon_loss.item()),
                "masked_loss": float(masked_loss.item()),
                "visible_loss": float(visible_loss.item()),
                "length_loss": float(length_loss.item()),
                "boundary_loss": float(boundary_loss.item()),
                "boundary_credit_loss": float(boundary_credit_loss.item()),
                "rate_loss": float(rate_loss.item()),
                "coding_rate": float(coding_rate),
                "coding_rate_loss": float(coding_loss.item()),
                "memory_vector_overlap_loss": float(memory_vector_overlap_loss.item()),
                "memory_gate_loss": float(memory_gate_loss.item()),
                "backbone_loss": float(backbone_loss.item()),
                "backbone_ppl": float(backbone_ppl),
                "backbone_units": float(backbone_units),
                "backbone_active_units": float(backbone_active_units),
                "actual_backbone_units_per_byte": float(backbone_units / float(valid_bytes.item())),
                "masked_backbone_units_per_byte": float(masked_readouts.float().sum().item() / float(valid_bytes.item())),
                "decoder_mask_acc": _safe_acc(pred, targets, masked_slot),
                "decoder_visible_acc": _safe_acc(pred, targets, visible_slot),
                "length_acc": _safe_acc(out.length_logits.argmax(dim=-1), length_target, out.chunks.chunk_mask),
                "backbone_mask_acc": float(backbone_mask_acc),
                "leakage_gap": float(leakage_gap),
                "readout_units_per_byte": float(rate.item() if hasattr(rate, "item") else rate),
                "soft_readout_units_per_byte": float(rate.item() if hasattr(rate, "item") else rate),
                "soft_emit_units_per_byte": float(out.aux["emit_units_per_byte"].item()),
                "extra_emit_mean": float(out.aux["readout_emit_mean"].item()) if out.aux.get("readout_emit_mean") is not None else 0.0,
                "hard_emit_fraction": float((compute_readouts.float().sum() / out.chunks.chunk_mask.unsqueeze(-1).expand_as(compute_readouts).float().sum().clamp(min=1.0)).item()),
                "active_chunks_per_byte": float(out.aux["active_chunks_per_byte"].item()),
                "readout_gate_mean": float(out.aux["readout_gate_mean"].item()) if out.aux.get("readout_gate_mean") is not None else 0.0,
                "readout_emit_mean": float(out.aux["readout_emit_mean"].item()) if out.aux.get("readout_emit_mean") is not None else 0.0,
                "hard_cut_fraction": float((out.policy.hard_cut & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "transition_fraction": float((out.policy.soft_transition & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "force_continue_fraction": float((out.policy.force_continue & valid).float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "confidence_mean": float((out.segmentor.confidence.float() * valid.float()).sum().item() / max(valid.float().sum().item(), 1.0)),
                "memory_enabled": bool(args.use_memory),
                "truncated_tokens": float(out.aux["truncated_tokens"].float().sum().item()) if out.aux.get("truncated_tokens") is not None else 0.0,
                "masked_byte_fraction": float(byte_mask.float().sum().item() / max(valid.float().sum().item(), 1.0)),
                "masked_chunk_fraction": float(masked_chunk_fraction.item()),
                "masked_readout_fraction": float(masked_readout_fraction.item()),
                "supervised_backbone_readout_fraction": float((supervised_backbone_readouts.float().sum() / out.chunks.chunk_mask.unsqueeze(-1).expand_as(supervised_backbone_readouts).float().sum().clamp(min=1.0)).item()),
                "backbone_supervised_byte_fraction": float((backbone_slot_mask.float().sum() / slot_mask.float().sum().clamp(min=1.0)).item()),
            }
            metrics.update(boundary_stats)
            metrics.update(boundary_credit_stats)
            metrics.update(memory_stats)
    return loss, metrics


@torch.no_grad()
def evaluate(model: FLUEDV33, backbone: LatentInfillBackbone | None, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    model.eval()
    if backbone is not None:
        backbone.eval()
    rows: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        _loss, metrics = step_model(model, backbone, batch, args, device)
        rows.append(metrics)
    model.train()
    if backbone is not None:
        backbone.train()
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


def run(args: argparse.Namespace) -> Dict[str, object]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    model = build_model(args).to(device)
    backbone = None
    if args.use_backbone:
        backbone = LatentInfillBackbone(
            args.d_z,
            args.backbone_hidden,
            args.backbone_layers,
            args.backbone_nhead,
            args.backbone_ffn_dim,
            args.max_chunks * args.max_readout_vectors,
            args.dropout,
        ).to(device)

    train_loader, eval_loader = make_dataloaders(args)
    params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters()) if backbone is not None else 0
    opt_params = list(model.parameters()) + (list(backbone.parameters()) if backbone is not None else [])
    if args.dry_run:
        result = {
            "model_version": "flued_v3_3",
            "experiment_name": args.experiment_name,
            "run_id": args.run_id,
            "params": params,
            "backbone_params": backbone_params,
            "total_trainable_params": params + backbone_params,
            "args": vars(args),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        (out_dir / "dry_run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    opt = build_optimizer(args, opt_params)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"
    start_step = 0
    latest_path = out_dir / "latest.pt"
    if args.resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device)
        if "model" in payload:
            model.load_state_dict(payload["model"])
        if backbone is not None and payload.get("backbone") is not None:
            backbone.load_state_dict(payload["backbone"])
        if payload.get("optimizer") is not None:
            opt.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            sched.load_state_dict(payload["scheduler"])
        start_step = int(payload.get("step", 0))
        print(f"resumed latest checkpoint from step={start_step}", flush=True)
    start = time.perf_counter()
    step = start_step
    model.train()
    if backbone is not None:
        backbone.train()
    train_iter = iter(train_loader)
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        collect_metrics = step % args.log_every == 0
        micro_metrics: List[Dict[str, float]] = []
        for _micro in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                loss, metrics = step_model(model, backbone, batch, args, device, collect_metrics=collect_metrics)
            (loss / grad_accum_steps).backward()
            if collect_metrics:
                micro_metrics.append(metrics)
        grad = torch.nn.utils.clip_grad_norm_(opt_params, args.grad_clip)
        opt.step()
        sched.step()
        if collect_metrics:
            metrics = _avg_metrics(micro_metrics)
            row = {"step": step, "lr": float(opt.param_groups[0]["lr"]), "grad": float(grad.item() if hasattr(grad, "item") else grad), **metrics}
            _append_jsonl(log_path, row)
            print(
                f"step={step} loss={row['loss']:.4f} dec_acc={row['decoder_mask_acc']:.3f} "
                f"bb_acc={row['backbone_mask_acc']:.3f} rate={row['readout_units_per_byte']:.3f}",
                flush=True,
            )
        if step > 0 and step % args.ckpt_every == 0:
            payload = {
                "model": model.state_dict(),
                "backbone": backbone.state_dict() if backbone is not None else None,
                "optimizer": opt.state_dict() if args.save_optimizer else None,
                "scheduler": sched.state_dict(),
                "args": vars(args),
                "step": step,
            }
            torch.save(payload, out_dir / f"step{step}.pt")
            torch.save(payload, out_dir / "latest.pt")
        step += 1

    eval_stats = evaluate(model, backbone, eval_loader, args, device)
    elapsed = time.perf_counter() - start
    result = {
        "model_version": "flued_v3_3",
        "experiment_name": args.experiment_name,
        "run_id": args.run_id,
        "params": params,
        "backbone_params": backbone_params,
        "steps": step,
        "elapsed_sec": elapsed,
        "steps_per_sec": step / max(elapsed, 1e-9),
        "args": vars(args),
        **{f"eval_{k}": v for k, v in eval_stats.items()},
    }
    torch.save({
        "model": model.state_dict(),
        "backbone": backbone.state_dict() if backbone is not None else None,
        "optimizer": opt.state_dict() if args.save_optimizer else None,
        "scheduler": sched.state_dict(),
        "args": vars(args),
        "step": step,
        "summary": result,
    }, out_dir / "latest.pt")
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="", help="Optional JSON config. CLI arguments override config values.")
    pre_args, _remaining = pre.parse_known_args()
    parser = argparse.ArgumentParser(description="Train FLUED v3.3 byte-to-latent interface", parents=[pre])
    parser.add_argument("--experiment-name", default="v33_manual")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--out-dir", default="checkpoints/v33_smoke")
    parser.add_argument("--streaming-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stream-samples-per-worker", type=int, default=100000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-z", type=int, default=256)
    parser.add_argument("--d-mem", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--max-chunks", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-readout-vectors", type=int, default=1)
    parser.add_argument("--chunk-mixer", choices=["mean", "delta_lite"], default="mean")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau-cut", type=float, default=0.90)
    parser.add_argument("--tau-trans", type=float, default=0.75)
    parser.add_argument("--tau-keep", type=float, default=0.65)
    parser.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-rank", type=int, default=0)
    parser.add_argument("--memory-top-k", type=int, default=4)
    parser.add_argument("--memory-build-mode", choices=["causal_current", "parallel_local"], default="parallel_local")
    parser.add_argument("--memory-visibility", choices=["past_only", "bidirectional_no_self", "all_visible"], default="bidirectional_no_self")
    parser.add_argument("--strict-masked-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--visible-loss-weight", type=float, default=0.25)
    parser.add_argument("--masked-loss-weight", type=float, default=1.0)
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.02)
    parser.add_argument("--boundary-credit-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-credit-backbone-weight", type=float, default=1.0)
    parser.add_argument("--rate-loss-weight", type=float, default=0.02)
    parser.add_argument("--rate-loss", choices=["upper", "l2"], default="upper")
    parser.add_argument("--target-rate", type=float, default=0.50)
    parser.add_argument("--coding-rate-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-vector-overlap-loss-weight", "--memory-redundancy-loss-weight", dest="memory_vector_overlap_loss_weight", type=float, default=0.0)
    parser.add_argument("--memory-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backbone-loss-weight", type=float, default=0.0)
    parser.add_argument("--active-only-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emit-compute-mode", choices=["all", "emitted", "masked_or_emitted"], default="all")
    parser.add_argument("--emit-threshold", type=float, default=0.5)
    parser.add_argument("--backbone-hidden", type=int, default=192)
    parser.add_argument("--backbone-layers", type=int, default=2)
    parser.add_argument("--backbone-nhead", type=int, default=4)
    parser.add_argument("--backbone-ffn-dim", type=int, default=768)
    parser.add_argument("--detach-backbone-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detach-backbone-keep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "fused_adamw", "foreach_adamw"], default="adamw")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    if pre_args.config:
        config_path = Path(pre_args.config)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            parser.error("--config must point to a JSON object")
        normalized = {str(k).replace("-", "_"): v for k, v in data.items()}
        if "memory_redundancy_loss_weight" in normalized and "memory_vector_overlap_loss_weight" not in normalized:
            normalized["memory_vector_overlap_loss_weight"] = normalized.pop("memory_redundancy_loss_weight")
        valid_keys = {action.dest for action in parser._actions}
        unknown = sorted(key for key in normalized if key not in valid_keys)
        if unknown:
            parser.error(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**normalized)
    args = parser.parse_args()
    if args.use_memory and args.memory_rank <= 0:
        parser.error("--use-memory requires --memory-rank > 0")
    if args.backbone_loss_weight > 0 and not args.use_backbone:
        parser.error("--backbone-loss-weight > 0 requires --use-backbone")
    run(args)


if __name__ == "__main__":
    main()
