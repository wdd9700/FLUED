"""Small probes for FLUED v3.3 architecture risks.

The probes are intentionally lightweight and do not train.  They are meant to
turn architectural doubts into measurable facts before changing the design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID  # noqa: E402
from flued.v33 import FLUEDV33, FLUEDV33Config  # noqa: E402
from tools.analysis.benchmark_v33_efficiency import DEFAULTS  # noqa: E402
from tools.train.train_v33 import (  # noqa: E402
    _compact_active_readout,
    _flatten_readout,
    make_byte_mask,
    make_targets,
    masked_readouts_from_slots,
    compute_readout_active_mask,
)


def _load_args(config_path: str) -> SimpleNamespace:
    values = dict(DEFAULTS)
    values.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    return SimpleNamespace(**values)


def _small_model(
    max_span: int = 128,
    max_readout_vectors: int = 16,
    chunk_mixer: str = "mean",
    memory_build_mode: str = "parallel_local",
    memory_visibility: str = "bidirectional_no_self",
) -> FLUEDV33:
    return FLUEDV33(
        FLUEDV33Config(
            d_model=64,
            d_z=32,
            d_mem=32,
            hidden=128,
            max_chunks=128,
            max_span=max_span,
            max_readout_vectors=max_readout_vectors,
            tau_cut=0.90,
            tau_trans=0.75,
            tau_keep=0.65,
            use_memory=True,
            memory_rank=2,
            memory_top_k=4,
            chunk_mixer=chunk_mixer,
            memory_build_mode=memory_build_mode,
            memory_visibility=memory_visibility,
        )
    ).eval()


@torch.no_grad()
def _probe_order_sensitivity_for_mixer(device: torch.device, chunk_mixer: str) -> dict:
    model = _small_model(max_span=8, max_readout_vectors=4, chunk_mixer=chunk_mixer).to(device)
    # IDs are byte+1.  Keep exactly the same byte multiset and no punctuation.
    ab = torch.tensor([[ord("a") + 1, ord("b") + 1, PAD_ID, PAD_ID]], device=device)
    ba = torch.tensor([[ord("b") + 1, ord("a") + 1, PAD_ID, PAD_ID]], device=device)
    out_ab = model(ab)
    out_ba = model(ba)
    pooled_ab = out_ab.interpreter.z_content[:, :1]
    pooled_ba = out_ba.interpreter.z_content[:, :1]
    readout_ab = out_ab.readout_z[:, :1]
    readout_ba = out_ba.readout_z[:, :1]
    return {
        "chunk_mixer": chunk_mixer,
        "z_content_max_abs_diff_ab_vs_ba": float((pooled_ab - pooled_ba).abs().max().item()),
        "readout_z_max_abs_diff_ab_vs_ba": float((readout_ab - readout_ba).abs().max().item()),
        "same_chunk_ids": bool(torch.equal(out_ab.chunks.chunk_ids, out_ba.chunks.chunk_ids)),
        "interpretation": "near_zero_diff_means_chunk_order_is_not_encoded_before_decoder",
    }


@torch.no_grad()
def probe_order_sensitivity(device: torch.device, chunk_mixer: str) -> dict:
    rows = {
        "mean": _probe_order_sensitivity_for_mixer(device, "mean"),
    }
    if chunk_mixer != "mean":
        rows[chunk_mixer] = _probe_order_sensitivity_for_mixer(device, chunk_mixer)
    return rows


@torch.no_grad()
def probe_masked_chunk_ratio(args: SimpleNamespace, device: torch.device, trials: int) -> dict:
    model = _small_model(max_span=args.max_span, max_readout_vectors=args.max_readout_vectors, chunk_mixer=args.chunk_mixer).to(device)
    rows = []
    for _ in range(trials):
        clean = torch.randint(1, 257, (int(args.batch_size), int(args.seq_len)), dtype=torch.long, device=device)
        valid = clean.ne(PAD_ID)
        byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
        src = clean.masked_fill(byte_mask, MASK_ID) if args.strict_masked_source else clean
        out = model(src)
        _targets, slot_mask, masked_slot = make_targets(clean, byte_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span)
        masked_chunks = masked_slot.any(dim=-1) & out.chunks.chunk_mask
        masked_readouts = masked_readouts_from_slots(masked_slot, out.chunks.chunk_mask, out.readout_z.size(2))
        chunk_count = out.chunks.chunk_mask.float().sum().clamp(min=1.0)
        readout_count = out.chunks.chunk_mask.unsqueeze(-1).expand_as(masked_readouts).float().sum().clamp(min=1.0)
        rows.append(
            {
                "masked_byte_fraction": float((byte_mask.float().sum() / valid.float().sum().clamp(min=1.0)).item()),
                "masked_chunk_fraction": float((masked_chunks.float().sum() / chunk_count).item()),
                "masked_readout_fraction": float((masked_readouts.float().sum() / readout_count).item()),
                "active_chunks": float(chunk_count.item()),
            }
        )
    return {
        "mask_prob": float(args.mask_prob),
        "mean_masked_byte_fraction": sum(r["masked_byte_fraction"] for r in rows) / len(rows),
        "mean_masked_chunk_fraction": sum(r["masked_chunk_fraction"] for r in rows) / len(rows),
        "mean_masked_readout_fraction": sum(r["masked_readout_fraction"] for r in rows) / len(rows),
        "mean_active_chunks": sum(r["active_chunks"] for r in rows) / len(rows),
    }


@torch.no_grad()
def probe_gate_vs_compute(args: SimpleNamespace, device: torch.device) -> dict:
    model = _small_model(max_span=args.max_span, max_readout_vectors=args.max_readout_vectors, chunk_mixer=args.chunk_mixer).to(device)
    clean = torch.randint(1, 257, (int(args.batch_size), int(args.seq_len)), dtype=torch.long, device=device)
    out = model(clean)
    byte_mask = make_byte_mask(clean.ne(PAD_ID), args.mask_prob, args.mask_span_min, args.mask_span_max)
    _targets, slot_mask, masked_slot = make_targets(clean, byte_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span)
    masked_readouts = masked_readouts_from_slots(masked_slot, out.chunks.chunk_mask, out.readout_z.size(2))
    compute_readouts = compute_readout_active_mask(
        out.chunks.chunk_mask,
        out.readout_emit,
        masked_readouts,
        args.emit_compute_mode,
        args.emit_threshold,
    )
    flat_z, flat_active_all, flat_pos = _flatten_readout(out.readout_z, out.readout_gate, out.chunks.chunk_mask)
    compact_z, compact_active, _compact_masked, _compact_pos = _compact_active_readout(
        flat_z,
        compute_readouts.reshape(compute_readouts.size(0), -1),
        masked_readouts.reshape(masked_readouts.size(0), -1),
        flat_pos,
    )
    valid_bytes = clean.ne(PAD_ID).float().sum().clamp(min=1.0)
    soft_units = float(out.readout_gate.float().sum().item())
    emit_units = float(out.readout_emit.float().sum().item())
    dense_units = float(flat_active_all.float().sum().item())
    actual_units = float(compact_z.size(0) * compact_z.size(1))
    active_units = float(compact_active.float().sum().item())
    gate = out.readout_gate[out.chunks.chunk_mask]
    emit = out.readout_emit[out.chunks.chunk_mask]
    extra_gate = gate[:, 1:].reshape(-1) if gate.numel() else gate.reshape(-1)
    extra_emit = emit[:, 1:].reshape(-1) if emit.numel() else emit.reshape(-1)
    return {
        "soft_readout_units_per_byte": soft_units / float(valid_bytes.item()),
        "soft_emit_units_per_byte": emit_units / float(valid_bytes.item()),
        "dense_backbone_units_per_byte": dense_units / float(valid_bytes.item()),
        "actual_backbone_units_per_byte": actual_units / float(valid_bytes.item()),
        "backbone_active_units_per_byte": active_units / float(valid_bytes.item()),
        "actual_over_soft": actual_units / max(soft_units, 1.0e-9),
        "actual_over_dense": actual_units / max(dense_units, 1.0e-9),
        "extra_gate_mean": float(extra_gate.mean().item()) if extra_gate.numel() else 0.0,
        "extra_gate_min": float(extra_gate.min().item()) if extra_gate.numel() else 0.0,
        "extra_gate_max": float(extra_gate.max().item()) if extra_gate.numel() else 0.0,
        "extra_emit_mean": float(extra_emit.mean().item()) if extra_emit.numel() else 0.0,
        "extra_emit_min": float(extra_emit.min().item()) if extra_emit.numel() else 0.0,
        "extra_emit_max": float(extra_emit.max().item()) if extra_emit.numel() else 0.0,
    }


@torch.no_grad()
def probe_memory_visibility(args: SimpleNamespace, device: torch.device) -> dict:
    model = _small_model(
        max_span=args.max_span,
        max_readout_vectors=args.max_readout_vectors,
        chunk_mixer=args.chunk_mixer,
        memory_build_mode=args.memory_build_mode,
        memory_visibility=args.memory_visibility,
    ).to(device)
    clean = torch.randint(1, 257, (int(args.batch_size), int(args.seq_len)), dtype=torch.long, device=device)
    out = model(clean)
    active = out.chunks.chunk_mask
    self_allowed = out.aux.get("memory_self_allowed_count")
    visible_slots = out.aux.get("memory_visible_slots")
    has_context = out.aux.get("memory_has_context")
    active_w = active.float()
    denom = active_w.sum().clamp(min=1.0)
    return {
        "memory_build_mode": args.memory_build_mode,
        "memory_visibility": args.memory_visibility,
        "active_chunks_mean": float(active_w.sum(dim=1).float().mean().item()),
        "has_context_frac": float((has_context.float() * active_w).sum().item() / denom.item()) if has_context is not None else 0.0,
        "self_allowed_mean": float((self_allowed.float() * active_w).sum().item() / denom.item()) if self_allowed is not None else 0.0,
        "visible_slots_mean": float((visible_slots.float() * active_w).sum().item() / denom.item()) if visible_slots is not None else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe FLUED v3.3 architecture risks")
    parser.add_argument("--config", default="configs/v33_full_300m_100m_corpus_v4.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--out-json", default="")
    ns = parser.parse_args()
    args = _load_args(ns.config)
    device = torch.device(ns.device if ns.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    result = {
        "config": ns.config,
        "device": str(device),
        "chunk_mixer": args.chunk_mixer,
        "order_sensitivity": probe_order_sensitivity(device, args.chunk_mixer),
        "masked_chunk_ratio": probe_masked_chunk_ratio(args, device, ns.trials),
        "gate_vs_compute": probe_gate_vs_compute(args, device),
        "memory_visibility": probe_memory_visibility(args, device),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if ns.out_json:
        Path(ns.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(ns.out_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
