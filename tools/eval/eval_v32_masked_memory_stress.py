"""Strict masked-source memory stress tests for FLUED-v3.2.1.

The older memory-case scripts evaluate clean reconstruction.  This script is
stricter: it masks repeated entities / identifiers in the raw byte input before
FLUED sees the sample, rebuilds segmentation from the masked source, and then
compares full / zero / shuffled / stale memory on the masked target bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID, text_to_byte_ids  # noqa: E402
from tools.analysis.train_v32_language_codec_2m import build_segments, complete_utf8_edge_valid, weak_boundary_starts  # noqa: E402
from tools.eval.eval_v32_language_codec_memory_ablation import (  # noqa: E402
    DEFAULT_MODES,
    _load_model,
    _resolve_checkpoint,
    forward_with_mode,
)


@dataclass(frozen=True)
class StressCase:
    label: str
    text: str
    terms: Tuple[str, ...]


DEFAULT_CASES: Tuple[StressCase, ...] = (
    StressCase(
        label="entity_repeat_en",
        text=(
            "Asterion-47 opened the archive with key KQ-91. Later the report says "
            "Asterion-47 must reuse key KQ-91 exactly."
        ),
        terms=("Asterion-47", "KQ-91"),
    ),
    StressCase(
        label="code_identifier",
        text=(
            "GraphCacheIndex maps pages to shards. GraphCacheIndexBuilder refreshes "
            "GraphCacheIndex after compact_index() updates GraphCacheIndex."
        ),
        terms=("GraphCacheIndex", "compact_index"),
    ),
    StressCase(
        label="version_number",
        text=(
            "Build ZX-9082 passed smoke tests with schema v17.4. The rollback note "
            "says ZX-9082 must keep schema v17.4 unchanged."
        ),
        terms=("ZX-9082", "v17.4"),
    ),
    StressCase(
        label="cjk_repeat",
        text=(
            "玄曜计划启动后，研究组记录了玄曜计划的编码结果。复核阶段要求玄曜计划保持同一命名。"
        ),
        terms=("玄曜计划",),
    ),
)


def _torch_load_jsonable(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _fmt(value: float, digits: int = 6) -> str:
    return "n/a" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _fmt_delta(value: float, digits: int = 6) -> str:
    return "n/a" if not math.isfinite(float(value)) else f"{float(value):+.{digits}f}"


def _parse_case(raw: str) -> StressCase:
    parts = raw.split("::", 2)
    if len(parts) != 3:
        raise ValueError("--case format: label::term1,term2::text")
    label, terms_raw, text = parts
    terms = tuple(term.strip() for term in terms_raw.split(",") if term.strip())
    if not label.strip() or not terms or not text:
        raise ValueError("--case requires non-empty label, terms, and text")
    return StressCase(label=label.strip(), terms=terms, text=text)


def _iter_cases(raw_cases: Sequence[str]) -> Tuple[StressCase, ...]:
    return tuple(_parse_case(item) for item in raw_cases) if raw_cases else DEFAULT_CASES


def _find_later_term_ranges(raw: bytes, terms: Sequence[str]) -> List[Tuple[int, int, str]]:
    ranges: List[Tuple[int, int, str]] = []
    for term in terms:
        needle = term.encode("utf-8")
        if not needle:
            continue
        positions: List[int] = []
        start = 0
        while True:
            pos = raw.find(needle, start)
            if pos < 0:
                break
            positions.append(pos)
            start = pos + 1
        for pos in positions[1:]:
            ranges.append((pos, pos + len(needle), term))
    return ranges


def _build_case_batch(case: StressCase, seq_len: int, min_span: int, max_span: int, max_units: int):
    ids = text_to_byte_ids(case.text)
    ids = ids[:seq_len]
    raw = bytes(max(0, min(255, tid - BYTE_OFFSET)) for tid in ids if BYTE_OFFSET <= tid < MASK_ID)
    src = torch.full((1, seq_len), PAD_ID, dtype=torch.long)
    if ids:
        src[0, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    clean_valid = complete_utf8_edge_valid(src, src.ne(PAD_ID))
    byte_mask = torch.zeros_like(clean_valid, dtype=torch.bool)
    ranges = _find_later_term_ranges(raw, case.terms)
    for begin, end, _term in ranges:
        lo = max(0, min(begin, seq_len))
        hi = max(lo, min(end, seq_len))
        byte_mask[0, lo:hi] = True
    byte_mask &= clean_valid
    masked_src = src.masked_fill(byte_mask, MASK_ID)
    valid = complete_utf8_edge_valid(masked_src, masked_src.ne(PAD_ID))
    starts = weak_boundary_starts(masked_src, valid, min_span, max_span)
    seg_ids, _masked_targets, masked_lengths, seg_mask = build_segments(masked_src, valid, starts, min(max_units, seq_len), max_span)
    targets, loss_mask, lengths = _targets_from_segments(src, byte_mask, seg_ids, seg_mask, max_span)
    lengths = torch.where(lengths.gt(0), lengths, masked_lengths)
    return {
        "src": src,
        "masked_src": masked_src,
        "valid": valid,
        "seg_ids": seg_ids,
        "seg_mask": seg_mask,
        "targets": targets,
        "loss_mask": loss_mask,
        "lengths": lengths,
        "byte_mask": byte_mask,
        "ranges": ranges,
    }


def _targets_from_segments(clean_src: torch.Tensor, byte_mask: torch.Tensor, seg_ids: torch.Tensor, seg_mask: torch.Tensor, max_span: int):
    bsz, seq_len = clean_src.shape
    max_units = seg_mask.size(1)
    targets = torch.full((bsz, max_units, max_span), PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros((bsz, max_units, max_span), dtype=torch.bool)
    lengths = torch.zeros((bsz, max_units), dtype=torch.long)
    cursor = torch.zeros((bsz, max_units), dtype=torch.long)
    for b in range(bsz):
        for t in range(seq_len):
            unit = int(seg_ids[b, t].item())
            if unit < 0 or unit >= max_units:
                continue
            slot = int(cursor[b, unit].item())
            if slot >= max_span:
                continue
            targets[b, unit, slot] = clean_src[b, t]
            loss_mask[b, unit, slot] = bool(byte_mask[b, t])
            cursor[b, unit] += 1
            lengths[b, unit] = max(int(lengths[b, unit].item()), slot + 1)
    return targets, loss_mask, lengths


def _stats(byte_logits: torch.Tensor, length_logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor, lengths: torch.Tensor, seg_mask: torch.Tensor, max_span: int) -> Dict[str, float]:
    ce = F.cross_entropy(
        byte_logits.float().reshape(-1, byte_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(targets)
    slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
    keep_mask = slot_mask & (~loss_mask)
    pred = byte_logits.argmax(dim=-1)
    length_target = (lengths.clamp(min=1, max=max_span) - 1).clamp(min=0)
    len_pred = length_logits.argmax(dim=-1)
    unit_mask = loss_mask.any(dim=-1) & seg_mask
    return {
        "masked_loss": float(ce[loss_mask].mean().item()) if loss_mask.any() else float("nan"),
        "masked_acc": float((pred[loss_mask] == targets[loss_mask]).float().mean().item()) if loss_mask.any() else float("nan"),
        "keep_acc": float((pred[keep_mask] == targets[keep_mask]).float().mean().item()) if keep_mask.any() else float("nan"),
        "length_acc": float(((len_pred == length_target) & seg_mask).float().sum().item() / max(float(seg_mask.sum().item()), 1.0)),
        "masked_length_acc": float(((len_pred == length_target) & unit_mask).float().sum().item() / max(float(unit_mask.sum().item()), 1.0)),
        "masked_slots": float(loss_mask.sum().item()),
        "active_units": float(seg_mask.sum().item()),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = _resolve_checkpoint(Path(args.checkpoint))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model, meta = _load_model(ckpt_path, device)
    # Avoid the CPU Transformer eval fast path that produced NaNs in earlier
    # strict masked-source checks. Dropout is 0 in fair v3.2.1 runs.
    model.train()
    ckpt_args = _torch_load_jsonable(meta.get("args", {}))
    seq_len = int(args.seq_len or ckpt_args.get("seq_len", 128) or 128)
    min_span = int(args.min_span or ckpt_args.get("min_span", 2) or 2)
    max_span = int(args.max_span or ckpt_args.get("max_span", getattr(model, "max_span", 16)) or 16)
    max_units = int(args.max_units or ckpt_args.get("max_units", seq_len) or seq_len)
    modes = [mode.strip() for mode in args.modes.replace(",", " ").split() if mode.strip()] or list(DEFAULT_MODES)
    if "full" not in modes:
        modes.insert(0, "full")
    for mode in modes:
        if mode not in DEFAULT_MODES:
            raise ValueError(f"unknown mode {mode}; choices: {DEFAULT_MODES}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    cases = _iter_cases(args.case)
    rows: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, float]] = {mode: {"loss_sum": 0.0, "correct": 0.0, "slots": 0.0} for mode in modes}
    previous_memory: Optional[torch.Tensor] = None
    for case in cases:
        batch = _build_case_batch(case, seq_len, min_span, max_span, max_units)
        if not batch["loss_mask"].any():
            rows.append({"case": case.label, "skipped": True, "reason": "no later term occurrence within seq_len"})
            continue
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        case_row: Dict[str, Any] = {
            "case": case.label,
            "terms": list(case.terms),
            "masked_ranges": [(int(a), int(b), term) for a, b, term in batch["ranges"]],
            "masked_slots": int(tensors["loss_mask"].sum().item()),
            "active_units": int(tensors["seg_mask"].sum().item()),
        }
        full_memory_for_stale: Optional[torch.Tensor] = None
        for mode in modes:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                byte_logits, length_logits, metrics = forward_with_mode(
                    model,
                    tensors["masked_src"],
                    tensors["valid"],
                    tensors["seg_ids"],
                    tensors["seg_mask"],
                    mode,
                    previous_memory=previous_memory,
                    generator=generator,
                )
            if mode == "full":
                full_memory_for_stale = metrics["full_memory"].detach()
            stats = _stats(byte_logits, length_logits, tensors["targets"], tensors["loss_mask"], tensors["lengths"], tensors["seg_mask"], max_span)
            case_row[mode] = stats
            if math.isfinite(stats["masked_loss"]):
                totals[mode]["loss_sum"] += stats["masked_loss"] * stats["masked_slots"]
                totals[mode]["slots"] += stats["masked_slots"]
                totals[mode]["correct"] += stats["masked_acc"] * stats["masked_slots"]
        if full_memory_for_stale is not None:
            previous_memory = full_memory_for_stale
        rows.append(case_row)

    summary: Dict[str, Any] = {}
    full_acc = float("nan")
    full_loss = float("nan")
    for mode in modes:
        slots = totals[mode]["slots"]
        acc = totals[mode]["correct"] / slots if slots else float("nan")
        loss = totals[mode]["loss_sum"] / slots if slots else float("nan")
        if mode == "full":
            full_acc, full_loss = acc, loss
        summary[mode] = {
            "masked_acc": acc,
            "masked_loss": loss,
            "delta_acc_vs_full": acc - full_acc if math.isfinite(acc) and math.isfinite(full_acc) else float("nan"),
            "delta_loss_vs_full": loss - full_loss if math.isfinite(loss) and math.isfinite(full_loss) else float("nan"),
            "masked_slots": slots,
        }

    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": meta.get("step"),
        "device": str(device),
        "strict_masked_source": True,
        "mask_policy": "later_occurrences_of_terms",
        "memory_enabled": bool(getattr(model, "memory_slots_per_chunk", 0) > 0),
        "memory_retrieval_mode": (
            str(getattr(model, "memory_retrieval_mode", "topk"))
            if getattr(model, "memory_slots_per_chunk", 0) > 0
            else "none"
        ),
        "seq_len": seq_len,
        "max_span": max_span,
        "modes": modes,
        "summary": summary,
        "cases": rows,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    modes = list(report.get("modes", []))
    lines = [
        "# FLUED v3.2 Strict Masked Memory Stress",
        "",
        f"- checkpoint: `{report.get('checkpoint')}`",
        f"- step: `{report.get('checkpoint_step')}`",
        f"- memory_enabled: `{report.get('memory_enabled')}`",
        f"- retrieval: `{report.get('memory_retrieval_mode')}`",
        f"- mask_policy: `{report.get('mask_policy')}`",
        "",
        "| mode | masked_acc | delta_acc | masked_loss | delta_loss | masked_slots |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary = report.get("summary", {})
    for mode in modes:
        row = summary.get(mode, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    _fmt(row.get("masked_acc", float("nan"))),
                    _fmt_delta(row.get("delta_acc_vs_full", float("nan"))),
                    _fmt(row.get("masked_loss", float("nan"))),
                    _fmt_delta(row.get("delta_loss_vs_full", float("nan"))),
                    str(int(row.get("masked_slots", 0) or 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases", []):
        lines.append(f"### {case.get('case')}")
        if case.get("skipped"):
            lines.append(f"- skipped: {case.get('reason')}")
            lines.append("")
            continue
        lines.append(f"- terms: `{', '.join(case.get('terms', []))}`")
        lines.append(f"- masked_slots: `{case.get('masked_slots')}`")
        lines.append("")
        lines.append("| mode | masked_acc | masked_loss | keep_acc | masked_length_acc |")
        lines.append("|---|---:|---:|---:|---:|")
        for mode in modes:
            row = case.get(mode, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        mode,
                        _fmt(row.get("masked_acc", float("nan"))),
                        _fmt(row.get("masked_loss", float("nan"))),
                        _fmt(row.get("keep_acc", float("nan"))),
                        _fmt(row.get("masked_length_acc", float("nan"))),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict masked-source memory stress eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=128)
    parser.add_argument("--modes", default="full,zero,shuffled,stale")
    parser.add_argument("--case", action="append", default=[], help="label::term1,term2::text")
    parser.add_argument("--seed", type=int, default=1234)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
