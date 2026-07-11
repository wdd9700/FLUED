"""Case-based memory diagnostics for FLUED-v3.2 language codec.

This evaluator keeps the v3.2 interface boundary intact: the external output is
``readout`` only, while causal summary slots and retrieved past memory remain
internal encoder state.  It probes whether that internal memory helps on
repeated entities, technical terms, identifiers, and cross-reference text where
history should matter more than in random corpus averages.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID, ByteReconstructionDataset  # noqa: E402
from tools.analysis.v3_2.train_v32_language_codec_2m import CodecCollator, move_codec_batch  # noqa: E402
from tools.eval.v3_2.eval_v32_language_codec_memory_ablation import (  # noqa: E402
    DEFAULT_MODES,
    _load_model,
    _resolve_checkpoint,
    _torch_load,
    forward_with_mode,
)


@dataclass(frozen=True)
class MemoryCase:
    label: str
    text: str
    terms: Tuple[str, ...]


DEFAULT_CASES: Tuple[MemoryCase, ...] = (
    MemoryCase(
        label="entity_repeat_en",
        text=(
            "Asterion-47 opened the archive. Later, Asterion-47 reused the same key "
            "inside a short report."
        ),
        terms=("Asterion-47", "key"),
    ),
    MemoryCase(
        label="code_identifier",
        text=(
            "GraphCacheIndex maps pages to shards. GraphCacheIndexBuilder refreshes "
            "GraphCacheIndex after compaction."
        ),
        terms=("GraphCacheIndex", "GraphCacheIndexBuilder", "compaction"),
    ),
    MemoryCase(
        label="cjk_reference",
        text=(
            "\u7384\u66dc\u8ba1\u5212\u542f\u52a8\u540e\uff0c\u7814\u7a76\u7ec4"
            "\u91cd\u590d\u5bf9\u7167\u7384\u66dc\u8ba1\u5212\u7684\u7f16\u7801"
            "\u7ed3\u679c\u3002"
        ),
        terms=("\u7384\u66dc\u8ba1\u5212", "\u7814\u7a76\u7ec4"),
    ),
    MemoryCase(
        label="version_number",
        text=(
            "Build ZX-9082 passed smoke tests. The rollback note says ZX-9082 "
            "must keep schema v17.4 unchanged."
        ),
        terms=("ZX-9082", "v17.4", "schema"),
    ),
)


def _arg_value(cli_value: Any, ckpt_args: Mapping[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    value = ckpt_args.get(key, default)
    return default if value is None else value


def _arg_int(cli_value: Any, ckpt_args: Mapping[str, Any], key: str, default: int) -> int:
    return int(_arg_value(cli_value, ckpt_args, key, default))


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _select_device(device_arg: Optional[str], ckpt_args: Mapping[str, Any]) -> torch.device:
    wanted = device_arg or str(ckpt_args.get("device", "cpu") or "cpu")
    if wanted == "cuda" and not torch.cuda.is_available():
        wanted = "cpu"
    return torch.device(wanted)


def _parse_modes(raw: str) -> Tuple[str, ...]:
    modes = tuple(mode.strip() for mode in raw.split(",") if mode.strip())
    allowed = set(DEFAULT_MODES)
    bad = [mode for mode in modes if mode not in allowed]
    if bad:
        raise ValueError(f"unknown memory modes: {', '.join(bad)}")
    return modes or DEFAULT_MODES


def _fmt(value: float, digits: int = 6) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_delta(value: float, digits: int = 6) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _fmt_int(value: float) -> str:
    if not _is_finite(value):
        return "n/a"
    return str(int(value))


def _parse_case(raw: str) -> MemoryCase:
    """Parse label::term1,term2::text or label::text."""

    parts = raw.split("::", 2)
    if len(parts) == 2:
        label, text = parts
        terms: Tuple[str, ...] = ()
    elif len(parts) == 3:
        label, terms_raw, text = parts
        terms = tuple(term.strip() for term in terms_raw.split(",") if term.strip())
    else:
        raise ValueError("--text must be label::text or label::term1,term2::text")
    label = label.strip()
    if not label:
        raise ValueError("case label must not be empty")
    if not text:
        raise ValueError("case text must not be empty")
    return MemoryCase(label=label, text=text, terms=terms)


def _iter_cases(raw_cases: Sequence[str]) -> Tuple[MemoryCase, ...]:
    if not raw_cases:
        return DEFAULT_CASES
    return tuple(_parse_case(item) for item in raw_cases)


def _source_bytes(src: torch.Tensor, valid: torch.Tensor) -> bytes:
    values: List[int] = []
    for token, keep in zip(src.tolist(), valid.tolist()):
        if not keep:
            continue
        if BYTE_OFFSET <= int(token) < MASK_ID:
            values.append(int(token) - BYTE_OFFSET)
    return bytes(values)


def _find_ranges(raw: bytes, terms: Sequence[str]) -> List[Tuple[int, int, str]]:
    ranges: List[Tuple[int, int, str]] = []
    for term in terms:
        needle = term.encode("utf-8")
        if not needle:
            continue
        start = 0
        while True:
            pos = raw.find(needle, start)
            if pos < 0:
                break
            ranges.append((pos, pos + len(needle), term))
            start = pos + 1
    return ranges


def _unit_masks(
    src: torch.Tensor,
    valid: torch.Tensor,
    seg_ids: torch.Tensor,
    seg_mask: torch.Tensor,
    terms: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Build all/later/entity segment masks for a batch."""

    masks = {
        "all": seg_mask.clone(),
        "later": torch.zeros_like(seg_mask, dtype=torch.bool),
        "entity": torch.zeros_like(seg_mask, dtype=torch.bool),
    }
    bsz, _seq_len = src.shape
    for b in range(bsz):
        active_units = seg_mask[b].nonzero(as_tuple=False).flatten()
        if active_units.numel() > 0:
            start = int(active_units.numel() // 2)
            masks["later"][b, active_units[start:]] = True

        raw = _source_bytes(src[b].cpu(), valid[b].cpu())
        ranges = _find_ranges(raw, terms)
        if not ranges:
            continue

        byte_offset = 0
        for t in range(src.size(1)):
            if not bool(valid[b, t]):
                continue
            token = int(src[b, t].item())
            if not (BYTE_OFFSET <= token < MASK_ID):
                continue
            unit = int(seg_ids[b, t].item())
            if unit < 0 or unit >= seg_mask.size(1):
                byte_offset += 1
                continue
            for begin, end, _term in ranges:
                if begin <= byte_offset < end:
                    masks["entity"][b, unit] = True
                    break
            byte_offset += 1
    return masks


def _empty_stats() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "slots": 0.0,
        "correct": 0.0,
        "units": 0.0,
        "length_correct": 0.0,
    }


def _update_subset_stats(
    stats: Dict[str, float],
    byte_logits: torch.Tensor,
    length_logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    segment_subset: torch.Tensor,
    max_span: int,
) -> None:
    slot_mask = targets.ne(PAD_ID) & segment_subset.unsqueeze(-1)
    if slot_mask.any():
        loss = F.cross_entropy(
            byte_logits.float().reshape(-1, byte_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(targets)
        stats["loss_sum"] += float(loss[slot_mask].sum().item())
        pred = byte_logits.argmax(dim=-1)
        stats["correct"] += float(((pred == targets) & slot_mask).sum().item())
        stats["slots"] += float(slot_mask.sum().item())

    if segment_subset.any():
        length_target = (lengths.clamp(min=1, max=max_span) - 1).clamp(min=0)
        length_pred = length_logits.argmax(dim=-1)
        stats["length_correct"] += float(((length_pred == length_target) & segment_subset).sum().item())
        stats["units"] += float(segment_subset.sum().item())


def _finalize_stats(stats: Mapping[str, float]) -> Dict[str, float]:
    slots = float(stats["slots"])
    units = float(stats["units"])
    return {
        "recon_loss": float(stats["loss_sum"]) / slots if slots else float("nan"),
        "recon_acc": float(stats["correct"]) / slots if slots else float("nan"),
        "length_acc": float(stats["length_correct"]) / units if units else float("nan"),
        "slots": slots,
        "units": units,
    }


def _make_batch(case: MemoryCase, seq_len: int, stride: int, collator: CodecCollator):
    dataset = ByteReconstructionDataset(texts=[case.text], seq_len=seq_len, stride=stride)
    items = [dataset[i] for i in range(len(dataset))]
    return collator(items)


@torch.no_grad()
def evaluate_cases(
    model,
    cases: Sequence[MemoryCase],
    modes: Sequence[str],
    *,
    seq_len: int,
    stride: int,
    min_span: int,
    max_span: int,
    max_units: int,
    device: torch.device,
    amp: bool,
    seed: int,
) -> Dict[str, Any]:
    collator = CodecCollator(min_span=min_span, max_span=max_span, max_units=max_units)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed + 4093)
    previous_full_memory: Optional[torch.Tensor] = None
    case_reports: List[Dict[str, Any]] = []

    for case in cases:
        batch = _make_batch(case, seq_len, stride, collator)
        src, starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
        del starts
        valid = src.ne(PAD_ID)
        subsets = _unit_masks(src.detach().cpu(), valid.detach().cpu(), seg_ids.detach().cpu(), seg_mask.detach().cpu(), case.terms)
        subsets = {name: mask.to(device) for name, mask in subsets.items()}

        stats_by_mode = {
            mode: {subset_name: _empty_stats() for subset_name in subsets}
            for mode in modes
        }
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            full_outputs = forward_with_mode(
                model,
                src,
                valid,
                seg_ids,
                seg_mask,
                "full",
                previous_memory=previous_full_memory,
                generator=shuffle_generator,
            )
            full_memory = full_outputs[2]["full_memory"].detach()
            for mode in modes:
                if mode == "full":
                    byte_logits, length_logits, _metrics = full_outputs
                else:
                    byte_logits, length_logits, _metrics = forward_with_mode(
                        model,
                        src,
                        valid,
                        seg_ids,
                        seg_mask,
                        mode,
                        previous_memory=previous_full_memory,
                        generator=shuffle_generator,
                    )
                for subset_name, subset_mask in subsets.items():
                    _update_subset_stats(
                        stats_by_mode[mode][subset_name],
                        byte_logits,
                        length_logits,
                        targets,
                        lengths,
                        subset_mask,
                        max_span,
                    )

        finalized = {
            mode: {
                subset_name: _finalize_stats(stats)
                for subset_name, stats in subset_stats.items()
            }
            for mode, subset_stats in stats_by_mode.items()
        }
        raw = _source_bytes(src[0].detach().cpu(), valid[0].detach().cpu()).decode("utf-8", errors="replace")
        case_reports.append(
            {
                "label": case.label,
                "terms": list(case.terms),
                "text": raw.strip(),
                "chunks": int(src.size(0)),
                "active_units": int(seg_mask.sum().item()),
                "results": finalized,
            }
        )
        previous_full_memory = full_memory

    return {"cases": case_reports}


def _effect_summary(case_report: Mapping[str, Any], subset: str) -> Dict[str, float]:
    results = case_report["results"]
    full = results["full"][subset]
    out = {
        "max_loss_delta_pct": 0.0,
        "max_acc_delta": 0.0,
    }
    for mode in ("zero", "shuffled", "stale"):
        if mode not in results:
            continue
        row = results[mode][subset]
        if _is_finite(row["recon_loss"]) and _is_finite(full["recon_loss"]) and full["recon_loss"] != 0:
            out["max_loss_delta_pct"] = max(
                out["max_loss_delta_pct"],
                abs(100.0 * (row["recon_loss"] - full["recon_loss"]) / full["recon_loss"]),
            )
        if _is_finite(row["recon_acc"]) and _is_finite(full["recon_acc"]):
            out["max_acc_delta"] = max(out["max_acc_delta"], abs(row["recon_acc"] - full["recon_acc"]))
    return out


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# FLUED v3.2 Memory Case Diagnostics")
    lines.append("")
    lines.append(f"- checkpoint: `{report['checkpoint_path']}`")
    lines.append(f"- checkpoint_step: `{report['checkpoint_step']}`")
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- pool_mode: `{report['pool_mode']}`")
    lines.append(f"- seq_len: `{report['eval_config']['seq_len']}`")
    lines.append(f"- max_span: `{report['eval_config']['max_span']}`")
    lines.append(
        "- interface note: `readout` remains the only external latent interface; "
        "causal `summary` slots and retrieved past `memory` are internal encoder state."
    )
    lines.append("")
    lines.append("## Case Summary")
    lines.append("")
    lines.append("| case | chunks | units | entity_units | entity_max_loss_delta_% | entity_max_acc_delta | later_max_loss_delta_% | later_max_acc_delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in report["cases"]:
        entity_units = case["results"]["full"]["entity"]["units"]
        entity_effect = _effect_summary(case, "entity")
        later_effect = _effect_summary(case, "later")
        lines.append(
            "| "
            + " | ".join(
                [
                    case["label"],
                    _fmt_int(case["chunks"]),
                    _fmt_int(case["active_units"]),
                    _fmt_int(entity_units),
                    _fmt(entity_effect["max_loss_delta_pct"], 3),
                    _fmt(entity_effect["max_acc_delta"], 6),
                    _fmt(later_effect["max_loss_delta_pct"], 3),
                    _fmt(later_effect["max_acc_delta"], 6),
                ]
            )
            + " |"
        )

    for case in report["cases"]:
        lines.append("")
        lines.append(f"## {case['label']}")
        lines.append("")
        lines.append(f"- text: `{case['text']}`")
        lines.append(f"- terms: `{', '.join(case['terms']) if case['terms'] else '(none)'}`")
        for subset in ("all", "later", "entity"):
            full = case["results"]["full"][subset]
            lines.append("")
            lines.append(f"### subset={subset}")
            lines.append("")
            lines.append("| mode | recon_loss | delta_loss | delta_loss_% | recon_acc | delta_acc | length_acc | slots | units |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for mode in report["modes"]:
                row = case["results"][mode][subset]
                loss_delta = row["recon_loss"] - full["recon_loss"]
                if _is_finite(row["recon_loss"]) and _is_finite(full["recon_loss"]) and full["recon_loss"] != 0:
                    loss_pct = 100.0 * loss_delta / full["recon_loss"]
                else:
                    loss_pct = float("nan")
                acc_delta = row["recon_acc"] - full["recon_acc"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            mode,
                            _fmt(row["recon_loss"]),
                            _fmt_delta(loss_delta),
                            _fmt_delta(loss_pct, 3),
                            _fmt(row["recon_acc"]),
                            _fmt_delta(acc_delta),
                            _fmt(row["length_acc"]),
                            _fmt_int(row["slots"]),
                            _fmt_int(row["units"]),
                        ]
                    )
                    + " |"
                )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> str:
    checkpoint_path = _resolve_checkpoint(Path(args.checkpoint))
    ckpt_for_device = _torch_load(checkpoint_path)
    ckpt_args = _dict_or_empty(ckpt_for_device.get("args", {}) if isinstance(ckpt_for_device, Mapping) else {})
    device = _select_device(args.device, ckpt_args)
    model, meta = _load_model(checkpoint_path, device)
    ckpt_args = meta["args"]

    seed = _arg_int(args.seed, ckpt_args, "seed", 1234)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    seq_len = _arg_int(args.seq_len, ckpt_args, "seq_len", 128)
    stride = _arg_int(args.stride, ckpt_args, "stride", seq_len)
    min_span = _arg_int(args.min_span, ckpt_args, "min_span", 2)
    max_span = _arg_int(args.max_span, ckpt_args, "max_span", 16)
    max_units = _arg_int(args.max_units, ckpt_args, "max_units", seq_len)
    if int(max_span) != int(model.max_span):
        raise ValueError(f"--max-span must match checkpoint max_span={model.max_span}")
    modes = _parse_modes(args.modes)
    amp = bool(_arg_value(args.amp, ckpt_args, "amp", False))
    cases = _iter_cases(args.text)

    results = evaluate_cases(
        model,
        cases,
        modes,
        seq_len=seq_len,
        stride=stride,
        min_span=min_span,
        max_span=max_span,
        max_units=max_units,
        device=device,
        amp=amp,
        seed=seed,
    )
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": meta.get("step"),
        "device": str(device),
        "pool_mode": str(getattr(model, "pool_mode", "unknown")),
        "modes": modes,
        "eval_config": {
            "seq_len": seq_len,
            "stride": stride,
            "min_span": min_span,
            "max_span": max_span,
            "max_units": max_units,
            "seed": seed,
        },
        **results,
    }
    markdown = _render_markdown(report)
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v3.2 codec memory on hand-picked cases")
    parser.add_argument("--checkpoint", default="checkpoint/latest.pt")
    parser.add_argument("--out-path", default="")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--device", default=None)
    parser.set_defaults(amp=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-span", type=int, default=None)
    parser.add_argument("--max-span", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Custom case: label::text or label::term1,term2::text. Can be repeated.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
