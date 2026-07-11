"""ROI and segmentation report for the FLUED-v3.1 language codec."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID
from tools.analysis.train_v31_language_codec_2m import (
    V31LanguageCodec2M,
    build_segments,
    complete_utf8_edge_valid,
    weak_boundary_starts,
)


BUILTIN_SAMPLES: List[Tuple[str, str]] = [
    (
        "zh",
        "\u8fd9\u4e2a\u7248\u672c\u9700\u8981\u770b\u6e05\u695a\u4e2d\u6587"
        "\u8bcd\u8bed\u3001\u6807\u70b9\u548c UTF-8 \u5b57\u8282\u8fb9\u754c"
        "\u7684\u5206\u6bb5\u60c5\u51b5\u3002",
    ),
    (
        "en",
        "The readout latent is the public codec interface, while summary and "
        "memory stay inside the FLUED encoder.",
    ),
    (
        "code",
        "def segment_lengths(raw: bytes):\n"
        "    starts = [0]\n"
        "    return [b - a for a, b in zip(starts, starts[1:])]\n",
    ),
    (
        "numbers",
        "run=31 step=12000 loss=1.482 boundary=0.731 units_per_byte=0.421 "
        "date=2026-07-02",
    ),
    (
        "mixed",
        "FLUED-v3.1 \u628a byte span -> readout latent -> byte span \u4f5c\u4e3a"
        " codec \u4e3b\u7ebf\uff0cAPINameLikeThis \u548c 4096-byte block "
        "\u8981\u540c\u65f6\u53ef\u89c6\u5316\u3002",
    ),
]


def _resolve_ckpt_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _as_int(args: Dict, key: str, default: int) -> int:
    value = args.get(key, default)
    if value is None:
        return default
    return int(value)


def _as_float(args: Dict, key: str, default: float) -> float:
    value = args.get(key, default)
    if value is None:
        return default
    return float(value)


def _load_model(path: Path, device: torch.device) -> Tuple[V31LanguageCodec2M, Dict]:
    ckpt_path = _resolve_ckpt_path(path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    model = V31LanguageCodec2M(
        d_model=_as_int(ckpt_args, "d_model", 192),
        hidden=_as_int(ckpt_args, "hidden", 192),
        nhead=_as_int(ckpt_args, "nhead", 4),
        encoder_layers=_as_int(ckpt_args, "encoder_layers", 2),
        ffn_dim=_as_int(ckpt_args, "ffn_dim", 768),
        max_span=_as_int(ckpt_args, "max_span", 16),
        refine_steps=_as_int(ckpt_args, "refine_steps", 1),
        dropout=_as_float(ckpt_args, "dropout", 0.0),
        pool_mode=ckpt_args.get("pool_mode", "mean"),
    )
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, {"path": str(ckpt_path), "checkpoint": ckpt, "args": ckpt_args}


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _encode_text(text: str, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, bytes]:
    raw = text.encode("utf-8")[: max(0, seq_len)]
    if not raw:
        ids = torch.full((1, 1), PAD_ID, dtype=torch.long, device=device)
    else:
        ids = torch.tensor([b + BYTE_OFFSET for b in raw], dtype=torch.long, device=device).unsqueeze(0)
    return ids, raw


def _is_utf8_continuation_byte(byte: int) -> bool:
    return 0x80 <= byte <= 0xBF


def _byte_type(byte: int) -> str:
    if _is_utf8_continuation_byte(byte):
        return "utf8_cont"
    if 0xC0 <= byte <= 0xDF:
        return "utf8_lead2"
    if 0xE0 <= byte <= 0xEF:
        return "utf8_lead3"
    if 0xF0 <= byte <= 0xF7:
        return "utf8_lead4"
    ch = chr(byte) if byte < 128 else ""
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "ascii_alpha"
    if ch.isspace():
        return "space"
    if ch in "+-*/%=<>!&|^~@#$\\:;.,?()[]{}_'\"`":
        return "punct_or_op"
    if byte < 128:
        return "ascii_other"
    return "utf8_other"


def _visible_char(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    if ch == " ":
        return "space"
    return ch


def _char_labels(text: str, raw_len: int) -> List[str]:
    labels = [""] * raw_len
    pos = 0
    for ch in text:
        ch_bytes = ch.encode("utf-8")
        if pos >= raw_len:
            break
        end = min(raw_len, pos + len(ch_bytes))
        labels[pos] = _visible_char(ch) if end == pos + len(ch_bytes) else "partial"
        for i in range(pos + 1, end):
            labels[i] = "cont"
        pos += len(ch_bytes)
    return labels


def _starts_from_probs(probs: Sequence[float], threshold: float) -> List[bool]:
    starts = [False] * len(probs)
    if starts:
        starts[0] = True
    for i, prob in enumerate(probs):
        if i > 0 and prob >= threshold:
            starts[i] = True
    return starts


def _utf8_len_from_start_byte(byte: int) -> int:
    if byte < 0x80:
        return 1
    if 0xC2 <= byte <= 0xDF:
        return 2
    if 0xE0 <= byte <= 0xEF:
        return 3
    if 0xF0 <= byte <= 0xF4:
        return 4
    return 1


def _constrained_starts_from_probs(
    raw: bytes,
    probs: Sequence[float],
    valid: Sequence[bool],
    threshold: float,
    min_span: int,
    max_span: int,
) -> List[bool]:
    """Decode executable segment starts from boundary probabilities.

    The model supplies boundary tendency. The codec runtime still has to obey
    hard byte-span constraints: do not start on UTF-8 continuation bytes and do
    not create a segment longer than max_span.
    """

    starts = [False] * len(raw)
    span_len = 0
    min_span = max(1, int(min_span))
    max_span = max(min_span, int(max_span))
    for i, byte in enumerate(raw):
        if i >= len(valid) or not valid[i]:
            span_len = 0
            continue
        can_start = not _is_utf8_continuation_byte(byte)
        if span_len == 0:
            starts[i] = True
            span_len = 1
            continue
        forced_break = span_len + _utf8_len_from_start_byte(byte) > max_span
        model_break = i < len(probs) and probs[i] >= threshold and span_len >= min_span
        if can_start and (forced_break or model_break):
            starts[i] = True
            span_len = 1
        else:
            span_len += 1
    return starts


def _tensor_starts_to_list(starts: torch.Tensor, n: int) -> List[bool]:
    if n <= 0:
        return []
    return [bool(x) for x in starts[0, :n].detach().cpu().tolist()]


def _list_starts_to_tensor(starts: Sequence[bool], device: torch.device) -> torch.Tensor:
    if not starts:
        return torch.zeros((1, 1), dtype=torch.bool, device=device)
    return torch.tensor([list(starts)], dtype=torch.bool, device=device)


def _spans_from_starts(starts: Sequence[bool], n: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    cut_points = [i for i, is_start in enumerate(starts[:n]) if is_start]
    if not cut_points or cut_points[0] != 0:
        cut_points.insert(0, 0)
    cut_points = sorted(set(i for i in cut_points if 0 <= i < n))
    cut_points.append(n)
    spans: List[Tuple[int, int]] = []
    for i in range(len(cut_points) - 1):
        start, end = cut_points[i], cut_points[i + 1]
        if start < end:
            spans.append((start, end))
    return spans


def _spans_from_starts_and_valid(starts: Sequence[bool], valid: Sequence[bool]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    current_start: int | None = None
    for i, ok in enumerate(valid):
        if not ok:
            if current_start is not None and current_start < i:
                spans.append((current_start, i))
            current_start = None
            continue
        if current_start is None or bool(starts[i]):
            if current_start is not None and current_start < i:
                spans.append((current_start, i))
            current_start = i
    if current_start is not None and current_start < len(valid):
        spans.append((current_start, len(valid)))
    return spans


def _length_stats(spans: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    lengths = [end - start for start, end in spans]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "hist": {}}
    hist = dict(sorted(Counter(lengths).items()))
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "hist": hist,
    }


def _continuation_start_count(raw: bytes, starts: Sequence[bool]) -> int:
    return sum(
        1
        for i, is_start in enumerate(starts[: len(raw)])
        if is_start and _is_utf8_continuation_byte(raw[i])
    )


def _decode_span(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("utf-8", errors="replace")


def _hex_span(raw: bytes, start: int, end: int) -> str:
    return " ".join(f"{b:02X}" for b in raw[start:end])


def _context(raw: bytes, idx: int, radius: int) -> str:
    left = max(0, idx - radius)
    right = min(len(raw), idx + radius + 1)
    return raw[left:right].decode("utf-8", errors="replace")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_inputs(text_args: Sequence[str]) -> List[Tuple[str, str]]:
    if text_args:
        return [(f"text_{i + 1}", text) for i, text in enumerate(text_args)]
    return list(BUILTIN_SAMPLES)


def _format_hist(hist: Dict[int, int]) -> str:
    if not hist:
        return ""
    return ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))


@torch.no_grad()
def _analyze_sample(
    model: V31LanguageCodec2M,
    name: str,
    text: str,
    cfg: argparse.Namespace,
    device: torch.device,
) -> Dict:
    ids, raw = _encode_text(text, cfg.seq_len, device)
    valid = complete_utf8_edge_valid(ids, ids.ne(PAD_ID))
    n = len(raw)
    if n == 0:
        return {
            "name": name,
            "text": text,
            "raw": raw,
            "bytes": [],
            "weak_spans": [],
            "model_spans": [],
            "stats": {},
            "top_boundaries": [],
            "truncated": False,
        }

    weak_t = weak_boundary_starts(ids, valid, cfg.min_span, cfg.max_span)
    weak_starts = _tensor_starts_to_list(weak_t, n)
    valid_list = [bool(x) for x in valid[0, :n].detach().cpu().tolist()]
    weak_spans = _spans_from_starts_and_valid(weak_starts, valid_list)

    max_units = min(cfg.max_units, ids.size(1))
    weak_seg_ids, _, _, weak_seg_mask = build_segments(ids, valid, weak_t, max_units, cfg.max_span)
    _, _, weak_metrics = model(ids, valid, weak_seg_ids, weak_seg_mask)
    probs = torch.sigmoid(weak_metrics["boundary_logits"]).float().squeeze(0)[:n].cpu().tolist()

    raw_model_starts = _starts_from_probs(probs, cfg.threshold)
    model_starts = _constrained_starts_from_probs(
        raw,
        probs,
        valid_list,
        cfg.threshold,
        cfg.min_span,
        cfg.max_span,
    )
    model_spans = _spans_from_starts_and_valid(model_starts, valid_list)
    model_t = _list_starts_to_tensor(model_starts, device)
    model_seg_ids, _, model_lengths, model_seg_mask = build_segments(ids, valid, model_t, max_units, cfg.max_span)
    _, length_logits, model_metrics = model(ids, valid, model_seg_ids, model_seg_mask)

    readout_units = int(model_seg_mask.sum().item())
    pred_lengths: List[int] = []
    if readout_units:
        pred = length_logits.argmax(dim=-1).squeeze(0)[:readout_units].detach().cpu().tolist()
        pred_lengths = [int(x) + 1 for x in pred]
    actual_model_lengths = [end - start for start, end in model_spans]
    stored_lengths = model_lengths.squeeze(0)[:readout_units].detach().cpu().tolist()
    stored_lengths = [int(x) for x in stored_lengths]

    labels = _char_labels(text, n)
    byte_rows = []
    for i, byte in enumerate(raw):
        byte_rows.append(
            {
                "i": i,
                "token": byte + BYTE_OFFSET,
                "hex": f"{byte:02X}",
                "type": _byte_type(byte),
                "char": labels[i] if i < len(labels) else "",
                "weak_start": weak_starts[i],
                "raw_model_start": raw_model_starts[i],
                "model_start": model_starts[i],
                "model_p": float(probs[i]),
                "valid": valid_list[i],
            }
        )

    top_indices = sorted(range(n), key=lambda i: probs[i], reverse=True)[: min(cfg.top_k, n)]
    top_boundaries = [
        {
            "i": i,
            "p": float(probs[i]),
            "hex": f"{raw[i]:02X}",
            "type": _byte_type(raw[i]),
            "weak_start": weak_starts[i],
            "raw_model_start": raw_model_starts[i],
            "model_start": model_starts[i],
            "context": _context(raw, i, cfg.context_radius),
        }
        for i in top_indices
    ]

    weak_len_stats = _length_stats(weak_spans)
    model_len_stats = _length_stats(model_spans)
    usable_probs = probs[1:] if len(probs) > 1 else probs
    readout_norm_mean = 0.0
    if readout_units:
        readout = model_metrics["readout"].squeeze(0)[:readout_units].float()
        readout_norm_mean = float(torch.linalg.vector_norm(readout, dim=-1).mean().item())

    stats = {
        "bytes": n,
        "threshold": cfg.threshold,
        "weak_units": len(weak_spans),
        "model_units": len(model_spans),
        "readout_units_used": readout_units,
        "weak_units_per_byte": len(weak_spans) / max(1, n),
        "model_units_per_byte": len(model_spans) / max(1, n),
        "weak_utf8_continuation_starts": _continuation_start_count(raw, weak_starts),
        "model_utf8_continuation_starts": _continuation_start_count(raw, model_starts),
        "raw_model_utf8_continuation_starts": _continuation_start_count(raw, raw_model_starts),
        "invalid_edge_bytes": int(n - sum(1 for x in valid_list if x)),
        "weak_len_min": weak_len_stats["min"],
        "weak_len_max": weak_len_stats["max"],
        "weak_len_mean": weak_len_stats["mean"],
        "model_len_min": model_len_stats["min"],
        "model_len_max": model_len_stats["max"],
        "model_len_mean": model_len_stats["mean"],
        "model_boundary_p_mean": _mean([float(x) for x in usable_probs]),
        "readout_l2_mean": readout_norm_mean,
    }

    segment_rows = []
    for unit, (start, end) in enumerate(model_spans):
        segment_rows.append(
            {
                "unit": unit,
                "start": start,
                "end": end,
                "len": end - start,
                "stored_len": stored_lengths[unit] if unit < len(stored_lengths) else None,
                "pred_len": pred_lengths[unit] if unit < len(pred_lengths) else None,
                "start_p": float(probs[start]) if start < len(probs) else 0.0,
                "text": _decode_span(raw, start, end),
                "hex": _hex_span(raw, start, min(end, start + cfg.max_span_hex_bytes)),
            }
        )

    return {
        "name": name,
        "text": text,
        "raw": raw,
        "bytes": byte_rows,
        "weak_spans": weak_spans,
        "model_spans": model_spans,
        "segment_rows": segment_rows,
        "weak_len_hist": weak_len_stats["hist"],
        "model_len_hist": model_len_stats["hist"],
        "stats": stats,
        "top_boundaries": top_boundaries,
        "truncated": readout_units < len(model_spans),
    }


def _md_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "\\r").replace("\n", "\\n")


def _fenced_text(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}"


def _render_markdown(report: Dict) -> str:
    lines: List[str] = []
    lines.append("# FLUED v3.1 Language Codec ROI")
    lines.append("")
    lines.append(f"- checkpoint: `{report['checkpoint_path']}`")
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- threshold: `{report['threshold']}`")
    lines.append("- model_start: constrained boundary decode from model probabilities, UTF-8 validity, and max_span.")
    lines.append(
        "- interface note: `readout` is the external latent interface; "
        "`summary` and `memory` are internal codec mechanisms."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| sample | bytes | model_units | units_per_byte | cont_starts | invalid_edge | len_min | len_max | len_mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["samples"]:
        s = item["stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item["name"]),
                    str(s.get("bytes", 0)),
                    str(s.get("model_units", 0)),
                    f"{float(s.get('model_units_per_byte', 0.0)):.4f}",
                    str(s.get("model_utf8_continuation_starts", 0)),
                    str(s.get("invalid_edge_bytes", 0)),
                    str(s.get("model_len_min", 0)),
                    str(s.get("model_len_max", 0)),
                    f"{float(s.get('model_len_mean', 0.0)):.2f}",
                ]
            )
            + " |"
        )
    lines.append("")

    for item in report["samples"]:
        stats = item["stats"]
        lines.append(f"## {item['name']}")
        lines.append("")
        lines.append("### Original")
        lines.append("")
        lines.append(_fenced_text(item["text"]))
        lines.append("")
        lines.append("### Boundary And Segment Stats")
        lines.append("")
        lines.append("| metric | weak | model |")
        lines.append("|---|---:|---:|")
        lines.append(f"| units | {stats.get('weak_units', 0)} | {stats.get('model_units', 0)} |")
        lines.append(
            f"| units_per_byte | {float(stats.get('weak_units_per_byte', 0.0)):.4f} | "
            f"{float(stats.get('model_units_per_byte', 0.0)):.4f} |"
        )
        lines.append(
            f"| utf8_continuation_starts | {stats.get('weak_utf8_continuation_starts', 0)} | "
            f"{stats.get('model_utf8_continuation_starts', 0)} |"
        )
        lines.append(f"| length_min | {stats.get('weak_len_min', 0)} | {stats.get('model_len_min', 0)} |")
        lines.append(f"| length_max | {stats.get('weak_len_max', 0)} | {stats.get('model_len_max', 0)} |")
        lines.append(
            f"| length_mean | {float(stats.get('weak_len_mean', 0.0)):.2f} | "
            f"{float(stats.get('model_len_mean', 0.0)):.2f} |"
        )
        lines.append("")
        lines.append(f"- weak length distribution: `{_format_hist(item['weak_len_hist'])}`")
        lines.append(f"- model length distribution: `{_format_hist(item['model_len_hist'])}`")
        lines.append(f"- model boundary sigmoid mean, excluding byte 0: `{stats.get('model_boundary_p_mean', 0.0):.4f}`")
        lines.append(f"- readout latent L2 mean over used model units: `{stats.get('readout_l2_mean', 0.0):.4f}`")
        lines.append(f"- raw threshold continuation starts before constraints: `{stats.get('raw_model_utf8_continuation_starts', 0)}`")
        lines.append(f"- invalid edge bytes masked out: `{stats.get('invalid_edge_bytes', 0)}`")
        if item["truncated"]:
            lines.append("- note: model spans exceeded `max_units`; readout tensors cover only the first used units.")
        lines.append("")

        lines.append("### Byte/Token Render")
        lines.append("")
        lines.append("| i | token | hex | type | char | valid | weak_start | model_p | raw_start | model_start |")
        lines.append("|---:|---:|---|---|---|---:|---:|---:|---:|---:|")
        for row in item["bytes"][: report["max_table_bytes"]]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["i"]),
                        str(row["token"]),
                        _md_cell(row["hex"]),
                        _md_cell(row["type"]),
                        _md_cell(row["char"]),
                        "1" if row.get("valid", True) else "0",
                        "1" if row["weak_start"] else "0",
                        f"{row['model_p']:.4f}",
                        "1" if row.get("raw_model_start", False) else "0",
                        "1" if row["model_start"] else "0",
                    ]
                )
                + " |"
            )
        if len(item["bytes"]) > report["max_table_bytes"]:
            lines.append("")
            lines.append(f"_byte table truncated at {report['max_table_bytes']} rows._")
        lines.append("")

        lines.append("### Top Model Boundary ROI")
        lines.append("")
        lines.append("| i | p | hex | type | weak_start | raw_start | model_start | context |")
        lines.append("|---:|---:|---|---|---:|---:|---:|---|")
        for row in item["top_boundaries"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["i"]),
                        f"{row['p']:.4f}",
                        _md_cell(row["hex"]),
                        _md_cell(row["type"]),
                        "1" if row["weak_start"] else "0",
                        "1" if row.get("raw_model_start", False) else "0",
                        "1" if row["model_start"] else "0",
                        _md_cell(row["context"]),
                    ]
                )
                + " |"
            )
        lines.append("")

        lines.append("### Segment Spans")
        lines.append("")
        lines.append("| unit | start | end | len | stored_len | pred_len | start_p | text | hex_prefix |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in item["segment_rows"][: report["max_spans"]]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["unit"]),
                        str(row["start"]),
                        str(row["end"]),
                        str(row["len"]),
                        _md_cell(row["stored_len"]),
                        _md_cell(row["pred_len"]),
                        f"{row['start_p']:.4f}",
                        _md_cell(row["text"]),
                        _md_cell(row["hex"]),
                    ]
                )
                + " |"
            )
        if len(item["segment_rows"]) > report["max_spans"]:
            lines.append("")
            lines.append(f"_segment table truncated at {report['max_spans']} rows._")
        lines.append("")
    return "\n".join(lines)


def _html_cell(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _render_html(report: Dict) -> str:
    parts: List[str] = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.45;color:#111;background:#fafafa}",
        "section{background:white;border:1px solid #ddd;border-radius:6px;padding:16px;margin:16px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}",
        "td,th{border:1px solid #ddd;padding:4px 6px;vertical-align:top}",
        "pre{white-space:pre-wrap;background:#f3f3f3;border:1px solid #ddd;padding:10px}",
        ".muted{color:#555}",
        "</style>",
        "<h1>FLUED v3.1 Language Codec ROI</h1>",
        "<ul>",
        f"<li>checkpoint: <code>{_html_cell(report['checkpoint_path'])}</code></li>",
        f"<li>device: <code>{_html_cell(report['device'])}</code></li>",
        f"<li>threshold: <code>{_html_cell(report['threshold'])}</code></li>",
        "<li>model_start: constrained boundary decode from model probabilities, UTF-8 validity, and max_span.</li>",
        "<li>interface note: <code>readout</code> is the external latent interface; "
        "<code>summary</code> and <code>memory</code> are internal codec mechanisms.</li>",
        "</ul>",
        "<h2>Summary</h2>",
        "<table><tr><th>sample</th><th>bytes</th><th>model_units</th><th>units_per_byte</th>"
        "<th>cont_starts</th><th>invalid_edge</th><th>len_min</th><th>len_max</th><th>len_mean</th></tr>",
    ]
    for item in report["samples"]:
        s = item["stats"]
        parts.append(
            "<tr>"
            f"<td>{_html_cell(item['name'])}</td>"
            f"<td>{_html_cell(s.get('bytes', 0))}</td>"
            f"<td>{_html_cell(s.get('model_units', 0))}</td>"
            f"<td>{float(s.get('model_units_per_byte', 0.0)):.4f}</td>"
            f"<td>{_html_cell(s.get('model_utf8_continuation_starts', 0))}</td>"
            f"<td>{_html_cell(s.get('invalid_edge_bytes', 0))}</td>"
            f"<td>{_html_cell(s.get('model_len_min', 0))}</td>"
            f"<td>{_html_cell(s.get('model_len_max', 0))}</td>"
            f"<td>{float(s.get('model_len_mean', 0.0)):.2f}</td>"
            "</tr>"
        )
    parts.append("</table>")

    for item in report["samples"]:
        stats = item["stats"]
        parts.extend(
            [
                f"<section><h2>{_html_cell(item['name'])}</h2>",
                "<h3>Original</h3>",
                f"<pre>{_html_cell(item['text'])}</pre>",
                "<h3>Boundary And Segment Stats</h3>",
                "<table><tr><th>metric</th><th>weak</th><th>model</th></tr>",
                f"<tr><td>units</td><td>{stats.get('weak_units', 0)}</td><td>{stats.get('model_units', 0)}</td></tr>",
                f"<tr><td>units_per_byte</td><td>{float(stats.get('weak_units_per_byte', 0.0)):.4f}</td>"
                f"<td>{float(stats.get('model_units_per_byte', 0.0)):.4f}</td></tr>",
                f"<tr><td>utf8_continuation_starts</td><td>{stats.get('weak_utf8_continuation_starts', 0)}</td>"
                f"<td>{stats.get('model_utf8_continuation_starts', 0)}</td></tr>",
                f"<tr><td>length_min</td><td>{stats.get('weak_len_min', 0)}</td><td>{stats.get('model_len_min', 0)}</td></tr>",
                f"<tr><td>length_max</td><td>{stats.get('weak_len_max', 0)}</td><td>{stats.get('model_len_max', 0)}</td></tr>",
                f"<tr><td>length_mean</td><td>{float(stats.get('weak_len_mean', 0.0)):.2f}</td>"
                f"<td>{float(stats.get('model_len_mean', 0.0)):.2f}</td></tr>",
                "</table>",
                f"<p class='muted'>weak length distribution: <code>{_html_cell(_format_hist(item['weak_len_hist']))}</code></p>",
                f"<p class='muted'>model length distribution: <code>{_html_cell(_format_hist(item['model_len_hist']))}</code></p>",
                f"<p class='muted'>model boundary sigmoid mean, excluding byte 0: "
                f"<code>{stats.get('model_boundary_p_mean', 0.0):.4f}</code>; readout L2 mean: "
                f"<code>{stats.get('readout_l2_mean', 0.0):.4f}</code></p>",
                f"<p class='muted'>raw threshold continuation starts before constraints: "
                f"<code>{stats.get('raw_model_utf8_continuation_starts', 0)}</code>; invalid edge bytes masked out: "
                f"<code>{stats.get('invalid_edge_bytes', 0)}</code></p>",
            ]
        )
        if item["truncated"]:
            parts.append("<p class='muted'>model spans exceeded <code>max_units</code>; readout tensors cover only the first used units.</p>")

        parts.append("<h3>Byte/Token Render</h3>")
        parts.append(
            "<table><tr><th>i</th><th>token</th><th>hex</th><th>type</th><th>char</th>"
            "<th>valid</th><th>weak_start</th><th>model_p</th><th>raw_start</th><th>model_start</th></tr>"
        )
        for row in item["bytes"][: report["max_table_bytes"]]:
            parts.append(
                "<tr>"
                f"<td>{row['i']}</td><td>{row['token']}</td><td>{_html_cell(row['hex'])}</td>"
                f"<td>{_html_cell(row['type'])}</td><td>{_html_cell(row['char'])}</td>"
                f"<td>{1 if row.get('valid', True) else 0}</td>"
                f"<td>{1 if row['weak_start'] else 0}</td><td>{row['model_p']:.4f}</td>"
                f"<td>{1 if row.get('raw_model_start', False) else 0}</td>"
                f"<td>{1 if row['model_start'] else 0}</td>"
                "</tr>"
            )
        parts.append("</table>")
        if len(item["bytes"]) > report["max_table_bytes"]:
            parts.append(f"<p class='muted'>byte table truncated at {report['max_table_bytes']} rows.</p>")

        parts.append("<h3>Top Model Boundary ROI</h3>")
        parts.append(
            "<table><tr><th>i</th><th>p</th><th>hex</th><th>type</th><th>weak_start</th>"
            "<th>raw_start</th><th>model_start</th><th>context</th></tr>"
        )
        for row in item["top_boundaries"]:
            parts.append(
                "<tr>"
                f"<td>{row['i']}</td><td>{row['p']:.4f}</td><td>{_html_cell(row['hex'])}</td>"
                f"<td>{_html_cell(row['type'])}</td><td>{1 if row['weak_start'] else 0}</td>"
                f"<td>{1 if row.get('raw_model_start', False) else 0}</td>"
                f"<td>{1 if row['model_start'] else 0}</td><td>{_html_cell(row['context'])}</td>"
                "</tr>"
            )
        parts.append("</table>")

        parts.append("<h3>Segment Spans</h3>")
        parts.append(
            "<table><tr><th>unit</th><th>start</th><th>end</th><th>len</th><th>stored_len</th>"
            "<th>pred_len</th><th>start_p</th><th>text</th><th>hex_prefix</th></tr>"
        )
        for row in item["segment_rows"][: report["max_spans"]]:
            parts.append(
                "<tr>"
                f"<td>{row['unit']}</td><td>{row['start']}</td><td>{row['end']}</td>"
                f"<td>{row['len']}</td><td>{_html_cell(row['stored_len'])}</td>"
                f"<td>{_html_cell(row['pred_len'])}</td><td>{row['start_p']:.4f}</td>"
                f"<td>{_html_cell(row['text'])}</td><td>{_html_cell(row['hex'])}</td>"
                "</tr>"
            )
        parts.append("</table>")
        if len(item["segment_rows"]) > report["max_spans"]:
            parts.append(f"<p class='muted'>segment table truncated at {report['max_spans']} rows.</p>")
        parts.append("</section>")
    return "\n".join(parts)


def _report_format(path: Path, requested: str) -> str:
    if requested in {"markdown", "md", "html"}:
        return "markdown" if requested == "md" else requested
    return "html" if path.suffix.lower() in {".html", ".htm"} else "markdown"


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    device = _select_device(args.device)
    model, loaded = _load_model(Path(args.ckpt), device)
    ckpt_args = loaded["args"]
    args.seq_len = args.seq_len if args.seq_len is not None else _as_int(ckpt_args, "seq_len", 128)
    args.min_span = args.min_span if args.min_span is not None else _as_int(ckpt_args, "min_span", 2)
    args.max_span = args.max_span if args.max_span is not None else _as_int(ckpt_args, "max_span", 16)
    args.max_units = args.max_units if args.max_units is not None else _as_int(ckpt_args, "max_units", args.seq_len)

    samples = [
        _analyze_sample(model, name, text, args, device)
        for name, text in _sample_inputs(args.text)
    ]
    report = {
        "checkpoint_path": loaded["path"],
        "checkpoint_step": loaded["checkpoint"].get("step"),
        "checkpoint_args": ckpt_args,
        "device": str(device),
        "threshold": args.threshold,
        "seq_len": args.seq_len,
        "min_span": args.min_span,
        "max_span": args.max_span,
        "max_units": args.max_units,
        "max_table_bytes": args.max_table_bytes,
        "max_spans": args.max_spans,
        "samples": samples,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _report_format(out_path, args.format)
    if fmt == "html":
        out_path.write_text(_render_html(report), encoding="utf-8")
    else:
        out_path.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"out_path": str(out_path), "format": fmt, "samples": len(samples)}, ensure_ascii=False))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FLUED-v3.1 language-codec ROI and segmentation")
    parser.add_argument("--ckpt", required=True, help="Path to latest.pt or to a run directory containing latest.pt")
    parser.add_argument("--out-path", required=True, help="Markdown or HTML report path")
    parser.add_argument("--text", action="append", default=[], help="Input sample text. Repeat --text for multiple samples.")
    parser.add_argument("--format", choices=["auto", "markdown", "md", "html"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--min-span", type=int, default=None)
    parser.add_argument("--max-span", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-table-bytes", type=int, default=256)
    parser.add_argument("--max-spans", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--context-radius", type=int, default=18)
    parser.add_argument("--max-span-hex-bytes", type=int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
