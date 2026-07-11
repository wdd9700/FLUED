"""ROI heatmaps for FLUED-v3.2 causal memory interpreter.

The heatmap colors each source byte by the per-segment loss delta between
``zero`` memory and ``full`` memory.  Positive delta means the retrieved past
memory helped that segment under the current decoder.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID, ByteReconstructionDataset  # noqa: E402
from tools.analysis.train_v32_language_codec_2m import CodecCollator, move_codec_batch  # noqa: E402
from tools.eval.eval_v32_language_codec_memory_ablation import (  # noqa: E402
    _load_model,
    _resolve_checkpoint,
    forward_with_mode,
)


@dataclass(frozen=True)
class Case:
    label: str
    text: str


DEFAULT_CASES: Tuple[Case, ...] = (
    Case(
        "entity_repeat_en",
        "Asterion-47 opened the archive. Later, Asterion-47 reused the same key inside a short report.",
    ),
    Case(
        "code_identifier",
        "GraphCacheIndex maps pages to shards. GraphCacheIndexBuilder refreshes GraphCacheIndex after compaction.",
    ),
    Case("cjk_reference", "玄曜计划启动后，研究组重复对照玄曜计划的编码结果。"),
    Case(
        "version_number",
        "Build ZX-9082 passed smoke tests. The rollback note says ZX-9082 must keep schema v17.4 unchanged.",
    ),
)


def _parse_cases(items: Sequence[str]) -> Tuple[Case, ...]:
    if not items:
        return DEFAULT_CASES
    cases: List[Case] = []
    for raw in items:
        if "::" in raw:
            label, text = raw.split("::", 1)
        else:
            label, text = f"case_{len(cases)}", raw
        label = label.strip()
        text = text.strip()
        if not label or not text:
            raise ValueError("--text must be label::text or raw text")
        cases.append(Case(label, text))
    return tuple(cases)


def _select_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        raw = "cpu"
    return torch.device(raw)


def _source_bytes(src: torch.Tensor, valid: torch.Tensor) -> bytes:
    values: List[int] = []
    for token, keep in zip(src.tolist(), valid.tolist()):
        if not keep:
            continue
        token_i = int(token)
        if BYTE_OFFSET <= token_i < MASK_ID:
            values.append(token_i - BYTE_OFFSET)
    return bytes(values)


def _byte_display(value: int) -> str:
    if value == 10:
        return "\\n"
    if value == 9:
        return "\\t"
    if 32 <= value <= 126:
        return html.escape(chr(value))
    return "·"


def _color(delta: float) -> str:
    clipped = max(-1.5, min(1.5, float(delta))) / 1.5
    if clipped >= 0:
        alpha = 0.18 + 0.62 * clipped
        return f"rgba(35, 132, 67, {alpha:.3f})"
    alpha = 0.18 + 0.62 * abs(clipped)
    return f"rgba(201, 84, 38, {alpha:.3f})"


def _loss_by_unit(
    byte_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    loss = F.cross_entropy(
        byte_logits.float().reshape(-1, byte_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(targets)
    slot_mask = targets.ne(PAD_ID)
    numer = (loss * slot_mask).sum(dim=-1)
    denom = slot_mask.sum(dim=-1).clamp(min=1)
    return numer / denom


def _unit_spans(
    src: torch.Tensor,
    valid: torch.Tensor,
    seg_ids: torch.Tensor,
    seg_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    raw = _source_bytes(src, valid)
    byte_to_unit: List[int] = []
    offset = 0
    for t in range(src.size(0)):
        if not bool(valid[t]):
            continue
        token = int(src[t].item())
        if not (BYTE_OFFSET <= token < MASK_ID):
            continue
        unit = int(seg_ids[t].item())
        if unit >= 0:
            byte_to_unit.append(unit)
        offset += 1
    spans: List[Dict[str, Any]] = []
    active_units = seg_mask.nonzero(as_tuple=False).flatten().tolist()
    for unit in active_units:
        positions = [i for i, u in enumerate(byte_to_unit) if u == int(unit)]
        if not positions:
            continue
        begin, end = min(positions), max(positions) + 1
        spans.append(
            {
                "unit": int(unit),
                "begin": begin,
                "end": end,
                "raw": raw[begin:end],
                "text": raw[begin:end].decode("utf-8", errors="replace"),
            }
        )
    return spans


@torch.no_grad()
def _analyze_case(
    model,
    case: Case,
    *,
    seq_len: int,
    stride: int,
    min_span: int,
    max_span: int,
    max_units: int,
    device: torch.device,
    amp: bool,
    generator: torch.Generator,
) -> Dict[str, Any]:
    dataset = ByteReconstructionDataset(texts=[case.text], seq_len=seq_len, stride=stride)
    batch = CodecCollator(min_span, max_span, max_units)([dataset[i] for i in range(len(dataset))])
    src, _starts, seg_ids, targets, _lengths, seg_mask = move_codec_batch(batch, device)
    valid = src.ne(PAD_ID)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        full_logits, _full_len, full_metrics = model(src, valid, seg_ids, seg_mask)
        zero_logits, _zero_len, _zero_metrics = forward_with_mode(
            model,
            src,
            valid,
            seg_ids,
            seg_mask,
            "zero",
            previous_memory=None,
            generator=generator,
        )
    full_loss = _loss_by_unit(full_logits, targets)
    zero_loss = _loss_by_unit(zero_logits, targets)
    delta = (zero_loss - full_loss).detach().float().cpu()
    boundary_p = torch.sigmoid(full_metrics["boundary_logits"]).detach().float().cpu()
    memory_norm = full_metrics["memory"].detach().float().norm(dim=-1).cpu()
    summary_norm = full_metrics["summary"].detach().float().norm(dim=-1).cpu()

    chunks: List[Dict[str, Any]] = []
    for b in range(src.size(0)):
        raw = _source_bytes(src[b].detach().cpu(), valid[b].detach().cpu())
        spans = _unit_spans(
            src[b].detach().cpu(),
            valid[b].detach().cpu(),
            seg_ids[b].detach().cpu(),
            seg_mask[b].detach().cpu(),
        )
        unit_delta = {int(u): float(delta[b, int(u)].item()) for u in seg_mask[b].nonzero(as_tuple=False).flatten().tolist()}
        byte_rows: List[Dict[str, Any]] = []
        byte_offset = 0
        for t in range(src.size(1)):
            if not bool(valid[b, t]):
                continue
            token = int(src[b, t].item())
            if not (BYTE_OFFSET <= token < MASK_ID):
                continue
            value = token - BYTE_OFFSET
            unit = int(seg_ids[b, t].item())
            byte_rows.append(
                {
                    "offset": byte_offset,
                    "value": value,
                    "unit": unit,
                    "delta": unit_delta.get(unit, 0.0),
                    "boundary": float(boundary_p[b, t].item()),
                }
            )
            byte_offset += 1
        seg_rows: List[Dict[str, Any]] = []
        for span in spans:
            unit = int(span["unit"])
            seg_rows.append(
                {
                    **span,
                    "delta": float(delta[b, unit].item()),
                    "full_loss": float(full_loss[b, unit].detach().float().cpu().item()),
                    "zero_loss": float(zero_loss[b, unit].detach().float().cpu().item()),
                    "memory_norm": float(memory_norm[b, unit].item()),
                    "summary_norm": float(summary_norm[b, unit].item()),
                }
            )
        chunks.append(
            {
                "raw_text": raw.decode("utf-8", errors="replace"),
                "bytes": byte_rows,
                "segments": seg_rows,
            }
        )
    return {"label": case.label, "text": case.text, "chunks": chunks}


def _render_case(case: Mapping[str, Any]) -> str:
    parts: List[str] = [f"<section><h2>{html.escape(str(case['label']))}</h2>"]
    parts.append(f"<p class='text'>{html.escape(str(case['text']))}</p>")
    for i, chunk in enumerate(case["chunks"]):
        parts.append(f"<h3>chunk {i}</h3>")
        chars: List[str] = []
        for row in chunk["bytes"]:
            style = _color(float(row["delta"]))
            border = "border-left:2px solid #111;" if float(row["boundary"]) >= 0.5 else ""
            title = (
                f"byte={row['offset']} hex={int(row['value']):02x} unit={row['unit']} "
                f"delta={float(row['delta']):+.4f} boundary={float(row['boundary']):.3f}"
            )
            chars.append(
                f"<span class='ch' title='{html.escape(title)}' style='background:{style};{border}'>"
                f"{_byte_display(int(row['value']))}</span>"
            )
        parts.append("<div class='heat'>" + "".join(chars) + "</div>")
        rows: List[str] = []
        for seg in sorted(chunk["segments"], key=lambda item: abs(float(item["delta"])), reverse=True):
            rows.append(
                "<tr>"
                f"<td>{int(seg['unit'])}</td>"
                f"<td>{int(seg['begin'])}:{int(seg['end'])}</td>"
                f"<td><code>{html.escape(str(seg['text']))}</code></td>"
                f"<td>{float(seg['full_loss']):.4f}</td>"
                f"<td>{float(seg['zero_loss']):.4f}</td>"
                f"<td>{float(seg['delta']):+.4f}</td>"
                f"<td>{float(seg['memory_norm']):.4f}</td>"
                f"<td>{float(seg['summary_norm']):.4f}</td>"
                "</tr>"
            )
        parts.append(
            "<table><tr><th>unit</th><th>byte span</th><th>text</th><th>full loss</th>"
            "<th>zero loss</th><th>zero-full</th><th>memory norm</th><th>summary norm</th></tr>"
            + "".join(rows)
            + "</table>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def _render_html(report: Mapping[str, Any]) -> str:
    css = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fafafa;color:#111}
section{background:white;border:1px solid #ddd;border-radius:6px;padding:16px;margin:16px 0}
.text{font-size:14px;color:#333}.heat{font-family:Consolas,monospace;font-size:18px;line-height:2.1;word-break:break-all;background:#fbfbfb;border:1px solid #ddd;padding:10px;border-radius:4px}
.ch{display:inline-block;min-width:9px;padding:1px 2px;margin:1px;color:#111;border-radius:2px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}code{white-space:pre-wrap}
.note{color:#555}
"""
    body = [
        "<!doctype html><meta charset='utf-8'>",
        f"<style>{css}</style>",
        "<h1>FLUED v3.2 ROI Heatmap</h1>",
        f"<p class='note'>checkpoint: <code>{html.escape(str(report['checkpoint']))}</code></p>",
        "<p class='note'>Color = zero-memory loss minus full-memory loss. Green means memory helped; orange means memory hurt. Left border marks boundary probability >= 0.5.</p>",
    ]
    body.extend(_render_case(case) for case in report["cases"])
    return "\n".join(body)


def run(args: argparse.Namespace) -> None:
    checkpoint = _resolve_checkpoint(Path(args.checkpoint))
    device = _select_device(args.device)
    model, _meta = _load_model(checkpoint, device)
    cases = _parse_cases(args.text)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    report = {
        "checkpoint": str(checkpoint),
        "cases": [
            _analyze_case(
                model,
                case,
                seq_len=args.seq_len,
                stride=args.stride,
                min_span=args.min_span,
                max_span=args.max_span,
                max_units=args.max_units,
                device=device,
                amp=args.amp,
                generator=generator,
            )
            for case in cases
        ],
    }
    html_text = _render_html(report)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render v3.2 codec ROI heatmaps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text", action="append", default=[], help="label::text or raw text; can be repeated")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
