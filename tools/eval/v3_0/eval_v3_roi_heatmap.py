"""Multi-scenario FLUED boundary ROI and heatmap evaluation."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID
from flued.model import FLUEDAutoencoder


SAMPLES: Dict[str, str] = {
    "zh_long": "昨天晚上，研究团队重新检查了实验日志，发现模型在处理专有名词和长距离指代时更倾向于保留边界，而在常见虚词附近会自动合并。",
    "en_long": "The compression module should preserve rare entity names while aggressively merging predictable function words and repeated local patterns.",
    "code": "def normalize_rate(values):\n    total = sum(values)\n    return [v / total for v in values if total > 0]\n",
    "math_digits": "The final score was 94.7%, p < 0.001, with loss=1.482 and budget_lambda=0.037 after 40,000 steps.",
    "mixed": "FLUED-v3 需要同时处理 ByteFlow-style coding rate、中文语义段、APINameLikeThis 和 2026-06-26 这样的结构。",
    "template": "订单编号 A1029 已确认。订单编号 A1030 已确认。订单编号 A1031 已确认。订单编号 A1032 已确认。",
    "entities": "OpenAI, NVIDIA, AutoDL, ByteFlow, SOMBRERO, FLEXITOKENS, and Fast BLT all imply different compression tradeoffs.",
}


def _infer_model_args(ckpt: Dict) -> Dict:
    state = ckpt.get("model", ckpt)
    emb = state["embedding.weight"]
    d_model = emb.shape[1]
    layer_ids = set()
    for key in state:
        if key.startswith("blocks."):
            try:
                layer_ids.add(int(key.split(".")[1]))
            except ValueError:
                pass
    num_layers = max(layer_ids) + 1 if layer_ids else 4
    ff_key = "blocks.0.ff_gate.weight"
    dim_ff = state[ff_key].shape[0] if ff_key in state else d_model * 4
    saved_args = ckpt.get("args", {})
    return {
        "d_model": int(saved_args.get("d_model", d_model)),
        "nhead": int(saved_args.get("nhead", max(1, d_model // 64))),
        "dim_feedforward": int(saved_args.get("dim_feedforward", dim_ff)),
        "swiglu_hidden": saved_args.get("swiglu_hidden", dim_ff),
        "num_layers": int(saved_args.get("num_layers", num_layers)),
        "max_seq_len": int(saved_args.get("max_seq_len", 512)),
        "assignment_window": int(saved_args.get("assignment_window", 128)),
        "target_compression": float(saved_args.get("target_compression", 0.3)),
        "compression_weight": float(saved_args.get("compression_weight", 0.1)),
        "lambda_utf8": float(saved_args.get("lambda_utf8", 0.05)),
        "lambda_cjk": float(saved_args.get("lambda_cjk", 0.05)),
        "lambda_type": float(saved_args.get("lambda_type", 0.02)),
    }


def _load_model(path: Path, device: torch.device) -> FLUEDAutoencoder:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model = FLUEDAutoencoder(**_infer_model_args(ckpt), dropout=0.0)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _encode_text(text: str, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, bytes]:
    raw = text.encode("utf-8")[:seq_len]
    ids = torch.tensor([b + 1 for b in raw], dtype=torch.long, device=device).unsqueeze(0)
    return ids, raw


def _byte_type(byte: int) -> str:
    if 0x80 <= byte <= 0xBF:
        return "utf8_cont"
    if 0xE4 <= byte <= 0xE9:
        return "cjk_lead"
    ch = chr(byte) if byte < 128 else ""
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    if ch in "+-*/%=<>!&|^~@#$\\:;.,?()[]{}_'\"`":
        return "op"
    if ch.isalpha():
        return "ascii_alpha"
    if byte < 128:
        return "ascii_other"
    return "utf8_other"


def _hard_spans(bp: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    cuts = [0]
    for i, p in enumerate(bp):
        if i > 0 and p > threshold:
            cuts.append(i)
    cuts.append(len(bp))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i] < cuts[i + 1]]


def _roi(raw: bytes, bp: np.ndarray, top_k: int, radius: int) -> List[Dict]:
    candidates = np.argsort(-bp)[: min(top_k, len(bp))]
    items = []
    for idx in sorted(int(i) for i in candidates):
        left = max(0, idx - radius)
        right = min(len(raw), idx + radius + 1)
        items.append({
            "byte_index": idx,
            "bp": float(bp[idx]),
            "byte": int(raw[idx]) if idx < len(raw) else None,
            "byte_type": _byte_type(raw[idx]) if idx < len(raw) else "",
            "context": raw[left:right].decode("utf-8", errors="replace"),
        })
    return items


def _type_stats(raw: bytes, bp: np.ndarray) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for i, b in enumerate(raw[: len(bp)]):
        buckets.setdefault(_byte_type(b), []).append(float(bp[i]))
    return {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for key, vals in sorted(buckets.items())
        if vals
    }


def _render_html(name: str, text: str, raw: bytes, bp: np.ndarray, spans: List[Tuple[int, int]], roi: List[Dict], out: Path) -> None:
    chars = []
    byte_i = 0
    for ch in text:
        ch_bytes = ch.encode("utf-8")
        if byte_i >= len(bp):
            break
        p = float(bp[byte_i])
        hue = int((1.0 - p) * 210)
        color = f"hsl({hue}, 80%, 42%)"
        title = f"byte={byte_i} bp={p:.3f}"
        chars.append(f'<span class="ch" title="{html.escape(title)}" style="background:{color}">{html.escape(ch)}</span>')
        byte_i += len(ch_bytes)

    span_rows = []
    for start, end in spans[:80]:
        segment = raw[start:end].decode("utf-8", errors="replace")
        span_rows.append(
            f"<tr><td>{start}</td><td>{end}</td><td>{end-start}</td><td>{html.escape(segment)}</td></tr>"
        )

    roi_rows = []
    for item in roi:
        roi_rows.append(
            "<tr>"
            f"<td>{item['byte_index']}</td><td>{item['bp']:.3f}</td><td>{item['byte_type']}</td>"
            f"<td>{html.escape(item['context'])}</td>"
            "</tr>"
        )

    out.write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>",
            "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.5;color:#111;background:#fafafa}",
            ".panel{background:white;border:1px solid #ddd;border-radius:6px;padding:16px;margin:14px 0}",
            ".heat{font-family:Consolas,monospace;font-size:18px;line-height:2.0;word-break:break-all}",
            ".ch{padding:2px 1px;margin:0 1px;color:white;border-radius:2px}",
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}",
            "</style>",
            f"<h1>FLUED ROI Heatmap: {html.escape(name)}</h1>",
            f"<p>bytes={len(raw)} bp_mean={float(bp.mean()):.4f} bp_std={float(bp.std()):.4f} soft_m/n={float(bp.mean()):.4f}</p>",
            "<div class='panel heat'>" + "".join(chars) + "</div>",
            "<div class='panel'><h2>Top Boundary ROI</h2><table><tr><th>byte</th><th>bp</th><th>type</th><th>context</th></tr>" + "".join(roi_rows) + "</table></div>",
            "<div class='panel'><h2>Hard Spans at threshold 0.5</h2><table><tr><th>start</th><th>end</th><th>len</th><th>text</th></tr>" + "".join(span_rows) + "</table></div>",
        ]),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate(model: FLUEDAutoencoder, out_dir: Path, seq_len: int, threshold: float, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, text in SAMPLES.items():
        ids, raw = _encode_text(text, seq_len, device)
        pad = ids == PAD_ID
        _, metrics = model.encode(ids, pad, skip_hard=True, boundary_src=ids)
        bp = metrics["boundary_probs"].float().squeeze(0).cpu().numpy()[: len(raw)]
        spans = _hard_spans(bp, threshold)
        roi = _roi(raw, bp, top_k=16, radius=16)
        item = {
            "sample": name,
            "text": text,
            "bytes": len(raw),
            "bp_mean": float(bp.mean()),
            "bp_std": float(bp.std()),
            "bp_p10": float(np.quantile(bp, 0.10)),
            "bp_p50": float(np.quantile(bp, 0.50)),
            "bp_p90": float(np.quantile(bp, 0.90)),
            "hard_m_over_n": float(len(spans) / max(1, len(raw))),
            "num_spans": len(spans),
            "type_stats": _type_stats(raw, bp),
            "roi": roi,
            "spans": [{"start": s, "end": e, "text": raw[s:e].decode("utf-8", errors="replace")} for s, e in spans],
        }
        summary.append(item)
        _render_html(name, text, raw, bp, spans, roi, out_dir / f"{name}.html")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(summary)} scenarios to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FLUED boundary ROI heatmaps on fixed scenarios")
    parser.add_argument("--flued-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.flued_ckpt), device)
    evaluate(model, Path(args.out_dir), args.seq_len, args.threshold, device)


if __name__ == "__main__":
    main()
