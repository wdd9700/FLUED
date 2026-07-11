"""ROI heatmaps for the minimal FLUED-v3 commit-controller prototype."""

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
from tools.analysis.v3_0.train_v3_commit_controller_small import V3CommitControllerSmall


SAMPLES: Dict[str, str] = {
    "zh_long": "昨天晚上，研究团队重新检查了实验日志，发现模型在处理专有名词和长距离指代时更倾向于保留边界，而在常见虚词附近会自动合并。",
    "en_long": "The compression module should preserve rare entity names while aggressively merging predictable function words and repeated local patterns.",
    "code": "def normalize_rate(values):\n    total = sum(values)\n    return [v / total for v in values if total > 0]\n",
    "math_digits": "The final score was 94.7%, p < 0.001, with loss=1.482 and budget_lambda=0.037 after 40,000 steps.",
    "mixed": "FLUED-v3 需要同时处理 ByteFlow-style coding rate、中文语义段、APINameLikeThis 和 2026-06-26 这样的结构。",
    "template": "订单编号 A1029 已确认。订单编号 A1030 已确认。订单编号 A1031 已确认。订单编号 A1032 已确认。",
    "entities": "OpenAI, NVIDIA, AutoDL, ByteFlow, SOMBRERO, FLEXITOKENS, and Fast BLT all imply different compression tradeoffs.",
}


def _load(path: Path, device: torch.device) -> V3CommitControllerSmall:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = V3CommitControllerSmall(
        d_model=int(args.get("d_model", 192)),
        hidden=int(args.get("hidden", 192)),
        controller_hidden=int(args.get("controller_hidden", 256)),
        decoder_input=str(args.get("decoder_input", "hidden_active_memory")),
        controller_memory_mode=str(args.get("controller_memory_mode", "raw")),
        update_cell=str(args.get("update_cell", "gru")),
        commit_stride=int(args.get("commit_stride", 1)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


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


def _hard_spans(p: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    cuts = [0]
    for i, val in enumerate(p):
        if i > 0 and val > threshold:
            cuts.append(i)
    cuts.append(len(p))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i] < cuts[i + 1]]


def _roi(raw: bytes, p: np.ndarray, top_k: int = 16, radius: int = 16) -> List[Dict]:
    idxs = np.argsort(-p)[: min(top_k, len(p))]
    out = []
    for idx in sorted(int(i) for i in idxs):
        left = max(0, idx - radius)
        right = min(len(raw), idx + radius + 1)
        out.append({
            "byte_index": idx,
            "commit": float(p[idx]),
            "byte": int(raw[idx]),
            "byte_type": _byte_type(raw[idx]),
            "context": raw[left:right].decode("utf-8", errors="replace"),
        })
    return out


def _render_html(name: str, text: str, raw: bytes, p: np.ndarray, spans: List[Tuple[int, int]], roi: List[Dict], out: Path) -> None:
    chars = []
    byte_i = 0
    for ch in text:
        ch_bytes = ch.encode("utf-8")
        if byte_i >= len(p):
            break
        val = float(p[byte_i])
        hue = int((1.0 - val) * 210)
        color = f"hsl({hue}, 80%, 42%)"
        title = f"byte={byte_i} commit={val:.3f}"
        chars.append(f'<span class="ch" title="{html.escape(title)}" style="background:{color}">{html.escape(ch)}</span>')
        byte_i += len(ch_bytes)
    roi_rows = "".join(
        f"<tr><td>{x['byte_index']}</td><td>{x['commit']:.3f}</td><td>{x['byte_type']}</td><td>{html.escape(x['context'])}</td></tr>"
        for x in roi
    )
    span_rows = "".join(
        f"<tr><td>{s}</td><td>{e}</td><td>{e-s}</td><td>{html.escape(raw[s:e].decode('utf-8', errors='replace'))}</td></tr>"
        for s, e in spans[:80]
    )
    out.write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fafafa;color:#111}.panel{background:white;border:1px solid #ddd;border-radius:6px;padding:16px;margin:14px 0}.heat{font-family:Consolas,monospace;font-size:18px;line-height:2;word-break:break-all}.ch{padding:2px 1px;margin:0 1px;color:white;border-radius:2px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}</style>",
            f"<h1>V3 Commit ROI: {html.escape(name)}</h1>",
            f"<p>bytes={len(raw)} mean={float(p.mean()):.4f} std={float(p.std()):.4f}</p>",
            "<div class='panel heat'>" + "".join(chars) + "</div>",
            "<div class='panel'><h2>Top Commit ROI</h2><table><tr><th>byte</th><th>commit</th><th>type</th><th>context</th></tr>" + roi_rows + "</table></div>",
            "<div class='panel'><h2>Hard Spans</h2><table><tr><th>start</th><th>end</th><th>len</th><th>text</th></tr>" + span_rows + "</table></div>",
        ]),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate(model: V3CommitControllerSmall, out_dir: Path, seq_len: int, threshold: float, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, text in SAMPLES.items():
        raw = text.encode("utf-8")[:seq_len]
        ids = torch.tensor([b + 1 for b in raw], dtype=torch.long, device=device).unsqueeze(0)
        valid = ids != PAD_ID
        _, metrics = model(ids, valid)
        p = metrics["commit_probs"].float().squeeze(0).cpu().numpy()[: len(raw)]
        usable = p.copy()
        if len(usable):
            usable[0] = np.nan
        spans = _hard_spans(p, threshold)
        item = {
            "sample": name,
            "bytes": len(raw),
            "commit_mean": float(np.nanmean(usable)),
            "commit_std": float(np.nanstd(usable)),
            "commit_p10": float(np.nanquantile(usable, 0.10)),
            "commit_p50": float(np.nanquantile(usable, 0.50)),
            "commit_p90": float(np.nanquantile(usable, 0.90)),
            "hard_m_over_n": float(len(spans) / max(1, len(raw))),
            "num_spans": len(spans),
            "roi": _roi(raw, p),
            "spans": [{"start": s, "end": e, "text": raw[s:e].decode("utf-8", errors="replace")} for s, e in spans],
        }
        summary.append(item)
        _render_html(name, text, raw, p, spans, item["roi"], out_dir / f"{name}.html")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(summary)} scenarios to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v3 commit-controller ROI heatmaps")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _load(Path(args.ckpt), device)
    evaluate(model, Path(args.out_dir), args.seq_len, args.threshold, device)


if __name__ == "__main__":
    main()
