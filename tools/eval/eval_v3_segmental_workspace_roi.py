"""ROI and latent-state diagnostics for FLUED-v3.1 segmental workspace."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID
from tools.analysis.train_v3_segmental_workspace_2m import SegmentalLatentWorkspace2M


SAMPLES: Dict[str, str] = {
    "zh_long": "昨天晚上，研究团队重新检查了实验日志，发现模型在处理专有名词和长距离指代时更倾向于保留边界，而在常见虚词附近会自动合并。",
    "en_long": "The compression module should preserve rare entity names while aggressively merging predictable function words and repeated local patterns.",
    "code": "def normalize_rate(values):\n    total = sum(values)\n    return [v / total for v in values if total > 0]\n",
    "math_digits": "The final score was 94.7%, p < 0.001, with loss=1.482 and budget_lambda=0.037 after 40,000 steps.",
    "mixed": "FLUED-v3 需要同时处理 ByteFlow-style coding rate、中文语义段、APINameLikeThis 和 2026-06-26 这样的结构。",
    "template": "订单编号 A1029 已确认。订单编号 A1030 已确认。订单编号 A1031 已确认。订单编号 A1032 已确认。",
    "entities": "OpenAI, NVIDIA, AutoDL, ByteFlow, SOMBRERO, FLEXITOKENS, and Fast BLT all imply different compression tradeoffs.",
}


def _load(path: Path, device: torch.device) -> Tuple[SegmentalLatentWorkspace2M, Dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = SegmentalLatentWorkspace2M(
        d_model=int(args.get("d_model", 192)),
        hidden=int(args.get("hidden", 192)),
        controller_hidden=int(args.get("controller_hidden", 256)),
        refine_steps=int(args.get("refine_steps", 4)),
        student_refine_steps=int(args.get("student_refine_steps", 1)),
        ar_correction_passes=int(args.get("ar_correction_passes", 2)),
        commit_stride=int(args.get("commit_stride", 1)),
        residual_mixer=str(args.get("residual_mixer", "attn")),
        memory_enabled=not bool(args.get("no_memory", False)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def _encode(text: str, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, bytes]:
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


def _hard_spans(p: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    cuts = [0]
    for i, val in enumerate(p):
        if i > 0 and val > threshold:
            cuts.append(i)
    cuts.append(len(p))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i] < cuts[i + 1]]


def _row_stats(prefix: str, x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "p10", "p50", "p90")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
    }


def _render_html(
    out: Path,
    sample: str,
    text: str,
    raw: bytes,
    commit: np.ndarray,
    value: np.ndarray,
    confidence: np.ndarray,
    alpha_last: np.ndarray,
    spans: List[Tuple[int, int]],
) -> None:
    chars = []
    byte_i = 0
    for ch in text:
        ch_bytes = ch.encode("utf-8")
        if byte_i >= len(commit):
            break
        c = float(commit[byte_i])
        v = float(value[byte_i])
        conf = float(confidence[byte_i])
        alpha = float(alpha_last[byte_i])
        hue = int((1.0 - c) * 210)
        light = max(28, min(78, 74 - int(max(0.0, v - 0.5) * 36)))
        title = f"byte={byte_i} commit={c:.3f} value={v:.3f} confidence={conf:.3f} alpha_last={alpha:.3f}"
        chars.append(
            f'<span class="ch" title="{html.escape(title)}" '
            f'style="background:hsl({hue},80%,{light}%);border-bottom:{max(1, int(conf * 5))}px solid #111">'
            f"{html.escape(ch)}</span>"
        )
        byte_i += len(ch_bytes)
    top = np.argsort(-commit)[: min(20, len(commit))]
    top_rows = []
    for idx in sorted(int(i) for i in top):
        left = max(0, idx - 18)
        right = min(len(raw), idx + 19)
        top_rows.append(
            "<tr>"
            f"<td>{idx}</td><td>{commit[idx]:.3f}</td><td>{value[idx]:.3f}</td>"
            f"<td>{confidence[idx]:.3f}</td><td>{alpha_last[idx]:.3f}</td>"
            f"<td>{_byte_type(raw[idx])}</td><td>{html.escape(raw[left:right].decode('utf-8', errors='replace'))}</td>"
            "</tr>"
        )
    span_rows = []
    for s, e in spans[:80]:
        span_rows.append(
            "<tr>"
            f"<td>{s}</td><td>{e}</td><td>{e-s}</td>"
            f"<td>{html.escape(raw[s:e].decode('utf-8', errors='replace'))}</td>"
            "</tr>"
        )
    out.write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>",
            "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f7;color:#111}",
            ".panel{background:white;border:1px solid #ddd;border-radius:6px;padding:16px;margin:14px 0}",
            ".heat{font-family:Consolas,monospace;font-size:18px;line-height:2.1;word-break:break-all}",
            ".ch{padding:2px 1px;margin:0 1px;color:white;border-radius:2px}",
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}",
            "</style>",
            f"<h1>FLUED v3.1 ROI: {html.escape(sample)}</h1>",
            "<p>Color follows commit probability. Darker shade indicates higher value score. Underline thickness follows confidence.</p>",
            "<div class='panel heat'>" + "".join(chars) + "</div>",
            "<div class='panel'><h2>Top Commit ROI</h2><table><tr><th>byte</th><th>commit</th><th>value</th><th>confidence</th><th>alpha_last</th><th>type</th><th>context</th></tr>" + "".join(top_rows) + "</table></div>",
            "<div class='panel'><h2>Hard Spans</h2><table><tr><th>start</th><th>end</th><th>len</th><th>text</th></tr>" + "".join(span_rows) + "</table></div>",
        ]),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate(model: SegmentalLatentWorkspace2M, out_dir: Path, seq_len: int, threshold: float, device: torch.device) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, text in SAMPLES.items():
        ids, raw = _encode(text, seq_len, device)
        valid = ids != PAD_ID
        _, metrics = model(ids, valid)
        n = len(raw)
        commit = metrics["commit_probs"].float().squeeze(0).cpu().numpy()[:n]
        value = torch.sigmoid(metrics["commit_value_logits"]).float().squeeze(0).cpu().numpy()[:n]
        confidence = torch.sigmoid(metrics["confidence_logits"]).float().squeeze(0).cpu().numpy()[:n]
        alpha_last = metrics["residual_alpha"].float().squeeze(0).cpu().numpy()[:n, -1]
        usable = np.arange(n) > 0
        spans = _hard_spans(commit, threshold)
        item = {
            "sample": name,
            "bytes": n,
            "hard_m_over_n": float(len(spans) / max(1, n)),
            "num_spans": len(spans),
            **_row_stats("commit", commit[usable]),
            **_row_stats("value", value[usable]),
            **_row_stats("confidence", confidence[usable]),
            **_row_stats("alpha_last", alpha_last[usable]),
        }
        summary.append(item)
        _render_html(out_dir / f"{name}.html", name, text, raw, commit, value, confidence, alpha_last, spans)
    result = {"summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "samples": len(summary)}, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FLUED-v3.1 segmental workspace ROI")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, _ = _load(Path(args.ckpt), device)
    evaluate(model, Path(args.out_dir), args.seq_len, args.threshold, device)


if __name__ == "__main__":
    main()
