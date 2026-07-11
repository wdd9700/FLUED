"""Pairwise commit-ROI stability diagnostics for FLUED-v3 checkpoints.

This is an eval-only tool. It compares commit probability regions before and
after light text perturbations for the minimal v3 commit-controller prototype.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID
from tools.analysis.v3_0.train_v3_commit_controller_small import V3CommitControllerSmall


DEFAULT_RAW_CKPT = Path(r"archive\v3_commit_controller_20260629\scale6m_seq128_raw_active_memory_15k\latest.pt")
DEFAULT_GATED_CKPT = Path(r"archive\v3_commit_controller_20260629\scale6m_seq128_gated_active_memory_15k\latest.pt")
DEFAULT_OUT_DIR = Path(r"archive\v3_diagnostics_20260629\stability")


BASE_SAMPLES: Dict[str, str] = {
    "zh_entities": "昨天晚上，研究团队重新检查了FLUED-v3实验日志，发现模型在处理专有名词和长距离指代时更倾向于保留边界。",
    "en_entities": "The compression module should preserve rare entity names while aggressively merging predictable function words.",
    "code": "def normalize_rate(values):\n    total = sum(values)\n    return [v / total for v in values if total > 0]\n",
    "math_digits": "The final score was 94.7%, p < 0.001, with loss=1.482 and budget_lambda=0.037 after 40000 steps.",
    "mixed": "FLUED-v3 需要同时处理 ByteFlow-style coding rate、中文语义段、APINameLikeThis 和 2026-06-26 这样的结构。",
    "template": "订单编号 A1029 已确认。订单编号 A1030 已确认。订单编号 A1031 已确认。订单编号 A1032 已确认。",
}


def _perturbations(name: str, text: str) -> List[Tuple[str, str]]:
    variants = [("append_clause", text.rstrip() + " 请再次核对。")]
    if name == "zh_entities":
        variants.append(("entity_swap", text.replace("FLUED-v3", "FLUED-v3.1")))
        variants.append(("punctuation", text.replace("，发现模型", "；发现模型")))
    elif name == "en_entities":
        variants.append(("entity_swap", text.replace("rare entity names", "rare entity IDs")))
        variants.append(("spacing", text.replace("while aggressively", "while  aggressively")))
    elif name == "code":
        variants.append(("identifier", text.replace("normalize_rate", "normalize_rates")))
        variants.append(("operator", text.replace("total > 0", "total >= 0")))
    elif name == "math_digits":
        variants.append(("number", text.replace("94.7%", "94.8%").replace("1.482", "1.481")))
        variants.append(("punctuation", text.replace("p < 0.001", "p<=0.001")))
    elif name == "mixed":
        variants.append(("date", text.replace("2026-06-26", "2026-06-29")))
        variants.append(("entity_case", text.replace("APINameLikeThis", "APINameLikeThat")))
    elif name == "template":
        variants.append(("digit_sequence", text.replace("A1031", "A1041")))
        variants.append(("template_word", text.replace("已确认", "已复核", 1)))
    return variants


def _load(path: Path, device: torch.device) -> V3CommitControllerSmall:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = V3CommitControllerSmall(
        d_model=int(args.get("d_model", 192)),
        hidden=int(args.get("hidden", 192)),
        controller_hidden=int(args.get("controller_hidden", 256)),
        decoder_input=str(args.get("decoder_input", "hidden_active_memory")),
        controller_memory_mode=str(args.get("controller_memory_mode", "raw")),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _commit_probs(model: V3CommitControllerSmall, text: str, seq_len: int, device: torch.device) -> Tuple[bytes, np.ndarray]:
    raw = text.encode("utf-8")[:seq_len]
    ids = torch.tensor([b + 1 for b in raw], dtype=torch.long, device=device).unsqueeze(0)
    valid = ids != PAD_ID
    _, metrics = model(ids, valid)
    commit = metrics["commit_probs"].float().squeeze(0).detach().cpu().numpy()[: len(raw)]
    return raw, commit


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


def _top_indices(p: np.ndarray, k: int, limit: int | None = None) -> List[int]:
    n = len(p) if limit is None else min(len(p), limit)
    if n <= 0:
        return []
    usable = p[:n].copy()
    if n:
        usable[0] = -np.inf
    kk = min(k, n)
    return [int(i) for i in np.argsort(-usable)[:kk] if np.isfinite(usable[int(i)])]


def _hard_set(p: np.ndarray, threshold: float, limit: int) -> set[int]:
    if limit <= 1:
        return set()
    return {int(i) for i, val in enumerate(p[:limit]) if i > 0 and float(val) > threshold}


def _window_set(indices: Iterable[int], radius: int, limit: int) -> set[int]:
    out: set[int] = set()
    for idx in indices:
        for j in range(max(0, idx - radius), min(limit, idx + radius + 1)):
            out.add(j)
    return out


def _tolerant_overlap(a: Sequence[int], b: Sequence[int], radius: int) -> float:
    if not a:
        return 0.0
    b_arr = np.asarray(list(b), dtype=np.int64)
    if b_arr.size == 0:
        return 0.0
    hits = 0
    for idx in a:
        if np.min(np.abs(b_arr - int(idx))) <= radius:
            hits += 1
    return hits / len(a)


def _mean_nearest_distance(a: Sequence[int], b: Sequence[int], cap: int) -> float:
    if not a or not b:
        return float(cap)
    b_arr = np.asarray(list(b), dtype=np.int64)
    vals = [min(cap, int(np.min(np.abs(b_arr - int(idx))))) for idx in a]
    return float(np.mean(vals))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or len(b) < 4:
        return float("nan")
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _roi(raw: bytes, p: np.ndarray, indices: Sequence[int], radius: int) -> List[Dict]:
    rows = []
    for idx in indices:
        left = max(0, int(idx) - radius)
        right = min(len(raw), int(idx) + radius + 1)
        byte = int(raw[int(idx)]) if int(idx) < len(raw) else -1
        rows.append({
            "byte_index": int(idx),
            "commit": float(p[int(idx)]),
            "byte": byte,
            "byte_type": _byte_type(byte) if byte >= 0 else "",
            "context": raw[left:right].decode("utf-8", errors="replace"),
        })
    return rows


def _stability_label(row: Dict) -> str:
    if row["topk_tolerant_overlap"] >= 0.75 and row["mean_abs_diff"] <= 0.08 and row["hard_jaccard"] >= 0.60:
        return "stable"
    if row["topk_tolerant_overlap"] >= 0.50 and row["mean_abs_diff"] <= 0.14:
        return "mixed"
    return "unstable"


def _compare(
    model_name: str,
    sample: str,
    perturbation: str,
    base_text: str,
    pert_text: str,
    base_raw: bytes,
    base_p: np.ndarray,
    pert_raw: bytes,
    pert_p: np.ndarray,
    threshold: float,
    top_k: int,
    radius: int,
) -> Dict:
    common = min(len(base_p), len(pert_p), len(base_raw), len(pert_raw))
    base_common = base_p[:common]
    pert_common = pert_p[:common]
    base_top = _top_indices(base_p, top_k, common)
    pert_top = _top_indices(pert_p, top_k, common)
    base_hard = _hard_set(base_p, threshold, common)
    pert_hard = _hard_set(pert_p, threshold, common)
    hard_union = base_hard | pert_hard
    base_win = _window_set(base_top, radius, common)
    pert_win = _window_set(pert_top, radius, common)
    roi_union = base_win | pert_win
    delta = np.abs(base_common - pert_common) if common else np.array([], dtype=np.float32)
    row = {
        "model": model_name,
        "sample": sample,
        "perturbation": perturbation,
        "base_bytes": len(base_raw),
        "perturbed_bytes": len(pert_raw),
        "common_bytes": common,
        "base_mean": float(np.mean(base_p)) if len(base_p) else float("nan"),
        "perturbed_mean": float(np.mean(pert_p)) if len(pert_p) else float("nan"),
        "mean_abs_diff": float(np.mean(delta)) if len(delta) else float("nan"),
        "p90_abs_diff": float(np.quantile(delta, 0.90)) if len(delta) else float("nan"),
        "max_abs_diff": float(np.max(delta)) if len(delta) else float("nan"),
        "corr": _safe_corr(base_common, pert_common),
        "topk_exact_jaccard": float(len(set(base_top) & set(pert_top)) / max(1, len(set(base_top) | set(pert_top)))),
        "topk_tolerant_overlap": _tolerant_overlap(base_top, pert_top, radius),
        "mean_top_nearest_distance": _mean_nearest_distance(base_top, pert_top, cap=max(1, common)),
        "hard_base_count": len(base_hard),
        "hard_perturbed_count": len(pert_hard),
        "hard_jaccard": float(len(base_hard & pert_hard) / max(1, len(hard_union))),
        "roi_mean_abs_diff": float(np.mean(delta[sorted(roi_union)])) if roi_union and len(delta) else float("nan"),
        "base_top_roi": _roi(base_raw, base_p, base_top, radius),
        "perturbed_top_roi": _roi(pert_raw, pert_p, pert_top, radius),
        "base_text": base_text,
        "perturbed_text": pert_text,
    }
    row["stability"] = _stability_label(row)
    return row


def _heat_html(text: str, p: np.ndarray, raw: bytes) -> str:
    cells = []
    byte_i = 0
    for ch in text:
        ch_len = len(ch.encode("utf-8"))
        if byte_i >= len(p):
            break
        val = float(p[byte_i])
        hue = int((1.0 - val) * 210)
        title = f"byte={byte_i} commit={val:.3f}"
        cells.append(f'<span class="ch" title="{html.escape(title)}" style="background:hsl({hue},78%,42%)">{html.escape(ch)}</span>')
        byte_i += ch_len
    if not cells:
        preview = html.escape(raw.decode("utf-8", errors="replace"))
        return f"<pre>{preview}</pre>"
    return "".join(cells)


def _render_pair_html(row: Dict, base_p: np.ndarray, pert_p: np.ndarray, base_raw: bytes, pert_raw: bytes, out: Path) -> None:
    roi_rows = []
    for side, items in (("base", row["base_top_roi"]), ("perturbed", row["perturbed_top_roi"])):
        for item in items:
            roi_rows.append(
                "<tr>"
                f"<td>{side}</td><td>{item['byte_index']}</td><td>{item['commit']:.3f}</td>"
                f"<td>{item['byte_type']}</td><td>{html.escape(item['context'])}</td>"
                "</tr>"
            )
    out.write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>",
            "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fbfbfb;color:#111;line-height:1.45}",
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#fff;border:1px solid #d8d8d8;border-radius:6px;padding:14px}",
            ".heat{font-family:Consolas,monospace;font-size:17px;line-height:2.0;word-break:break-all}.ch{padding:2px 1px;margin:0 1px;color:white;border-radius:2px}",
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#eee}",
            "</style>",
            f"<h1>{html.escape(row['model'])} / {html.escape(row['sample'])} / {html.escape(row['perturbation'])}</h1>",
            f"<p><span class='badge'>{html.escape(row['stability'])}</span> common={row['common_bytes']} mean_abs_diff={row['mean_abs_diff']:.4f} top_overlap={row['topk_tolerant_overlap']:.3f} hard_jaccard={row['hard_jaccard']:.3f}</p>",
            "<div class='grid'>",
            "<div class='panel'><h2>Base</h2><div class='heat'>" + _heat_html(row["base_text"], base_p, base_raw) + "</div></div>",
            "<div class='panel'><h2>Perturbed</h2><div class='heat'>" + _heat_html(row["perturbed_text"], pert_p, pert_raw) + "</div></div>",
            "</div>",
            "<div class='panel'><h2>Top Commit ROI</h2><table><tr><th>side</th><th>byte</th><th>commit</th><th>type</th><th>context</th></tr>" + "".join(roi_rows) + "</table></div>",
        ]),
        encoding="utf-8",
    )


def _render_index(rows: List[Dict], out: Path) -> None:
    body = []
    for row in sorted(rows, key=lambda r: (r["model"], r["stability"], r["sample"], r["perturbation"])):
        link = html.escape(row["html"])
        body.append(
            "<tr>"
            f"<td>{html.escape(row['model'])}</td><td>{html.escape(row['sample'])}</td><td>{html.escape(row['perturbation'])}</td>"
            f"<td>{html.escape(row['stability'])}</td><td>{row['mean_abs_diff']:.4f}</td><td>{row['topk_tolerant_overlap']:.3f}</td>"
            f"<td>{row['hard_jaccard']:.3f}</td><td>{row['mean_top_nearest_distance']:.2f}</td><td><a href='{link}'>view</a></td>"
            "</tr>"
        )
    out.write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fafafa;color:#111}table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}td,th{border:1px solid #ddd;padding:6px;text-align:left}th{background:#eee}</style>",
            "<h1>FLUED-v3 Commit ROI Stability</h1>",
            "<table><tr><th>model</th><th>sample</th><th>perturbation</th><th>stability</th><th>mean abs diff</th><th>top overlap</th><th>hard jaccard</th><th>mean top shift</th><th>html</th></tr>",
            "".join(body),
            "</table>",
        ]),
        encoding="utf-8",
    )


def _csv_rows(rows: List[Dict]) -> List[Dict]:
    keys = [
        "model", "sample", "perturbation", "stability", "base_bytes", "perturbed_bytes", "common_bytes",
        "base_mean", "perturbed_mean", "mean_abs_diff", "p90_abs_diff", "max_abs_diff", "corr",
        "topk_exact_jaccard", "topk_tolerant_overlap", "mean_top_nearest_distance",
        "hard_base_count", "hard_perturbed_count", "hard_jaccard", "roi_mean_abs_diff", "html",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _aggregate(rows: List[Dict], ckpts: Dict[str, str], args: argparse.Namespace) -> Dict:
    by_model: Dict[str, Dict] = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        by_model[model] = {
            "n": len(subset),
            "stable": sum(1 for row in subset if row["stability"] == "stable"),
            "mixed": sum(1 for row in subset if row["stability"] == "mixed"),
            "unstable": sum(1 for row in subset if row["stability"] == "unstable"),
            "mean_abs_diff": float(np.mean([row["mean_abs_diff"] for row in subset])),
            "topk_tolerant_overlap": float(np.mean([row["topk_tolerant_overlap"] for row in subset])),
            "hard_jaccard": float(np.mean([row["hard_jaccard"] for row in subset])),
            "most_unstable": [
                {
                    "sample": row["sample"],
                    "perturbation": row["perturbation"],
                    "stability": row["stability"],
                    "mean_abs_diff": row["mean_abs_diff"],
                    "topk_tolerant_overlap": row["topk_tolerant_overlap"],
                    "hard_jaccard": row["hard_jaccard"],
                    "html": row["html"],
                }
                for row in sorted(subset, key=lambda r: (r["stability"] != "unstable", r["topk_tolerant_overlap"], -r["mean_abs_diff"]))[:6]
            ],
        }
    return {
        "diagnostic": "v3_commit_roi_perturbation_stability",
        "checkpoints": ckpts,
        "seq_len": args.seq_len,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "radius": args.radius,
        "models": by_model,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpts = {"raw": str(Path(args.raw_ckpt)), "gated": str(Path(args.gated_ckpt))}
    models = {
        "raw": _load(Path(args.raw_ckpt), device),
        "gated": _load(Path(args.gated_ckpt), device),
    }
    rows: List[Dict] = []
    for model_name, model in models.items():
        cache: Dict[Tuple[str, str], Tuple[bytes, np.ndarray]] = {}
        for sample, base_text in BASE_SAMPLES.items():
            cache[(sample, "base")] = _commit_probs(model, base_text, args.seq_len, device)
            base_raw, base_p = cache[(sample, "base")]
            for perturbation, pert_text in _perturbations(sample, base_text):
                pert_raw, pert_p = _commit_probs(model, pert_text, args.seq_len, device)
                row = _compare(
                    model_name,
                    sample,
                    perturbation,
                    base_text,
                    pert_text,
                    base_raw,
                    base_p,
                    pert_raw,
                    pert_p,
                    args.threshold,
                    args.top_k,
                    args.radius,
                )
                html_name = f"{model_name}_{sample}_{perturbation}.html"
                row["html"] = html_name
                _render_pair_html(row, base_p, pert_p, base_raw, pert_raw, out_dir / html_name)
                rows.append(row)

    summary = _aggregate(rows, ckpts, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_csv_rows(rows)[0].keys()))
        writer.writeheader()
        writer.writerows(_csv_rows(rows))
    _render_index(rows, out_dir / "index.html")
    print(f"wrote {len(rows)} pairwise comparisons to {out_dir}")
    for model_name, item in summary["models"].items():
        print(
            f"{model_name}: stable={item['stable']} mixed={item['mixed']} unstable={item['unstable']} "
            f"mean_abs_diff={item['mean_abs_diff']:.4f} top_overlap={item['topk_tolerant_overlap']:.3f} "
            f"hard_jaccard={item['hard_jaccard']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose FLUED-v3 commit ROI stability under light text perturbations.")
    parser.add_argument("--raw-ckpt", default=str(DEFAULT_RAW_CKPT))
    parser.add_argument("--gated-ckpt", default=str(DEFAULT_GATED_CKPT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--radius", type=int, default=2, help="Byte tolerance for top ROI overlap.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
