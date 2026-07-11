"""Diagnose weak byte-shape priors in FLUED-v3 commit controllers.

This is a read-only evaluator for trained v3 commit-controller checkpoints. It
does not import or modify the training entrypoint beyond reusing the model
class, and writes JSON/CSV/HTML artifacts for comparing raw vs gated controllers.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID
from tools.analysis.train_v3_commit_controller_small import V3CommitControllerSmall


OPERATOR_BYTES = set(b"+-*/%=<>!&|^~")
PUNCT_BYTES = set(ord(c) for c in string.punctuation.encode("ascii").decode("ascii"))


SAMPLES: Dict[str, str] = {
    "code_camel_ops": "def parseHTTPStatus(apiResponseCode):\n    ok = value_count >= 10 and errorRate < 0.05\n    return ok\n",
    "mixed_entities": "FLUED-v3 compares rawActiveMemory, gatedActiveMemory, APINameLikeThis, and 2026-06-29 commit traces.",
    "zh_punct_utf8": "研究日志显示：模型在中文标点、换行，以及 UTF-8 continuation byte 附近可能产生边界捷径。\n下一行继续测试。",
    "digits_template": "订单 A1029=94.7%, A1030=95.1%, A1031=93.8%; loss=1.482, lambda=0.037, steps=15000.",
}


GOLD_SEGMENTS: Dict[str, List[Tuple[str, str]]] = {
    "code_camel_ops": [
        ("keyword", "def"),
        ("function_name", "parseHTTPStatus"),
        ("argument", "apiResponseCode"),
        ("statement", "ok = value_count >= 10 and errorRate < 0.05"),
        ("return", "return ok"),
    ],
    "mixed_entities": [
        ("entity", "FLUED-v3"),
        ("predicate", "compares"),
        ("term", "rawActiveMemory"),
        ("term", "gatedActiveMemory"),
        ("term", "APINameLikeThis"),
        ("date", "2026-06-29"),
        ("object", "commit traces"),
    ],
    "zh_punct_utf8": [
        ("topic", "研究日志显示"),
        ("claim", "模型在中文标点、换行"),
        ("claim", "以及 UTF-8 continuation byte 附近"),
        ("risk", "可能产生边界捷径"),
        ("next_sentence", "下一行继续测试"),
    ],
    "digits_template": [
        ("record", "订单 A1029=94.7%"),
        ("record", "A1030=95.1%"),
        ("record", "A1031=93.8%"),
        ("metric", "loss=1.482"),
        ("metric", "lambda=0.037"),
        ("metric", "steps=15000"),
    ],
}


@dataclass
class EvalBatch:
    commit: np.ndarray
    ce: np.ndarray
    byte_values: np.ndarray
    model_name: str


def _safe_float(x: float) -> float | None:
    if x is None or math.isnan(float(x)) or math.isinf(float(x)):
        return None
    return float(x)


def _load_model(path: Path, device: torch.device) -> Tuple[V3CommitControllerSmall, Dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = V3CommitControllerSmall(
        d_model=int(args.get("d_model", 192)),
        hidden=int(args.get("hidden", 192)),
        controller_hidden=int(args.get("controller_hidden", 256)),
        decoder_input=str(args.get("decoder_input", "active_memory")),
        controller_memory_mode=str(args.get("controller_memory_mode", "raw")),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    meta = {
        "path": str(path),
        "step": int(ckpt.get("step", 0)),
        "args": args,
        "summary": ckpt.get("summary", {}),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return model, meta


def _iter_random_chunks(path: Path, seq_len: int, num_chunks: int, seed: int) -> Iterable[bytes]:
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"empty data file: {path}")
    rng = random.Random(seed)
    if len(data) <= seq_len:
        for _ in range(num_chunks):
            yield data
        return
    max_start = len(data) - seq_len
    for _ in range(num_chunks):
        start = rng.randint(0, max_start)
        yield data[start : start + seq_len]


def _target_next(ids: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    target = torch.full_like(ids, PAD_ID)
    target[:, :-1] = ids[:, 1:]
    mask = valid.clone()
    mask[:, :-1] = valid[:, :-1] & valid[:, 1:]
    mask[:, -1] = False
    return target, mask


@torch.no_grad()
def _collect(
    model: V3CommitControllerSmall,
    name: str,
    chunks: List[bytes],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> EvalBatch:
    commit_parts: List[np.ndarray] = []
    ce_parts: List[np.ndarray] = []
    byte_parts: List[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch_raw = chunks[start : start + batch_size]
        ids_np = np.zeros((len(batch_raw), seq_len), dtype=np.int64)
        raw_np = np.full((len(batch_raw), seq_len), -1, dtype=np.int16)
        for i, raw in enumerate(batch_raw):
            trimmed = raw[:seq_len]
            raw_np[i, : len(trimmed)] = np.frombuffer(trimmed, dtype=np.uint8).astype(np.int16)
            ids_np[i, : len(trimmed)] = raw_np[i, : len(trimmed)] + 1
        ids = torch.from_numpy(ids_np).to(device)
        valid = ids != PAD_ID
        logits, metrics = model(ids, valid)
        target, target_mask = _target_next(ids, valid)
        ce = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(target)
        usable = target_mask.clone()
        usable[:, 0] = False
        commit_parts.append(metrics["commit_probs"][usable].detach().float().cpu().numpy())
        ce_parts.append(ce[usable].detach().float().cpu().numpy())
        byte_parts.append(raw_np[usable.cpu().numpy()].astype(np.int16))
    return EvalBatch(
        commit=np.concatenate(commit_parts),
        ce=np.concatenate(ce_parts),
        byte_values=np.concatenate(byte_parts),
        model_name=name,
    )


def _feature_matrix(raw: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[str]]:
    b = raw.astype(np.int16)
    prev_b = np.concatenate([np.array([-1], dtype=np.int16), b[:-1]])
    next_b = np.concatenate([b[1:], np.array([-1], dtype=np.int16)])

    is_digit = (b >= ord("0")) & (b <= ord("9"))
    prev_digit = (prev_b >= ord("0")) & (prev_b <= ord("9"))
    next_digit = (next_b >= ord("0")) & (next_b <= ord("9"))
    is_lower = (b >= ord("a")) & (b <= ord("z"))
    is_upper = (b >= ord("A")) & (b <= ord("Z"))
    prev_lower = (prev_b >= ord("a")) & (prev_b <= ord("z"))
    next_upper = (next_b >= ord("A")) & (next_b <= ord("Z"))

    punct = np.isin(b, np.fromiter(PUNCT_BYTES, dtype=np.int16))
    operator = np.isin(b, np.fromiter(OPERATOR_BYTES, dtype=np.int16))
    feats = {
        "punctuation": punct,
        "newline": (b == 10) | (b == 13),
        "space_or_tab": (b == 32) | (b == 9),
        "utf8_continuation": (b >= 0x80) & (b <= 0xBF),
        "utf8_lead": (b >= 0xC0) & (b <= 0xF7),
        "camel_boundary": (is_upper & prev_lower) | (is_lower & next_upper),
        "digit_run": is_digit & (prev_digit | next_digit),
        "single_digit": is_digit & ~(prev_digit | next_digit),
        "operator": operator,
        "punct_non_operator": punct & ~operator,
        "ascii_alpha": ((b >= ord("a")) & (b <= ord("z"))) | ((b >= ord("A")) & (b <= ord("Z"))),
        "ascii_other": (b >= 0) & (b < 128) & ~punct & ~is_digit & ~is_lower & ~is_upper & (b != 32) & (b != 9) & (b != 10) & (b != 13),
    }
    order = [
        "punctuation",
        "punct_non_operator",
        "operator",
        "newline",
        "space_or_tab",
        "utf8_continuation",
        "utf8_lead",
        "camel_boundary",
        "digit_run",
        "single_digit",
        "ascii_alpha",
        "ascii_other",
    ]
    return feats, order


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _feature_stats(batch: EvalBatch, threshold: float, top_frac: float) -> Tuple[List[Dict], Dict]:
    feats, order = _feature_matrix(batch.byte_values)
    commit = batch.commit.astype(np.float64)
    ce = batch.ce.astype(np.float64)
    top_k = max(1, int(round(commit.size * top_frac)))
    top_mask = np.zeros(commit.size, dtype=bool)
    top_mask[np.argpartition(-commit, top_k - 1)[:top_k]] = True

    rows: List[Dict] = []
    for name in order:
        mask = feats[name].astype(bool)
        inv = ~mask
        n = int(mask.sum())
        if n == 0 or inv.sum() == 0:
            mean_on = mean_off = lift = corr = hard_lift = top_enrich = float("nan")
            ce_lift = float("nan")
        else:
            mean_on = float(commit[mask].mean())
            mean_off = float(commit[inv].mean())
            lift = mean_on - mean_off
            corr = _corr(mask.astype(np.float64), commit)
            hard_lift = float((commit[mask] > threshold).mean() - (commit[inv] > threshold).mean())
            top_enrich = float((top_mask[mask].mean()) / max(top_mask.mean(), 1e-12))
            ce_lift = float(ce[mask].mean() - ce[inv].mean())
        rows.append({
            "model": batch.model_name,
            "feature": name,
            "n": n,
            "prevalence": n / max(1, commit.size),
            "commit_mean_on": _safe_float(mean_on),
            "commit_mean_off": _safe_float(mean_off),
            "commit_lift": _safe_float(lift),
            "commit_ratio": _safe_float(mean_on / mean_off) if mean_off and not math.isnan(mean_off) else None,
            "point_biserial_corr": _safe_float(corr),
            "hard_rate_lift": _safe_float(hard_lift),
            "top_commit_enrichment": _safe_float(top_enrich),
            "next_byte_ce_lift": _safe_float(ce_lift),
        })

    x = np.stack([feats[name].astype(np.float64) for name in order], axis=1)
    x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    y = commit
    ridge = 1e-4 * np.eye(x.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    pred = x @ beta
    total_var = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / max(total_var, 1e-12)
    max_abs_corr = max(abs(r["point_biserial_corr"] or 0.0) for r in rows)
    max_abs_lift = max(abs(r["commit_lift"] or 0.0) for r in rows)
    dominance = {
        "model": batch.model_name,
        "positions": int(commit.size),
        "commit_mean": float(commit.mean()),
        "commit_std": float(commit.std()),
        "commit_p10": float(np.quantile(commit, 0.10)),
        "commit_p50": float(np.quantile(commit, 0.50)),
        "commit_p90": float(np.quantile(commit, 0.90)),
        "hard_rate": float((commit > threshold).mean()),
        "weak_prior_linear_r2": float(max(0.0, min(1.0, r2))),
        "max_abs_feature_corr": float(max_abs_corr),
        "max_abs_feature_lift": float(max_abs_lift),
        "ce_corr": _safe_float(_corr(commit, ce)),
        "top_features_by_abs_corr": sorted(
            rows,
            key=lambda r: abs(r["point_biserial_corr"] or 0.0),
            reverse=True,
        )[:5],
    }
    return rows, dominance


def _byte_value_stats(batch: EvalBatch, threshold: float, top_frac: float) -> List[Dict]:
    commit = batch.commit.astype(np.float64)
    ce = batch.ce.astype(np.float64)
    raw = batch.byte_values.astype(np.int16)
    top_k = max(1, int(round(commit.size * top_frac)))
    top_mask = np.zeros(commit.size, dtype=bool)
    top_mask[np.argpartition(-commit, top_k - 1)[:top_k]] = True
    global_mean = float(commit.mean())
    global_hard = float((commit > threshold).mean())
    global_top = float(top_mask.mean())
    rows: List[Dict] = []
    for byte in range(256):
        mask = raw == byte
        n = int(mask.sum())
        if n == 0:
            continue
        mean = float(commit[mask].mean())
        hard = float((commit[mask] > threshold).mean())
        top = float(top_mask[mask].mean())
        label = _byte_label(byte)
        char_group = "other"
        if ord("a") <= byte <= ord("z"):
            char_group = "lowercase"
        elif ord("A") <= byte <= ord("Z"):
            char_group = "uppercase"
        elif ord("0") <= byte <= ord("9"):
            char_group = "digit"
        elif byte in PUNCT_BYTES:
            char_group = "punctuation"
        elif byte in (9, 10, 13, 32):
            char_group = "whitespace"
        elif 0x80 <= byte <= 0xBF:
            char_group = "utf8_continuation"
        elif 0xC0 <= byte <= 0xF7:
            char_group = "utf8_lead"
        rows.append({
            "model": batch.model_name,
            "byte": byte,
            "label": label,
            "group": char_group,
            "n": n,
            "prevalence": n / max(1, commit.size),
            "commit_mean": mean,
            "commit_lift_vs_global": mean - global_mean,
            "hard_rate": hard,
            "hard_lift_vs_global": hard - global_hard,
            "top_commit_rate": top,
            "top_commit_enrichment": top / max(global_top, 1e-12),
            "next_byte_ce_mean": float(ce[mask].mean()),
        })
    return rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "feature",
        "n",
        "prevalence",
        "commit_mean_on",
        "commit_mean_off",
        "commit_lift",
        "commit_ratio",
        "point_biserial_corr",
        "hard_rate_lift",
        "top_commit_enrichment",
        "next_byte_ce_lift",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _write_byte_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "model",
        "byte",
        "label",
        "group",
        "n",
        "prevalence",
        "commit_mean",
        "commit_lift_vs_global",
        "hard_rate",
        "hard_lift_vs_global",
        "top_commit_rate",
        "top_commit_enrichment",
        "next_byte_ce_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _color(val: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    hue = int(210 - 210 * t)
    return f"hsl({hue}, 78%, 42%)"


@torch.no_grad()
def _sample_records(
    models: Dict[str, V3CommitControllerSmall],
    seq_len: int,
    device: torch.device,
) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for sample_name, text in SAMPLES.items():
        raw = text.encode("utf-8")[:seq_len]
        ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
        ids[0, : len(raw)] = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1).to(device)
        valid = ids != PAD_ID
        out[sample_name] = {}
        for model_name, model in models.items():
            logits, metrics = model(ids, valid)
            target, target_mask = _target_next(ids, valid)
            ce = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).view_as(target)
            p = metrics["commit_probs"].squeeze(0).float().cpu().numpy()[: len(raw)]
            c = ce.squeeze(0).float().cpu().numpy()[: len(raw)]
            feats, order = _feature_matrix(np.frombuffer(raw, dtype=np.uint8).astype(np.int16))
            top_idx = np.argsort(-p)[: min(18, len(p))]
            out[sample_name][model_name] = {
                "text": text,
                "raw_hex": raw.hex(),
                "commit": p.tolist(),
                "ce": c.tolist(),
                "byte_values": list(raw),
                "features": {k: feats[k].astype(int).tolist() for k in order},
                "top": [
                    {
                        "byte_index": int(i),
                        "commit": float(p[i]),
                        "ce": float(c[i]),
                        "byte": int(raw[i]),
                        "char": bytes([raw[i]]).decode("utf-8", errors="replace"),
                        "features": [k for k in order if feats[k][i]],
                    }
                    for i in sorted(top_idx)
                ],
            }
    return out


def _render_index(out_dir: Path, summary: Dict, rows: List[Dict], sample_records: Dict[str, Dict[str, Dict]]) -> None:
    model_cards = []
    for name, dom in summary["dominance"].items():
        tops = ", ".join(
            f"{r['feature']} corr={r['point_biserial_corr']:.3f} lift={r['commit_lift']:.3f}"
            for r in dom["top_features_by_abs_corr"][:3]
            if r["point_biserial_corr"] is not None
        )
        model_cards.append(
            f"<div class='card'><h2>{html.escape(name)}</h2>"
            f"<p>mean={dom['commit_mean']:.3f} std={dom['commit_std']:.3f} hard={dom['hard_rate']:.3f}</p>"
            f"<p>weak-prior linear R2={dom['weak_prior_linear_r2']:.3f}, max |corr|={dom['max_abs_feature_corr']:.3f}</p>"
            f"<p>{html.escape(tops)}</p></div>"
        )
    conclusion = html.escape(summary["conclusion"])
    links = "".join(
        f"<li><a href='{html.escape(name)}.html'>{html.escape(name)}</a> "
        f"<span class='muted'>reference cuts included</span></li>"
        for name in sample_records
    )
    out_dir.joinpath("index.html").write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f4;color:#161616}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}.card{background:white;border:1px solid #d9d9d2;border-radius:6px;padding:14px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:5px;text-align:right}td:first-child,th:first-child{text-align:left}.note{font-size:18px;font-weight:600}.muted{color:#666;font-size:12px}</style>",
            "<h1>FLUED-v3 Soft Prior Diagnostics</h1>",
            f"<p class='note'>{conclusion}</p>",
            "<div class='cards'>" + "".join(model_cards) + "</div>",
            "<h2>Sample Heatmaps</h2><ul>" + links + "</ul>",
            "<p><a href='feature_lifts.html'>Feature lift table</a> · <a href='byte_value_lifts.html'>Byte value lift table</a> · <a href='summary.csv'>summary.csv</a> · <a href='byte_value_summary.csv'>byte_value_summary.csv</a> · <a href='summary.json'>summary.json</a></p>",
        ]),
        encoding="utf-8",
    )

    by_feature: Dict[str, Dict[str, Dict]] = {}
    for row in rows:
        by_feature.setdefault(row["feature"], {})[row["model"]] = row
    model_names = list(summary["dominance"].keys())
    table_rows = []
    for feat, models in by_feature.items():
        cells = [f"<td>{html.escape(feat)}</td>"]
        for model in model_names:
            row = models.get(model, {})
            lift = row.get("commit_lift")
            corr = row.get("point_biserial_corr")
            enrich = row.get("top_commit_enrichment")
            cells.append(
                f"<td>{'' if lift is None else f'{lift:.4f}'}</td>"
                f"<td>{'' if corr is None else f'{corr:.4f}'}</td>"
                f"<td>{'' if enrich is None else f'{enrich:.3f}'}</td>"
            )
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "<tr><th>feature</th>" + "".join(f"<th>{m} lift</th><th>{m} corr</th><th>{m} top enrich</th>" for m in model_names) + "</tr>"
    out_dir.joinpath("feature_lifts.html").write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;text-align:right}td:first-child,th:first-child{text-align:left}</style>",
            "<h1>Feature Lifts</h1><table>",
            headers,
            "".join(table_rows),
            "</table>",
        ]),
        encoding="utf-8",
    )

    byte_rows = summary.get("byte_value_rows", [])
    byte_subset = [
        r for r in byte_rows
        if r["group"] in {"lowercase", "uppercase", "digit", "punctuation", "whitespace"}
    ]
    byte_subset.sort(key=lambda r: (r["model"], r["group"], -abs(r["commit_lift_vs_global"])))
    byte_table = []
    for r in byte_subset:
        byte_table.append(
            "<tr>"
            f"<td>{html.escape(r['model'])}</td><td>{r['byte']}</td><td>{html.escape(r['label'])}</td>"
            f"<td>{html.escape(r['group'])}</td><td>{r['n']}</td>"
            f"<td>{r['commit_mean']:.4f}</td><td>{r['commit_lift_vs_global']:.4f}</td>"
            f"<td>{r['hard_rate']:.4f}</td><td>{r['top_commit_enrichment']:.3f}</td>"
            "</tr>"
        )
    out_dir.joinpath("byte_value_lifts.html").write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;text-align:right}td:nth-child(1),td:nth-child(3),td:nth-child(4),th:nth-child(1),th:nth-child(3),th:nth-child(4){text-align:left}</style>",
            "<h1>Byte Value Lifts</h1>",
            "<table><tr><th>model</th><th>byte</th><th>label</th><th>group</th><th>n</th><th>commit mean</th><th>lift</th><th>hard rate</th><th>top enrich</th></tr>",
            "".join(byte_table),
            "</table>",
        ]),
        encoding="utf-8",
    )


def _render_sample_pages(out_dir: Path, sample_records: Dict[str, Dict[str, Dict]]) -> None:
    for sample_name, model_records in sample_records.items():
        text = next(iter(model_records.values()))["text"]
        raw = text.encode("utf-8")
        gold_html = _render_gold_segments(sample_name, text)
        panels = []
        for model_name, rec in model_records.items():
            p = np.array(rec["commit"], dtype=np.float64)
            lo, hi = float(np.quantile(p, 0.05)), float(np.quantile(p, 0.95))
            spans = []
            byte_i = 0
            for ch in text:
                ch_raw = ch.encode("utf-8")
                if byte_i >= len(p):
                    break
                end = min(len(p), byte_i + len(ch_raw))
                vals = p[byte_i:end]
                val = float(vals.max()) if vals.size else float(p[byte_i])
                title = (
                    f"bytes={byte_i}-{end - 1} raw={' '.join(str(x) for x in ch_raw)} "
                    f"commit_max={val:.3f} commit_mean={float(vals.mean()):.3f}"
                )
                label = "\u2424" if ch == "\n" else html.escape(ch)
                spans.append(f"<span title='{html.escape(title)}' style='background:{_color(val, lo, hi)}'>{label}</span>")
                byte_i += len(ch_raw)
            top_rows = "".join(
                "<tr>"
                f"<td>{x['byte_index']}</td><td>{x['commit']:.3f}</td><td>{x['ce']:.3f}</td>"
                f"<td>{x['byte']}</td><td>{html.escape(_byte_label(x['byte']))}</td><td>{html.escape(','.join(x['features']))}</td>"
                "</tr>"
                for x in rec["top"]
            )
            panels.append(
                f"<section><h2>{html.escape(model_name)}</h2>"
                f"<p>mean={p.mean():.3f} std={p.std():.3f} p90={np.quantile(p,0.90):.3f}</p>"
                "<div class='heat'>" + "".join(spans) + "</div>"
                "<table><tr><th>byte</th><th>commit</th><th>CE</th><th>raw</th><th>byte label</th><th>features</th></tr>"
                + top_rows + "</table></section>"
            )
        out_dir.joinpath(f"{sample_name}.html").write_text(
            "\n".join([
                "<!doctype html><meta charset='utf-8'>",
                "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#fbfbfb;color:#111}.heat{font-family:Consolas,monospace;font-size:18px;line-height:2.1;word-break:break-all;background:white;border:1px solid #ddd;border-radius:6px;padding:12px}.heat span{color:white;padding:2px 1px;margin:0 1px;border-radius:2px}section{margin:18px 0}.gold{background:#fff;border:1px solid #d5d5cc;border-radius:6px;padding:12px}.seg{display:inline-flex;align-items:baseline;gap:6px;border:1px solid #cfcfc7;border-left:4px solid #2d6cdf;border-radius:4px;padding:6px 8px;margin:4px;background:#fbfbf7}.tag{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.04em}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{border:1px solid #ddd;padding:5px}</style>",
                f"<h1>{html.escape(sample_name)}</h1>",
                f"<pre>{html.escape(text)}</pre>",
                f"<p>bytes={len(raw)}</p>",
                gold_html,
                "".join(panels),
            ]),
            encoding="utf-8",
        )


def _render_gold_segments(sample_name: str, text: str) -> str:
    segments = GOLD_SEGMENTS.get(sample_name, [])
    if not segments:
        return ""
    chips = []
    search_from = 0
    for tag, segment in segments:
        char_start = text.find(segment, search_from)
        if char_start < 0:
            char_start = text.find(segment)
        if char_start < 0:
            byte_start = -1
            byte_end = -1
        else:
            char_end = char_start + len(segment)
            byte_start = len(text[:char_start].encode("utf-8"))
            byte_end = len(text[:char_end].encode("utf-8"))
            search_from = char_end
        byte_label = "unmatched" if byte_start < 0 else f"b{byte_start}-{max(byte_start, byte_end - 1)}"
        chips.append(
            "<span class='seg'>"
            f"<span class='tag'>{html.escape(tag)}</span>"
            f"<span>{html.escape(segment)}</span>"
            f"<span class='tag'>{html.escape(byte_label)}</span>"
            "</span>"
        )
    return (
        "<section class='gold'><h2>Reference Cuts (GPT-5.5)</h2>"
        "<p>These are semantic or structural units, not byte-shape triggers.</p>"
        "<div>" + "".join(chips) + "</div></section>"
    )


def _byte_label(byte: int) -> str:
    if byte == 10:
        return "\\n"
    if byte == 13:
        return "\\r"
    if byte == 9:
        return "\\t"
    if 32 <= byte < 127:
        return chr(byte)
    if 0x80 <= byte <= 0xBF:
        return f"UTF-8 continuation 0x{byte:02X}"
    if 0xC0 <= byte <= 0xF7:
        return f"UTF-8 lead 0x{byte:02X}"
    return f"0x{byte:02X}"


def _make_conclusion(dominance: Dict[str, Dict]) -> str:
    raw = dominance.get("raw")
    gated = dominance.get("gated")
    if not raw or not gated:
        return "Diagnostics completed, but raw/gated comparison names were not both present."
    raw_score = raw["weak_prior_linear_r2"] + raw["max_abs_feature_corr"] + raw["max_abs_feature_lift"]
    gated_score = gated["weak_prior_linear_r2"] + gated["max_abs_feature_corr"] + gated["max_abs_feature_lift"]
    if raw_score > gated_score * 1.15:
        worse = "raw"
    elif gated_score > raw_score * 1.15:
        worse = "gated"
    else:
        worse = "similar"
    level = max(raw["weak_prior_linear_r2"], gated["weak_prior_linear_r2"])
    if level >= 0.18:
        severity = "strong"
    elif level >= 0.08:
        severity = "moderate"
    else:
        severity = "limited"
    if worse == "similar":
        return f"Weak byte-shape priors explain a {severity} share of commit variance, and raw/gated are broadly similar."
    return f"Weak byte-shape priors explain a {severity} share of commit variance; {worse} is more affected by the measured shortcuts."


def run(args: argparse.Namespace) -> Dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model_specs = {
        "raw": Path(args.raw_ckpt),
        "gated": Path(args.gated_ckpt),
    }
    models: Dict[str, V3CommitControllerSmall] = {}
    meta: Dict[str, Dict] = {}
    for name, path in model_specs.items():
        models[name], meta[name] = _load_model(path, device)

    data_path = Path(args.data_path or meta["raw"]["args"].get("data_path", ""))
    if not data_path.exists():
        raise FileNotFoundError(f"data path not found: {data_path}")
    chunks = list(_iter_random_chunks(data_path, args.seq_len, args.num_chunks, args.seed))

    all_rows: List[Dict] = []
    all_byte_rows: List[Dict] = []
    dominance: Dict[str, Dict] = {}
    for name, model in models.items():
        batch = _collect(model, name, chunks, args.batch_size, args.seq_len, device)
        rows, dom = _feature_stats(batch, args.threshold, args.top_frac)
        all_rows.extend(rows)
        all_byte_rows.extend(_byte_value_stats(batch, args.threshold, args.top_frac))
        dominance[name] = dom

    sample_records = _sample_records(models, args.seq_len, device)
    summary = {
        "artifact": "FLUED-v3 weak semantic prior diagnostics",
        "data_path": str(data_path),
        "num_chunks": args.num_chunks,
        "seq_len": args.seq_len,
        "threshold": args.threshold,
        "top_frac": args.top_frac,
        "models": meta,
        "dominance": dominance,
        "feature_rows": all_rows,
        "byte_value_rows": all_byte_rows,
    }
    summary["conclusion"] = _make_conclusion(dominance)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", all_rows)
    _write_byte_csv(out_dir / "byte_value_summary.csv", all_byte_rows)
    (out_dir / "sample_records.json").write_text(json.dumps(sample_records, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_index(out_dir, summary, all_rows, sample_records)
    _render_sample_pages(out_dir, sample_records)
    print(json.dumps({
        "out_dir": str(out_dir),
        "conclusion": summary["conclusion"],
        "raw": dominance["raw"],
        "gated": dominance["gated"],
    }, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose weak byte-shape priors in v3 commit checkpoints")
    parser.add_argument("--raw-ckpt", required=True)
    parser.add_argument("--gated-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
