"""Span-level diagnostics for FLUED-v3 commit-controller checkpoints.

This evaluator does not train. It measures whether commit probabilities at
anchor position t align with short future spans t+1..t+H for H in 4/8/16.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset, PAD_ID
from tools.analysis.train_v3_commit_controller_small import V3CommitControllerSmall, _load_texts


DEFAULT_CKPTS = {
    "raw": r"K:\FLUED_archive\v3_commit_controller_20260629\scale6m_seq128_raw_active_memory_15k\latest.pt",
    "gated": r"K:\FLUED_archive\v3_commit_controller_20260629\scale6m_seq128_gated_active_memory_15k\latest.pt",
}
DEFAULT_OUT_DIR = r"K:\FLUED_archive\v3_diagnostics_20260629\span_objective"


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
    return model, ckpt


def _target_next(src: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    target = torch.full_like(src, PAD_ID)
    target[:, :-1] = src[:, 1:]
    mask = valid.clone()
    mask[:, :-1] = valid[:, :-1] & valid[:, 1:]
    mask[:, -1] = False
    return target, mask


def _corr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    xv = x[mask].float()
    yv = y[mask].float()
    if xv.numel() <= 4:
        return float("nan")
    xv = (xv - xv.mean()) / xv.std(unbiased=False).clamp(min=1e-6)
    yv = (yv - yv.mean()) / yv.std(unbiased=False).clamp(min=1e-6)
    return float((xv * yv).mean().item())


def _top_enrichment(score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, density: float) -> float:
    s = score[mask].float()
    t = target[mask].float()
    n = s.numel()
    if n < 8:
        return float("nan")
    k = max(1, min(n, int(round(n * density))))
    top_s = torch.topk(s, k=k).indices
    top_t = torch.topk(t, k=k).indices
    sel = torch.zeros(n, dtype=torch.bool, device=s.device)
    hard = torch.zeros(n, dtype=torch.bool, device=s.device)
    sel[top_s] = True
    hard[top_t] = True
    return float(((sel & hard).float().mean().item()) / max(k / n, 1e-6))


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return sum(vals) / max(1, len(vals))


def _quantiles(x: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    vals = x[mask].float()
    if vals.numel() == 0:
        return {k: float("nan") for k in ("mean", "std", "p10", "p50", "p90", "p99")}
    qs = torch.quantile(vals, torch.tensor([0.10, 0.50, 0.90, 0.99], device=vals.device))
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "p10": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p90": float(qs[2].item()),
        "p99": float(qs[3].item()),
    }


def _span_metrics(
    ce_next: torch.Tensor,
    ok_next: torch.Tensor,
    next_mask: torch.Tensor,
    valid: torch.Tensor,
    horizons: List[int],
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    out = {}
    bsz, seq_len = ce_next.shape
    for horizon in horizons:
        span_ce = torch.zeros_like(ce_next)
        span_ok = torch.zeros_like(ok_next.float())
        span_mask = torch.ones_like(next_mask)
        for offset in range(horizon):
            shifted_ce = torch.zeros_like(ce_next)
            shifted_ok = torch.zeros_like(ok_next.float())
            shifted_mask = torch.zeros_like(next_mask)
            if offset < seq_len:
                shifted_ce[:, : seq_len - offset] = ce_next[:, offset:]
                shifted_ok[:, : seq_len - offset] = ok_next[:, offset:].float()
                shifted_mask[:, : seq_len - offset] = next_mask[:, offset:]
            span_ce = span_ce + shifted_ce
            span_ok = span_ok + shifted_ok
            span_mask = span_mask & shifted_mask
        span_mask[:, 0] = False
        span_mask = span_mask & valid
        out[horizon] = (span_ce, span_ok / float(horizon), span_mask)
    return out


def _byte_context(ids: torch.Tensor, anchor: int, radius: int = 32) -> str:
    vals = ids.detach().cpu().tolist()
    raw = bytes(max(0, min(255, int(v) - 1)) for v in vals if v != PAD_ID)
    left = max(0, anchor - radius)
    right = min(len(raw), anchor + radius + 1)
    return raw[left:right].decode("utf-8", errors="replace")


@torch.no_grad()
def evaluate_model(
    label: str,
    model: V3CommitControllerSmall,
    loader: DataLoader,
    device: torch.device,
    horizons: List[int],
    max_batches: int,
    roi_top_k: int,
) -> Tuple[List[Dict], List[Dict]]:
    aggregate_vals: Dict[int, Dict[str, List[float]]] = {
        h: defaultdict(list) for h in horizons
    }
    aggregate_vals[1] = defaultdict(list)
    roi_candidates: List[Dict] = []

    for batch_idx, (src, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        src = src.to(device)
        valid = src != PAD_ID
        logits, metrics = model(src, valid)
        target, next_mask = _target_next(src, valid)
        ce_next = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(target)
        pred = logits.argmax(dim=-1)
        ok_next = pred.eq(target) & next_mask
        commit = metrics["commit_probs"].float()
        usable_next = next_mask.clone()
        usable_next[:, 0] = False

        commit_q = _quantiles(commit, usable_next)
        density = max(0.01, min(0.99, commit_q["mean"]))
        aggregate_vals[1]["ce_per_byte"].append(float(ce_next[usable_next].mean().item()))
        aggregate_vals[1]["acc_per_byte"].append(float(ok_next[usable_next].float().mean().item()))
        aggregate_vals[1]["commit_mean"].append(commit_q["mean"])
        aggregate_vals[1]["commit_std"].append(commit_q["std"])
        aggregate_vals[1]["commit_corr_ce"].append(_corr(commit, ce_next, usable_next))
        aggregate_vals[1]["commit_hard_ce_enrichment"].append(_top_enrichment(commit, ce_next, usable_next, density))

        for horizon, (span_ce, span_acc, span_mask) in _span_metrics(ce_next, ok_next, next_mask, valid, horizons).items():
            vals = aggregate_vals[horizon]
            vals["cum_ce"].append(float(span_ce[span_mask].mean().item()))
            vals["ce_per_byte"].append(float((span_ce[span_mask] / horizon).mean().item()))
            vals["acc_per_byte"].append(float(span_acc[span_mask].mean().item()))
            vals["commit_mean"].append(commit_q["mean"])
            vals["commit_std"].append(commit_q["std"])
            vals["commit_corr_span_ce"].append(_corr(commit, span_ce, span_mask))
            vals["commit_hard_span_enrichment"].append(_top_enrichment(commit, span_ce, span_mask, density))

            flat_commit = commit[span_mask].detach().float()
            flat_ce = span_ce[span_mask].detach().float()
            if flat_commit.numel() >= 8:
                k = max(1, int(round(flat_commit.numel() * density)))
                top_idx = torch.topk(flat_commit, k=k).indices
                low_idx = torch.topk(-flat_commit, k=k).indices
                vals["top_commit_cum_ce"].append(float(flat_ce[top_idx].mean().item()))
                vals["low_commit_cum_ce"].append(float(flat_ce[low_idx].mean().item()))
                vals["top_minus_low_cum_ce"].append(float(flat_ce[top_idx].mean().item() - flat_ce[low_idx].mean().item()))

            if horizon == max(horizons) and roi_top_k > 0:
                score = commit * span_mask.float()
                for b in range(src.size(0)):
                    row_score = score[b]
                    valid_count = int(span_mask[b].sum().item())
                    if valid_count == 0:
                        continue
                    k = min(roi_top_k, valid_count)
                    for idx in torch.topk(row_score, k=k).indices.detach().cpu().tolist():
                        if not bool(span_mask[b, idx].item()):
                            continue
                        roi_candidates.append({
                            "model": label,
                            "batch": batch_idx,
                            "row": b,
                            "anchor": int(idx),
                            "commit": float(commit[b, idx].item()),
                            "span16_cum_ce": float(span_ce[b, idx].item()),
                            "span16_ce_per_byte": float(span_ce[b, idx].item() / horizon),
                            "span16_acc": float(span_acc[b, idx].item()),
                            "context": _byte_context(src[b], int(idx)),
                        })

    rows: List[Dict] = []
    for horizon in [1] + horizons:
        vals = aggregate_vals[horizon]
        row = {"model": label, "horizon": horizon}
        for key, xs in vals.items():
            row[key] = _safe_mean(xs)
        rows.append(row)

    roi_candidates.sort(key=lambda x: (-x["commit"], x["span16_ce_per_byte"]))
    return rows, roi_candidates[: max(1, roi_top_k * 4)]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_rows(rows: List[Dict]) -> List[Dict]:
    by_key = {(r["model"], int(r["horizon"])): r for r in rows}
    out = []
    horizons = sorted({int(r["horizon"]) for r in rows})
    for h in horizons:
        if ("raw", h) not in by_key or ("gated", h) not in by_key:
            continue
        raw = by_key[("raw", h)]
        gated = by_key[("gated", h)]
        row = {"horizon": h}
        for key in sorted(set(raw) | set(gated)):
            if key in {"model", "horizon"}:
                continue
            if isinstance(raw.get(key), (int, float)) and isinstance(gated.get(key), (int, float)):
                row[f"raw_{key}"] = raw[key]
                row[f"gated_{key}"] = gated[key]
                row[f"gated_minus_raw_{key}"] = gated[key] - raw[key]
        out.append(row)
    return out


def _render_html(out_dir: Path, rows: List[Dict], comparison: List[Dict], roi_rows: List[Dict], meta: Dict) -> None:
    def table(items: List[Dict]) -> str:
        if not items:
            return "<p>No rows.</p>"
        keys = list(items[0].keys())
        head = "".join(f"<th>{html.escape(str(k))}</th>" for k in keys)
        body = []
        for item in items:
            cells = []
            for key in keys:
                val = item.get(key, "")
                if isinstance(val, float):
                    val = f"{val:.6g}"
                cells.append(f"<td>{html.escape(str(val))}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "<table><tr>" + head + "</tr>" + "\n".join(body) + "</table>"

    html_text = "\n".join([
        "<!doctype html><meta charset='utf-8'>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111;background:#f7f7f7}"
        "section{background:white;border:1px solid #ddd;border-radius:6px;padding:16px;margin:14px 0}"
        "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}"
        "th{background:#eee}.ctx{font-family:Consolas,monospace;white-space:pre-wrap}</style>",
        "<h1>FLUED v3 Span Objective Diagnostics</h1>",
        f"<p>data={html.escape(str(meta.get('data_path')))} max_batches={meta.get('max_batches')} horizons={meta.get('horizons')}</p>",
        "<section><h2>Aggregate</h2>" + table(rows) + "</section>",
        "<section><h2>Raw vs Gated</h2>" + table(comparison) + "</section>",
        "<section><h2>Top Commit ROI Examples</h2>" + table(roi_rows) + "</section>",
    ])
    (out_dir / "report.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> Dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    raw_ckpt = Path(args.raw_ckpt)
    gated_ckpt = Path(args.gated_ckpt)

    if args.data_path:
        data_path = args.data_path
    else:
        probe = torch.load(raw_ckpt, map_location="cpu", weights_only=False)
        data_path = str(probe.get("args", {}).get("data_path", ""))
    if not data_path:
        raise RuntimeError("data path was not provided and is missing from checkpoint args")

    texts = _load_texts(data_path, args.eval_max_lines)
    ds = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_rows: List[Dict] = []
    roi_rows: List[Dict] = []
    ckpt_meta: Dict[str, Dict] = {}
    for label, ckpt_path in (("raw", raw_ckpt), ("gated", gated_ckpt)):
        model, ckpt = _load_model(ckpt_path, device)
        ckpt_meta[label] = {
            "path": str(ckpt_path),
            "step": ckpt.get("step"),
            "args": ckpt.get("args", {}),
            "summary": ckpt.get("summary", {}),
        }
        rows, roi = evaluate_model(
            label=label,
            model=model,
            loader=loader,
            device=device,
            horizons=args.horizons,
            max_batches=args.max_batches,
            roi_top_k=args.roi_top_k,
        )
        all_rows.extend(rows)
        roi_rows.extend(roi)

    comparison = _comparison_rows(all_rows)
    meta = {
        "data_path": data_path,
        "out_dir": str(out_dir),
        "device": str(device),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "horizons": args.horizons,
        "checkpoints": ckpt_meta,
    }
    result = {
        "meta": meta,
        "aggregate": all_rows,
        "comparison": comparison,
        "roi_examples": roi_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "aggregate.csv", all_rows)
    _write_csv(out_dir / "comparison.csv", comparison)
    _write_csv(out_dir / "roi_examples.csv", roi_rows)
    _render_html(out_dir, all_rows, comparison, roi_rows, meta)
    print(json.dumps({"out_dir": str(out_dir), "aggregate": all_rows, "comparison": comparison}, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FLUED-v3 short-span objective alignment")
    parser.add_argument("--raw-ckpt", default=DEFAULT_CKPTS["raw"])
    parser.add_argument("--gated-ckpt", default=DEFAULT_CKPTS["gated"])
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument("--eval-max-lines", type=int, default=30000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--roi-top-k", type=int, default=8)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
