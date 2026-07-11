"""Analyze detailed metrics.jsonl from surprise-assisted FLUED runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


FIELDS = [
    "run",
    "bucket",
    "n",
    "step_min",
    "step_max",
    "denoise_rate",
    "future_mask_rate",
    "clean_rate",
    "acc",
    "recon",
    "ce_p90",
    "soft_mn",
    "bp_std",
    "bp_p10",
    "bp_p50",
    "bp_p90",
    "bp_gt_05",
    "bp_residual_corr",
    "bp_residual_enrichment",
    "signal_std",
    "signal_p90",
    "signal_p99",
    "signal_residual_corr",
    "signal_residual_enrichment",
    "surprise_mse",
    "surprise_byte",
    "rate_boundary",
    "rate_boundary_entropy",
    "rate_latent",
    "coding_rate_value",
    "coding_rate_loss",
    "adaptive_rate_lambda",
    "contrast",
    "total_grad_norm",
    "boundary_grad_norm",
    "surprise_grad_norm",
]


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _avg(rows: List[Dict], key: str) -> str:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    if not vals:
        return ""
    return f"{mean(vals):.6f}"


def _summarize(run: str, bucket: str, rows: List[Dict]) -> Dict[str, str]:
    steps = [int(r["step"]) for r in rows if "step" in r]
    denoise_vals = [1.0 if r.get("use_denoise") else 0.0 for r in rows]
    future_vals = [float(r.get("task_future_mask", r.get("use_future_mask", 0.0))) for r in rows]
    clean_vals = [float(r.get("task_clean", 0.0)) for r in rows]
    out: Dict[str, str] = {
        "run": run,
        "bucket": bucket,
        "n": str(len(rows)),
        "step_min": str(min(steps)) if steps else "",
        "step_max": str(max(steps)) if steps else "",
        "denoise_rate": f"{mean(denoise_vals):.6f}" if denoise_vals else "",
        "future_mask_rate": f"{mean(future_vals):.6f}" if future_vals else "",
        "clean_rate": f"{mean(clean_vals):.6f}" if clean_vals else "",
    }
    for key in FIELDS:
        if key in out:
            continue
        out[key] = _avg(rows, key)
    return out


def _window_bucket(step: int, window: int) -> str:
    start = ((step - 1) // window) * window
    end = start + window
    return f"{start}-{end}"


def analyze_run(run: str, path: Path, window: int) -> List[Dict[str, str]]:
    rows = _load_jsonl(path)
    output: List[Dict[str, str]] = []

    by_window: Dict[str, List[Dict]] = defaultdict(list)
    by_denoise: Dict[str, List[Dict]] = defaultdict(list)
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        step = int(row.get("step", 0))
        by_window[_window_bucket(step, window)].append(row)
        by_denoise["denoise" if row.get("use_denoise") else "clean"].append(row)
        if row.get("task_future_mask"):
            by_task["future_mask"].append(row)
        elif row.get("task_denoise") or row.get("use_denoise"):
            by_task["denoise_task"].append(row)
        else:
            by_task["clean_task"].append(row)

    for bucket in sorted(by_window, key=lambda x: int(x.split("-", 1)[0])):
        output.append(_summarize(run, bucket, by_window[bucket]))
    for bucket in ["clean", "denoise"]:
        if by_denoise.get(bucket):
            output.append(_summarize(run, bucket, by_denoise[bucket]))
    for bucket in ["clean_task", "denoise_task", "future_mask"]:
        if by_task.get(bucket):
            output.append(_summarize(run, bucket, by_task[bucket]))
    output.append(_summarize(run, "all", rows))
    return output


def write_csv(rows: Iterable[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detailed surprise-assisted metrics.jsonl")
    parser.add_argument("--native", type=Path, default=None, help="Path to native metrics.jsonl")
    parser.add_argument("--v3", type=Path, default=None, help="Path to v3 metrics.jsonl")
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    if args.native:
        rows.extend(analyze_run("native", args.native, args.window))
    if args.v3:
        rows.extend(analyze_run("v3", args.v3, args.window))
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
