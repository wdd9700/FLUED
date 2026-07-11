"""Summarize surprise-assisted FLUED-small runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


FIELDS = [
    "run",
    "mode",
    "steps",
    "model_params",
    "surprise_params",
    "last_acc",
    "last_soft_mn",
    "last_bp_std",
    "last_bp_corr",
    "last_bp_enrichment",
    "last_signal_corr",
    "last_budget",
    "last_budget_weight",
    "eval_acc",
    "eval_soft_mn",
    "eval_bp_std",
    "eval_bp_residual_corr",
    "eval_bp_residual_enrichment",
    "eval_signal_residual_corr",
    "eval_signal_residual_enrichment",
]


def load_runs(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        last = data.get("last", {})
        rows.append({
            "run": path.parent.name,
            "mode": data.get("mode", ""),
            "steps": data.get("steps", ""),
            "model_params": data.get("model_params", ""),
            "surprise_params": data.get("surprise_params", ""),
            "last_acc": last.get("acc", ""),
            "last_soft_mn": last.get("soft_mn", ""),
            "last_bp_std": last.get("bp_std", ""),
            "last_bp_corr": last.get("bp_residual_corr", ""),
            "last_bp_enrichment": last.get("bp_residual_enrichment", ""),
            "last_signal_corr": last.get("signal_residual_corr", ""),
            "last_budget": last.get("budget", ""),
            "last_budget_weight": last.get("budget_weight", ""),
            "eval_acc": data.get("eval_acc", ""),
            "eval_soft_mn": data.get("eval_soft_mn", ""),
            "eval_bp_std": data.get("eval_bp_std", ""),
            "eval_bp_residual_corr": data.get("eval_bp_residual_corr", ""),
            "eval_bp_residual_enrichment": data.get("eval_bp_residual_enrichment", ""),
            "eval_signal_residual_corr": data.get("eval_signal_residual_corr", ""),
            "eval_signal_residual_enrichment": data.get("eval_signal_residual_enrichment", ""),
        })
    return rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r["run"] for r in rows]
    eval_acc = [float(r["eval_acc"] or 0.0) for r in rows]
    eval_mn = [float(r["eval_soft_mn"] or 0.0) for r in rows]
    eval_corr = [float(r["eval_bp_residual_corr"] or 0.0) for r in rows]
    eval_enrich = [float(r["eval_bp_residual_enrichment"] or 0.0) for r in rows]

    x = range(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=140)
    for ax, values, title in [
        (axes[0, 0], eval_acc, "eval reconstruction accuracy"),
        (axes[0, 1], eval_mn, "eval soft m/n"),
        (axes[1, 0], eval_corr, "boundary-residual correlation"),
        (axes[1, 1], eval_enrich, "boundary-residual enrichment"),
    ]:
        ax.bar(list(x), values)
        ax.set_title(title)
        ax.set_xticks(list(x), labels, rotation=25, ha="right")
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize surprise-assisted FLUED-small runs")
    parser.add_argument("--run-root", default="checkpoints/v3_surprise_assisted_5080")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    rows = load_runs(root)
    write_csv(rows, out_dir / "summary.csv")
    try:
        write_plot(rows, out_dir / "summary.png")
    except Exception as exc:
        print(f"plot failed: {exc}")
    print(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
