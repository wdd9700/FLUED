"""Plot FLUED v3-FMC small-run diagnostics.

Inputs:
  checkpoints/v3_fmc_small_5080/<run>/run.log
  checkpoints/v3_fmc_small_5080/<run>/fmc_probe.json

Outputs:
  summary.csv
  train_curves.png
  type_bp.png
  fmc_probe.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


STEP_RE = re.compile(
    r"step=\s*(?P<step>\d+).*?"
    r"loss=(?P<loss>[-+0-9.eEnNaA]+).*?"
    r"recon=(?P<recon>[-+0-9.eEnNaA]+).*?"
    r"comp=(?P<comp>[-+0-9.eEnNaA]+).*?"
    r"latent=(?P<latent>[-+0-9.eEnNaA]+).*?"
    r"recon_acc=(?P<recon_acc>[-+0-9.eEnNaA]+).*?"
    r"soft_m/n=(?P<soft_mn>[-+0-9.eEnNaA]+).*?"
    r"bp_mean=(?P<bp_mean>[-+0-9.eEnNaA]+).*?"
    r"bp_std=(?P<bp_std>[-+0-9.eEnNaA]+).*?"
    r"bhead_gnorm=(?P<bhead_gnorm>[-+0-9.eEnNaA]+).*?"
    r"denoise=(?P<denoise>[-+0-9.eEnNaA]+)"
)

EVAL_RE = re.compile(
    r"E1 eval.*?reconstruction_accuracy=(?P<eval_acc>[-+0-9.eEnNaA]+).*?m/n=(?P<eval_mn>[-+0-9.eEnNaA]+)"
)

TYPE_RE = re.compile(
    r"step=\s*(?P<step>\d+).*?\[type_bp\].*?"
    r"utf8=(?P<utf8>[-+0-9.eEnNaA]+|N/A).*?"
    r"ascii=(?P<ascii>[-+0-9.eEnNaA]+|N/A).*?"
    r"cjk=(?P<cjk>[-+0-9.eEnNaA]+|N/A).*?"
    r"op=(?P<op>[-+0-9.eEnNaA]+|N/A).*?"
    r"digit=(?P<digit>[-+0-9.eEnNaA]+|N/A)"
)


def _f(x: str) -> float:
    if x == "N/A":
        return float("nan")
    return float(x)


def parse_run(run_dir: Path) -> Dict:
    rows: List[Dict] = []
    type_rows: List[Dict] = []
    eval_result: Dict = {}
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return {"name": run_dir.name, "rows": rows, "type_rows": type_rows, "eval": eval_result}

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            d = {k: _f(v) for k, v in m.groupdict().items()}
            d["step"] = int(d["step"])
            rows.append(d)
            continue
        t = TYPE_RE.search(line)
        if t:
            d = {k: _f(v) for k, v in t.groupdict().items()}
            d["step"] = int(d["step"])
            type_rows.append(d)
            continue
        e = EVAL_RE.search(line)
        if e:
            eval_result = {k: _f(v) for k, v in e.groupdict().items()}

    probe_path = run_dir / "fmc_probe.json"
    probe = {}
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))

    causal_probe_path = run_dir / "causal_surprise_probe.json"
    causal_probe = {}
    if causal_probe_path.exists():
        causal_probe = json.loads(causal_probe_path.read_text(encoding="utf-8"))

    return {
        "name": run_dir.name,
        "rows": rows,
        "type_rows": type_rows,
        "eval": eval_result,
        "probe": probe,
        "causal_probe": causal_probe,
    }


def write_summary(runs: List[Dict], out_dir: Path) -> None:
    modes = sorted({
        mode
        for run in runs
        for mode in run.get("probe", {}).get("fmc_modes", {}).keys()
    })
    fields = [
        "run", "last_step", "last_recon_acc", "last_soft_mn", "last_bp_std",
        "eval_acc", "eval_mn",
        "native_density", "native_residual_enrichment", "native_residual_corr",
        "causal_learned_corr", "causal_bigram_corr", "causal_native_corr",
        "causal_learned_enrichment", "causal_bigram_enrichment", "causal_native_enrichment",
    ]
    for mode in modes:
        fields.extend([
            f"{mode}_density",
            f"{mode}_residual_enrichment",
            f"{mode}_residual_corr",
        ])
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            last = run["rows"][-1] if run["rows"] else {}
            probe = run.get("probe", {})
            causal_probe = run.get("causal_probe", {})
            ev = run.get("eval", {})
            row = {
                "run": run["name"],
                "last_step": last.get("step", ""),
                "last_recon_acc": last.get("recon_acc", ""),
                "last_soft_mn": last.get("soft_mn", ""),
                "last_bp_std": last.get("bp_std", ""),
                "eval_acc": ev.get("eval_acc", ""),
                "eval_mn": ev.get("eval_mn", ""),
                "native_density": probe.get("native_density", ""),
                "native_residual_enrichment": probe.get("native_residual_enrichment", ""),
                "native_residual_corr": probe.get("native_residual_corr", ""),
                "causal_learned_corr": causal_probe.get("learned_corr", ""),
                "causal_bigram_corr": causal_probe.get("bigram_corr", ""),
                "causal_native_corr": causal_probe.get("native_corr", ""),
                "causal_learned_enrichment": causal_probe.get("learned_enrichment", ""),
                "causal_bigram_enrichment": causal_probe.get("bigram_enrichment", ""),
                "causal_native_enrichment": causal_probe.get("native_enrichment", ""),
            }
            for mode in modes:
                stats = probe.get("fmc_modes", {}).get(mode, {})
                row[f"{mode}_density"] = stats.get("density", "")
                row[f"{mode}_residual_enrichment"] = stats.get("residual_enrichment", "")
                row[f"{mode}_residual_corr"] = stats.get("residual_corr", "")
            writer.writerow(row)


def make_plots(runs: List[Dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    metrics = [
        ("recon_acc", "train reconstruction accuracy"),
        ("soft_mn", "soft m/n"),
        ("bp_std", "boundary probability std"),
        ("bhead_gnorm", "boundary-head grad norm"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for run in runs:
            rows = run["rows"]
            if not rows:
                continue
            ax.plot([r["step"] for r in rows], [r[metric] for r in rows], label=run["name"])
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "train_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    type_metrics = ["utf8", "ascii", "cjk", "op", "digit"]
    for run in runs:
        rows = run["type_rows"]
        if not rows:
            continue
        for metric in type_metrics:
            ax.plot(
                [r["step"] for r in rows],
                [r[metric] for r in rows],
                label=f"{run['name']}:{metric}",
                linewidth=1.0,
            )
    ax.set_title("type boundary probabilities")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "type_bp.png")
    plt.close(fig)

    labels = []
    mode_names = sorted({
        mode
        for run in runs
        for mode in run.get("probe", {}).get("fmc_modes", {}).keys()
    })
    enrich_by_mode = {mode: [] for mode in mode_names}
    corr_by_mode = {mode: [] for mode in mode_names}
    native_enrich = []
    native_corr = []
    for run in runs:
        probe = run.get("probe", {})
        if not probe:
            continue
        labels.append(run["name"])
        native_enrich.append(probe.get("native_residual_enrichment", 0.0))
        native_corr.append(probe.get("native_residual_corr", 0.0))
        modes = probe.get("fmc_modes", {})
        for mode in mode_names:
            stats = modes.get(mode, {})
            enrich_by_mode[mode].append(stats.get("residual_enrichment", 0.0))
            corr_by_mode[mode].append(stats.get("residual_corr", 0.0))

    if labels:
        x = range(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=140)
        series = [("native", native_enrich)] + [(mode, enrich_by_mode[mode]) for mode in mode_names]
        width = min(0.16, 0.8 / max(len(series), 1))
        offsets = [(i - (len(series) - 1) / 2) * width for i in range(len(series))]
        for offset, (label, values) in zip(offsets, series):
            axes[0].bar([i + offset for i in x], values, width, label=label)
        axes[0].set_title("residual top-k enrichment")
        axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
        axes[0].legend()
        axes[0].grid(alpha=0.25, axis="y")

        series = [("native", native_corr)] + [(mode, corr_by_mode[mode]) for mode in mode_names]
        for offset, (label, values) in zip(offsets, series):
            axes[1].bar([i + offset for i in x], values, width, label=label)
        axes[1].set_title("score-residual correlation")
        axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
        axes[1].legend()
        axes[1].grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "fmc_probe.png")
        plt.close(fig)

    causal_labels = []
    learned_corr = []
    bigram_corr = []
    native_corr = []
    learned_enrich = []
    bigram_enrich = []
    native_enrich = []
    for run in runs:
        probe = run.get("causal_probe", {})
        if not probe:
            continue
        causal_labels.append(run["name"])
        learned_corr.append(probe.get("learned_corr", 0.0))
        bigram_corr.append(probe.get("bigram_corr", 0.0))
        native_corr.append(probe.get("native_corr", 0.0))
        learned_enrich.append(probe.get("learned_enrichment", 0.0))
        bigram_enrich.append(probe.get("bigram_enrichment", 0.0))
        native_enrich.append(probe.get("native_enrichment", 0.0))

    if causal_labels:
        x = range(len(causal_labels))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=140)
        width = 0.24
        for ax, values, title in [
            (axes[0], [native_corr, bigram_corr, learned_corr], "causal surprise correlation"),
            (axes[1], [native_enrich, bigram_enrich, learned_enrich], "causal surprise enrichment"),
        ]:
            ax.bar([i - width for i in x], values[0], width, label="native")
            ax.bar(list(x), values[1], width, label="bigram")
            ax.bar([i + width for i in x], values[2], width, label="learned causal")
            ax.set_title(title)
            ax.set_xticks(list(x), causal_labels, rotation=20, ha="right")
            ax.legend()
            ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "causal_surprise_probe.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot v3-FMC small-run diagnostics")
    parser.add_argument("--run-root", default="checkpoints/v3_fmc_small_5080")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [parse_run(p) for p in sorted(run_root.iterdir()) if p.is_dir() and not p.name.startswith("analysis")]
    write_summary(runs, out_dir)
    try:
        make_plots(runs, out_dir)
    except Exception as exc:
        print(f"plotting failed: {exc}")
    print(f"wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
