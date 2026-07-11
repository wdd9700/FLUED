"""Summarize FLUED-v3.1 parallel diffusion sweep runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _last_jsonl(path: Path) -> Dict:
    last = {}
    if not path.exists():
        return last
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _score(row: Dict) -> float:
    loss = _f(row.get("deploy_eval_loss"))
    future = _f(row.get("deploy_eval_future_loss"))
    acc = _f(row.get("deploy_eval_acc"))
    std = _f(row.get("deploy_eval_commit_std"))
    corr = _f(row.get("deploy_eval_commit_corr"))
    value = _f(row.get("last_value_corr"))
    delta = _f(row.get("last_ar_delta_loss"))
    if math.isnan(loss):
        return float("inf")
    return (
        loss
        + 0.25 * (future if math.isfinite(future) else 0.0)
        - 0.35 * (acc if math.isfinite(acc) else 0.0)
        - 0.15 * max(0.0, std if math.isfinite(std) else 0.0)
        - 0.10 * max(0.0, corr if math.isfinite(corr) else 0.0)
        - 0.10 * max(0.0, value if math.isfinite(value) else 0.0)
        + 0.25 * max(0.0, delta if math.isfinite(delta) else 0.0)
    )


def _write_csv(path: Path, rows: List[Dict]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path) -> Dict:
    rows: List[Dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        summary = _load_json(child / "summary.json")
        if not summary:
            continue
        last = _last_jsonl(child / "metrics.jsonl")
        args = _load_json(child / "args.json")
        row = {
            "variant": child.name,
            "path": str(child),
            "params": summary.get("params"),
            "steps": summary.get("steps"),
            "step_schedule": summary.get("step_schedule"),
            "deploy_eval_loss": summary.get("deploy_eval_loss"),
            "deploy_eval_acc": summary.get("deploy_eval_acc"),
            "deploy_eval_future_loss": summary.get("deploy_eval_future_loss"),
            "deploy_eval_commit_mn": summary.get("deploy_eval_commit_mn"),
            "deploy_eval_commit_std": summary.get("deploy_eval_commit_std"),
            "deploy_eval_commit_corr": summary.get("deploy_eval_commit_corr"),
            "deploy_eval_commit_enrich": summary.get("deploy_eval_commit_enrich"),
            "deploy_eval_denoise_loss": summary.get("deploy_eval_denoise_loss"),
            "deploy_eval_future_mask_loss": summary.get("deploy_eval_future_mask_loss"),
            "multi_eval_loss": summary.get("multi_eval_loss"),
            "multi_eval_acc": summary.get("multi_eval_acc"),
            "multi_eval_future_loss": summary.get("multi_eval_future_loss"),
            "last_recon": last.get("recon"),
            "last_acc": last.get("acc"),
            "last_future_loss": last.get("future_loss"),
            "last_value_corr": last.get("value_corr"),
            "last_ar_delta_loss": last.get("ar_delta_loss"),
            "last_boundary_steps": last.get("boundary_steps"),
            "last_memory_steps": last.get("memory_steps"),
            "last_readout_steps": last.get("readout_steps"),
            "last_ar_passes": last.get("ar_passes"),
            "last_commit_mn": last.get("commit_mn"),
            "last_commit_std": last.get("commit_std"),
            "last_rate_lambda": last.get("rate_lambda"),
            "args": args,
        }
        row["one_minus_multi_loss"] = _f(row["deploy_eval_loss"]) - _f(row["multi_eval_loss"])
        row["rank_score"] = _score(row)
        rows.append(row)
    rows.sort(key=lambda x: _f(x.get("rank_score")))
    result = {"root": str(root), "num_variants": len(rows), "rows": rows}
    (root / "sweep_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows_csv = [{k: v for k, v in row.items() if k != "args"} for row in rows]
    _write_csv(root / "sweep_summary.csv", rows_csv)
    lines = [
        "# FLUED v3.1 Parallel Diffusion Sweep",
        "",
        f"Root: `{root}`",
        "",
        "| rank | variant | deploy_loss | multi_loss | gap | future | acc | m/n | std | value_corr | delta | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | {variant} | {loss:.4f} | {multi:.4f} | {gap:.4f} | {future:.4f} | {acc:.4f} | "
            "{mn:.3f} | {std:.3f} | {value:.3f} | {delta:.4f} | {score:.4f} |".format(
                rank=i,
                variant=row["variant"],
                loss=_f(row["deploy_eval_loss"]),
                multi=_f(row["multi_eval_loss"]),
                gap=_f(row["one_minus_multi_loss"]),
                future=_f(row["deploy_eval_future_loss"]),
                acc=_f(row["deploy_eval_acc"]),
                mn=_f(row["deploy_eval_commit_mn"]),
                std=_f(row["deploy_eval_commit_std"]),
                value=_f(row["last_value_corr"]),
                delta=_f(row["last_ar_delta_loss"]),
                score=_f(row["rank_score"]),
            )
        )
    (root / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "num_variants": len(rows), "best": rows[0]["variant"] if rows else None}, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v3.1 diffusion sweep")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run(Path(args.root))


if __name__ == "__main__":
    main()
