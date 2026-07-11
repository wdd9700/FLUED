"""Summarize FLUED-v3.1 segmental workspace matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_last_jsonl(path: Path) -> Dict:
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _safe(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _score(row: Dict) -> float:
    # Lower is better. This is a ranking aid, not a paper metric.
    eval_loss = _safe(row.get("eval_loss"))
    future = _safe(row.get("eval_future_loss"))
    student = _safe(row.get("eval_student_loss"))
    commit_std = _safe(row.get("eval_commit_std"))
    commit_corr = _safe(row.get("eval_commit_corr"))
    value_corr = _safe(row.get("last_value_corr"))
    if math.isnan(eval_loss):
        return float("inf")
    return (
        eval_loss
        + 0.30 * (future if not math.isnan(future) else 0.0)
        + 0.20 * (student if not math.isnan(student) else 0.0)
        - 0.20 * max(0.0, commit_std if not math.isnan(commit_std) else 0.0)
        - 0.15 * max(0.0, commit_corr if not math.isnan(commit_corr) else 0.0)
        - 0.10 * max(0.0, value_corr if not math.isnan(value_corr) else 0.0)
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
        summary = _read_json(child / "summary.json")
        if not summary:
            continue
        last_metrics = _read_last_jsonl(child / "metrics.jsonl")
        row = {
            "variant": child.name,
            "path": str(child),
            "params": summary.get("params"),
            "steps": summary.get("steps"),
            "refine_steps": summary.get("refine_steps"),
            "student_refine_steps": summary.get("student_refine_steps"),
            "ar_correction_passes": summary.get("ar_correction_passes"),
            "residual_mixer": summary.get("residual_mixer"),
            "memory_enabled": summary.get("memory_enabled"),
            "eval_loss": summary.get("eval_loss"),
            "eval_student_loss": summary.get("eval_student_loss"),
            "eval_future_loss": summary.get("eval_future_loss"),
            "eval_acc": summary.get("eval_acc"),
            "eval_commit_mn": summary.get("eval_commit_mn"),
            "eval_commit_std": summary.get("eval_commit_std"),
            "eval_commit_corr": summary.get("eval_commit_corr"),
            "eval_commit_enrich": summary.get("eval_commit_enrich"),
            "eval_denoise_loss": summary.get("eval_denoise_loss"),
            "eval_future_mask_loss": summary.get("eval_future_mask_loss"),
            "last_recon": last_metrics.get("recon"),
            "last_student_loss": last_metrics.get("student_loss"),
            "last_future_loss": last_metrics.get("future_loss"),
            "last_value_corr": last_metrics.get("value_corr"),
            "last_commit_mn": last_metrics.get("commit_mn"),
            "last_commit_std": last_metrics.get("commit_std"),
            "last_rate_lambda": last_metrics.get("rate_lambda"),
            "last_alpha_last_mean": last_metrics.get("residual_alpha_last_mean"),
        }
        row["rank_score"] = _score(row)
        rows.append(row)
    rows.sort(key=lambda r: _safe(r.get("rank_score")))

    out = {
        "root": str(root),
        "num_variants": len(rows),
        "rows": rows,
    }
    (root / "matrix_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(root / "matrix_summary.csv", rows)
    lines = [
        "# FLUED v3.1 Segmental Workspace Matrix Summary",
        "",
        f"Root: `{root}`",
        "",
        "| rank | variant | eval_loss | future | student | acc | m/n | std | corr | value_corr | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {variant} | {eval_loss:.4f} | {future:.4f} | {student:.4f} | {acc:.4f} | "
            "{mn:.3f} | {std:.3f} | {corr:.3f} | {value_corr:.3f} | {score:.4f} |".format(
                rank=i,
                variant=row["variant"],
                eval_loss=_safe(row.get("eval_loss")),
                future=_safe(row.get("eval_future_loss")),
                student=_safe(row.get("eval_student_loss")),
                acc=_safe(row.get("eval_acc")),
                mn=_safe(row.get("eval_commit_mn")),
                std=_safe(row.get("eval_commit_std")),
                corr=_safe(row.get("eval_commit_corr")),
                value_corr=_safe(row.get("last_value_corr")),
                score=_safe(row.get("rank_score")),
            )
        )
    (root / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "num_variants": len(rows), "best": rows[0]["variant"] if rows else None}, indent=2, ensure_ascii=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FLUED-v3.1 segmental workspace matrix")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    run(Path(args.root))


if __name__ == "__main__":
    main()
