"""Gate and select the v3.4 boundary-calibration screening matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/v3_4/v34_boundary_calibration_5k.json")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    matrix = json.loads((repo / args.matrix).read_text(encoding="utf-8"))
    root = Path(args.root)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    by_id = {item["id"]: item for item in matrix["experiments"]}
    for run_id, experiment in by_id.items():
        run_dir = root / run_id
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "train_log.jsonl"
        if not summary_path.exists() or not log_path.exists():
            rows.append({"run": run_id, "complete": False, "passed": False, "reason": "missing output"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dynamic = [row for row in logs if row.get("boundary_mode") == "confidence_threshold"]
        early_dynamic = dynamic[:10]
        tail = dynamic[-10:] if dynamic else logs[-10:]
        requested = float(summary.get(
            "eval_requested_eligible_hard_cut_fraction",
            summary.get("eval_requested_hard_cut_fraction", 0.0),
        ))
        target = float(summary.get("eval_boundary_rate_target_cut_fraction", 0.0))
        ratio = requested / max(target, 1.0e-9)
        correlation = float(summary.get("eval_confidence_rate_correlation", 0.0))
        continuation = float(summary.get("eval_boundary_cont_mean", 1.0))
        punctuation = float(summary.get("eval_boundary_punct_mean", -1.0))
        neutral = float(summary.get("eval_boundary_neutral_mean", 1.0))
        overflow = float(summary.get("eval_cut_capacity_overflow_per_byte", 1.0))
        truncated = float(summary.get("eval_truncated_tokens", 1.0))
        calibration_gap = float(summary.get("eval_boundary_rate_calibration_probability_gap", 1.0))
        tail_grad = _mean(tail, "segmentor_head_grad_norm")
        early_correlation = _mean(early_dynamic, "confidence_rate_correlation")
        tail_correlation = _mean(tail, "confidence_rate_correlation")
        correlation_drop = early_correlation - tail_correlation
        positive_constraint_fraction = (
            sum(float(row.get("boundary_compute_constraint", 1.0)) > 0.0 for row in dynamic)
            / max(len(dynamic), 1)
        )
        tail_dual = _mean(tail, "boundary_compute_dual")
        passed = (
            int(summary.get("steps", 0)) >= 5000
            and truncated == 0.0
            and overflow == 0.0
            and continuation <= -0.90
            and abs(neutral) <= 0.10
            and 0.35 <= punctuation <= 0.65
            and correlation >= 0.65
            and tail_correlation >= 0.65
            and correlation_drop <= 0.10
            and target >= 0.005
            and 0.75 <= ratio <= 1.25
            and calibration_gap <= 0.10
            and tail_grad > 0.0
            and positive_constraint_fraction <= 0.05
            and tail_dual < 19.0
        )
        score = (
            abs(math.log(max(ratio, 1.0e-6)))
            + (1.0 - max(min(correlation, 1.0), -1.0))
            + abs(neutral)
            + abs(punctuation - 0.5)
            + 2.0 * calibration_gap
            + 10.0 * overflow
            - 0.1 * float(summary.get("eval_identity_acc", 0.0))
            - 0.1 * float(summary.get("eval_completion_mask_acc", 0.0))
        )
        rows.append({
            "run": run_id,
            "complete": True,
            "passed": passed,
            "score": score,
            "requested_cut": requested,
            "target_cut": target,
            "cut_ratio": ratio,
            "correlation": correlation,
            "continuation_mean": continuation,
            "punctuation_mean": punctuation,
            "neutral_mean": neutral,
            "overflow_per_byte": overflow,
            "calibration_gap": calibration_gap,
            "tail_segmentor_grad": tail_grad,
            "early_correlation": early_correlation,
            "tail_correlation": tail_correlation,
            "correlation_drop": correlation_drop,
            "positive_constraint_fraction": positive_constraint_fraction,
            "tail_dual": tail_dual,
            "identity_acc": float(summary.get("eval_identity_acc", 0.0)),
            "completion_acc": float(summary.get("eval_completion_mask_acc", 0.0)),
            "preserve_acc": float(summary.get("eval_completion_preserve_acc", 0.0)),
            "completion_ppl": float(summary.get("eval_masked_byte_pseudo_ppl", 0.0)),
            "actual_latent_per_byte": float(summary.get("eval_actual_backbone_units_per_byte", 0.0)),
        })

    baseline = next((row for row in rows if row.get("run") == "b0_aggregate_prior_no_calibration" and row.get("complete")), None)
    if baseline:
        for row in rows:
            if not row.get("passed") or row["run"] == baseline["run"]:
                continue
            task_ratios = [
                row[key] / max(baseline[key], 1.0e-9)
                for key in ("identity_acc", "completion_acc", "preserve_acc")
            ]
            row["min_task_ratio_vs_control"] = min(task_ratios)
            if row["min_task_ratio_vs_control"] < 0.90:
                row["passed"] = False
                row["reason"] = "task metrics jointly/regressively exceed 10% gate"
    passed_rows = sorted((row for row in rows if row.get("passed")), key=lambda row: row["score"])
    winner = passed_rows[0] if passed_rows else None
    payload = {"matrix": matrix["matrix_name"], "rows": rows, "winner": winner}
    (output / "boundary_calibration_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "run", "passed", "requested_cut", "target_cut", "cut_ratio", "correlation",
        "continuation_mean", "punctuation_mean", "neutral_mean", "completion_ppl",
        "actual_latent_per_byte", "score",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if winner:
        lines.extend(["", f"Winner: `{winner['run']}`"])
        selected = {
            key: value
            for key, value in {
                **matrix.get("base", {}),
                **by_id[winner["run"]]["overrides"],
            }.items()
            if key in {
                "boundary_bridge_gradient_scale",
                "boundary_rate_alignment_weight",
                "boundary_continuation_loss_weight",
                "boundary_punctuation_loss_weight",
                "boundary_neutral_loss_weight",
                "boundary_rate_calibration_weight",
                "boundary_rate_density_weight",
                "boundary_rate_margin_weight",
                "boundary_rate_positive_margin",
                "boundary_calibration_temperature",
                "boundary_target_bytes_per_chunk",
                "max_chunks",
                "max_span",
                "boundary_dual_lr",
                "boundary_dual_max",
                "boundary_budget_augmented_weight",
            }
        }
        (output / "selected_overrides.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        lines.extend(["", "No experiment passed the calibration gate."])
    (output / "boundary_calibration_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if winner is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
