"""Summarize FLUED-v3.1 language-codec training runs.

The script reads one or more run directories produced by
``train_v31_language_codec_2m.py`` and emits a markdown diagnostic report.  It
only depends on the standard library and torch; checkpoints are loaded on CPU
to inspect args/summary metadata without touching GPU memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


LOG_NAME = "train_log.jsonl"
SUMMARY_NAME = "summary.json"
CKPT_NAME = "latest.pt"

ACCURACY_KEYS = ("recon_acc", "length_acc", "boundary_acc")
LOG_NUMERIC_KEYS = (
    "loss",
    "recon_loss",
    "length_loss",
    "boundary_loss",
    "recon_acc",
    "length_acc",
    "boundary_acc",
    "units_per_byte",
    "grad",
    "lr",
    "steps_per_sec",
    "recent_steps_per_sec",
    "samples_per_sec",
    "recent_samples_per_sec",
    "bytes_per_sec",
    "max_memory_allocated_mb",
)
CONFIG_KEYS = (
    "seq_len",
    "stride",
    "batch_size",
    "max_steps",
    "log_every",
    "ckpt_every",
    "max_eval_batches",
    "streaming_train",
    "stream_samples_per_worker",
    "max_lines",
    "eval_max_lines",
    "d_model",
    "hidden",
    "nhead",
    "encoder_layers",
    "ffn_dim",
    "dropout",
    "refine_steps",
    "pool_mode",
    "min_span",
    "max_span",
    "max_units",
    "length_loss_weight",
    "boundary_loss_weight",
    "lr",
    "weight_decay",
    "warmup_steps",
    "grad_clip",
    "amp",
    "device",
    "num_workers",
    "seed",
)


def _to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: Any) -> bool:
    return math.isfinite(_to_float(value))


def _fmt(value: Any, digits: int = 4) -> str:
    x = _to_float(value)
    if not math.isfinite(x):
        return "n/a"
    if abs(x) >= 10000 or (0 < abs(x) < 0.0001):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _fmt_int(value: Any) -> str:
    x = _to_float(value)
    if not math.isfinite(x):
        return "n/a"
    return str(int(x))


def _md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _code(value: Any) -> str:
    text = str(value).replace("`", "'")
    return f"`{text}`"


def _mean(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _stdev(values: Sequence[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if len(finite) < 2:
        return 0.0 if finite else float("nan")
    return statistics.pstdev(finite)


def _read_json(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"could not parse {path.name}: {exc}")
        return {}
    if not isinstance(data, dict):
        warnings.append(f"{path.name} is not a JSON object")
        return {}
    return data


def _read_jsonl(path: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        warnings.append(f"missing {LOG_NAME}")
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                warnings.append(f"{LOG_NAME}:{line_no} parse error: {exc}")
                continue
            if not isinstance(row, dict):
                warnings.append(f"{LOG_NAME}:{line_no} is not a JSON object")
                continue
            rows.append(row)
    if not rows:
        warnings.append(f"{LOG_NAME} has no usable rows")
    return rows


def _load_checkpoint_meta(path: Path, warnings: List[str]) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        try:
            ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=torch.device("cpu"))
    except Exception as exc:
        warnings.append(f"could not load {CKPT_NAME} metadata on CPU: {exc}")
        return {"error": str(exc)}

    if not isinstance(ckpt, dict):
        warnings.append(f"{CKPT_NAME} is not a dictionary checkpoint")
        return {}

    args = ckpt.get("args", {})
    summary = ckpt.get("summary", {})
    if not isinstance(args, dict):
        args = {}
    if not isinstance(summary, dict):
        summary = {}
    meta = {
        "args": args,
        "summary": summary,
        "step": ckpt.get("step"),
        "params": ckpt.get("params", summary.get("params")),
        "has_model_state": "model" in ckpt,
    }
    if "model" in ckpt:
        del ckpt["model"]
    return meta


def _is_run_dir(path: Path) -> bool:
    return any((path / name).exists() for name in (LOG_NAME, SUMMARY_NAME, CKPT_NAME))


def _resolve_run_dirs(inputs: Iterable[Path]) -> Tuple[List[Path], List[str]]:
    run_dirs: List[Path] = []
    notes: List[str] = []
    for raw in inputs:
        path = raw.expanduser()
        if _is_run_dir(path):
            run_dirs.append(path)
            continue
        if path.is_dir():
            children = [child for child in sorted(path.iterdir()) if child.is_dir() and _is_run_dir(child)]
            if children:
                run_dirs.extend(children)
                notes.append(f"expanded {path} to {len(children)} child run(s)")
                continue
        run_dirs.append(path)
    return run_dirs, notes


def _series(rows: Sequence[Dict[str, Any]], key: str) -> List[Tuple[Optional[int], float]]:
    values: List[Tuple[Optional[int], float]] = []
    for row in rows:
        value = _to_float(row.get(key))
        if math.isfinite(value):
            step_value = row.get("step")
            step = int(step_value) if _is_finite(step_value) else None
            values.append((step, value))
    return values


def _trend(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = _series(rows, key)
    if not values:
        return {
            "first": float("nan"),
            "final": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "early_avg": float("nan"),
            "late_avg": float("nan"),
            "delta": float("nan"),
            "late_minus_early": float("nan"),
            "min_step": None,
            "max_step": None,
        }
    nums = [value for _, value in values]
    window = max(1, min(5, len(nums) // 4 if len(nums) >= 8 else len(nums)))
    min_idx = min(range(len(values)), key=lambda i: values[i][1])
    max_idx = max(range(len(values)), key=lambda i: values[i][1])
    return {
        "first": nums[0],
        "final": nums[-1],
        "min": nums[min_idx],
        "max": nums[max_idx],
        "mean": _mean(nums),
        "std": _stdev(nums),
        "early_avg": _mean(nums[:window]),
        "late_avg": _mean(nums[-window:]),
        "delta": nums[-1] - nums[0],
        "late_minus_early": _mean(nums[-window:]) - _mean(nums[:window]),
        "min_step": values[min_idx][0],
        "max_step": values[max_idx][0],
    }


def _merged_summary(summary_json: Dict[str, Any], ckpt_meta: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    ckpt_summary = ckpt_meta.get("summary", {})
    if isinstance(ckpt_summary, dict):
        merged.update(ckpt_summary)
    merged.update(summary_json)
    if "params" not in merged and ckpt_meta.get("params") is not None:
        merged["params"] = ckpt_meta.get("params")
    if "steps" not in merged and ckpt_meta.get("step") is not None:
        merged["steps"] = ckpt_meta.get("step")
    return merged


def _summary_eval(summary: Dict[str, Any], key: str) -> float:
    return _to_float(summary.get(f"eval_{key}"))


def _verdict(
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    trends: Dict[str, Dict[str, Any]],
    unit_checks: Dict[str, Any],
    numerical: Dict[str, Any],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    loss = trends["loss"]
    recon = _summary_eval(summary, "recon_acc")
    if not math.isfinite(recon):
        recon = trends["recon_acc"]["final"]
    length = _summary_eval(summary, "length_acc")
    if not math.isfinite(length):
        length = trends["length_acc"]["final"]

    first_loss = loss["first"]
    final_loss = loss["final"]
    enough_loss_points = len(_series(rows, "loss")) >= 2
    loss_decreased = enough_loss_points and math.isfinite(first_loss) and math.isfinite(final_loss) and final_loss < first_loss

    if numerical["has_nonfinite"]:
        reasons.append("non-finite value found in training log")
    if numerical["has_explosion"]:
        reasons.extend(numerical["explosion_reasons"])
    if unit_checks["collapsed"]:
        reasons.append("units_per_byte collapsed")
    if not enough_loss_points:
        reasons.append("not enough loss points to prove a downward trend")
    elif not loss_decreased:
        reasons.append("loss did not decrease from first to final log row")
    else:
        reasons.append(f"loss decreased {_fmt(first_loss)} -> {_fmt(final_loss)}")

    if math.isfinite(recon):
        if recon > 0.5:
            reasons.append(f"recon_acc above threshold: {_fmt(recon)} > 0.5000")
        else:
            reasons.append(f"recon_acc below threshold: {_fmt(recon)} <= 0.5000")
    else:
        reasons.append("recon_acc unavailable")

    if math.isfinite(length):
        if length > 0.7:
            reasons.append(f"length_acc above threshold: {_fmt(length)} > 0.7000")
        else:
            reasons.append(f"length_acc below threshold: {_fmt(length)} <= 0.7000")
    else:
        reasons.append("length_acc unavailable")

    if numerical["has_nonfinite"] or numerical["has_explosion"] or unit_checks["collapsed"]:
        return "FAIL", reasons
    if not rows and not summary:
        return "FAIL", reasons
    if enough_loss_points and not loss_decreased:
        return "FAIL", reasons
    if loss_decreased and math.isfinite(recon) and math.isfinite(length) and recon > 0.5 and length > 0.7:
        return "PASS", reasons
    if loss_decreased and not unit_checks["collapsed"]:
        return "NEEDS_MORE_STEPS", reasons
    return "FAIL", reasons


def _numerical_checks(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    nonfinite: List[str] = []
    for idx, row in enumerate(rows):
        step = row.get("step", idx)
        for key in LOG_NUMERIC_KEYS:
            if key not in row:
                continue
            value = _to_float(row.get(key))
            if not math.isfinite(value):
                nonfinite.append(f"step={step} {key}={row.get(key)!r}")

    loss_values = [value for _, value in _series(rows, "loss")]
    grad_values = [value for _, value in _series(rows, "grad")]
    reasons: List[str] = []
    if loss_values:
        first = loss_values[0]
        final = loss_values[-1]
        max_loss = max(loss_values)
        if any(abs(value) > 1e6 for value in loss_values):
            reasons.append("loss magnitude exceeded 1e6")
        if math.isfinite(first) and first > 0 and math.isfinite(final) and final > max(first * 2.0, first + 1.0):
            reasons.append(f"final loss exploded relative to first loss: {_fmt(first)} -> {_fmt(final)}")
        if math.isfinite(first) and first > 0 and max_loss > max(first * 10.0, 50.0):
            reasons.append(f"max loss spike: {_fmt(max_loss)}")
    if grad_values and max(abs(value) for value in grad_values) > 1e4:
        reasons.append(f"grad norm exceeded 1e4: {_fmt(max(abs(value) for value in grad_values))}")
    return {
        "has_nonfinite": bool(nonfinite),
        "nonfinite_examples": nonfinite[:8],
        "has_explosion": bool(reasons),
        "explosion_reasons": reasons,
    }


def _unit_checks(trend: Dict[str, Any], args: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    values = [value for _, value in _series(args.get("_rows", []), "units_per_byte")]
    final = _summary_eval(summary, "units_per_byte")
    if not math.isfinite(final):
        final = trend["final"]
    mean = trend["mean"]
    std = trend["std"]
    max_span = _to_float(args.get("max_span"))
    expected_full_span = 1.0 / max_span if math.isfinite(max_span) and max_span > 0 else float("nan")

    collapsed = False
    if math.isfinite(final):
        if final <= 0:
            collapsed = True
        elif math.isfinite(expected_full_span):
            collapsed = final < expected_full_span * 0.5
        else:
            first = trend["first"]
            collapsed = final < 0.01 or (math.isfinite(first) and first > 0 and final < first * 0.5)

    cv = std / mean if math.isfinite(std) and math.isfinite(mean) and mean != 0 else float("nan")
    fixed_span_signal = False
    if math.isfinite(expected_full_span) and math.isfinite(mean) and expected_full_span > 0:
        near_full_span = abs(mean - expected_full_span) / expected_full_span <= 0.15
        low_variation = math.isfinite(cv) and cv <= 0.05
        fixed_span_signal = near_full_span and low_variation

    return {
        "final": final,
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": trend["min"],
        "max": trend["max"],
        "expected_full_span": expected_full_span,
        "collapsed": collapsed,
        "fixed_span_signal": fixed_span_signal,
        "count": len(values),
    }


def analyze_run(path: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    rows = _read_jsonl(path / LOG_NAME, warnings)
    summary_json = _read_json(path / SUMMARY_NAME, warnings)
    if not (path / SUMMARY_NAME).exists():
        warnings.append(f"missing {SUMMARY_NAME}")
    ckpt_meta = _load_checkpoint_meta(path / CKPT_NAME, warnings)
    if not (path / CKPT_NAME).exists():
        warnings.append(f"missing {CKPT_NAME}")

    args = ckpt_meta.get("args", {})
    if not isinstance(args, dict):
        args = {}
    args_with_rows = dict(args)
    args_with_rows["_rows"] = rows
    summary = _merged_summary(summary_json, ckpt_meta)

    trends = {key: _trend(rows, key) for key in ("loss", *ACCURACY_KEYS, "units_per_byte", "grad")}
    unit_checks = _unit_checks(trends["units_per_byte"], args_with_rows, summary)
    numerical = _numerical_checks(rows)
    verdict, reasons = _verdict(rows, summary, trends, unit_checks, numerical)

    return {
        "path": path,
        "rows": rows,
        "summary_json": summary_json,
        "summary": summary,
        "ckpt_meta": ckpt_meta,
        "args": args,
        "trends": trends,
        "unit_checks": unit_checks,
        "numerical": numerical,
        "warnings": warnings,
        "verdict": verdict,
        "reasons": reasons,
    }


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _overview_rows(analyses: Sequence[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for item in analyses:
        summary = item["summary"]
        trends = item["trends"]
        recon = _summary_eval(summary, "recon_acc")
        if not math.isfinite(recon):
            recon = trends["recon_acc"]["final"]
        length = _summary_eval(summary, "length_acc")
        if not math.isfinite(length):
            length = trends["length_acc"]["final"]
        boundary = _summary_eval(summary, "boundary_acc")
        if not math.isfinite(boundary):
            boundary = trends["boundary_acc"]["final"]
        unit_final = item["unit_checks"]["final"]
        rows.append(
            [
                _md_escape(item["path"].name),
                item["verdict"],
                _fmt_int(summary.get("steps", item["ckpt_meta"].get("step"))),
                str(len(item["rows"])),
                f"{_fmt(trends['loss']['first'])} -> {_fmt(trends['loss']['final'])}",
                _fmt(trends["loss"]["min"]),
                _fmt(recon),
                _fmt(length),
                _fmt(boundary),
                _fmt(unit_final),
                _fmt(summary.get("train_steps_per_sec")),
                _fmt(summary.get("train_samples_per_sec")),
                _fmt(summary.get("max_memory_allocated_mb")),
            ]
        )
    return rows


def _config_rows(item: Dict[str, Any]) -> List[List[str]]:
    args = item["args"]
    summary = item["summary"]
    ckpt = item["ckpt_meta"]
    rows: List[List[str]] = [
        ["params", _md_escape(summary.get("params", ckpt.get("params", "n/a")))],
        ["steps", _md_escape(summary.get("steps", ckpt.get("step", "n/a")))],
    ]
    for key in (
        "train_elapsed_sec",
        "train_steps_per_sec",
        "train_samples_per_sec",
        "train_bytes_per_sec",
        "max_memory_allocated_mb",
    ):
        if key in summary:
            rows.append([key, _md_escape(summary[key])])
    for key in CONFIG_KEYS:
        if key in args:
            rows.append([key, _md_escape(args[key])])
    return rows


def _metric_trend_rows(item: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    summary = item["summary"]
    for key in ("loss", *ACCURACY_KEYS):
        trend = item["trends"][key]
        eval_value = _summary_eval(summary, key)
        rows.append(
            [
                key,
                _fmt(trend["first"]),
                _fmt(trend["final"]),
                _fmt(trend["min"]),
                _fmt(trend["max"]),
                _fmt(trend["early_avg"]),
                _fmt(trend["late_avg"]),
                _fmt(trend["delta"]),
                _fmt(eval_value),
            ]
        )
    return rows


def render_markdown(analyses: Sequence[Dict[str, Any]], notes: Sequence[str]) -> str:
    lines: List[str] = [
        "# FLUED v3.1 Language Codec Diagnostics",
        "",
    ]
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        _table(
            (
                "run",
                "verdict",
                "steps",
                "log_rows",
                "loss first->final",
                "min_loss",
                "recon",
                "length",
                "boundary",
                "units/byte",
                "steps/s",
                "samples/s",
                "max_mem_mb",
            ),
            _overview_rows(analyses),
        )
    )
    lines.append("")

    for item in analyses:
        path = item["path"]
        lines.extend([f"## {_md_escape(path.name)}", "", f"Path: {_code(path)}", ""])

        lines.extend(["### Verdict", ""])
        lines.append(f"**{item['verdict']}**")
        lines.append("")
        for reason in item["reasons"]:
            lines.append(f"- {reason}")
        if item["unit_checks"]["fixed_span_signal"]:
            lines.append("- possible fixed-span shortcut signal: units_per_byte is near 1/max_span with low variation")
        length_eval = _summary_eval(item["summary"], "length_acc")
        recon_eval = _summary_eval(item["summary"], "recon_acc")
        if not math.isfinite(length_eval):
            length_eval = item["trends"]["length_acc"]["final"]
        if not math.isfinite(recon_eval):
            recon_eval = item["trends"]["recon_acc"]["final"]
        if math.isfinite(length_eval) and math.isfinite(recon_eval) and length_eval > 0.7 and recon_eval <= 0.5:
            lines.append("- length_head is learning faster than reconstruction; treat high length_acc alone as insufficient")
        lines.append("")

        lines.extend(["### Config", ""])
        lines.extend(_table(("key", "value"), _config_rows(item)))
        lines.append("")

        lines.extend(["### Loss And Accuracy Trends", ""])
        lines.extend(
            _table(
                (
                    "metric",
                    "first",
                    "final",
                    "min",
                    "max",
                    "early_avg",
                    "late_avg",
                    "delta",
                    "eval",
                ),
                _metric_trend_rows(item),
            )
        )
        min_step = item["trends"]["loss"]["min_step"]
        if min_step is not None:
            lines.append("")
            lines.append(f"Minimum logged loss occurs at step `{min_step}`.")
        lines.append("")

        unit = item["unit_checks"]
        lines.extend(["### Units Per Byte", ""])
        lines.extend(
            _table(
                ("mean", "std", "cv", "min", "max", "final/eval", "1/max_span", "collapsed", "fixed_span_signal"),
                (
                    (
                        _fmt(unit["mean"]),
                        _fmt(unit["std"]),
                        _fmt(unit["cv"]),
                        _fmt(unit["min"]),
                        _fmt(unit["max"]),
                        _fmt(unit["final"]),
                        _fmt(unit["expected_full_span"]),
                        str(unit["collapsed"]),
                        str(unit["fixed_span_signal"]),
                    ),
                ),
            )
        )
        lines.append("")

        numerical = item["numerical"]
        lines.extend(["### Numerical Health", ""])
        lines.append(f"- non_finite: `{numerical['has_nonfinite']}`")
        lines.append(f"- explosion: `{numerical['has_explosion']}`")
        for example in numerical["nonfinite_examples"]:
            lines.append(f"- non_finite_example: `{_md_escape(example)}`")
        for reason in numerical["explosion_reasons"]:
            lines.append(f"- explosion_reason: {_md_escape(reason)}")
        if not numerical["has_nonfinite"] and not numerical["has_explosion"]:
            lines.append("- no NaN/Inf/explosion signal found in parsed log rows")
        lines.append("")

        if item["warnings"]:
            lines.extend(["### Warnings", ""])
            for warning in item["warnings"]:
                lines.append(f"- {_md_escape(warning)}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize FLUED-v3.1 language codec runs")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directory or parent containing child run directories")
    parser.add_argument("--out-path", type=Path, default=None, help="Write markdown report here instead of stdout")
    args = parser.parse_args(argv)

    run_dirs, notes = _resolve_run_dirs(args.run_dirs)
    analyses = [analyze_run(path) for path in run_dirs]
    report = render_markdown(analyses, notes)

    if args.out_path:
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        args.out_path.write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
