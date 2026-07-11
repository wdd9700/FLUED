"""Summarize strict masked-source backbone runs for FLUED v3-family codecs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SUMMARY_NAME = "summary.json"


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    if digits == 0:
        return str(int(round(x)))
    if abs(x) >= 10000 or (0 < abs(x) < 0.0001):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_run_dir(path: Path) -> bool:
    return (path / SUMMARY_NAME).exists()


def resolve_runs(paths: Iterable[Path]) -> Tuple[List[Path], List[str]]:
    runs: List[Path] = []
    notes: List[str] = []
    for path in paths:
        path = path.expanduser()
        if _is_run_dir(path):
            runs.append(path)
            continue
        if path.is_dir():
            children = [child for child in sorted(path.iterdir()) if child.is_dir() and _is_run_dir(child)]
            if children:
                notes.append(f"expanded {path} to {len(children)} child run(s)")
                runs.extend(children)
                continue
        runs.append(path)
    return runs, notes


def load_rows(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = _read_json(run_dir / SUMMARY_NAME)
        row = {
            "run": run_dir.name,
            "path": str(run_dir),
            "mode": summary.get("mode", ""),
            "family": summary.get("codec_family", ""),
            "codec_name": summary.get("codec_name", ""),
            "codec_ckpt": summary.get("codec_ckpt", ""),
            "codec_memory_enabled": summary.get("codec_memory_enabled", False),
            "codec_pool_mode": summary.get("codec_pool_mode", ""),
            "codec_memory_retrieval_mode": summary.get("codec_memory_retrieval_mode", ""),
            "steps": summary.get("steps"),
            "params": summary.get("params"),
            "mask_acc": summary.get("eval_mask_byte_acc"),
            "byte_ce": summary.get("eval_byte_loss", summary.get("eval_loss")),
            "loss": summary.get("eval_loss"),
            "keep_acc": summary.get("eval_keep_byte_acc", summary.get("eval_keep_byte_acc_copy", summary.get("eval_keep_byte_acc_model"))),
            "length_acc": summary.get("eval_mask_length_acc"),
            "mask_frac": summary.get("eval_masked_byte_fraction"),
            "units_per_byte": summary.get("eval_units_per_byte"),
            "steps_per_sec": summary.get("train_steps_per_sec"),
        }
        rows.append(row)
    return rows


def _float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def add_deltas(rows: List[Dict[str, Any]]) -> Tuple[float | None, float | None]:
    byte_rows = [row for row in rows if row.get("mode") == "byte" and math.isfinite(_float(row.get("mask_acc")))]
    if not byte_rows:
        for row in rows:
            row["delta_acc_vs_byte"] = None
            row["delta_ce_vs_byte"] = None
        return None, None
    baseline = max(byte_rows, key=lambda row: _float(row.get("mask_acc")))
    byte_acc = _float(baseline.get("mask_acc"))
    byte_ce = _float(baseline.get("byte_ce"))
    for row in rows:
        acc = _float(row.get("mask_acc"))
        ce = _float(row.get("byte_ce"))
        row["delta_acc_vs_byte"] = acc - byte_acc if math.isfinite(acc) else None
        row["delta_ce_vs_byte"] = byte_ce - ce if math.isfinite(byte_ce) and math.isfinite(ce) else None
    return byte_acc, byte_ce


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "run",
        "mode",
        "family",
        "codec_name",
        "codec_memory_enabled",
        "codec_pool_mode",
        "codec_memory_retrieval_mode",
        "steps",
        "params",
        "mask_acc",
        "delta_acc_vs_byte",
        "byte_ce",
        "delta_ce_vs_byte",
        "keep_acc",
        "length_acc",
        "mask_frac",
        "units_per_byte",
        "steps_per_sec",
        "path",
        "codec_ckpt",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def render_md(rows: List[Dict[str, Any]], notes: List[str], byte_acc: float | None, byte_ce: float | None) -> str:
    rows_sorted = sorted(rows, key=lambda row: (_float(row.get("mask_acc")) if row.get("mode") != "byte" else -1.0), reverse=True)
    lines: List[str] = ["# FLUED v3 Strict Masked-Source Backbone Full Table", ""]
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("This table trains the same small infill backbone for each archived codec checkpoint. Masking happens on input bytes before FLUED sees the sequence.")
    lines.append("")
    if byte_acc is not None:
        lines.append(f"Byte baseline: mask_acc `{byte_acc:.4f}`, CE `{byte_ce:.4f}`.")
        lines.append("")
    lines.extend(
        [
            "| rank | run | family | memory | pool | steps | mask_acc | delta_acc | byte_CE | delta_CE | keep_acc | len_acc | mask_frac | steps/s |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    rank = 0
    for row in rows_sorted:
        if row.get("mode") == "byte":
            label = "byte_baseline"
        else:
            rank += 1
            label = str(rank)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row.get("run", "")),
                    str(row.get("family", "")),
                    str(row.get("codec_memory_enabled", "")),
                    str(row.get("codec_pool_mode", "")),
                    _fmt(row.get("steps"), 0),
                    _fmt(row.get("mask_acc")),
                    _fmt(row.get("delta_acc_vs_byte")),
                    _fmt(row.get("byte_ce")),
                    _fmt(row.get("delta_ce_vs_byte")),
                    _fmt(row.get("keep_acc")),
                    _fmt(row.get("length_acc")),
                    _fmt(row.get("mask_frac")),
                    _fmt(row.get("steps_per_sec")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- `delta_acc` is latent mask accuracy minus byte-baseline mask accuracy; positive means the frozen FLUED readout made the small backbone's masked-byte task easier.")
    lines.append("- `delta_CE` is byte-baseline CE minus latent CE; positive means lower cross entropy than the byte baseline.")
    lines.append("- Memory-enabled codecs may produce memory-conditioned readout internally, but the external backbone only receives readout latents.")
    lines.append("")
    lines.append("## Run Paths")
    lines.append("")
    for row in sorted(rows, key=lambda item: str(item.get("run", ""))):
        lines.append(f"- `{row.get('run')}`: `{row.get('path')}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v3 strict masked-source backbone sweep")
    parser.add_argument("runs", nargs="+", help="Run directories or parent directories")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    run_dirs, notes = resolve_runs(Path(x) for x in args.runs)
    rows = load_rows(run_dirs)
    byte_acc, byte_ce = add_deltas(rows)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.out_md).parent if args.out_md else Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "strict_backbone_full_table.csv")
    (out_dir / "strict_backbone_full_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_md(rows, notes, byte_acc, byte_ce)
    md_path = Path(args.out_md) if args.out_md else out_dir / "strict_backbone_full_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"runs": len(rows), "out_md": str(md_path), "out_csv": str(out_dir / "strict_backbone_full_table.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
