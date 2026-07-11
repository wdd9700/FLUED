"""Summarize FLUED-v3.2 minimal-backbone runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SUMMARY_NAME = "summary.json"
LOG_NAME = "train_log.jsonl"


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


def _code(value: Any) -> str:
    text = str(value).replace("`", "'")
    return f"`{text}`"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_last_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                last = row
    return last


def _is_run_dir(path: Path) -> bool:
    return (path / SUMMARY_NAME).exists() or (path / LOG_NAME).exists()


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


def load_run(path: Path) -> Dict[str, Any]:
    summary = _read_json(path / SUMMARY_NAME)
    last = _read_last_log(path / LOG_NAME)
    return {
        "path": path,
        "name": path.name,
        "summary": summary,
        "last": last,
    }


def _is_strict_masked_source(summary: Dict[str, Any]) -> bool:
    return summary.get("task") == "strict_masked_source"


def verdict(run: Dict[str, Any], best_byte_baseline: float | None) -> str:
    s = run["summary"]
    mode = s.get("mode", "")
    acc = float(s.get("eval_mask_byte_acc", float("nan")))
    if not math.isfinite(acc):
        return "FAIL"
    if mode == "byte":
        if _is_strict_masked_source(s):
            return "BYTE_MASKED_SOURCE_BASELINE"
        if s.get("eval_masked_units", 0):
            return "BYTE_SEGMENT_BASELINE"
        return "BYTE_RANDOM_REFERENCE"
    strict = _is_strict_masked_source(s)
    if best_byte_baseline is None:
        return "LATENT_NO_BASELINE"
    if acc > best_byte_baseline + 0.01:
        return "LATENT_BEATS_BYTE_MASKED_SOURCE" if strict else "LATENT_BEATS_BYTE_SEGMENT"
    if acc >= best_byte_baseline - 0.005:
        return "TIED_WITH_BYTE_MASKED_SOURCE" if strict else "TIED_WITH_BYTE_SEGMENT"
    return "LATENT_BELOW_BYTE_MASKED_SOURCE" if strict else "LATENT_BELOW_BYTE_SEGMENT"


def render(runs: List[Dict[str, Any]], notes: List[str]) -> str:
    byte_baseline_accs = [
        float(run["summary"].get("eval_mask_byte_acc"))
        for run in runs
        if run["summary"].get("mode") == "byte"
        and (
            _is_strict_masked_source(run["summary"])
            or float(run["summary"].get("eval_masked_units", 0) or 0) > 0
        )
    ]
    best_byte_baseline = max(byte_baseline_accs) if byte_baseline_accs else None
    has_strict = any(_is_strict_masked_source(run["summary"]) for run in runs)

    lines: List[str] = ["# FLUED v3.2 Minimal Backbone Summary", ""]
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "| run | verdict | mode | steps | params | mask_acc | keep_acc | mask_frac | loss | steps/s | byte_aux | codec |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run in runs:
        s = run["summary"]
        keep = s.get("eval_keep_byte_acc", s.get("eval_keep_byte_acc_copy", s.get("eval_keep_byte_acc_model")))
        lines.append(
            "| "
            + " | ".join(
                [
                    run["name"],
                    verdict(run, best_byte_baseline),
                    str(s.get("mode", "n/a")),
                    _fmt(s.get("steps"), 0),
                    _fmt(s.get("params"), 0),
                    _fmt(s.get("eval_mask_byte_acc")),
                    _fmt(keep),
                    _fmt(s.get("eval_masked_byte_fraction")),
                    _fmt(s.get("eval_loss")),
                    _fmt(s.get("train_steps_per_sec")),
                    _fmt(s.get("eval_byte_loss_aux")),
                    "yes" if s.get("codec_ckpt") else "no",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `BYTE_RANDOM_REFERENCE` is a legacy clean-source byte reference and is not a strict masked-source baseline.")
    lines.append("- `BYTE_SEGMENT_BASELINE` is legacy segment-level masking over clean segmentation.")
    lines.append("- `BYTE_MASKED_SOURCE_BASELINE` masks byte/span positions before any FLUED encode; this is the strict baseline.")
    if has_strict:
        lines.append("- Strict latent runs must beat `BYTE_MASKED_SOURCE_BASELINE`; clean segment/readout results are historical only.")
    else:
        lines.append("- Legacy `LATENT_BEATS_BYTE_SEGMENT` only means the old clean-readout task improved over the old segment baseline.")
    lines.append("- Latent runs still decode through the frozen FLUED decoder; the backbone receives readout latent only, not raw FLUED memory tensors.")
    lines.append("- In memory-enabled codec runs, that readout may still be memory-conditioned inside FLUED.")
    lines.append("")
    if best_byte_baseline is not None:
        label = "masked-source" if has_strict else "segment"
        lines.append(f"Best byte {label} baseline mask_acc: `{best_byte_baseline:.4f}`.")
        lines.append("")

    best_latent = None
    for run in runs:
        s = run["summary"]
        if s.get("mode") != "latent":
            continue
        acc = float(s.get("eval_mask_byte_acc", float("nan")))
        if math.isfinite(acc) and (best_latent is None or acc > best_latent[1]):
            best_latent = (run, acc)
    if best_latent is not None:
        lines.append(f"Best latent run: `{best_latent[0]['name']}` with mask_acc `{best_latent[1]:.4f}`.")
        lines.append("")

    lines.append("## Run Paths")
    lines.append("")
    for run in runs:
        lines.append(f"- {run['name']}: {_code(run['path'])}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v3.2 minimal-backbone runs")
    parser.add_argument("runs", nargs="+", help="Run directories or parent directories")
    parser.add_argument("--out-path", default="")
    args = parser.parse_args()

    run_dirs, notes = resolve_runs(Path(x) for x in args.runs)
    runs = [load_run(path) for path in run_dirs]
    text = render(runs, notes)
    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({"out_path": str(out), "runs": len(runs)}, ensure_ascii=False))
    else:
        print(text)


if __name__ == "__main__":
    main()
