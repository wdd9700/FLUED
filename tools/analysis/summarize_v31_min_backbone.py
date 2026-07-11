"""Summarize FLUED-v3.1 minimal-backbone runs."""

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


def verdict(run: Dict[str, Any], best_byte_segment: float | None) -> str:
    s = run["summary"]
    mode = s.get("mode", "")
    acc = float(s.get("eval_mask_byte_acc", float("nan")))
    if not math.isfinite(acc):
        return "FAIL"
    if mode == "byte":
        if s.get("eval_masked_units", 0):
            return "BYTE_SEGMENT_BASELINE"
        return "BYTE_RANDOM_REFERENCE"
    if best_byte_segment is None:
        return "LATENT_NO_BASELINE"
    if acc > best_byte_segment + 0.01:
        return "LATENT_BEATS_BYTE_SEGMENT"
    if acc >= best_byte_segment - 0.005:
        return "TIED_WITH_BYTE_SEGMENT"
    return "LATENT_BELOW_BYTE_SEGMENT"


def render(runs: List[Dict[str, Any]], notes: List[str]) -> str:
    byte_segment_accs = [
        float(run["summary"].get("eval_mask_byte_acc"))
        for run in runs
        if run["summary"].get("mode") == "byte" and float(run["summary"].get("eval_masked_units", 0) or 0) > 0
    ]
    best_byte_segment = max(byte_segment_accs) if byte_segment_accs else None

    lines: List[str] = ["# FLUED v3.1 Minimal Backbone Summary", ""]
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
                    verdict(run, best_byte_segment),
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
    lines.append("- `BYTE_RANDOM_REFERENCE` masks individual bytes and is easier than segment-level infill.")
    lines.append("- `BYTE_SEGMENT_BASELINE` masks the same kind of contiguous segment spans used by the latent backbone.")
    lines.append("- `LATENT_BEATS_BYTE_SEGMENT` is the first useful signal that readout latents can reduce backbone burden for this task.")
    lines.append("- Latent runs still decode through the frozen FLUED decoder; the backbone never receives FLUED memory.")
    lines.append("")
    if best_byte_segment is not None:
        lines.append(f"Best byte segment baseline mask_acc: `{best_byte_segment:.4f}`.")
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
    parser = argparse.ArgumentParser(description="Summarize v3.1 minimal-backbone runs")
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
