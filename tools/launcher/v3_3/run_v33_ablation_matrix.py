"""Run a FLUED v3.3 ablation matrix from one JSON file.

The matrix file has three top-level fields:

{
  "matrix_name": "v33_2m_core",
  "base": {"seq_len": 128, "...": "..."},
  "experiments": [
    {"id": "no_memory_codec", "overrides": {"use_memory": false}},
    {"id": "memory_rank4_codec", "overrides": {"use_memory": true, "memory_rank": 4}}
  ]
}

The launcher writes a resolved config into each run directory and calls
tools/train/v3_3/train_v33.py.  It deliberately stays thin: train_v33.py remains the
single source of truth for model/training arguments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train" / "v3_3" / "train_v33.py"


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _select(experiments: Iterable[Dict[str, Any]], only: str) -> List[Dict[str, Any]]:
    rows = list(experiments)
    if not only:
        return rows
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    return [row for row in rows if str(row.get("id", "")) in wanted]


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    next_config = dict(config)
    if args.data_path:
        next_config["data_path"] = args.data_path
    if args.device:
        next_config["device"] = args.device
    if args.batch_size is not None:
        next_config["batch_size"] = args.batch_size
    if args.max_steps is not None:
        next_config["max_steps"] = args.max_steps
    if args.num_workers is not None:
        next_config["num_workers"] = args.num_workers
    if args.amp is not None:
        next_config["amp"] = bool(args.amp)
    return next_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FLUED v3.3 ablation matrix")
    parser.add_argument("--matrix", default="configs/v3_3/v33_ablation_2m.json")
    parser.add_argument("--out-root", default="checkpoints")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", default="", help="Comma-separated experiment ids")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    matrix_path = (REPO_ROOT / args.matrix).resolve() if not Path(args.matrix).is_absolute() else Path(args.matrix)
    matrix = _load_json(matrix_path)
    matrix_name = str(matrix.get("matrix_name") or matrix_path.stem)
    base = matrix.get("base", {})
    experiments = matrix.get("experiments", [])
    if not isinstance(base, dict) or not isinstance(experiments, list):
        raise ValueError("matrix JSON requires object field 'base' and list field 'experiments'")

    selected = _select(experiments, args.only)
    if not selected:
        raise SystemExit("No experiments selected")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = (REPO_ROOT / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    print(f"matrix={matrix_name} experiments={len(selected)} out_root={out_root}", flush=True)

    for row in selected:
        exp_id = str(row.get("id", "")).strip()
        if not exp_id:
            raise ValueError("Each experiment must have a non-empty id")
        overrides = row.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Experiment {exp_id} overrides must be an object")

        config = _merge(base, overrides)
        config = _apply_cli_overrides(config, args)
        run_dir = out_root / matrix_name / exp_id
        config["experiment_name"] = matrix_name
        config["run_id"] = exp_id
        config["out_dir"] = str(run_dir)

        summary_path = run_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[skip] {exp_id}: summary exists at {summary_path}", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        generated_config = run_dir / f"resolved_input_{timestamp}.json"
        generated_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = [args.python, str(TRAIN_SCRIPT), "--config", str(generated_config)]
        print("[run] " + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
