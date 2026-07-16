"""Run the long-horizon v3.4 attribution matrices and post-run probes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_LAUNCHER = REPO_ROOT / "tools" / "launcher" / "v3_4" / "run_v34_pos_ar_matrix.py"
CURVE_ANALYZER = REPO_ROOT / "tools" / "analysis" / "v3_4" / "analyze_v34_5k_curves.py"
MEMORY_PROBE = REPO_ROOT / "tools" / "analysis" / "v3_4" / "probe_v34_memory_interventions.py"


def write_state(path: Path, **values: object) -> None:
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    previous.update(values)
    previous["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")


def run_logged(cmd: list[str], log_path: Path, state_path: Path, stage: str) -> None:
    write_state(state_path, status="running", stage=stage, command=cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {stage}\n")
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] END {stage} "
            f"exit={process.returncode}\n"
        )
    if process.returncode:
        write_state(state_path, status="failed", stage=stage, exit_code=process.returncode)
        raise subprocess.CalledProcessError(process.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--root", default=r"L:\FLUED_archive\v34_attribution_matrices_20260715")
    parser.add_argument("--skip-post-analysis", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    boundary_root = root / "boundary_schedule_40k"
    memory_root = root / "memory_usage_20k"
    analysis_root = root / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "orchestrator_state.json"

    matrices = (
        (
            "boundary_schedule_40k",
            "configs/v3_4/v34_boundary_schedule_40k_attribution.json",
            boundary_root,
        ),
        (
            "memory_usage_20k",
            "configs/v3_4/v34_memory_usage_supervision_20k.json",
            memory_root,
        ),
    )
    write_state(state_path, status="running", stage="initializing", root=str(root))
    for stage, matrix, out_root in matrices:
        run_logged(
            [
                args.python,
                str(MATRIX_LAUNCHER),
                "--matrix",
                matrix,
                "--out-root",
                str(out_root),
            ],
            root / f"{stage}.log",
            state_path,
            stage,
        )

    if not args.skip_post_analysis:
        for stage, run_root in (
            ("analyze_boundary_curves", boundary_root),
            ("analyze_memory_curves", memory_root),
        ):
            run_logged(
                [
                    args.python,
                    str(CURVE_ANALYZER),
                    str(run_root),
                    "--out-dir",
                    str(analysis_root / stage),
                ],
                root / "post_analysis.log",
                state_path,
                stage,
            )

        for run_dir in sorted(path for path in memory_root.iterdir() if path.is_dir()):
            checkpoint = run_dir / "latest.pt"
            summary = run_dir / "summary.json"
            if not checkpoint.exists() or not summary.exists():
                continue
            stage = f"memory_intervention_{run_dir.name}"
            run_logged(
                [
                    args.python,
                    str(MEMORY_PROBE),
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(analysis_root / stage),
                    "--max-eval-batches",
                    "32",
                ],
                root / "post_analysis.log",
                state_path,
                stage,
            )

    write_state(state_path, status="complete", stage="complete")


if __name__ == "__main__":
    main()
