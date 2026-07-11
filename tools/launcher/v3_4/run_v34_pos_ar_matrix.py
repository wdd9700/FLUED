"""Run the four FLUED v3.4 position x AR probe experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN = REPO_ROOT / "tools" / "train" / "v3_4" / "train_v34_pos_ar_probe.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/v3_4/v34_pos_ar_40m_probe.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-root", default="outputs/v34_pos_ar_40m_probe")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--only", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()
    matrix = json.loads((REPO_ROOT / args.matrix).read_text(encoding="utf-8"))
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    for experiment in matrix["experiments"]:
        if wanted and experiment["id"] not in wanted:
            continue
        config = {**matrix["base"], **experiment["overrides"]}
        if args.max_steps is not None:
            config["max_steps"] = args.max_steps
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        run_dir = Path(args.out_root) / experiment["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and not args.rerun_complete and not args.dry_run:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(previous.get("steps", 0)) >= int(config["max_steps"]):
                print(f"[skip-complete] {experiment['id']} steps={previous['steps']}", flush=True)
                continue
        config.update({"run_id": experiment["id"], "out_dir": str(run_dir)})
        resolved = run_dir / "resolved_input.json"
        resolved.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = [args.python, str(TRAIN), "--config", str(resolved)]
        if args.dry_run:
            cmd.append("--dry-run")
        print("[run] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
