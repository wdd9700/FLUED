"""Summarize FLUED v3.3 ablation runs into JSON and Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


DISPLAY_KEYS = [
    "run_id",
    "params",
    "backbone_params",
    "steps",
    "steps_per_sec",
    "eval_loss",
    "eval_decoder_mask_acc",
    "eval_decoder_visible_acc",
    "eval_backbone_mask_acc",
    "eval_leakage_gap",
    "eval_readout_units_per_byte",
    "eval_length_acc",
    "eval_boundary_loss",
    "eval_rate_loss",
]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.1f}"
        return f"{value:.4f}"
    return str(value)


def _read_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    args = data.get("args", {})
    if isinstance(args, dict):
        data["use_memory"] = args.get("use_memory")
        data["use_backbone"] = args.get("use_backbone")
        data["mask_prob"] = args.get("mask_prob")
        data["target_rate"] = args.get("target_rate")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FLUED v3.3 ablation runs")
    parser.add_argument("--root", default="checkpoints/v33_2m_core")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    root = (REPO_ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    summaries = sorted(root.glob("*/summary.json"))
    rows: List[Dict[str, Any]] = [_read_summary(path) for path in summaries]
    if not rows:
        raise SystemExit(f"No summary.json files found under {root}")

    out_json = Path(args.out_json) if args.out_json else root / "ablation_summary.json"
    out_md = Path(args.out_md) if args.out_md else root / "ablation_summary.md"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    header = ["run_id", "memory", "backbone", "mask_prob", "target_rate", *DISPLAY_KEYS[1:]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [
            row.get("run_id", ""),
            row.get("use_memory", ""),
            row.get("use_backbone", ""),
            row.get("mask_prob", ""),
            row.get("target_rate", ""),
            *[row.get(key, "") for key in DISPLAY_KEYS[1:]],
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
