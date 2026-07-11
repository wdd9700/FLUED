"""Summarize v3.4 ablations with a controlled within-chunk order probe."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID, text_to_byte_ids  # noqa: E402
from tools.train.train_v34_pos_ar_probe import build_model  # noqa: E402


def _relative_delta(x: torch.Tensor, y: torch.Tensor) -> float:
    denom = max(0.5 * (x.float().norm().item() + y.float().norm().item()), 1.0e-9)
    return float((x - y).float().norm().item() / denom)


@torch.no_grad()
def controlled_local_order_probe(model) -> dict[str, float]:
    """Hold chunk membership fixed and change only within-chunk byte order."""

    def local(text: str):
        ids = torch.tensor([text_to_byte_ids(text)], dtype=torch.long)
        lookup = model.byte_lookup if model.config.use_structured_lookup else model.plain_byte_lookup
        x = lookup(ids)
        width = model.config.max_span
        spans = x.new_zeros((1, 1, width, x.size(-1)))
        mask = torch.zeros((1, 1, width), dtype=torch.bool)
        spans[:, :, : x.size(1)] = x.unsqueeze(1)
        mask[:, :, : x.size(1)] = True
        memory = model.memory_pool(spans, mask)
        readout = model.readout_pool(spans, mask)
        if model.config.use_ar:
            memory, readout, _ = model.ar(spans, mask, memory, readout)
        return memory, readout

    base_m, base_r = local("reads")
    swap_m, swap_r = local("raeds")
    subst_m, subst_r = local("rxyds")
    memory_swap = _relative_delta(base_m, swap_m)
    memory_subst = _relative_delta(base_m, subst_m)
    readout_swap = _relative_delta(base_r, swap_r)
    readout_subst = _relative_delta(base_r, subst_r)
    return {
        "local_memory_swap_delta": memory_swap,
        "local_memory_substitute_delta": memory_subst,
        "local_memory_swap_to_substitute": memory_swap / max(memory_subst, 1.0e-9),
        "local_readout_swap_delta": readout_swap,
        "local_readout_substitute_delta": readout_subst,
        "local_readout_swap_to_substitute": readout_swap / max(readout_subst, 1.0e-9),
    }


def _args_with_defaults(raw: dict) -> Namespace:
    defaults = {
        "use_structured_lookup": True,
        "use_memory": True,
        "use_logic_prior": True,
        "use_boundary_bridge": True,
        "boundary_temperature": 0.15,
    }
    return Namespace(**{**defaults, **raw})


def _curve_stats(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tail = rows[-5:]
    result = {}
    for key in ("identity_acc", "completion_mask_acc", "requested_hard_cut_fraction", "hard_cut_fraction"):
        result[f"tail_{key}"] = sum(float(row[key]) for row in tail) / len(tail)
    for threshold in (0.10, 0.25, 0.50):
        hit = next((int(row["step"]) for row in rows if row["requested_hard_cut_fraction"] >= threshold), -1)
        result[f"first_requested_cut_ge_{threshold:.2f}"] = hit
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for root_text in args.roots:
        for checkpoint in sorted(Path(root_text).glob("*/latest.pt")):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(_args_with_defaults(payload["args"]))
            model.load_state_dict(payload["model"], strict=False)
            model.eval()
            summary = dict(payload.get("summary") or json.loads((checkpoint.parent / "summary.json").read_text()))
            row = {
                "run": checkpoint.parent.name,
                "checkpoint": str(checkpoint),
                **summary,
                **controlled_local_order_probe(model),
                **_curve_stats(checkpoint.parent / "train_log.jsonl"),
            }
            rows.append(row)
    (output / "v34_ablation_full.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = [
        "run",
        "eval_identity_acc",
        "eval_completion_mask_acc",
        "eval_completion_preserve_acc",
        "eval_requested_hard_cut_fraction",
        "eval_hard_cut_fraction",
        "local_memory_swap_to_substitute",
        "local_readout_swap_to_substitute",
        "eval_memory_gate_mean",
        "train_steps_per_sec",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = [row["run"]] + [f"{float(row[key]):.4f}" for key in columns[1:]]
        lines.append("| " + " | ".join(values) + " |")
    (output / "v34_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
