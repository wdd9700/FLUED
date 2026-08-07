"""Eval S1.0-era milestone checkpoints on the FIXED eval stream: fair
metric-vs-step trajectories for curve-shape analysis (end-slope = ceiling
signal; matched-step scores = current-pace comparison).

All runs share the canonical data/eval config (same corpus, seed,
eval_mask_seed), so the 128 eval windows and their masks are identical
across runs; each run's own resolved_config.json drives the model shape
(incl. backbone_readout for k=1 arms).

Usage:
  python eval_s1_milestones.py --data-path <corpus> run_dir_a [run_dir_b ...]
  -> writes <run_dir>/milestone_evals.jsonl (one row per milestone)
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import make_dataloaders  # noqa: E402
from tools.train.v3_6.train_v36 import build_model  # noqa: E402
import tools.train.v3_6.train_v36_s1 as s1  # noqa: E402

KEEP = (
    "eval_direct_acc",
    "eval_backbone_acc",
    "eval_backbone_unmasked_acc",
    "eval_backbone_masked_acc",
    "eval_predict_cos",
    "eval_state_norm",
    "eval_direct_ppl",
    "eval_backbone_ppl",
)


def eval_run(run_dir: Path, data_path: str, device: torch.device) -> None:
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg["data_path"] = data_path
    cfg["data_manifest"] = ""
    ns = Namespace(**cfg)
    if s1._BPE_TOKENIZER is None and getattr(ns, "mask_mode", "mixed") == "mixed":
        s1._load_bpe_tokenizer(ns.bpe_tokenizer_path)
    model = build_model(ns).to(device)
    _, eval_loader = make_dataloaders(ns)

    ckpts = sorted(run_dir.glob("step_*.pt")) + [run_dir / "latest.pt"]
    out_path = run_dir / "milestone_evals.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for ck in ckpts:
            if not ck.exists():
                continue
            payload = torch.load(ck, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            metrics = s1.evaluate(model, eval_loader, ns, device)
            row = {"checkpoint": ck.name, "step": int(payload.get("step", 0))}
            row.update({k: round(float(metrics[k]), 6) for k in KEEP if k in metrics})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(row, flush=True)
    print(f"[milestone-eval] {run_dir.name}: wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rd in args.run_dirs:
        eval_run(Path(rd), args.data_path, device)


if __name__ == "__main__":
    main()
