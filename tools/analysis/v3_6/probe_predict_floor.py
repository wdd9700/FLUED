"""Probe: predict_cos adjacent-chunk similarity floor (E35 metric caveat).

predict_cos compares backbone_out[i] against decoder_in(content[i+1]).
Adjacent chunks of natural text are highly redundant, so even a do-nothing
backbone (identity: output your own chunk's conditioning) scores a high
cosine. This probe measures that floor -- cos(decoder_in(content[i]),
decoder_in(content[i+1])) on the eval stream -- so small predict_cos
differences between arms (E35: 0.903..0.953) can be read against the floor
instead of against zero.

Usage: py -3.14 probe_predict_floor.py <checkpoint> [--max-batches 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_6.train_v36 import build_model, build_parser  # noqa: E402
from tools.train.v3_3.train_v33 import make_dataloaders  # noqa: E402
from flued.data import PAD_ID  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-batches", type=int, default=16)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/canonical_v36.json")
    pre_args, _ = pre.parse_known_args()
    tp = build_parser()
    tp.set_defaults(**json.loads(Path(pre_args.config).read_text(encoding="utf-8")))
    args, _ = tp.parse_known_args([])
    args.data_path = parser.parse_known_args()[0].data_path
    args.data_manifest = ""
    ckpt_arg = parser.parse_args().checkpoint
    max_batches = parser.parse_args().max_batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_arg, map_location=device, weights_only=False)
    saved_args = payload.get("args", {})
    for key in ("backbone_mode", "state_channel", "kda_impl", "summarizer_type", "per_chunk_readout"):
        if key in saved_args:
            setattr(args, key, saved_args[key])
    model = build_model(args).to(device)
    current = model.state_dict()
    compatible = {k: v for k, v in payload["model"].items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    model.eval()
    print(f"[floor] loaded {len(compatible)}/{len(payload['model'])} tensors from {ckpt_arg} (step {payload.get('step')})")

    _, eval_loader = make_dataloaders(args)
    floors, selfs = [], []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            ids = batch[0].to(device)
            out = model(ids)
            content = model.decoder_in(out.package.mean(dim=2)).float()
            pair = out.chunks.chunk_mask[:, :-1] & out.chunks.chunk_mask[:, 1:]
            if not pair.any():
                continue
            a, b = content[:, :-1][pair], content[:, 1:][pair]
            floors.append(F.cosine_similarity(a, b, dim=-1).mean().item())
            selfs.append(F.cosine_similarity(a, a, dim=-1).mean().item())
    floor = sum(floors) / max(len(floors), 1)
    print(json.dumps({
        "checkpoint": ckpt_arg,
        "batches": len(floors),
        "adjacent_cos_floor": floor,
        "self_cos_check": sum(selfs) / max(len(selfs), 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
