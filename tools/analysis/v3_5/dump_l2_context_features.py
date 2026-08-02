"""Dump cross-chunk context features for the v3.5 L2 utility dataset.

Replays the same frozen body and eval stream as build_l2_offline_utility_dataset.py
(same checkpoint, same batch order) on CLEAN inputs (uniform boundary => identical
chunking), takes each chunk's slot-0 readout as its content vector, and emits
per-chunk context features keyed by (batch_index, sample_in_batch, chunk_index):

- sim_prev1: cosine to previous chunk
- max_sim_prev: max cosine to any earlier chunk (window-level redundancy)
- sim_next1: cosine to next chunk
- sim_window_mean: cosine to the mean of all other chunks (topic typicality)
- readout_norm: L2 norm of the chunk vector
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import make_dataloaders  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import build_model  # noqa: E402

PAD_ID = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(device=cli.device, max_eval_batches=cli.batches, num_workers=0)
    args = Namespace(**config)
    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    _, eval_loader = make_dataloaders(args)

    out_rows: dict[str, dict] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader):
            if batch_index >= cli.batches:
                break
            clean = batch[0].to(device)
            valid = clean.ne(PAD_ID)
            out = model(clean)
            readout = out.readout_z[:, :, 0, :].float()
            chunk_mask = out.chunks.chunk_mask
            for b in range(clean.size(0)):
                n_chunks = int(chunk_mask[b].sum().item())
                if n_chunks == 0:
                    continue
                vecs = readout[b, :n_chunks]
                vecs = F.normalize(vecs, dim=-1)
                norms = readout[b, :n_chunks].norm(dim=-1)
                sim = vecs @ vecs.T
                mean_dir = F.normalize(vecs.mean(dim=0, keepdim=True), dim=-1)
                sim_to_mean = (vecs @ mean_dir.T).squeeze(-1)
                for c in range(n_chunks):
                    prev = sim[c, c - 1].item() if c > 0 else -2.0
                    nxt = sim[c, c + 1].item() if c < n_chunks - 1 else -2.0
                    max_prev = sim[c, :c].max().item() if c > 0 else -2.0
                    key = f"{batch_index}_{b}_{c}"
                    out_rows[key] = {
                        "sim_prev1": float(prev),
                        "max_sim_prev": float(max_prev),
                        "sim_next1": float(nxt),
                        "sim_window_mean": float(sim_to_mean[c].item()),
                        "readout_norm": float(norms[c].item()),
                        "n_chunks": n_chunks,
                    }
            print(f"[ctx] batch {batch_index} done, rows={len(out_rows)}", flush=True)

    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows), encoding="utf-8")
    print(json.dumps({"rows": len(out_rows), "out": str(out_path.resolve())}))


if __name__ == "__main__":
    main()
