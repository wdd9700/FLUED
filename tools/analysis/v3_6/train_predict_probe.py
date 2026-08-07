"""T4: train a lightweight probe read-head on FROZEN checkpoints to measure
byte-level decodability of the latent prediction (generation-line prerequisite).

Zero-shot decoder reuse gave byte acc 0.099 / BPB 10.1 (E28) -- the latent
prediction cannot be read for free. This probe trains a small MLP from the
frozen backbone's predict representation (backbone_out[i] in per-chunk arms)
to chunk i+1's bytes, giving a byte-acc/BPB estimate that is honestly
comparable (with caveats: next-chunk vs next-byte task, compressed vs
uncompressed context) to the H-Net reproduction anchor (0.653 BPB).

Usage:
  python train_predict_probe.py <checkpoint> --data-path <corpus> [--steps 3000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import make_dataloaders, make_targets  # noqa: E402
from tools.train.v3_6.train_v36 import build_model  # noqa: E402
import tools.train.v3_6.train_v36_s1 as s1  # noqa: E402
from flued.data import PAD_ID  # noqa: E402


class ProbeHead(nn.Module):
    """backbone_out vector (d) -> per-slot byte logits for the next chunk."""

    def __init__(self, d_in: int, hidden: int, max_span: int) -> None:
        super().__init__()
        self.max_span = max_span
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, max_span * 258),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(*x.shape[:-1], self.max_span, 258)


@torch.no_grad()
def featurize(model, ids, ns):
    """Frozen full stack forward -> (content, backbone_out, chunks)."""
    out = s1.s1_forward(model, ids, ns)
    return out["content"], out["backbone_out"], out["chunks"]


def run_split(model, probe, loader, ns, device, steps, opt=None):
    t0 = time.time()
    ce_sum, acc_sum, n_sum = 0.0, 0.0, 0
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= steps:
            break
        clean = batch[0].to(device)
        with torch.no_grad():
            byte_mask = torch.zeros_like(clean, dtype=torch.bool)
            content, backbone_out, chunks = featurize(model, clean, ns)
            targets, slot_mask, _ = make_targets(
                clean, byte_mask, chunks.chunk_ids, chunks.offsets, ns.max_chunks, ns.max_span
            )
        pair_mask = chunks.chunk_mask[:, :-1] & chunks.chunk_mask[:, 1:]
        if not pair_mask.any():
            continue
        src = backbone_out if backbone_out.size(1) > 1 else content
        if src.size(1) <= 1:  # k=1 arms: fall back to per-chunk content features
            src = content
        feats = src[:, :-1][pair_mask]
        tgt = targets[:, 1:][pair_mask]  # (N, span)
        tmask = slot_mask[:, 1:][pair_mask] & tgt.ne(PAD_ID)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = probe(feats)
        ce = F.cross_entropy(
            logits.reshape(-1, 258).float(), tgt.reshape(-1).clamp(min=0),
            reduction="none", ignore_index=PAD_ID,
        ).reshape_as(tgt)
        ce = (ce * tmask).sum() / tmask.sum().clamp(min=1)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            ce.backward()
            opt.step()
        acc = ((logits.argmax(dim=-1) == tgt).float() * tmask).sum() / tmask.sum().clamp(min=1)
        ce_sum += float(ce) * int(tmask.sum())
        acc_sum += float(acc) * int(tmask.sum())
        n_sum += int(tmask.sum())
        if opt is not None and (bi + 1) % 200 == 0:
            print(f"[probe] step {bi+1}/{steps} ce {ce_sum/max(n_sum,1):.4f} acc {acc_sum/max(n_sum,1):.4f} ({(bi+1)/max(time.time()-t0,1e-9):.1f}/s)", flush=True)
    return ce_sum / max(n_sum, 1), acc_sum / max(n_sum, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.checkpoint).parent
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg["data_path"] = args.data_path
    cfg["data_manifest"] = ""
    ns = Namespace(**cfg)
    if getattr(ns, "mask_mode", "mixed") == "mixed":
        s1._load_bpe_tokenizer(ns.bpe_tokenizer_path)
    model = build_model(ns).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    # Feature source: per-chunk arms expose one predict vector per chunk
    # (d_backbone); k=1 (final) arms fall back to per-chunk content (d_pack).
    d_in = ns.d_pack if getattr(ns, "backbone_readout", "per_chunk") == "final" else ns.d_backbone
    probe = ProbeHead(d_in, args.hidden, ns.max_span).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader, eval_loader = make_dataloaders(ns)
    print(f"[probe] training read-head on {run_dir.name} (step {payload.get('step')}), d_in={d_in}")
    run_split(model, probe, train_loader, ns, device, args.steps, opt=opt)
    ce, acc = run_split(model, probe, eval_loader, ns, device, args.eval_batches, opt=None)
    result = {
        "checkpoint": args.checkpoint,
        "probe_steps": args.steps,
        "eval_next_chunk_byte_ce": round(ce, 4),
        "eval_next_chunk_bpb": round(ce / 0.6931471805599453, 4),
        "eval_next_chunk_byte_acc": round(acc, 4),
        "reference": {"hnet_repro_bpb": 0.653, "zeroshot_decoder_reuse_bpb": 10.1},
        "caveat": "next-chunk (compressed ctx) vs next-byte (full ctx) task difference; probe = upper-bound read, not the deployed decoder",
    }
    out_path = run_dir / "predict_probe_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
