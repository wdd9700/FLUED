"""Position-resolved backbone accuracy probe for k=1 arms (E38 follow-up).

The k=1 backbone conditions the whole window on the FINAL state readout only.
If the flat ~0.2 completion plateau is a state-carry limit, accuracy should
decay with distance from the final chunk (recency); if it is uniform-low,
the limit is elsewhere (readout mechanism / task form). This probe reports
backbone (and direct, for contrast) accuracy bucketed by chunks-from-end.

Usage: python probe_position_acc.py <checkpoint> --data-path <corpus> [--max-batches 16]
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

from tools.train.v3_3.train_v33 import make_dataloaders, make_targets  # noqa: E402
from tools.train.v3_6.train_v36 import build_model  # noqa: E402
import tools.train.v3_6.train_v36_s1 as s1  # noqa: E402
from flued.data import PAD_ID  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-batches", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.checkpoint).parent
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg["data_path"] = args.data_path
    cfg["data_manifest"] = ""
    ns = Namespace(**cfg)
    if s1._BPE_TOKENIZER is None and getattr(ns, "mask_mode", "mixed") == "mixed":
        s1._load_bpe_tokenizer(ns.bpe_tokenizer_path)
    model = build_model(ns).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    _, eval_loader = make_dataloaders(ns)

    max_back = 32
    hit_bb = torch.zeros(max_back, dtype=torch.float64)
    hit_di = torch.zeros(max_back, dtype=torch.float64)
    cnt = torch.zeros(max_back, dtype=torch.float64)
    with torch.no_grad():
        for bi, batch in enumerate(eval_loader):
            if bi >= args.max_batches:
                break
            clean = batch[0].to(device)
            byte_mask = s1.make_mixed_mask(
                clean.cpu(), ns.mask_prob, char_frac=ns.mask_char_frac,
                char_span_max=ns.mask_char_span_max, tokenizer=s1._BPE_TOKENIZER,
                generator=s1._MASK_GENERATOR,
            ).to(device)
            source = clean.masked_fill(byte_mask, 257)
            out = s1.s1_forward(model, source, ns)
            chunks = out["chunks"]
            targets, slot_mask, _ = make_targets(
                clean, byte_mask, chunks.chunk_ids, chunks.offsets, ns.max_chunks, ns.max_span
            )
            pred_bb = out["logits_backbone"].argmax(dim=-1)
            pred_di = out["logits_direct"].argmax(dim=-1)
            real = chunks.chunk_mask.long().sum(dim=1).clamp(min=1) - 1  # last real chunk idx
            bsz = chunks.chunk_mask.size(0)
            for b in range(bsz):
                for i in range(int(real[b].item()) + 1):
                    sel = slot_mask[b, i]
                    tgt_b = targets[b, i][sel]
                    valid_t = tgt_b.ne(PAD_ID)
                    if not valid_t.any():
                        continue
                    dist = int(real[b].item()) - i
                    if dist >= max_back:
                        continue
                    hit_bb[dist] += (pred_bb[b, i][sel][valid_t] == tgt_b[valid_t]).sum().item()
                    hit_di[dist] += (pred_di[b, i][sel][valid_t] == tgt_b[valid_t]).sum().item()
                    cnt[dist] += int(valid_t.sum())
    table = [
        {
            "chunks_from_end": d,
            "slots": int(cnt[d]),
            "backbone_acc": round(float(hit_bb[d] / cnt[d].clamp(min=1)), 4),
            "direct_acc": round(float(hit_di[d] / cnt[d].clamp(min=1)), 4),
        }
        for d in range(max_back)
        if cnt[d] > 0
    ]
    print(json.dumps({"checkpoint": args.checkpoint, "step": int(payload.get("step", 0)), "per_position": table}, ensure_ascii=False))


if __name__ == "__main__":
    main()
