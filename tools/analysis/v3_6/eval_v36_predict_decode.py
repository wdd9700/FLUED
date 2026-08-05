"""FLUED v3.6 S1.0: byte-level decode evaluation of the latent prediction path.

S1.0 trains prediction in latent space only: MSE(backbone_out[i],
decoder_in(content[i+1]).detach()). This script measures what that latent
prediction is worth in BYTES: condition the decoder on backbone_out[i] (plus
the position embedding of chunk i+1) and decode chunk i+1's bytes, scored
against the clean text.

Reported per eval set:
* predict_byte_acc / predict_bpb: all valid slots of chunk i+1;
* predict_byte_acc_unmasked / predict_bpb_unmasked: only slots that were NOT
  masked in the encoder input (pure prediction; masked slots mix in
  mask-filling difficulty).

Caveat for external comparison (T4): the decoder was never trained on this
conditioning — this is a zero-shot reuse of the backbone->decoder channel, so
the numbers are a LOWER bound on prediction-path quality. H-Net's 0.653 BPB
is next-byte with full uncompressed context; ours is next-chunk through the
compressed state. Task differences must be stated when comparing.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID  # noqa: E402
from tools.train.v3_3.train_v33 import make_byte_mask, make_dataloaders, make_targets  # noqa: E402
from tools.train.v3_6.train_v36 import build_model  # noqa: E402
import tools.train.v3_6.train_v36_s1 as s1  # noqa: E402

LN2 = math.log(2.0)


class _Totals:
    def __init__(self) -> None:
        self.ce_sum = {"all": 0.0, "unmasked": 0.0}
        self.hits = {"all": 0.0, "unmasked": 0.0}
        self.count = {"all": 0.0, "unmasked": 0.0}
        self.pairs = 0.0

    def add(self, key: str, ce: torch.Tensor, pred: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor) -> None:
        wf = w.to(ce.dtype)
        self.ce_sum[key] += float((ce * wf).sum().item())
        self.hits[key] += float(((pred == tgt) & w).sum().item())
        self.count[key] += float(wf.sum().item())

    def as_dict(self) -> dict[str, float]:
        out = {}
        for key in ("all", "unmasked"):
            n = max(self.count[key], 1.0)
            out[f"predict_byte_acc_{key}"] = self.hits[key] / n
            out[f"predict_bpb_{key}"] = self.ce_sum[key] / n / LN2
        out["chunk_pairs"] = self.pairs
        return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(device=cli.device, max_eval_batches=cli.max_eval_batches, num_workers=0)
    args = Namespace(**config)
    if not getattr(args, "per_chunk_readout", False):
        raise SystemExit("predict-decode eval requires a per_chunk_readout (S1.0-style) checkpoint")
    if getattr(args, "mask_mode", "byte_span") == "mixed":
        s1._load_bpe_tokenizer(
            getattr(args, "bpe_tokenizer_path", "checkpoints/bpe_tokenizer_128k_v4/tokenizer.json")
        )
    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    _, eval_loader = make_dataloaders(args)

    torch.manual_seed(args.eval_mask_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_mask_seed)
    s1._MASK_GENERATOR.manual_seed(args.eval_mask_seed)

    totals = _Totals()
    for batch_index, batch in enumerate(eval_loader):
        if batch_index >= args.max_eval_batches:
            break
        clean = batch[0].to(device)
        valid = clean.ne(PAD_ID)
        if getattr(args, "mask_mode", "byte_span") == "mixed":
            byte_mask = s1.make_mixed_mask(
                clean.cpu(),
                args.mask_prob,
                char_frac=getattr(args, "mask_char_frac", 0.4),
                char_span_max=getattr(args, "mask_char_span_max", 3),
                tokenizer=s1._BPE_TOKENIZER,
                generator=s1._MASK_GENERATOR,
            ).to(device)
        else:
            byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
        source = clean.masked_fill(byte_mask, 257)  # MASK_ID
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = s1.s1_forward(model, source, args)
        chunks = out["chunks"]
        targets, slot_mask, masked_slot = make_targets(
            clean, byte_mask, chunks.chunk_ids, chunks.offsets, args.max_chunks, args.max_span
        )
        backbone_out = out["backbone_out"].float()
        n_chunks = chunks.chunk_mask.size(1)
        pos = model.chunk_pos.weight.unsqueeze(0)[:, :n_chunks].float()
        pair_mask = chunks.chunk_mask[:, :-1] & chunks.chunk_mask[:, 1:]
        if not pair_mask.any():
            continue
        for i in range(n_chunks - 1):
            pm = pair_mask[:, i]
            if not pm.any():
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                cond = backbone_out[:, i].unsqueeze(1) + pos[:, i + 1].unsqueeze(1)
                logits = model.decoder(cond, chunks.token_mask[:, i + 1].unsqueeze(1))
            logits = logits.squeeze(1).float()  # (B, span, V)
            tgt = targets[:, i + 1]
            slots = slot_mask[:, i + 1] & pm.unsqueeze(1)
            unmasked = slots & ~masked_slot[:, i + 1]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1).clamp(min=0, max=257),
                ignore_index=PAD_ID, reduction="none",
            ).reshape(tgt.shape)
            pred = logits.argmax(dim=-1)
            totals.add("all", ce, pred, tgt, slots)
            totals.add("unmasked", ce, pred, tgt, unmasked)
            totals.pairs += float(pm.sum().item())

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "max_eval_batches": cli.max_eval_batches,
        "eval_mask_seed": args.eval_mask_seed,
        "mask_mode": getattr(args, "mask_mode", "byte_span"),
        "note": (
            "zero-shot decoder reuse: condition on backbone_out[i], decode chunk i+1; "
            "lower bound on prediction-path quality. H-Net 0.653 BPB is next-byte "
            "uncompressed; ours is next-chunk through the compressed state."
        ),
        **totals.as_dict(),
    }
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predict_decode_eval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
