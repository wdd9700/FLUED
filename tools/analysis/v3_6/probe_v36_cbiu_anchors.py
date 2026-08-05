"""FLUED v3.6 CBIU anchor probe: offline rich/null references for the beta write gate.

The v3.6 CBIU action object is the beta write gate (v3.6 spec section 10).
Anchors are the two extreme write configurations on a frozen checkpoint:

* rich_all_readouts: beta forced to 1.0 on every valid chunk (full write);
* null_fallback_only: beta forced to 0.0 on every chunk (no write; the KDA
  state stays zero and the readout package degenerates to a constant).

Three paired risks are recomputed with identical bytes and masks per batch
(bits per target byte), S1.0 three-task semantics (2026-08-05):

* reconstruction_bpb: direct-path CE over all valid slots against the
  AS-ENCODED sequence (MASK_ID at masked slots — codec fidelity);
* completion_bpb: backbone-path CE over masked slots (clean targets);
* preservation_bpb: backbone-path CE over unmasked slots (clean targets).

The mask generator follows the checkpoint's resolved config (``mask_mode``):
``byte_span`` (legacy 1-8B random spans) or ``mixed`` (v36.3: 40% whole UTF-8
chars + 60% whole BPE words). The JSON schema matches
tools/train/v3_4/cbiu.py (modes + RISK_NAMES + null > rich separability
check) so the anchor file loads through CBIUState.from_anchor_file unchanged.
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
RISK_NAMES = ("reconstruction_bpb", "completion_bpb", "preservation_bpb")
PROTOCOL = "CBIU_V36_BETA_ANCHORS_S1_20260805"
MODES = ("rich_all_readouts", "null_fallback_only")
MODE_BETA = {"rich_all_readouts": 1.0, "null_fallback_only": 0.0}
MODE_SEMANTICS = {
    "rich_all_readouts": "beta=1.0 on every valid chunk (full write into the KDA state)",
    "null_fallback_only": "beta=0.0 on every chunk (no write; state stays zero, readout degenerates)",
}


class RiskTotals:
    def __init__(self) -> None:
        self.loss_sum = {name: 0.0 for name in RISK_NAMES}
        self.targets = {name: 0.0 for name in RISK_NAMES}
        self.chunks = 0.0
        self.batches = 0

    def add(self, name: str, values: torch.Tensor, weight: torch.Tensor) -> None:
        w = weight.to(values.dtype)
        self.loss_sum[name] += float((values * w).sum().item())
        self.targets[name] += float(w.sum().item())

    def as_dict(self) -> dict[str, float]:
        return {
            name: self.loss_sum[name] / max(self.targets[name], 1.0) / LN2 for name in RISK_NAMES
        }


def _ce_none(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = targets.reshape(-1).clamp(min=0, max=257)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=PAD_ID, reduction="none")
    return ce.reshape(targets.shape)


@torch.no_grad()
def _run_mode(model, eval_loader, args, device, beta_value: float) -> RiskTotals:
    totals = RiskTotals()
    original_forward = model.write_head.forward

    def patched_forward(memory: torch.Tensor) -> dict[str, torch.Tensor]:
        gates = original_forward(memory)
        gates["beta"] = torch.full_like(gates["beta"], beta_value)
        return gates

    model.write_head.forward = patched_forward
    mask_mode = getattr(args, "mask_mode", "byte_span")
    try:
        torch.manual_seed(args.eval_mask_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_mask_seed)
        s1._MASK_GENERATOR.manual_seed(args.eval_mask_seed)
        for batch_index, batch in enumerate(eval_loader):
            if batch_index >= args.max_eval_batches:
                break
            clean = batch[0].to(device)
            valid = clean.ne(PAD_ID)
            if mask_mode == "mixed":
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
                out = model(source)
            targets, slot_mask, masked_slot = make_targets(
                clean, byte_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span
            )
            # as-encoded targets for the direct path (S1.0 fidelity semantics)
            encoded_targets, encoded_slot_mask, _ = make_targets(
                source, torch.zeros_like(byte_mask), out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span
            )
            ce_direct = _ce_none(out.logits_direct, encoded_targets)
            ce_backbone = _ce_none(out.logits_backbone, targets)
            unmasked_slot = slot_mask & ~masked_slot
            totals.add("reconstruction_bpb", ce_direct, encoded_slot_mask)
            totals.add("completion_bpb", ce_backbone, masked_slot)
            totals.add("preservation_bpb", ce_backbone, unmasked_slot)
            totals.chunks += float(out.chunks.chunk_mask.float().sum().item())
            totals.batches += 1
    finally:
        model.write_head.forward = original_forward
    return totals


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

    rows = {}
    for mode in MODES:
        totals = _run_mode(model, eval_loader, args, device, MODE_BETA[mode])
        row = totals.as_dict()
        row["batches"] = totals.batches
        row["chunks_per_sample_mean"] = totals.chunks / max(totals.batches, 1) / args.batch_size
        rows[mode] = row
        print(f"[v36-cbiu] {mode}: {row}", flush=True)

    rich, null = rows["rich_all_readouts"], rows["null_fallback_only"]
    invalid = [name for name in RISK_NAMES if not null[name] > rich[name]]

    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "max_eval_batches": cli.max_eval_batches,
        "eval_mask_seed": args.eval_mask_seed,
        "mask_mode": getattr(args, "mask_mode", "byte_span"),
        "risk_semantics": "S1.0: reconstruction=as-encoded fidelity, completion/preservation=backbone vs clean",
        "action_object": "beta_write_gate",
        "mode_semantics": MODE_SEMANTICS,
        "transmitted_scalars": args.readout_queries * args.d_pack,
        "invalid_anchor_dimensions": invalid,
        "modes": rows,
    }

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cbiu_v36_anchors.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"invalid_anchor_dimensions": invalid}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
