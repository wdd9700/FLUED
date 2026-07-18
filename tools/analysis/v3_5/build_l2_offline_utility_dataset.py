"""Build the v3.5 L2 offline counterfactual utility dataset on a frozen body.

Every record is one scored emit action `(sample, chunk, extra slot)` with:
- chunk content features (byte-class ratios, unigram entropy, position);
- paired on/off three-risk BPB from strict re-computation;
- multi-mask-draw aggregation fields (draw id per record; aggregation happens in L3);
- quality/net utility under frozen rich/null anchors.

The body (FLUED + temporary backbone) is never updated. Utility labels are detached.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import LatentInfillBackbone, make_byte_mask, make_dataloaders  # noqa: E402
from tools.train.v3_4.cbiu import CBIUState, cbiu_keep_utility  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    _score_cbiu_emit_actions,
    build_model,
)

PAD_ID = 0
MASK_ID = 257


def _restore_runtime_state(model, payload: dict) -> dict:
    state = payload.get("runtime_boundary_state", {})
    if state:
        model.config.boundary_mode = state.get("mode", model.config.boundary_mode)
        model.coding_rate_selector.mode = state.get(
            "coding_rate_mode", model.coding_rate_selector.mode
        )
        model.config.boundary_blend_alpha = float(
            state.get("blend_alpha", model.config.boundary_blend_alpha)
        )
    return {
        "mode": model.config.boundary_mode,
        "coding_rate_mode": model.coding_rate_selector.mode,
        "blend_alpha": float(model.config.boundary_blend_alpha),
    }


def _byte_class_ratios(values: torch.Tensor) -> dict[str, float]:
    total = max(int(values.numel()), 1)
    ascii_letter = ((values >= 0x41) & (values <= 0x5A)) | ((values >= 0x61) & (values <= 0x7A))
    digit = (values >= 0x30) & (values <= 0x39)
    space = (values == 0x20) | (values == 0x0A) | (values == 0x09)
    punct = (values >= 0x21) & (values <= 0x2F) | (values >= 0x3A) & (values <= 0x40) | (
        (values >= 0x5B) & (values <= 0x60)
    ) | ((values >= 0x7B) & (values <= 0x7E))
    high = values >= 0x80
    return {
        "letter_ratio": float(ascii_letter.float().mean().item()),
        "digit_ratio": float(digit.float().mean().item()),
        "space_ratio": float(space.float().mean().item()),
        "punct_ratio": float(punct.float().mean().item()),
        "high_byte_ratio": float(high.float().mean().item()),
    }


def _unigram_entropy(values: torch.Tensor) -> float:
    counts = torch.bincount(values.flatten(), minlength=256).float()
    probs = counts / counts.sum().clamp(min=1.0)
    nz = probs[probs > 0]
    return float(-(nz * nz.log2()).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--anchor-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--mask-draws", type=int, default=8)
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
    runtime_state = _restore_runtime_state(model, payload)
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    ).to(device)
    backbone.load_state_dict(payload["backbone"], strict=True)
    model.eval()
    backbone.eval()
    state = CBIUState.from_anchor_file(cli.anchor_file, args.cbiu_compute_budget, device)
    if "cbiu_state" in payload:
        state.load_state_dict(payload["cbiu_state"], device)
    _, eval_loader = make_dataloaders(args)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "l2_offline_utility_dataset.jsonl"
    meta = {
        "protocol": "V35_L2_OFFLINE_UTILITY_20260717",
        "checkpoint": str(checkpoint.resolve()),
        "anchor_file": str(Path(cli.anchor_file).resolve()),
        "runtime_boundary_state": runtime_state,
        "batches": cli.batches,
        "mask_draws": cli.mask_draws,
        "records": 0,
    }

    records = 0
    with dataset_path.open("w", encoding="utf-8") as sink:
        for draw in range(cli.mask_draws):
            draw_seed = int(args.eval_mask_seed) + 1000 * draw
            torch.manual_seed(draw_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(draw_seed)
            for batch_index, batch in enumerate(eval_loader):
                if batch_index >= cli.batches:
                    break
                clean = batch[0].to(device)
                valid = clean.ne(PAD_ID)
                byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
                source = clean.masked_fill(byte_mask, MASK_ID)
                with torch.no_grad():
                    training_out = model(source)
                for slot in range(1, args.readout_vectors):
                    global_step = (
                        (draw * cli.batches + batch_index) * (args.readout_vectors - 1) + slot - 1
                    ) * max(args.emit_value_every, 1)
                    scored = _score_cbiu_emit_actions(
                        model, backbone, clean, byte_mask, training_out, args, global_step
                    )
                    action_batches = scored["batch_indices"]
                    if action_batches.numel() == 0:
                        continue
                    action_chunks = scored["training_chunk_indices"]
                    utility = cbiu_keep_utility(
                        scored["on_risks"],
                        scored["off_risks"],
                        scored["on_cost"],
                        scored["off_cost"],
                        state,
                        args.cbiu_augmented_weight,
                    )
                    for row in range(action_batches.numel()):
                        b = int(action_batches[row].item())
                        c = int(action_chunks[row].item())
                        positions = (
                            training_out.chunks.chunk_ids[b].eq(c) & valid[b]
                        ).nonzero(as_tuple=False).flatten()
                        if positions.numel() == 0:
                            continue
                        chunk_bytes = (clean[b, positions] - 1).clamp(min=0, max=255)
                        features = _byte_class_ratios(chunk_bytes)
                        features["unigram_entropy"] = _unigram_entropy(chunk_bytes)
                        features["byte_len"] = int(positions.numel())
                        features["chunk_index"] = c
                        features["byte_anchor"] = float(positions.float().mean().item())
                        record = {
                            "draw": draw,
                            "batch_index": batch_index,
                            "sample_in_batch": b,
                            "chunk_index": c,
                            "slot": slot,
                            "features": features,
                            "risk_on": [float(v) for v in scored["on_risks"][row].tolist()],
                            "risk_off": [float(v) for v in scored["off_risks"][row].tolist()],
                            "cost_on": float(scored["on_cost"][row].item()),
                            "cost_off": float(scored["off_cost"][row].item()),
                            "quality_utility": float(utility["quality_utility"][row].item()),
                            "net_utility": float(utility["net_utility"][row].item()),
                        }
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records += 1
            print(f"[l2] draw {draw} done, records={records}", flush=True)

    meta["records"] = records
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
