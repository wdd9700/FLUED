"""Audit per-action CBIU calibration on frozen v3.4 checkpoints."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID  # noqa: E402
from tools.analysis.v3_4.probe_v34_cbiu import _restore_runtime_state  # noqa: E402
from tools.train.v3_3.train_v33 import LatentInfillBackbone, make_byte_mask, make_dataloaders  # noqa: E402
from tools.train.v3_4.cbiu import CBIUState, cbiu_keep_utility  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    _score_cbiu_emit_actions,
    build_model,
)


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum().div(denom.clamp(min=1.0e-12)).item())


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.bool()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if positives == 0 or negatives == 0:
        return None
    ranks = _rank(scores) + 1.0
    positive_rank_sum = ranks[labels].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc.item())


def _ece(probability: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    error = probability.new_zeros(())
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = probability.ge(lower) & (
            probability.le(upper) if index == bins - 1 else probability.lt(upper)
        )
        if bool(selected.any()):
            weight = selected.float().mean()
            error = error + weight * (
                probability[selected].mean() - labels[selected].float().mean()
            ).abs()
    return float(error.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--anchor-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-mask-seed", type=int, default=-1)
    cli = parser.parse_args()

    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(device=cli.device, max_eval_batches=cli.max_eval_batches, num_workers=0)
    if cli.eval_mask_seed >= 0:
        config["eval_mask_seed"] = cli.eval_mask_seed
    args = Namespace(**config)
    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    _restore_runtime_state(model, payload)
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

    probabilities: list[torch.Tensor] = []
    quality_utilities: list[torch.Tensor] = []
    net_utilities: list[torch.Tensor] = []
    slots: list[torch.Tensor] = []
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(args.eval_mask_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_mask_seed)
        for batch_index, batch in enumerate(eval_loader):
            if batch_index >= cli.max_eval_batches:
                break
            clean = batch[0].to(device)
            valid = clean.ne(PAD_ID)
            byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
            source = clean.masked_fill(byte_mask, MASK_ID)
            training_out = model(source)
            for slot in range(1, args.readout_vectors):
                global_step = (batch_index * (args.readout_vectors - 1) + slot - 1) * args.emit_value_every
                scored = _score_cbiu_emit_actions(
                    model,
                    backbone,
                    clean,
                    byte_mask,
                    training_out,
                    args,
                    global_step,
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
                logits = training_out.emit_logits[action_batches, action_chunks, slot]
                probabilities.append(torch.sigmoid(logits).detach().cpu())
                quality_utilities.append(utility["quality_utility"].detach().cpu())
                net_utilities.append(utility["net_utility"].detach().cpu())
                slots.append(torch.full_like(logits.detach().cpu(), slot, dtype=torch.long))

    probability = torch.cat(probabilities)
    quality = torch.cat(quality_utilities)
    net = torch.cat(net_utilities)
    slot_ids = torch.cat(slots)
    labels = net.gt(0)
    count = probability.numel()
    top_k = max(1, count // 4)
    predicted_top = set(torch.topk(probability, top_k).indices.tolist())
    utility_top = set(torch.topk(net, top_k).indices.tolist())
    result = {
        "protocol": "CBIU_V1_ACTION_CALIBRATION_20260717",
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "anchor_file": str(Path(cli.anchor_file).resolve()),
        "eval_mask_seed": int(args.eval_mask_seed),
        "examples": count,
        "positive_fraction": float(labels.float().mean().item()),
        "predicted_probability_mean": float(probability.mean().item()),
        "quality_utility_mean": float(quality.mean().item()),
        "net_utility_mean": float(net.mean().item()),
        "spearman_quality": _correlation(_rank(probability), _rank(quality)),
        "spearman_net": _correlation(_rank(probability), _rank(net)),
        "auc_net_positive": _auc(probability, labels),
        "brier": float((probability - labels.float()).square().mean().item()),
        "ece_10bin": _ece(probability, labels),
        "sign_accuracy_at_0_5": float(probability.ge(0.5).eq(labels).float().mean().item()),
        "top_quartile_overlap": len(predicted_top & utility_top) / top_k,
        "per_slot": {},
    }
    for slot in range(1, args.readout_vectors):
        selected = slot_ids.eq(slot)
        if not bool(selected.any()):
            continue
        slot_labels = labels[selected]
        result["per_slot"][str(slot)] = {
            "examples": int(selected.sum().item()),
            "probability": float(probability[selected].mean().item()),
            "net_utility": float(net[selected].mean().item()),
            "positive_fraction": float(slot_labels.float().mean().item()),
            "auc": _auc(probability[selected], slot_labels),
        }

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cbiu_action_calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
