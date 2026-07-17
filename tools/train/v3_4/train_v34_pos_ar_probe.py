"""Train the FLUED v3.4 RoPE x small-AR probe.

The two supervised paths deliberately share the same masked source:

* identity: encoder/decoder must reproduce the observed stream, including MASK;
* completion: the temporary backbone replaces affected readouts and must recover
  the clean bytes while preserving visible bytes in the same chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_VOCAB_SIZE, MASK_ID, PAD_ID, text_to_byte_ids  # noqa: E402
from flued.v34 import FLUEDV34, FLUEDV34Config  # noqa: E402
from tools.train.v3_4.cbiu import (  # noqa: E402
    CBIUState,
    RISK_NAMES as CBIU_RISK_NAMES,
    cbiu_keep_utility,
    update_cbiu_dual,
)
from tools.train.v3_3.train_v33 import (  # noqa: E402
    LatentInfillBackbone,
    _append_jsonl,
    _avg_metrics,
    _compact_active_readout,
    _cosine_with_warmup,
    _safe_acc,
    _scatter_compact_readout,
    build_optimizer,
    make_byte_mask,
    make_dataloaders,
    make_targets,
    masked_readouts_from_slots,
)


def build_model(args: argparse.Namespace) -> FLUEDV34:
    model = FLUEDV34(
        FLUEDV34Config(
            d_model=args.d_model,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
            segmentor_layers=args.segmentor_layers,
            interpreter_layers=args.interpreter_layers,
            memory_rank=args.memory_rank,
            readout_vectors=args.readout_vectors,
            ar_hidden=args.ar_hidden,
            use_position=args.use_position,
            position_strategy=getattr(args, "position_strategy", "layered_rope"),
            prompt_position_scale=getattr(args, "prompt_position_scale", 0.1),
            use_prompt_alibi=getattr(args, "use_prompt_alibi", False),
            use_ar=args.use_ar,
            use_structured_lookup=args.use_structured_lookup,
            use_memory=args.use_memory,
            use_boundary_bridge=args.use_boundary_bridge,
            memory_use_position=getattr(args, "memory_use_position", True),
            memory_position_mode=getattr(args, "memory_position_mode", "legacy"),
            memory_residual_scale=getattr(args, "memory_residual_scale", 0.1),
            memory_context_norm=getattr(args, "memory_context_norm", "none"),
            memory_scale_mode=getattr(args, "memory_scale_mode", "fixed"),
            memory_scale_max=getattr(args, "memory_scale_max", 0.1),
            memory_access_mode=getattr(args, "memory_access_mode", "other_only"),
            current_memory_mode=getattr(args, "current_memory_mode", "off"),
            current_memory_scale=getattr(args, "current_memory_scale", 0.03),
            current_memory_scale_max=getattr(args, "current_memory_scale_max", 0.1),
            boundary_mode=getattr(args, "boundary_mode", "threshold"),
            coding_rate_dim=getattr(args, "boundary_coding_rate_dim", 16),
            coding_rate_epsilon=getattr(args, "boundary_coding_rate_epsilon", 1.0),
            coding_rate_temperature=getattr(args, "boundary_coding_rate_temperature", 0.15),
            coding_rate_mode=getattr(args, "boundary_coding_rate_mode", "exact"),
            boundary_blend_alpha=getattr(args, "boundary_blend_alpha", 1.0),
            fixed_chunk_budget=getattr(args, "fixed_chunk_budget", 0),
            bytes_per_chunk_budget=getattr(args, "bytes_per_chunk_budget", 16),
            use_emit_controller=getattr(args, "use_emit_controller", False),
            emit_forward_mode=getattr(args, "emit_forward_mode", "hard_st"),
            emit_initial_probability=getattr(args, "emit_initial_probability", 0.1),
            emit_threshold=getattr(args, "emit_threshold", 0.5),
            emit_controller_hidden=getattr(args, "emit_controller_hidden", 0),
            emit_controller_slot_embedding=getattr(args, "emit_controller_slot_embedding", False),
            max_chunks=args.max_chunks,
            max_span=args.max_span,
            tau_cut=args.tau_cut,
            tau_trans=args.tau_trans,
            boundary_temperature=args.boundary_temperature,
            boundary_bridge_gradient_scale=getattr(args, "boundary_bridge_gradient_scale", 1.0),
            noise_scale=args.noise_scale,
            decoder_mode=getattr(args, "decoder_mode", "legacy_independent"),
        )
    )
    # The rate dual is training-only state saved outside the model state dict.
    # Register a non-persistent default so evaluation-only construction follows
    # the same code path without requiring the training loop to attach it.
    model.register_buffer("boundary_rate_dual", torch.zeros(()), persistent=False)
    return model


def apply_boundary_curriculum(model: FLUEDV34, args: argparse.Namespace, step: int) -> bool:
    """Apply a hard switch or a fixed-budget uniform-to-rate score transition."""
    switch_step = int(getattr(args, "boundary_curriculum_switch_step", 0))
    if switch_step <= 0 or step < switch_step:
        return False
    target_mode = getattr(args, "boundary_curriculum_mode", "marginal_rate_topk")
    target_rate_mode = getattr(args, "boundary_curriculum_coding_rate_mode", "l2")
    transition_steps = int(getattr(args, "boundary_curriculum_transition_steps", 0))
    old_mode = model.config.boundary_mode
    if transition_steps > 0 and step < switch_step + transition_steps:
        progress = (step - switch_step) / float(transition_steps)
        alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
        active_mode = (
            "uniform_confidence_blend"
            if target_mode == "confidence_threshold"
            else "uniform_l2_blend"
        )
    else:
        alpha = 1.0
        active_mode = target_mode
    changed = old_mode != active_mode or model.coding_rate_selector.mode != target_rate_mode
    model.config.boundary_mode = active_mode
    model.config.coding_rate_mode = target_rate_mode
    model.config.boundary_blend_alpha = alpha
    model.coding_rate_selector.mode = target_rate_mode
    # step_model owns auxiliary-loss accounting and historically read args.
    # Keep both runtime views synchronized across curriculum transitions.
    args.boundary_mode = active_mode
    args.boundary_coding_rate_mode = target_rate_mode
    args.boundary_blend_alpha = alpha
    return changed


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, BYTE_VOCAB_SIZE),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(targets)


def _mean_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def _mean_per_chunk(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum(dim=-1) / weight.sum(dim=-1).clamp(min=1.0)


def _run_completion(
    model,
    backbone,
    readout: torch.Tensor,
    active_readouts: torch.Tensor,
    affected_readouts: torch.Tensor,
    chunk_mask: torch.Tensor,
):
    bsz, chunks, readouts, dim = readout.shape
    flat_z = readout.reshape(bsz, chunks * readouts, dim)
    flat_active = active_readouts.reshape(bsz, -1)
    flat_affected = affected_readouts.reshape(bsz, -1) & flat_active
    positions = torch.arange(flat_z.size(1), device=flat_z.device).view(1, -1).expand(bsz, -1)
    compact_z, compact_active, compact_affected, compact_pos = _compact_active_readout(
        flat_z, flat_active, flat_affected, positions
    )
    predicted_compact = backbone(
        compact_z,
        compact_active,
        compact_affected,
        position_ids=compact_pos,
    )
    predicted_flat = _scatter_compact_readout(predicted_compact, compact_pos, flat_z.size(1))
    predicted = predicted_flat.reshape_as(readout)
    completed = torch.where((affected_readouts & active_readouts).unsqueeze(-1), predicted, readout)
    logits = model.decode(completed, chunk_mask, active_readouts)
    return logits, compact_z.size(1)


def _strict_affected_readouts(
    masked_slot: torch.Tensor,
    chunk_mask: torch.Tensor,
    active_readouts: torch.Tensor,
) -> torch.Tensor:
    """Map masked bytes only to executable readouts, with fallback writable."""

    affected_chunks = masked_slot.any(dim=-1) & chunk_mask
    affected = masked_readouts_from_slots(
        masked_slot,
        chunk_mask,
        active_readouts.size(-1),
    ) & active_readouts
    affected = affected.clone()
    affected[..., 0] |= affected_chunks
    return affected & active_readouts


def _toggle_cbiu_actions(
    base_z: torch.Tensor,
    candidates: torch.Tensor,
    base_active: torch.Tensor,
    batch_indices: torch.Tensor,
    chunk_indices: torch.Tensor,
    slot: int,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = base_z.clone()
    active = base_active.clone()
    if batch_indices.numel() == 0:
        return z, active
    if enabled:
        z[batch_indices, chunk_indices, slot] = candidates[batch_indices, chunk_indices, slot]
        active[batch_indices, chunk_indices, slot] = True
    else:
        z[batch_indices, chunk_indices, slot] = 0.0
        active[batch_indices, chunk_indices, slot] = False
    return z, active


def _gather_action_chunks(
    values: torch.Tensor,
    batch_indices: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> torch.Tensor:
    return values[batch_indices, chunk_indices]


@torch.no_grad()
def _score_cbiu_emit_actions(
    model: FLUEDV34,
    backbone: LatentInfillBackbone,
    clean: torch.Tensor,
    byte_mask: torch.Tensor,
    training_out,
    args: argparse.Namespace,
    global_step: int,
) -> dict[str, torch.Tensor | int]:
    """Score one paired extra-readout action per sample with frozen execution noise.

    The same byte anchor identifies the action in clean, masked-score and
    gradient-carrying training graphs.  Only samples with reconstruction,
    completion and visible-preservation support produce an emit target.
    """

    was_model_training = model.training
    was_backbone_training = backbone.training
    old_collect = bool(getattr(model, "collect_memory_diagnostics", False))
    model.eval()
    backbone.eval()
    model.collect_memory_diagnostics = False
    try:
        valid = clean.ne(PAD_ID)
        src = clean.masked_fill(byte_mask, MASK_ID)
        masked_out = model(src)
        clean_out = model(clean)
        zero_mask = torch.zeros_like(byte_mask)
        clean_targets, clean_slot_mask, _ = make_targets(
            clean,
            zero_mask,
            clean_out.chunks.chunk_ids,
            clean_out.chunks.offsets,
            args.max_chunks,
            args.max_span,
        )
        masked_targets, masked_slot_mask, masked_slot = make_targets(
            clean,
            byte_mask,
            masked_out.chunks.chunk_ids,
            masked_out.chunks.offsets,
            args.max_chunks,
            args.max_span,
        )
        affected_chunks = masked_slot.any(dim=-1) & masked_out.chunks.chunk_mask
        preserve_slot = masked_slot_mask & affected_chunks.unsqueeze(-1) & ~masked_slot
        masked_base_active = masked_out.emit_hard.clone()
        masked_base_active[..., 0] |= masked_out.chunks.chunk_mask
        valid_bytes = valid.float().sum(dim=1).clamp(min=1.0)
        policy_cost = masked_base_active.float().sum(dim=(1, 2)) / valid_bytes
        mapped_affected = masked_readouts_from_slots(
            masked_slot,
            masked_out.chunks.chunk_mask,
            masked_out.readout_z.size(2),
        )
        slot = 1 + (global_step // max(args.emit_value_every, 1)) % (masked_out.readout_z.size(2) - 1)
        eligible_chunks = (
            affected_chunks
            & preserve_slot.any(dim=-1)
            & ~mapped_affected[:, :, slot]
        )

        batch_ids: list[int] = []
        masked_chunks: list[int] = []
        clean_chunks: list[int] = []
        training_chunks: list[int] = []
        anchor_positions: list[int] = []
        for batch_index in range(clean.size(0)):
            choices = eligible_chunks[batch_index].nonzero(as_tuple=False).flatten()
            if choices.numel() == 0:
                continue
            offset = (global_step // max(args.emit_value_every, 1) + 7 * batch_index) % choices.numel()
            masked_chunk = int(choices[offset].item())
            positions = (
                masked_out.chunks.chunk_ids[batch_index].eq(masked_chunk) & valid[batch_index]
            ).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            anchor = int(positions[0].item())
            clean_chunk = int(clean_out.chunks.chunk_ids[batch_index, anchor].item())
            training_chunk = int(training_out.chunks.chunk_ids[batch_index, anchor].item())
            if (
                clean_chunk < 0
                or clean_chunk >= clean_out.chunks.chunk_mask.size(1)
                or training_chunk < 0
                or training_chunk >= training_out.chunks.chunk_mask.size(1)
                or not bool(clean_out.chunks.chunk_mask[batch_index, clean_chunk])
                or not bool(training_out.chunks.chunk_mask[batch_index, training_chunk])
            ):
                continue
            batch_ids.append(batch_index)
            masked_chunks.append(masked_chunk)
            clean_chunks.append(clean_chunk)
            training_chunks.append(training_chunk)
            anchor_positions.append(anchor)

        batch_indices = torch.tensor(batch_ids, device=clean.device, dtype=torch.long)
        masked_chunk_indices = torch.tensor(masked_chunks, device=clean.device, dtype=torch.long)
        clean_chunk_indices = torch.tensor(clean_chunks, device=clean.device, dtype=torch.long)
        training_chunk_indices = torch.tensor(training_chunks, device=clean.device, dtype=torch.long)
        anchor_tensor = torch.tensor(anchor_positions, device=clean.device, dtype=torch.long)
        if batch_indices.numel() == 0:
            return {
                "slot": slot,
                "batch_indices": batch_indices,
                "training_chunk_indices": training_chunk_indices,
                "anchor_positions": anchor_tensor,
                "policy_cost": policy_cost,
            }

        clean_base_active = clean_out.emit_hard.clone()
        clean_base_active[..., 0] |= clean_out.chunks.chunk_mask
        clean_on_z, clean_on_active = _toggle_cbiu_actions(
            clean_out.readout_z,
            clean_out.readout_candidates,
            clean_base_active,
            batch_indices,
            clean_chunk_indices,
            slot,
            True,
        )
        clean_off_z, clean_off_active = _toggle_cbiu_actions(
            clean_out.readout_z,
            clean_out.readout_candidates,
            clean_base_active,
            batch_indices,
            clean_chunk_indices,
            slot,
            False,
        )
        masked_on_z, masked_on_active = _toggle_cbiu_actions(
            masked_out.readout_z,
            masked_out.readout_candidates,
            masked_base_active,
            batch_indices,
            masked_chunk_indices,
            slot,
            True,
        )
        masked_off_z, masked_off_active = _toggle_cbiu_actions(
            masked_out.readout_z,
            masked_out.readout_candidates,
            masked_base_active,
            batch_indices,
            masked_chunk_indices,
            slot,
            False,
        )

        clean_on_logits = model.decode(
            clean_on_z,
            clean_out.chunks.chunk_mask,
            clean_on_active,
        )
        clean_off_logits = model.decode(
            clean_off_z,
            clean_out.chunks.chunk_mask,
            clean_off_active,
        )
        affected_on = _strict_affected_readouts(
            masked_slot,
            masked_out.chunks.chunk_mask,
            masked_on_active,
        )
        affected_off = _strict_affected_readouts(
            masked_slot,
            masked_out.chunks.chunk_mask,
            masked_off_active,
        )
        masked_on_logits, _ = _run_completion(
            model,
            backbone,
            masked_on_z,
            masked_on_active,
            affected_on,
            masked_out.chunks.chunk_mask,
        )
        masked_off_logits, _ = _run_completion(
            model,
            backbone,
            masked_off_z,
            masked_off_active,
            affected_off,
            masked_out.chunks.chunk_mask,
        )

        ln2 = math.log(2.0)
        clean_on_risk = _gather_action_chunks(
            _mean_per_chunk(_ce(clean_on_logits, clean_targets), clean_slot_mask),
            batch_indices,
            clean_chunk_indices,
        ) / ln2
        clean_off_risk = _gather_action_chunks(
            _mean_per_chunk(_ce(clean_off_logits, clean_targets), clean_slot_mask),
            batch_indices,
            clean_chunk_indices,
        ) / ln2
        masked_on_ce = _ce(masked_on_logits, masked_targets)
        masked_off_ce = _ce(masked_off_logits, masked_targets)
        fill_on_risk = _gather_action_chunks(
            _mean_per_chunk(masked_on_ce, masked_slot),
            batch_indices,
            masked_chunk_indices,
        ) / ln2
        fill_off_risk = _gather_action_chunks(
            _mean_per_chunk(masked_off_ce, masked_slot),
            batch_indices,
            masked_chunk_indices,
        ) / ln2
        keep_on_risk = _gather_action_chunks(
            _mean_per_chunk(masked_on_ce, preserve_slot),
            batch_indices,
            masked_chunk_indices,
        ) / ln2
        keep_off_risk = _gather_action_chunks(
            _mean_per_chunk(masked_off_ce, preserve_slot),
            batch_indices,
            masked_chunk_indices,
        ) / ln2
        on_cost = masked_on_active.float().sum(dim=(1, 2)) / valid_bytes
        off_cost = masked_off_active.float().sum(dim=(1, 2)) / valid_bytes
        return {
            "slot": slot,
            "batch_indices": batch_indices,
            "training_chunk_indices": training_chunk_indices,
            "anchor_positions": anchor_tensor,
            "on_risks": torch.stack((clean_on_risk, fill_on_risk, keep_on_risk), dim=-1),
            "off_risks": torch.stack((clean_off_risk, fill_off_risk, keep_off_risk), dim=-1),
            "on_cost": on_cost[batch_indices],
            "off_cost": off_cost[batch_indices],
            "policy_cost": policy_cost,
        }
    finally:
        model.collect_memory_diagnostics = old_collect
        model.train(was_model_training)
        backbone.train(was_backbone_training)


def boundary_prior_losses_v34(
    confidence: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    collect_metrics: bool = True,
):
    """Settled signed-confidence priors for v3.4.

    UTF-8 continuation bytes target -1, punctuation and whitespace target 0.5,
    and all remaining character-boundary candidates have zero mean. Hard
    thresholds are fixed execution policy, not direct regression targets.
    """
    raw = (token_ids - 1).clamp(min=0, max=255)
    utf8_cont = raw.ge(0x80) & raw.le(0xBF) & valid
    whitespace = ((raw == 10) | (raw == 13) | (raw == 32) | (raw == 9)) & valid
    punct = (
        (raw.ge(0x21) & raw.le(0x2F))
        | (raw.ge(0x3A) & raw.le(0x40))
        | (raw.ge(0x5B) & raw.le(0x60))
        | (raw.ge(0x7B) & raw.le(0x7E))
    ) & valid
    first = torch.zeros_like(valid)
    first_idx = valid.float().argmax(dim=1)
    first[torch.arange(valid.size(0), device=valid.device), first_idx] = valid.any(dim=1)
    weak_prior = (whitespace | punct) & ~utf8_cont
    neutral = valid & ~utf8_cont & ~weak_prior & ~first
    conf = confidence.float()

    cont_w = utf8_cont.float()
    weak_w = weak_prior.float()
    neutral_w = neutral.float()
    cont_loss = _mean_masked(F.smooth_l1_loss(conf, -torch.ones_like(conf), reduction="none"), utf8_cont)
    weak_loss = _mean_masked(F.smooth_l1_loss(conf, torch.full_like(conf, 0.5), reduction="none"), weak_prior)
    neutral_mean = (conf * neutral_w).sum() / neutral_w.sum().clamp(min=1.0)
    loss = cont_loss + weak_loss + neutral_mean.square()
    stats = {}
    if collect_metrics:
        stats = {
            "boundary_cont_loss": float(cont_loss.item()),
            "boundary_punct_loss": float(weak_loss.item()),
            "boundary_neutral_mean": float(neutral_mean.item()),
            "boundary_cont_mean": float((conf * cont_w).sum().item() / cont_w.sum().clamp(min=1.0).item()),
            "boundary_punct_mean": float((conf * weak_w).sum().item() / weak_w.sum().clamp(min=1.0).item()),
            "boundary_continue_target": -1.0,
            "boundary_punct_target": 0.5,
        }
    components = {
        "continuation": cont_loss,
        "punctuation": weak_loss,
        "neutral_mean": neutral_mean.square(),
    }
    return loss, stats, components


def projected_coding_rate_loss(
    readout: torch.Tensor,
    chunk_mask: torch.Tensor,
    projection_dim: int = 64,
):
    """Low-cost coding-rate proxy over active readout vectors."""
    active = readout[chunk_mask].reshape(-1, readout.size(-1)).float()
    if active.size(0) < 2:
        zero = readout.new_zeros(())
        return zero, zero
    dim = min(int(projection_dim), active.size(-1))
    # autocast may lower adaptive pooling back to bf16; the log-determinant
    # must stay in float32 for both support and numerical stability.
    with torch.amp.autocast(device_type=readout.device.type, enabled=False):
        projected = F.adaptive_avg_pool1d(active.float().unsqueeze(1), dim).squeeze(1).float()
    projected = projected - projected.mean(dim=0, keepdim=True)
    covariance = (
        projected.transpose(0, 1) @ projected / max(active.size(0) - 1, 1)
    ).float()
    eye = torch.eye(dim, device=readout.device, dtype=covariance.dtype)
    sign, logabsdet = torch.linalg.slogdet(eye + covariance)
    rate = torch.where(sign > 0, logabsdet / dim, logabsdet.new_zeros(()))
    return -rate.to(readout.dtype), rate.detach().to(readout.dtype)


def boundary_rate_alignment_loss(
    confidence: torch.Tensor,
    marginal_rate: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align signed confidence with a detached, sample-normalized rate target."""
    raw = (token_ids - 1).clamp(min=0, max=255)
    continuation = raw.ge(0x80) & raw.le(0xBF) & valid
    eligible = valid & ~continuation
    if eligible.size(1):
        eligible = eligible.clone()
        eligible[:, 0] = False
    weight = eligible.to(marginal_rate.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (marginal_rate * weight).sum(dim=1, keepdim=True) / count
    variance = ((marginal_rate - mean).square() * weight).sum(dim=1, keepdim=True) / count
    target = torch.tanh((marginal_rate - mean) / variance.sqrt().clamp(min=1.0e-5)).detach()
    if not eligible.any():
        zero = confidence.new_zeros(())
        return zero, zero, zero, zero, zero
    loss = F.smooth_l1_loss(confidence[eligible], target[eligible])
    centered_conf = confidence[eligible].float() - confidence[eligible].float().mean()
    centered_target = target[eligible].float() - target[eligible].float().mean()
    correlation = (
        (centered_conf * centered_target).mean()
        / (centered_conf.square().mean().sqrt() * centered_target.square().mean().sqrt()).clamp(min=1.0e-6)
    )
    target_cut_fraction = target[eligible].gt(0.9).float().mean()
    return loss, target[eligible].float().mean(), correlation, target_cut_fraction, target


def boundary_threshold_calibration_loss(
    confidence: torch.Tensor,
    target: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    tau_cut: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calibrate threshold-crossing probability without discrete pseudo-labels."""
    if temperature <= 0:
        raise ValueError("boundary calibration temperature must be positive")
    raw = (token_ids - 1).clamp(min=0, max=255)
    continuation = raw.ge(0x80) & raw.le(0xBF) & valid
    eligible = valid & ~continuation
    if eligible.size(1):
        eligible = eligible.clone()
        eligible[:, 0] = False
    if not eligible.any():
        zero = confidence.new_zeros(())
        return zero, zero
    target_probability = torch.sigmoid(
        (target[eligible].float() - float(tau_cut)) / float(temperature)
    ).detach()
    prediction_logits = (
        confidence[eligible].float() - float(tau_cut)
    ) / float(temperature)
    loss = F.binary_cross_entropy_with_logits(prediction_logits, target_probability)
    prediction_probability = torch.sigmoid(prediction_logits)
    probability_gap = (prediction_probability - target_probability).abs().mean()
    return loss, probability_gap


def boundary_threshold_density_loss(
    confidence: torch.Tensor,
    target: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    tau_cut: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the dynamic coding-rate cut count with hard-forward/soft-backward gradients."""
    if temperature <= 0:
        raise ValueError("boundary density temperature must be positive")
    raw = (token_ids - 1).clamp(min=0, max=255)
    continuation = raw.ge(0x80) & raw.le(0xBF) & valid
    eligible = valid & ~continuation
    if eligible.size(1):
        eligible = eligible.clone()
        eligible[:, 0] = False
    eligible_f = eligible.float()
    denominator = eligible_f.sum().clamp(min=1.0)

    soft_cut = torch.sigmoid((confidence.float() - float(tau_cut)) / float(temperature))
    hard_cut = confidence.gt(float(tau_cut)).float()
    priced_cut = soft_cut + (hard_cut - soft_cut).detach()
    target_cut = target.detach().gt(float(tau_cut)).float()
    predicted_density = (priced_cut * eligible_f).sum() / denominator
    target_density = (target_cut * eligible_f).sum() / denominator
    if not eligible.any():
        zero = confidence.new_zeros(())
        return zero, zero
    density_gap = predicted_density - target_density
    loss = F.smooth_l1_loss(
        predicted_density,
        target_density,
        beta=0.01,
    )
    return loss, density_gap.abs().detach()


def boundary_threshold_positive_margin_loss(
    confidence: torch.Tensor,
    target: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    tau_cut: float,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push coding-rate-positive positions across the fixed execution threshold."""
    if temperature <= 0:
        raise ValueError("boundary margin temperature must be positive")
    if margin < 0:
        raise ValueError("boundary positive margin must be non-negative")
    raw = (token_ids - 1).clamp(min=0, max=255)
    continuation = raw.ge(0x80) & raw.le(0xBF) & valid
    eligible = valid & ~continuation
    if eligible.size(1):
        eligible = eligible.clone()
        eligible[:, 0] = False
    positive = eligible & target.detach().gt(float(tau_cut))
    if not positive.any():
        zero = confidence.new_zeros(())
        return zero, zero
    required = float(tau_cut) + float(margin)
    shortfall = required - confidence.float()[positive]
    loss = float(temperature) * F.softplus(shortfall / float(temperature)).mean()
    return loss, F.relu(shortfall).mean().detach()


def boundary_rate_minimum_ratio_loss(
    confidence: torch.Tensor,
    target: torch.Tensor,
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    tau_cut: float,
    temperature: float,
    minimum_ratio: float,
    dual_value: torch.Tensor,
    augmented_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforce a batch-average lower cut ratio with hard-forward/soft-backward counts."""
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("boundary minimum rate ratio must be in (0, 1]")
    raw = (token_ids - 1).clamp(min=0, max=255)
    continuation = raw.ge(0x80) & raw.le(0xBF) & valid
    eligible = valid & ~continuation
    if eligible.size(1):
        eligible = eligible.clone()
        eligible[:, 0] = False
    eligible_f = eligible.float()
    denominator = eligible_f.sum().clamp(min=1.0)
    soft_cut = torch.sigmoid((confidence.float() - float(tau_cut)) / float(temperature))
    hard_cut = confidence.gt(float(tau_cut)).float()
    priced_cut = soft_cut + (hard_cut - soft_cut).detach()
    target_cut = target.detach().gt(float(tau_cut)).float()
    predicted_density = (priced_cut * eligible_f).sum() / denominator
    target_density = ((target_cut * eligible_f).sum() / denominator).detach()
    constraint = float(minimum_ratio) * target_density - predicted_density
    violation = F.relu(constraint)
    loss = dual_value.detach() * violation + 0.5 * float(augmented_weight) * violation.square()
    return loss, constraint, predicted_density.detach(), target_density


def boundary_compute_budget_loss(
    confidence: torch.Tensor,
    valid: torch.Tensor,
    force_continue: torch.Tensor,
    tau_cut: float,
    temperature: float,
    bytes_per_chunk: int,
    dual_value: torch.Tensor,
    augmented_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Price batch-average soft chunk creation without fixing per-sample K."""
    if bytes_per_chunk <= 0:
        raise ValueError("boundary target bytes_per_chunk must be positive")
    eligible = valid & ~force_continue
    soft_cut = torch.sigmoid((confidence.float() - float(tau_cut)) / float(temperature))
    soft_cut = soft_cut * eligible.float()
    hard_cut = confidence.gt(float(tau_cut)).float() * eligible.float()
    if soft_cut.size(1):
        soft_cut = soft_cut.clone()
        hard_cut = hard_cut.clone()
        soft_cut[:, 0] = 0.0
        hard_cut[:, 0] = 0.0
    # The cost ledger must see exactly the hard execution count in forward,
    # while the segmentor still receives the continuous sigmoid derivative.
    priced_cut = soft_cut + (hard_cut - soft_cut).detach()
    valid_bytes = valid.float().sum(dim=1).clamp(min=1.0)
    priced_chunks = 1.0 + priced_cut.sum(dim=1)
    target_chunks = torch.maximum(
        valid_bytes / float(bytes_per_chunk),
        torch.ones_like(valid_bytes),
    )
    priced_density = (priced_chunks / valid_bytes).mean()
    soft_density = ((1.0 + soft_cut.sum(dim=1)) / valid_bytes).mean().detach()
    target_density = (target_chunks / valid_bytes).mean().detach()
    constraint = priced_density - target_density
    positive_violation = F.relu(constraint)
    # This is an upper compute bound, not a target rate. Once a batch is below
    # budget, the controller must stop pushing toward fewer chunks; task and
    # structural signals remain free to choose any cheaper segmentation.
    loss = dual_value.detach() * positive_violation + 0.5 * float(augmented_weight) * positive_violation.square()
    return loss, constraint, priced_density, target_density, soft_density


def step_model(
    model: FLUEDV34,
    backbone: LatentInfillBackbone,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    collect_metrics: bool = True,
    global_step: int = -1,
    cbiu_state: CBIUState | None = None,
    update_cbiu_state: bool = True,
):
    clean = batch[0].to(device, non_blocking=device.type == "cuda")
    valid = clean.ne(PAD_ID)
    byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    src = clean.masked_fill(byte_mask, MASK_ID)
    model.collect_memory_diagnostics = bool(
        collect_metrics and args.training_scope != "emit_only"
    )
    out = model(src)

    zero_mask = torch.zeros_like(byte_mask)
    identity_targets, slot_mask, _ = make_targets(
        src, zero_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span
    )
    clean_targets, clean_slot_mask, masked_slot = make_targets(
        clean, byte_mask, out.chunks.chunk_ids, out.chunks.offsets, args.max_chunks, args.max_span
    )
    slot_mask = slot_mask & clean_slot_mask

    identity_ce = _ce(out.byte_logits, identity_targets)
    identity_loss = _mean_masked(identity_ce, slot_mask)

    affected_chunks = masked_slot.any(dim=-1) & out.chunks.chunk_mask
    if args.completion_mask_granularity == "readout":
        affected_readouts = masked_readouts_from_slots(
            masked_slot,
            out.chunks.chunk_mask,
            out.readout_z.size(2),
        )
    else:
        affected_readouts = affected_chunks.unsqueeze(-1).expand(-1, -1, out.readout_z.size(2))
    active_readouts = out.emit_hard if args.use_emit_controller else out.chunks.chunk_mask.unsqueeze(-1).expand_as(affected_readouts)
    # A masked byte must retain a writable latent slot even when the emit
    # controller would otherwise silence that optional readout.
    active_readouts = active_readouts | affected_readouts
    completed_logits, backbone_padded_units = _run_completion(
        model,
        backbone,
        out.readout_z,
        active_readouts,
        affected_readouts,
        out.chunks.chunk_mask,
    )
    completion_ce = _ce(completed_logits, clean_targets)
    masked_loss = _mean_masked(completion_ce, masked_slot)
    preserve_slot = slot_mask & affected_chunks.unsqueeze(-1) & ~masked_slot
    preserve_loss = _mean_masked(completion_ce, preserve_slot)

    boundary_loss, boundary_stats, boundary_components = boundary_prior_losses_v34(
        out.segmentor.confidence,
        clean,
        valid,
        collect_metrics=collect_metrics,
    )
    if args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}:
        boundary_rate_loss, boundary_rate_target_mean, confidence_rate_correlation, boundary_rate_target_cut_fraction, boundary_rate_target = (
            boundary_rate_alignment_loss(
                out.segmentor.confidence,
                out.aux["marginal_coding_rate"],
                clean,
                valid,
            )
        )
    else:
        boundary_rate_loss = out.readout_z.new_zeros(())
        boundary_rate_target_mean = out.readout_z.new_zeros(())
        confidence_rate_correlation = out.readout_z.new_zeros(())
        boundary_rate_target_cut_fraction = out.readout_z.new_zeros(())
        boundary_rate_target = out.segmentor.confidence.detach().new_zeros(
            out.segmentor.confidence.shape
        )
    if args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}:
        boundary_calibration_loss, boundary_calibration_probability_gap = (
            boundary_threshold_calibration_loss(
                out.segmentor.confidence,
                boundary_rate_target,
                clean,
                valid,
                args.tau_cut,
                args.boundary_calibration_temperature,
            )
        )
    else:
        boundary_calibration_loss = out.readout_z.new_zeros(())
        boundary_calibration_probability_gap = out.readout_z.new_zeros(())
    if args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}:
        boundary_density_loss, boundary_density_gap = boundary_threshold_density_loss(
            out.segmentor.confidence,
            boundary_rate_target,
            clean,
            valid,
            args.tau_cut,
            args.boundary_calibration_temperature,
        )
    else:
        boundary_density_loss = out.readout_z.new_zeros(())
        boundary_density_gap = out.readout_z.new_zeros(())
    if args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}:
        boundary_margin_loss, boundary_margin_shortfall = boundary_threshold_positive_margin_loss(
            out.segmentor.confidence,
            boundary_rate_target,
            clean,
            valid,
            args.tau_cut,
            args.boundary_rate_positive_margin,
            args.boundary_calibration_temperature,
        )
    else:
        boundary_margin_loss = out.readout_z.new_zeros(())
        boundary_margin_shortfall = out.readout_z.new_zeros(())
    if args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}:
        boundary_rate_dual_loss, boundary_rate_dual_constraint, boundary_rate_dual_predicted, boundary_rate_dual_target = (
            boundary_rate_minimum_ratio_loss(
                out.segmentor.confidence,
                boundary_rate_target,
                clean,
                valid,
                args.tau_cut,
                args.boundary_calibration_temperature,
                args.boundary_rate_min_ratio,
                model.boundary_rate_dual.clone(),
                args.boundary_rate_dual_augmented_weight,
            )
        )
    else:
        boundary_rate_dual_loss = out.readout_z.new_zeros(())
        boundary_rate_dual_constraint = out.readout_z.new_zeros(())
        boundary_rate_dual_predicted = out.readout_z.new_zeros(())
        boundary_rate_dual_target = out.readout_z.new_zeros(())
    if args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}:
        boundary_budget_loss, boundary_budget_constraint, requested_budget_chunks_per_byte, boundary_target_chunks_per_byte, requested_soft_chunks_per_byte = (
            boundary_compute_budget_loss(
                out.segmentor.confidence,
                valid,
                out.policy.force_continue,
                args.tau_cut,
                args.boundary_temperature,
                args.boundary_target_bytes_per_chunk,
                model.boundary_compute_dual.clone(),
                args.boundary_budget_augmented_weight,
            )
        )
    else:
        boundary_budget_loss = out.readout_z.new_zeros(())
        boundary_budget_constraint = out.readout_z.new_zeros(())
        requested_budget_chunks_per_byte = out.readout_z.new_zeros(())
        requested_soft_chunks_per_byte = out.readout_z.new_zeros(())
        boundary_target_chunks_per_byte = out.readout_z.new_zeros(())
    if args.boundary_mode == "threshold":
        coding_rate_loss, coding_rate = projected_coding_rate_loss(
            out.readout_z, out.chunks.chunk_mask, args.coding_rate_dim
        )
    else:
        coding_rate_loss = out.readout_z.new_zeros(())
        selected_rate = out.aux["marginal_coding_rate"][out.policy.hard_cut]
        coding_rate = selected_rate.float().mean().to(out.readout_z.dtype) if selected_rate.numel() else coding_rate_loss
    memory_gate_mean = out.aux["memory_gate_mean"]
    memory_context_norm = out.aux["memory_context_norm"]
    memory_residual_ratio = out.aux["memory_residual_ratio"]
    memory_usage_loss = (
        F.relu(args.memory_usage_min - memory_gate_mean).square()
        + F.relu(memory_gate_mean - args.memory_usage_max).square()
    ) if args.use_memory else memory_gate_mean.new_zeros(())
    emit_value_loss = out.readout_z.new_zeros(())
    emit_value_mean = out.readout_z.new_zeros(())
    emit_target_mean = out.readout_z.new_zeros(())
    sampled_emit_slot = 0
    cbiu_valid_actions = 0
    cbiu_quality_utility_mean = out.readout_z.new_zeros(())
    cbiu_cost_utility_mean = out.readout_z.new_zeros(())
    cbiu_constraint = out.readout_z.new_zeros(())
    cbiu_predicted_probability_mean = out.readout_z.new_zeros(())
    cbiu_brier = out.readout_z.new_zeros(())
    cbiu_sign_accuracy = out.readout_z.new_zeros(())
    cbiu_on_risk_means = out.readout_z.new_zeros((3,))
    cbiu_off_risk_means = out.readout_z.new_zeros((3,))
    cbiu_rho_on_mean = out.readout_z.new_zeros(())
    cbiu_rho_off_mean = out.readout_z.new_zeros(())
    cbiu_dominant_risk = out.readout_z.new_zeros((3,))
    if (
        args.use_emit_controller
        and out.readout_z.size(2) > 1
        and global_step >= 0
        and global_step % args.emit_value_every == 0
    ):
        if args.emit_target_mode == "cbiu":
            if cbiu_state is None:
                raise RuntimeError("emit_target_mode=cbiu requires initialized CBIU state")
            scored = _score_cbiu_emit_actions(
                model,
                backbone,
                clean,
                byte_mask,
                out,
                args,
                global_step,
            )
            sampled_emit_slot = int(scored["slot"])
            action_batches = scored["batch_indices"]
            action_chunks = scored["training_chunk_indices"]
            cbiu_valid_actions = int(action_batches.numel())
            if cbiu_valid_actions > 0:
                utility = cbiu_keep_utility(
                    scored["on_risks"],
                    scored["off_risks"],
                    scored["on_cost"],
                    scored["off_cost"],
                    cbiu_state,
                    args.cbiu_augmented_weight,
                )
                value = utility["net_utility"].detach()
                target = torch.sigmoid(value / args.cbiu_utility_temperature)
                predicted_logits = out.emit_logits[
                    action_batches,
                    action_chunks,
                    sampled_emit_slot,
                ]
                emit_value_loss = F.binary_cross_entropy_with_logits(predicted_logits, target)
                predicted_probability = torch.sigmoid(predicted_logits.detach())
                useful = value.gt(0).to(predicted_probability.dtype)
                emit_value_mean = value.mean()
                emit_target_mean = target.mean()
                cbiu_quality_utility_mean = utility["quality_utility"].mean()
                cbiu_cost_utility_mean = utility["cost_utility"].mean()
                cbiu_predicted_probability_mean = predicted_probability.mean()
                cbiu_brier = (predicted_probability - useful).square().mean()
                cbiu_sign_accuracy = predicted_probability.ge(0.5).eq(useful.bool()).float().mean()
                cbiu_on_risk_means = scored["on_risks"].mean(dim=0)
                cbiu_off_risk_means = scored["off_risks"].mean(dim=0)
                cbiu_rho_on_mean = utility["rho_on"].mean()
                cbiu_rho_off_mean = utility["rho_off"].mean()
                dominant = utility["on_normalized"].argmax(dim=-1)
                cbiu_dominant_risk = torch.stack(
                    [dominant.eq(index).float().mean() for index in range(3)]
                )
            if model.training and update_cbiu_state:
                cbiu_constraint = update_cbiu_dual(
                    cbiu_state,
                    scored.get("policy_cost", out.readout_z.new_zeros((1,))),
                    args.cbiu_dual_lr,
                    args.cbiu_dual_max,
                ).to(out.readout_z.dtype)
        else:
            sampled_emit_slot = 1 + (global_step // args.emit_value_every) % (out.readout_z.size(2) - 1)
            on_z = out.readout_z.clone()
            off_z = out.readout_z.clone()
            on_z[:, :, sampled_emit_slot] = out.readout_candidates[:, :, sampled_emit_slot]
            off_z[:, :, sampled_emit_slot] = 0.0
            on_active = active_readouts.clone()
            off_active = active_readouts.clone()
            on_active[:, :, sampled_emit_slot] = out.chunks.chunk_mask
            off_active[:, :, sampled_emit_slot] = False

            on_identity_logits = model.decode(on_z, out.chunks.chunk_mask, on_active)
            off_identity_logits = model.decode(off_z, out.chunks.chunk_mask, off_active)
            on_completed_logits, _ = _run_completion(
                model, backbone, on_z, on_active, affected_readouts, out.chunks.chunk_mask
            )
            off_completed_logits, _ = _run_completion(
                model, backbone, off_z, off_active, affected_readouts, out.chunks.chunk_mask
            )
            on_identity = _mean_per_chunk(_ce(on_identity_logits, identity_targets), slot_mask)
            off_identity = _mean_per_chunk(_ce(off_identity_logits, identity_targets), slot_mask)
            on_completion_ce = _ce(on_completed_logits, clean_targets)
            off_completion_ce = _ce(off_completed_logits, clean_targets)
            on_task = (
                args.identity_loss_weight * on_identity
                + args.completion_loss_weight * _mean_per_chunk(on_completion_ce, masked_slot)
                + args.preserve_loss_weight * _mean_per_chunk(on_completion_ce, preserve_slot)
            )
            off_task = (
                args.identity_loss_weight * off_identity
                + args.completion_loss_weight * _mean_per_chunk(off_completion_ce, masked_slot)
                + args.preserve_loss_weight * _mean_per_chunk(off_completion_ce, preserve_slot)
            )
            removal_delta = (off_task - on_task).detach()

            chunk_rate = removal_delta.new_zeros(out.chunks.chunk_mask.shape)
            starts = out.policy.hard_cut & valid
            b_idx, t_idx = starts.nonzero(as_tuple=True)
            c_idx = out.chunks.chunk_ids[b_idx, t_idx]
            in_range = c_idx.ge(0) & c_idx.lt(out.chunks.chunk_mask.size(1))
            chunk_rate[b_idx[in_range], c_idx[in_range]] = out.aux["marginal_coding_rate"][
                b_idx[in_range], t_idx[in_range]
            ].detach().float()
            active_rate = chunk_rate[out.chunks.chunk_mask]
            rate_scale = active_rate.abs().mean().clamp(min=1.0e-6)
            normalized_rate = chunk_rate / rate_scale

            active_count = active_readouts.float().sum(dim=(1, 2), keepdim=False)
            max_count = out.chunks.chunk_mask.float().sum(dim=1) * out.readout_z.size(2)
            marginal_compute_cost = args.emit_compute_cost_weight * (
                1.0 + active_count / max_count.clamp(min=1.0)
            ).unsqueeze(1)
            value = (
                removal_delta
                + args.emit_rate_value_weight * normalized_rate
                - marginal_compute_cost
            )
            target = torch.sigmoid(value / args.emit_value_temperature).detach()
            active_chunks = out.chunks.chunk_mask
            emit_value_loss = F.binary_cross_entropy_with_logits(
                out.emit_logits[:, :, sampled_emit_slot][active_chunks],
                target[active_chunks],
            )
            emit_value_mean = value[active_chunks].mean()
            emit_target_mean = target[active_chunks].mean()
    continuation_weight = (
        args.boundary_loss_weight
        if args.boundary_continuation_loss_weight is None
        else args.boundary_continuation_loss_weight
    )
    punctuation_weight = (
        args.boundary_loss_weight
        if args.boundary_punctuation_loss_weight is None
        else args.boundary_punctuation_loss_weight
    )
    neutral_weight = (
        args.boundary_loss_weight
        if args.boundary_neutral_loss_weight is None
        else args.boundary_neutral_loss_weight
    )
    weighted_boundary_prior = (
        continuation_weight * boundary_components["continuation"]
        + punctuation_weight * boundary_components["punctuation"]
        + neutral_weight * boundary_components["neutral_mean"]
    )
    identity_scale = float(args.decoder_loss_scale)
    completion_scale = 1.0
    if model.training and global_step >= 0 and args.training_scope != "emit_only":
        if args.decoder_warmup_steps > 0 and global_step < args.decoder_warmup_steps:
            completion_scale = 0.0
        elif args.decoder_alternating_period > 0:
            if (global_step // args.decoder_alternating_period) % 2 == 0:
                completion_scale = 0.0
            else:
                identity_scale = 0.0
    loss = (
        identity_scale * args.identity_loss_weight * identity_loss
        + completion_scale * args.completion_loss_weight * masked_loss
        + completion_scale * args.preserve_loss_weight * preserve_loss
        + weighted_boundary_prior
        + args.boundary_rate_alignment_weight * boundary_rate_loss
        + args.boundary_rate_calibration_weight * boundary_calibration_loss
        + args.boundary_rate_density_weight * boundary_density_loss
        + args.boundary_rate_margin_weight * boundary_margin_loss
        + boundary_rate_dual_loss
        + boundary_budget_loss
        + args.coding_rate_loss_weight * coding_rate_loss
        + args.memory_usage_loss_weight * memory_usage_loss
        + args.emit_value_loss_weight * emit_value_loss
        + args.ar_delta_loss_weight * out.ar_delta
    )
    if args.training_scope == "emit_only":
        # Pure policy attribution: the interface and temporary backbone stay
        # frozen, and no task-gradient shortcut is allowed into the gate.
        loss = args.emit_value_loss_weight * emit_value_loss + 0.0 * out.emit_logits.sum()
    if (
        model.training
        and global_step >= 0
        and args.boundary_mode in {"confidence_threshold", "uniform_confidence_blend"}
    ):
        with torch.no_grad():
            model.boundary_compute_dual.add_(
                float(args.boundary_dual_lr) * boundary_budget_constraint.detach()
            ).clamp_(min=0.0, max=float(args.boundary_dual_max))
            model.boundary_rate_dual.add_(
                float(args.boundary_rate_dual_lr) * boundary_rate_dual_constraint.detach()
            ).clamp_(min=0.0, max=float(args.boundary_rate_dual_max))

    metrics: Dict[str, float] = {}
    if collect_metrics:
        identity_pred = out.byte_logits.argmax(dim=-1)
        completed_pred = completed_logits.argmax(dim=-1)
        bytes_n = valid.float().sum().clamp(min=1.0)
        clean_raw = (clean - 1).clamp(min=0, max=255)
        eligible_boundary = valid & ~(clean_raw.ge(0x80) & clean_raw.le(0xBF))
        if eligible_boundary.size(1):
            eligible_boundary = eligible_boundary.clone()
            eligible_boundary[:, 0] = False
        eligible_boundary_n = eligible_boundary.float().sum().clamp(min=1.0)
        metrics = {
            "loss": float(loss.item()),
            "identity_loss": float(identity_loss.item()),
            "completion_masked_loss": float(masked_loss.item()),
            "completion_preserve_loss": float(preserve_loss.item()),
            "identity_weight_effective": float(identity_scale * args.identity_loss_weight),
            "completion_weight_effective": float(completion_scale * args.completion_loss_weight),
            "emit_warmup_active": float(bool(getattr(model, "emit_warmup_active", False))),
            "identity_acc": _safe_acc(identity_pred, identity_targets, slot_mask),
            "identity_mask_is_mask_acc": _safe_acc(identity_pred, identity_targets, masked_slot),
            "completion_mask_acc": _safe_acc(completed_pred, clean_targets, masked_slot),
            "completion_preserve_acc": _safe_acc(completed_pred, clean_targets, preserve_slot),
            "completion_ppl": float(torch.exp(masked_loss.detach().float().clamp(max=20)).item()),
            "masked_byte_pseudo_ppl": float(torch.exp(masked_loss.detach().float().clamp(max=20)).item()),
            "boundary_confidence_controls_execution": float(
                args.boundary_mode in {"threshold", "confidence_threshold"}
            ),
            "boundary_confidence_controls_soft_bridge": float(
                args.boundary_mode in {"threshold", "confidence_threshold", "uniform_confidence_blend"}
            ),
            "coding_score_controls_execution": float(args.boundary_mode == "marginal_rate_topk"),
            "uniform_budget_controls_execution": float(
                args.boundary_mode in {"uniform_budget", "uniform_l2_blend", "uniform_confidence_blend"}
            ),
            "boundary_rate_alignment_loss": float(boundary_rate_loss.item()),
            "boundary_rate_calibration_loss": float(boundary_calibration_loss.item()),
            "boundary_rate_calibration_probability_gap": float(
                boundary_calibration_probability_gap.item()
            ),
            "boundary_rate_density_loss": float(boundary_density_loss.item()),
            "boundary_rate_density_gap": float(boundary_density_gap.item()),
            "boundary_rate_margin_loss": float(boundary_margin_loss.item()),
            "boundary_rate_margin_shortfall": float(boundary_margin_shortfall.item()),
            "boundary_rate_dual_loss": float(boundary_rate_dual_loss.item()),
            "boundary_rate_dual_constraint": float(boundary_rate_dual_constraint.item()),
            "boundary_rate_dual_predicted": float(boundary_rate_dual_predicted.item()),
            "boundary_rate_dual_target": float(boundary_rate_dual_target.item()),
            "boundary_rate_dual": float(model.boundary_rate_dual.item()),
            "boundary_rate_target_mean": float(boundary_rate_target_mean.item()),
            "confidence_rate_correlation": float(confidence_rate_correlation.item()),
            "boundary_rate_target_cut_fraction": float(boundary_rate_target_cut_fraction.item()),
            "boundary_compute_budget_loss": float(boundary_budget_loss.item()),
            "boundary_compute_constraint": float(boundary_budget_constraint.item()),
            "boundary_compute_dual": float(model.boundary_compute_dual.item()),
            "requested_soft_chunks_per_byte": float(requested_soft_chunks_per_byte.item()),
            "requested_budget_chunks_per_byte": float(requested_budget_chunks_per_byte.item()),
            "boundary_target_chunks_per_byte": float(boundary_target_chunks_per_byte.item()),
            "hard_cut_fraction": float((out.policy.hard_cut & valid).float().sum().item() / bytes_n.item()),
            "eligible_hard_cut_fraction": float(
                (out.policy.hard_cut & eligible_boundary).float().sum().item()
                / eligible_boundary_n.item()
            ),
            "requested_hard_cut_fraction": float(
                (out.aux["requested_hard_cut"] & valid).float().sum().item() / bytes_n.item()
            ),
            "requested_eligible_hard_cut_fraction": float(
                (out.aux["requested_hard_cut"] & eligible_boundary).float().sum().item()
                / eligible_boundary_n.item()
            ),
            "cut_capacity_overflow": float(out.aux["cut_capacity_overflow"].float().sum().item()),
            "cut_capacity_overflow_per_byte": float(
                out.aux["cut_capacity_overflow"].float().sum().item() / bytes_n.item()
            ),
            "chunks_per_byte": float(out.chunks.chunk_mask.float().sum().item() / bytes_n.item()),
            "forced_max_span_chunks_per_byte": float(
                (
                    out.chunks.chunk_mask.float().sum()
                    - (out.policy.hard_cut & valid).float().sum()
                ).clamp(min=0).item()
                / bytes_n.item()
            ),
            "ar_delta": float(out.ar_delta.item()),
            "coding_rate": float(coding_rate.item()),
            "memory_gate_mean": float(memory_gate_mean.item()),
            "memory_context_norm": float(memory_context_norm.item()),
            "memory_context_raw_norm": float(out.aux["memory_context_raw_norm"].item()),
            "memory_effective_scale": float(out.aux["memory_effective_scale"].item()),
            "memory_residual_ratio": float(memory_residual_ratio.item()),
            "memory_attention_current_share": float(out.aux["memory_attention_current_share"].item()),
            "memory_attention_other_share": float(out.aux["memory_attention_other_share"].item()),
            "current_channel_attention_share": float(out.aux["current_channel_attention_share"].item()),
            "current_memory_context_norm": float(out.aux["current_memory_context_norm"].item()),
            "current_memory_context_raw_norm": float(out.aux["current_memory_context_raw_norm"].item()),
            "current_memory_effective_scale": float(out.aux["current_memory_effective_scale"].item()),
            "current_memory_readout_cosine": float(out.aux["current_memory_readout_cosine"].item()),
            "current_memory_contribution_share": float(out.aux["current_memory_contribution_share"].item()),
            "memory_usage_loss": float(memory_usage_loss.item()),
            "emit_value_loss": float(emit_value_loss.item()),
            "emit_value_mean": float(emit_value_mean.item()),
            "emit_target_mean": float(emit_target_mean.item()),
            "sampled_emit_slot": float(sampled_emit_slot),
            "cbiu_valid_actions": float(cbiu_valid_actions),
            "cbiu_quality_utility_mean": float(cbiu_quality_utility_mean.item()),
            "cbiu_cost_utility_mean": float(cbiu_cost_utility_mean.item()),
            "cbiu_net_utility_mean": float(emit_value_mean.item()) if args.emit_target_mode == "cbiu" else 0.0,
            "cbiu_compute_constraint": float(cbiu_constraint.item()),
            "cbiu_compute_dual": float(cbiu_state.compute_dual.item()) if cbiu_state is not None else 0.0,
            "cbiu_predicted_probability_mean": float(cbiu_predicted_probability_mean.item()),
            "cbiu_brier": float(cbiu_brier.item()),
            "cbiu_sign_accuracy": float(cbiu_sign_accuracy.item()),
            "cbiu_rho_on_mean": float(cbiu_rho_on_mean.item()),
            "cbiu_rho_off_mean": float(cbiu_rho_off_mean.item()),
            **{
                f"cbiu_{name}_on_bpb": float(cbiu_on_risk_means[index].item())
                for index, name in enumerate(CBIU_RISK_NAMES)
            },
            **{
                f"cbiu_{name}_off_bpb": float(cbiu_off_risk_means[index].item())
                for index, name in enumerate(CBIU_RISK_NAMES)
            },
            **{
                f"cbiu_dominant_{name}_fraction": float(cbiu_dominant_risk[index].item())
                for index, name in enumerate(CBIU_RISK_NAMES)
            },
            "soft_readout_units_per_byte": float(out.emit_soft.sum().item() / bytes_n.item()),
            "policy_readout_units_per_byte": float(out.emit_hard.float().sum().item() / bytes_n.item()),
            "actual_backbone_units_per_byte": float(active_readouts.float().sum().item() / bytes_n.item()),
            "backbone_padded_units_per_byte": float(
                backbone_padded_units * clean.size(0) / bytes_n.item()
            ),
            "truncated_tokens": float(out.chunks.pack_info["truncated_tokens"].float().sum().item()),
        }
        metrics.update(boundary_stats)
    return loss, metrics


def _relative_delta(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).float().norm().item() / max(0.5 * (x.float().norm().item() + y.float().norm().item()), 1.0e-9))


@torch.no_grad()
def order_probe(model: FLUEDV34, seq_len: int, device: torch.device) -> Dict[str, float]:
    base = "the model reads from cache and returns the value."
    swap = "the model raeds from cache and returns the value."
    substitute = "the model rxyds from cache and returns the value."

    def encode(text: str):
        ids = text_to_byte_ids(text)[:seq_len]
        src = torch.tensor([ids + [PAD_ID] * (seq_len - len(ids))], device=device)
        return model(src)

    was_training = model.training
    model.eval()
    base_out, swap_out, subst_out = encode(base), encode(swap), encode(substitute)
    if was_training:
        model.train()
    chunk = int(base_out.chunks.chunk_ids[0, 11].item())
    memory_swap = _relative_delta(base_out.memory_z[:, chunk], swap_out.memory_z[:, chunk])
    memory_subst = _relative_delta(base_out.memory_z[:, chunk], subst_out.memory_z[:, chunk])
    readout_swap = _relative_delta(base_out.readout_z[:, chunk], swap_out.readout_z[:, chunk])
    readout_subst = _relative_delta(base_out.readout_z[:, chunk], subst_out.readout_z[:, chunk])
    return {
        "order_same_hard_cut": float(torch.equal(base_out.policy.hard_cut, swap_out.policy.hard_cut)),
        "order_memory_swap_delta": memory_swap,
        "order_memory_substitute_delta": memory_subst,
        "order_memory_swap_to_substitute": memory_swap / max(memory_subst, 1.0e-9),
        "order_readout_swap_delta": readout_swap,
        "order_readout_substitute_delta": readout_subst,
        "order_readout_swap_to_substitute": readout_swap / max(readout_subst, 1.0e-9),
    }


@torch.no_grad()
def evaluate(model, backbone, loader, args, device, cbiu_state: CBIUState | None = None) -> Dict[str, float]:
    model.eval()
    backbone.eval()
    previous_emit_warmup = getattr(model, "emit_warmup_active", False)
    model.emit_warmup_active = False
    rows: List[Dict[str, float]] = []
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(args.eval_mask_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_mask_seed)
        for index, batch in enumerate(loader):
            if index >= args.max_eval_batches:
                break
            eval_step = index * args.emit_value_every if args.use_emit_controller else -1
            _loss, metrics = step_model(
                model,
                backbone,
                batch,
                args,
                device,
                collect_metrics=True,
                global_step=eval_step,
                cbiu_state=cbiu_state,
                update_cbiu_state=False,
            )
            rows.append(metrics)
    result = _avg_metrics(rows)
    result.update(order_probe(model, args.seq_len, device))
    model.emit_warmup_active = previous_emit_warmup
    model.train()
    backbone.train()
    return result


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        if getattr(args, "deterministic", False):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = out_dir / "latest.pt"

    model = build_model(args).to(device)
    model.boundary_rate_dual = torch.zeros((), device=device)
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    ).to(device)
    if args.emit_target_mode == "cbiu" and args.decoder_mode != "shared_inverse":
        raise ValueError("online CBIU requires decoder_mode=shared_inverse so hard emit changes execution")
    cbiu_state = None
    if args.emit_target_mode == "cbiu":
        if not args.cbiu_anchor_file:
            raise ValueError("emit_target_mode=cbiu requires --cbiu-anchor-file")
        cbiu_state = CBIUState.from_anchor_file(
            args.cbiu_anchor_file,
            args.cbiu_compute_budget,
            device,
        )
        if args.cbiu_require_anchor_checkpoint_match and args.init_checkpoint:
            expected = Path(cbiu_state.anchor_checkpoint).resolve()
            actual = Path(args.init_checkpoint).resolve()
            if expected != actual:
                raise RuntimeError(
                    "CBIU anchor checkpoint does not match initialization checkpoint: "
                    f"anchor={expected}, init={actual}"
                )
    if args.init_checkpoint and not (args.resume and latest_path.exists()) and not args.dry_run:
        init_path = Path(args.init_checkpoint)
        init_payload = torch.load(init_path, map_location=device, weights_only=False)
        if args.reset_emit_controller:
            init_model = {
                key: value
                for key, value in init_payload["model"].items()
                if not key.startswith("emit_controller.")
            }
            missing, unexpected = model.load_state_dict(init_model, strict=False)
            if unexpected or any(not key.startswith("emit_controller.") for key in missing):
                raise RuntimeError(
                    f"unexpected partial initialization mismatch: missing={missing}, unexpected={unexpected}"
                )
            model.emit_controller.reset_parameters()
        else:
            model.load_state_dict(init_payload["model"])
        backbone.load_state_dict(init_payload["backbone"])
        print(f"[init-checkpoint] {init_path}", flush=True)
    model_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters())
    result_base = {
        "model_version": "flued_v3_4",
        "run_id": args.run_id,
        "model_params": model_params,
        "backbone_params": backbone_params,
        "total_params": model_params + backbone_params,
        "args": vars(args),
    }
    if args.dry_run:
        print(json.dumps(result_base, ensure_ascii=False, indent=2))
        return result_base

    train_loader, eval_loader = make_dataloaders(args)
    if args.training_scope == "emit_only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in model.emit_controller.parameters():
            parameter.requires_grad_(True)
        params = list(model.emit_controller.parameters())
    else:
        params = list(model.parameters()) + list(backbone.parameters())
    result_base["trainable_params"] = sum(parameter.numel() for parameter in params)
    optimizer = build_optimizer(args, params)
    scheduler = _cosine_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    start_step = 0
    if args.resume and latest_path.exists():
        resume_payload = torch.load(latest_path, map_location=device, weights_only=False)
        if "optimizer" in resume_payload and "scheduler" in resume_payload:
            try:
                model.load_state_dict(resume_payload["model"])
            except RuntimeError as exc:
                raise RuntimeError(
                    "training checkpoint is not exactly compatible with the current v3.4 model; "
                    "start a new output directory or pass --no-resume. Evaluation tools allow "
                    "only explicitly inactive legacy fields."
                ) from exc
            backbone.load_state_dict(resume_payload["backbone"])
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            start_step = int(resume_payload.get("step", 0))
            if "torch_rng_state" in resume_payload:
                torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state" in resume_payload:
                torch.cuda.set_rng_state(resume_payload["cuda_rng_state"].cpu(), device)
            if "boundary_rate_dual" in resume_payload:
                model.boundary_rate_dual.copy_(resume_payload["boundary_rate_dual"].to(device))
            if cbiu_state is not None:
                if "cbiu_state" not in resume_payload:
                    raise RuntimeError("CBIU checkpoint has no protocol state; refusing silent reset")
                cbiu_state.load_state_dict(resume_payload["cbiu_state"], device)
            print(f"[resume] {latest_path} at completed_step={start_step}", flush=True)

    def checkpoint_payload(completed_step: int, summary: dict | None = None) -> dict:
        payload = {
            "model": model.state_dict(),
            "backbone": backbone.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "step": int(completed_step),
            "torch_rng_state": torch.get_rng_state(),
            "summary": summary or {},
            "runtime_boundary_state": {
                "mode": model.config.boundary_mode,
                "coding_rate_mode": model.coding_rate_selector.mode,
                "blend_alpha": float(model.config.boundary_blend_alpha),
            },
            "boundary_rate_dual": model.boundary_rate_dual.detach().cpu(),
        }
        if cbiu_state is not None:
            payload["cbiu_state"] = cbiu_state.state_dict()
        if device.type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        return payload
    train_iter = iter(train_loader)
    log_path = out_dir / "train_log.jsonl"
    model.train()
    backbone.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    if apply_boundary_curriculum(model, args, start_step):
        print(
            f"[boundary-curriculum] resume in {model.config.boundary_mode}/"
            f"{model.coding_rate_selector.mode} at step={start_step}",
            flush=True,
        )
    for step in range(start_step, args.max_steps):
        model.emit_warmup_active = bool(
            args.use_emit_controller and args.emit_warmup_steps > 0 and step < args.emit_warmup_steps
        )
        if apply_boundary_curriculum(model, args, step):
            print(
                f"[boundary-curriculum] switch at step={step}: "
                f"{model.config.boundary_mode}/{model.coding_rate_selector.mode}",
                flush=True,
            )
        optimizer.zero_grad(set_to_none=True)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        collect = step % args.log_every == 0
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            loss, metrics = step_model(
                model,
                backbone,
                batch,
                args,
                device,
                collect_metrics=collect,
                global_step=step,
                cbiu_state=cbiu_state,
            )
        loss.backward()
        if collect:
            segmentor_head_grads = [
                parameter.grad.float().square().sum()
                for parameter in model.segmentor_head.parameters()
                if parameter.grad is not None
            ]
            metrics["segmentor_head_grad_norm"] = float(
                torch.stack(segmentor_head_grads).sum().sqrt().item()
                if segmentor_head_grads
                else 0.0
            )
        if collect and getattr(model, "_diagnostic_byte_input", None) is not None:
            byte_grad = model._diagnostic_byte_input.grad
            metrics["current_byte_input_grad_rms"] = (
                float(byte_grad.float().square().mean().sqrt().item()) if byte_grad is not None else 0.0
            )
        grad = nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if collect:
            elapsed = time.perf_counter() - start
            row = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "grad": float(grad),
                "steps_per_sec": (step + 1) / max(elapsed, 1e-9),
                "boundary_mode": model.config.boundary_mode,
                "boundary_coding_rate_mode": model.coding_rate_selector.mode,
                "boundary_blend_alpha": float(model.config.boundary_blend_alpha),
                **metrics,
            }
            _append_jsonl(log_path, row)
            print(
                f"step={step} loss={row['loss']:.4f} identity={row['identity_acc']:.3f} "
                f"complete={row['completion_mask_acc']:.3f} keep={row['completion_preserve_acc']:.3f} "
                f"speed={row['steps_per_sec']:.2f}",
                flush=True,
            )
        completed_step = step + 1
        if args.checkpoint_every > 0 and completed_step % args.checkpoint_every == 0:
            payload = checkpoint_payload(completed_step)
            torch.save(payload, latest_path)
            if args.milestone_every > 0 and completed_step % args.milestone_every == 0:
                torch.save(payload, out_dir / f"step_{completed_step:06d}.pt")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_elapsed = time.perf_counter() - start
    steps_run_this_process = max(args.max_steps - start_step, 0)
    train_peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    eval_stats = evaluate(model, backbone, eval_loader, args, device, cbiu_state=cbiu_state)
    elapsed = time.perf_counter() - start
    result = {
        **result_base,
        "steps": args.max_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_run_this_process / max(elapsed, 1.0e-9),
        "train_elapsed_sec": train_elapsed,
        "train_steps_per_sec": steps_run_this_process / max(train_elapsed, 1.0e-9),
        "train_peak_memory_gb": train_peak_memory_gb,
        "final_boundary_mode": model.config.boundary_mode,
        "final_boundary_coding_rate_mode": model.coding_rate_selector.mode,
        "final_boundary_blend_alpha": float(model.config.boundary_blend_alpha),
        **{f"eval_{key}": value for key, value in eval_stats.items()},
    }
    payload = checkpoint_payload(args.max_steps, result)
    torch.save(payload, latest_path)
    final_milestone = out_dir / f"step_{args.max_steps:06d}.pt"
    if not final_milestone.exists():
        torch.save(payload, final_milestone)
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults = {}
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--run-id", default="v34_probe")
    parser.add_argument("--out-dir", default="checkpoints/v34_probe")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--streaming-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-samples-per-worker", type=int, default=20000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-mask-seed", type=int, default=1042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--segmentor-layers", type=int, default=5)
    parser.add_argument("--interpreter-layers", type=int, default=3)
    parser.add_argument("--memory-rank", type=int, default=4)
    parser.add_argument("--readout-vectors", type=int, default=4)
    parser.add_argument("--ar-hidden", type=int, default=128)
    parser.add_argument("--use-position", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--position-strategy",
        choices=["layered_rope", "prompt_additive", "prompt_plus_local_rope", "none"],
        default="layered_rope",
    )
    parser.add_argument("--prompt-position-scale", type=float, default=0.1)
    parser.add_argument("--use-prompt-alibi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-structured-lookup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-boundary-bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-use-position", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--memory-position-mode",
        choices=["legacy", "none", "chunk_rope", "byte_alibi"],
        default="legacy",
    )
    parser.add_argument("--memory-residual-scale", type=float, default=0.1)
    parser.add_argument("--memory-context-norm", choices=["none", "layernorm"], default="none")
    parser.add_argument("--memory-scale-mode", choices=["fixed", "bounded"], default="fixed")
    parser.add_argument("--memory-scale-max", type=float, default=0.1)
    parser.add_argument(
        "--memory-access-mode",
        choices=["other_only", "all", "none"],
        default="other_only",
    )
    parser.add_argument(
        "--current-memory-mode",
        choices=["off", "separate_detached", "separate_e2e"],
        default="off",
    )
    parser.add_argument("--current-memory-scale", type=float, default=0.03)
    parser.add_argument("--current-memory-scale-max", type=float, default=0.1)
    parser.add_argument(
        "--boundary-mode",
        choices=["threshold", "confidence_threshold", "marginal_rate_topk", "uniform_budget", "uniform_l2_blend", "uniform_confidence_blend"],
        default="threshold",
    )
    parser.add_argument("--boundary-coding-rate-dim", type=int, default=16)
    parser.add_argument("--boundary-coding-rate-epsilon", type=float, default=1.0)
    parser.add_argument("--boundary-coding-rate-temperature", type=float, default=0.15)
    parser.add_argument("--boundary-coding-rate-mode", choices=["exact", "diag", "l2"], default="exact")
    parser.add_argument("--boundary-blend-alpha", type=float, default=1.0)
    parser.add_argument("--boundary-curriculum-switch-step", type=int, default=0)
    parser.add_argument("--boundary-curriculum-transition-steps", type=int, default=0)
    parser.add_argument(
        "--boundary-curriculum-mode",
        choices=["threshold", "confidence_threshold", "marginal_rate_topk", "uniform_budget", "uniform_l2_blend", "uniform_confidence_blend"],
        default="marginal_rate_topk",
    )
    parser.add_argument("--boundary-curriculum-coding-rate-mode", choices=["exact", "diag", "l2"], default="diag")
    parser.add_argument("--fixed-chunk-budget", type=int, default=0)
    parser.add_argument("--bytes-per-chunk-budget", type=int, default=16)
    parser.add_argument("--use-emit-controller", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--emit-forward-mode", choices=["hard_st", "soft"], default="hard_st")
    parser.add_argument("--emit-initial-probability", type=float, default=0.1)
    parser.add_argument("--emit-warmup-steps", type=int, default=0)
    parser.add_argument("--emit-threshold", type=float, default=0.5)
    parser.add_argument("--emit-controller-hidden", type=int, default=0)
    parser.add_argument("--emit-controller-slot-embedding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-emit-controller", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--max-span", type=int, default=128)
    parser.add_argument("--tau-cut", type=float, default=0.9)
    parser.add_argument("--tau-trans", type=float, default=0.75)
    parser.add_argument("--boundary-temperature", type=float, default=0.15)
    parser.add_argument("--boundary-bridge-gradient-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument(
        "--completion-mask-granularity",
        choices=["chunk", "readout"],
        default="chunk",
        help="chunk reproduces historical v3.4 runs; readout restores the v3.3 slot-local mapping",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=["legacy_independent", "shared_inverse"],
        default="legacy_independent",
    )
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--completion-loss-weight", type=float, default=2.0)
    parser.add_argument("--preserve-loss-weight", type=float, default=0.5)
    parser.add_argument("--decoder-warmup-steps", type=int, default=0)
    parser.add_argument("--decoder-alternating-period", type=int, default=0)
    parser.add_argument("--decoder-loss-scale", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.02)
    parser.add_argument("--boundary-continuation-loss-weight", type=float, default=None)
    parser.add_argument("--boundary-punctuation-loss-weight", type=float, default=None)
    parser.add_argument("--boundary-neutral-loss-weight", type=float, default=None)
    parser.add_argument("--boundary-rate-alignment-weight", type=float, default=0.02)
    parser.add_argument("--boundary-rate-calibration-weight", type=float, default=0.0)
    parser.add_argument("--boundary-rate-density-weight", type=float, default=0.0)
    parser.add_argument("--boundary-rate-margin-weight", type=float, default=0.0)
    parser.add_argument("--boundary-rate-positive-margin", type=float, default=0.02)
    parser.add_argument("--boundary-rate-min-ratio", type=float, default=0.8)
    parser.add_argument("--boundary-rate-dual-lr", type=float, default=0.0)
    parser.add_argument("--boundary-rate-dual-max", type=float, default=1.0)
    parser.add_argument("--boundary-rate-dual-augmented-weight", type=float, default=0.0)
    parser.add_argument("--boundary-calibration-temperature", type=float, default=0.1)
    parser.add_argument("--boundary-target-bytes-per-chunk", type=int, default=16)
    parser.add_argument("--boundary-dual-lr", type=float, default=0.05)
    parser.add_argument("--boundary-dual-max", type=float, default=20.0)
    parser.add_argument("--boundary-budget-augmented-weight", type=float, default=1.0)
    parser.add_argument("--coding-rate-loss-weight", type=float, default=0.01)
    parser.add_argument("--coding-rate-dim", type=int, default=64)
    parser.add_argument("--memory-usage-loss-weight", type=float, default=0.02)
    parser.add_argument("--memory-usage-min", type=float, default=0.20)
    parser.add_argument("--memory-usage-max", type=float, default=0.50)
    parser.add_argument("--emit-value-loss-weight", type=float, default=0.1)
    parser.add_argument("--emit-value-every", type=int, default=4)
    parser.add_argument("--emit-target-mode", choices=["legacy", "cbiu"], default="legacy")
    parser.add_argument("--emit-rate-value-weight", type=float, default=0.05)
    parser.add_argument("--emit-compute-cost-weight", type=float, default=0.05)
    parser.add_argument("--emit-value-temperature", type=float, default=0.25)
    parser.add_argument("--cbiu-anchor-file", default="")
    parser.add_argument("--cbiu-require-anchor-checkpoint-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cbiu-compute-budget", type=float, default=0.18)
    parser.add_argument("--cbiu-dual-lr", type=float, default=0.0)
    parser.add_argument("--cbiu-dual-max", type=float, default=20.0)
    parser.add_argument("--cbiu-augmented-weight", type=float, default=0.0)
    parser.add_argument("--cbiu-utility-temperature", type=float, default=0.25)
    parser.add_argument("--ar-delta-loss-weight", type=float, default=0.01)
    parser.add_argument("--backbone-hidden", type=int, default=384)
    parser.add_argument("--backbone-layers", type=int, default=3)
    parser.add_argument("--backbone-nhead", type=int, default=8)
    parser.add_argument("--backbone-ffn-dim", type=int, default=1024)
    parser.add_argument("--optimizer", choices=["fused_adamw", "foreach_adamw", "adamw"], default="fused_adamw")
    parser.add_argument("--training-scope", choices=["joint", "emit_only"], default="joint")
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--milestone-every", type=int, default=2500)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
