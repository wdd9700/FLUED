"""Train the FLUED v3.4 RoPE x small-AR probe.

The two supervised paths deliberately share the same masked source:

* identity: encoder/decoder must reproduce the observed stream, including MASK;
* completion: the temporary backbone replaces affected readouts and must recover
  the clean bytes while preserving visible bytes in the same chunks.
"""

from __future__ import annotations

import argparse
import json
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
from flued.v34 import FLUEDV34Probe, FLUEDV34ProbeConfig  # noqa: E402
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
)


def build_model(args: argparse.Namespace) -> FLUEDV34Probe:
    return FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
            segmentor_layers=args.segmentor_layers,
            interpreter_layers=args.interpreter_layers,
            memory_rank=args.memory_rank,
            readout_vectors=args.readout_vectors,
            ar_hidden=args.ar_hidden,
            use_position=args.use_position,
            use_ar=args.use_ar,
            use_structured_lookup=args.use_structured_lookup,
            use_memory=args.use_memory,
            use_logic_prior=args.use_logic_prior,
            use_boundary_bridge=args.use_boundary_bridge,
            boundary_mode=getattr(args, "boundary_mode", "threshold"),
            coding_rate_dim=getattr(args, "boundary_coding_rate_dim", 16),
            coding_rate_epsilon=getattr(args, "boundary_coding_rate_epsilon", 1.0),
            coding_rate_temperature=getattr(args, "boundary_coding_rate_temperature", 0.15),
            coding_rate_mode=getattr(args, "boundary_coding_rate_mode", "exact"),
            fixed_chunk_budget=getattr(args, "fixed_chunk_budget", 0),
            bytes_per_chunk_budget=getattr(args, "bytes_per_chunk_budget", 16),
            use_emit_controller=getattr(args, "use_emit_controller", False),
            emit_forward_mode=getattr(args, "emit_forward_mode", "hard_st"),
            emit_initial_probability=getattr(args, "emit_initial_probability", 0.1),
            max_chunks=args.max_chunks,
            max_span=args.max_span,
            tau_cut=args.tau_cut,
            tau_trans=args.tau_trans,
            boundary_temperature=args.boundary_temperature,
            noise_scale=args.noise_scale,
        )
    )


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
    logits = model.decoder(completed, chunk_mask)
    return logits, compact_z.size(1)


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
    return loss, stats


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


def step_model(
    model: FLUEDV34Probe,
    backbone: LatentInfillBackbone,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    collect_metrics: bool = True,
    global_step: int = -1,
):
    clean = batch[0].to(device, non_blocking=device.type == "cuda")
    valid = clean.ne(PAD_ID)
    byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    src = clean.masked_fill(byte_mask, MASK_ID)
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
    affected_readouts = affected_chunks.unsqueeze(-1).expand(-1, -1, out.readout_z.size(2))
    active_readouts = out.emit_hard if args.use_emit_controller else out.chunks.chunk_mask.unsqueeze(-1).expand_as(affected_readouts)
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

    boundary_loss, boundary_stats = boundary_prior_losses_v34(
        out.segmentor.confidence,
        src,
        valid,
        collect_metrics=collect_metrics,
    )
    if args.boundary_mode == "threshold":
        coding_rate_loss, coding_rate = projected_coding_rate_loss(
            out.readout_z, out.chunks.chunk_mask, args.coding_rate_dim
        )
    else:
        coding_rate_loss = out.readout_z.new_zeros(())
        selected_rate = out.aux["marginal_coding_rate"][out.policy.hard_cut]
        coding_rate = selected_rate.float().mean().to(out.readout_z.dtype) if selected_rate.numel() else coding_rate_loss
    memory_gate_mean = out.aux["memory_gate_mean"]
    memory_usage_loss = (
        F.relu(args.memory_usage_min - memory_gate_mean).square()
        + F.relu(memory_gate_mean - args.memory_usage_max).square()
    )
    emit_value_loss = out.readout_z.new_zeros(())
    emit_value_mean = out.readout_z.new_zeros(())
    emit_target_mean = out.readout_z.new_zeros(())
    sampled_emit_slot = 0
    if (
        args.use_emit_controller
        and out.readout_z.size(2) > 1
        and global_step >= 0
        and global_step % args.emit_value_every == 0
    ):
        sampled_emit_slot = 1 + (global_step // args.emit_value_every) % (out.readout_z.size(2) - 1)
        on_z = out.readout_z.clone()
        off_z = out.readout_z.clone()
        on_z[:, :, sampled_emit_slot] = out.readout_candidates[:, :, sampled_emit_slot]
        off_z[:, :, sampled_emit_slot] = 0.0
        on_active = active_readouts.clone()
        off_active = active_readouts.clone()
        on_active[:, :, sampled_emit_slot] = out.chunks.chunk_mask
        off_active[:, :, sampled_emit_slot] = False

        on_identity_logits = model.decoder(on_z, out.chunks.chunk_mask)
        off_identity_logits = model.decoder(off_z, out.chunks.chunk_mask)
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
    loss = (
        args.identity_loss_weight * identity_loss
        + args.completion_loss_weight * masked_loss
        + args.preserve_loss_weight * preserve_loss
        + args.boundary_loss_weight * boundary_loss
        + args.coding_rate_loss_weight * coding_rate_loss
        + args.memory_usage_loss_weight * memory_usage_loss
        + args.emit_value_loss_weight * emit_value_loss
        + args.ar_delta_loss_weight * out.ar_delta
    )

    metrics: Dict[str, float] = {}
    if collect_metrics:
        identity_pred = out.byte_logits.argmax(dim=-1)
        completed_pred = completed_logits.argmax(dim=-1)
        bytes_n = valid.float().sum().clamp(min=1.0)
        metrics = {
            "loss": float(loss.item()),
            "identity_loss": float(identity_loss.item()),
            "completion_masked_loss": float(masked_loss.item()),
            "completion_preserve_loss": float(preserve_loss.item()),
            "identity_acc": _safe_acc(identity_pred, identity_targets, slot_mask),
            "identity_mask_is_mask_acc": _safe_acc(identity_pred, identity_targets, masked_slot),
            "completion_mask_acc": _safe_acc(completed_pred, clean_targets, masked_slot),
            "completion_preserve_acc": _safe_acc(completed_pred, clean_targets, preserve_slot),
            "completion_ppl": float(torch.exp(masked_loss.detach().float().clamp(max=20)).item()),
            "hard_cut_fraction": float((out.policy.hard_cut & valid).float().sum().item() / bytes_n.item()),
            "requested_hard_cut_fraction": float(
                (out.aux["requested_hard_cut"] & valid).float().sum().item() / bytes_n.item()
            ),
            "cut_capacity_overflow": float(out.aux["cut_capacity_overflow"].float().sum().item()),
            "chunks_per_byte": float(out.chunks.chunk_mask.float().sum().item() / bytes_n.item()),
            "ar_delta": float(out.ar_delta.item()),
            "coding_rate": float(coding_rate.item()),
            "memory_gate_mean": float(memory_gate_mean.item()),
            "memory_usage_loss": float(memory_usage_loss.item()),
            "emit_value_loss": float(emit_value_loss.item()),
            "emit_value_mean": float(emit_value_mean.item()),
            "emit_target_mean": float(emit_target_mean.item()),
            "sampled_emit_slot": float(sampled_emit_slot),
            "soft_readout_units_per_byte": float(out.emit_soft.sum().item() / bytes_n.item()),
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
def order_probe(model: FLUEDV34Probe, seq_len: int, device: torch.device) -> Dict[str, float]:
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
def evaluate(model, backbone, loader, args, device) -> Dict[str, float]:
    model.eval()
    backbone.eval()
    rows: List[Dict[str, float]] = []
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
        )
        rows.append(metrics)
    result = _avg_metrics(rows)
    result.update(order_probe(model, args.seq_len, device))
    model.train()
    backbone.train()
    return result


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    model = build_model(args).to(device)
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    ).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters())
    result_base = {
        "model_version": "flued_v3_4_position_ar_probe",
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
    params = list(model.parameters()) + list(backbone.parameters())
    optimizer = build_optimizer(args, params)
    scheduler = _cosine_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    start_step = 0
    latest_path = out_dir / "latest.pt"
    if args.resume and latest_path.exists():
        resume_payload = torch.load(latest_path, map_location=device, weights_only=False)
        if "optimizer" in resume_payload and "scheduler" in resume_payload:
            model.load_state_dict(resume_payload["model"])
            backbone.load_state_dict(resume_payload["backbone"])
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            start_step = int(resume_payload.get("step", 0))
            if "torch_rng_state" in resume_payload:
                torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state" in resume_payload:
                torch.cuda.set_rng_state(resume_payload["cuda_rng_state"].cpu(), device)
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
        }
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
    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        collect = step % args.log_every == 0
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            loss, metrics = step_model(
                model, backbone, batch, args, device, collect_metrics=collect, global_step=step
            )
        loss.backward()
        grad = nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if collect:
            elapsed = time.perf_counter() - start
            row = {"step": step, "lr": optimizer.param_groups[0]["lr"], "grad": float(grad), "steps_per_sec": (step + 1) / max(elapsed, 1e-9), **metrics}
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
    eval_stats = evaluate(model, backbone, eval_loader, args, device)
    elapsed = time.perf_counter() - start
    result = {
        **result_base,
        "steps": args.max_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_run_this_process / max(elapsed, 1.0e-9),
        "train_elapsed_sec": train_elapsed,
        "train_steps_per_sec": steps_run_this_process / max(train_elapsed, 1.0e-9),
        "train_peak_memory_gb": train_peak_memory_gb,
        **{f"eval_{key}": value for key, value in eval_stats.items()},
    }
    payload = checkpoint_payload(args.max_steps, result)
    torch.save(payload, latest_path)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--segmentor-layers", type=int, default=5)
    parser.add_argument("--interpreter-layers", type=int, default=3)
    parser.add_argument("--memory-rank", type=int, default=4)
    parser.add_argument("--readout-vectors", type=int, default=4)
    parser.add_argument("--ar-hidden", type=int, default=128)
    parser.add_argument("--use-position", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-structured-lookup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-logic-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-boundary-bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-mode", choices=["threshold", "marginal_rate_topk", "uniform_budget"], default="threshold")
    parser.add_argument("--boundary-coding-rate-dim", type=int, default=16)
    parser.add_argument("--boundary-coding-rate-epsilon", type=float, default=1.0)
    parser.add_argument("--boundary-coding-rate-temperature", type=float, default=0.15)
    parser.add_argument("--boundary-coding-rate-mode", choices=["exact", "l2"], default="exact")
    parser.add_argument("--fixed-chunk-budget", type=int, default=0)
    parser.add_argument("--bytes-per-chunk-budget", type=int, default=16)
    parser.add_argument("--use-emit-controller", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--emit-forward-mode", choices=["hard_st", "soft"], default="hard_st")
    parser.add_argument("--emit-initial-probability", type=float, default=0.1)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--max-span", type=int, default=128)
    parser.add_argument("--tau-cut", type=float, default=0.9)
    parser.add_argument("--tau-trans", type=float, default=0.75)
    parser.add_argument("--boundary-temperature", type=float, default=0.15)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--completion-loss-weight", type=float, default=2.0)
    parser.add_argument("--preserve-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.02)
    parser.add_argument("--coding-rate-loss-weight", type=float, default=0.01)
    parser.add_argument("--coding-rate-dim", type=int, default=64)
    parser.add_argument("--memory-usage-loss-weight", type=float, default=0.02)
    parser.add_argument("--memory-usage-min", type=float, default=0.20)
    parser.add_argument("--memory-usage-max", type=float, default=0.50)
    parser.add_argument("--emit-value-loss-weight", type=float, default=0.1)
    parser.add_argument("--emit-value-every", type=int, default=4)
    parser.add_argument("--emit-rate-value-weight", type=float, default=0.05)
    parser.add_argument("--emit-compute-cost-weight", type=float, default=0.05)
    parser.add_argument("--emit-value-temperature", type=float, default=0.25)
    parser.add_argument("--ar-delta-loss-weight", type=float, default=0.01)
    parser.add_argument("--backbone-hidden", type=int, default=384)
    parser.add_argument("--backbone-layers", type=int, default=3)
    parser.add_argument("--backbone-nhead", type=int, default=8)
    parser.add_argument("--backbone-ffn-dim", type=int, default=1024)
    parser.add_argument("--optimizer", choices=["fused_adamw", "foreach_adamw", "adamw"], default="fused_adamw")
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
