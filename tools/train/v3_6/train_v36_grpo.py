"""FLUED v3.6 S0.5: GRPO fine-tuning of boundary cuts and the beta write gate.

Pre-registered design (v3.6 spec section 10 + appendix A, NLA migration):

* Action space: per-byte Bernoulli cut decisions in the tanh confidence space
  (p = sigmoid((confidence - tau_cut) / cut_temperature), free positions only:
  valid, non-UTF8-continuation, non-forced-first) and per-chunk beta write
  gates perturbed in logit space (a = logit(beta) + beta_sigma * eps).
* Group sampling: each prompt is expanded to G samples sharing the same byte
  mask (same-bytes-same-mask pairing discipline); one wide (B*G) forward.
* Reward: negative anchor-normalized robust risk (max over reconstruction /
  completion / preservation BPB, normalized by the offline rich/null anchors
  from probe_v36_cbiu_anchors.py). Advantage is group-standardized, no value
  network (GRPO). Per NLA (transformer-circuits.pub/2026/nla), constraints do
  NOT go through the group-relative reward: the boundary rate constraint is a
  differentiable direct loss on the expected cut count E[count] = sum of
  cut_prob over free positions (the smooth relaxation of the deployable hard
  count), rate_weight * relu(E[count]/chunk_budget - 1). Round history:
  R1 no rate term -> boundaries hit the capacity cap; R2 sampled-count penalty
  (Monte-Carlo estimate of E[count]) worked in-sample but the temperature
  smear decoupled it from the hard rule; R3 deterministic hard-count penalty
  had zero within-group variance and was annihilated by group-relative
  advantage. E[count] as a direct loss is the exact, differentiable form.
* Gradient paths: sampled cuts and beta are injected detached; only the
  log-probs carry RL gradients. The task CE keeps training decoder / backbone /
  summarizer / write-head k,v paths. The soft boundary bridge is retired here
  (real policy gradients replace the approximate gradient path).
* Control arm discipline (spec section 11): the decoder-readaptation control is
  plain two-task continued training from the same snapshot via train_v36.py
  with identical step count -- no new code needed.

Anchors are instrument calibration, not curriculum. Guards per run:
truncated_tokens == 0, cut_capacity_overflow tracked, NaN skip counter,
pre-clip grad norm, boundary drift stats (hard_cut_fraction, chunks_per_sample).
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID  # noqa: E402
from flued.v34.model import FLUEDV34Probe  # noqa: E402
from tools.train.v3_3.train_v33 import (  # noqa: E402
    _append_jsonl,
    _cosine_with_warmup,
    _safe_acc,
    build_optimizer,
    make_byte_mask,
    make_dataloaders,
    make_targets,
)
from tools.train.v3_4.cbiu import CBIUState, normalize_cbiu_risks, robust_cbiu_risk  # noqa: E402
from tools.train.v3_6.train_v36 import build_model, evaluate  # noqa: E402

LN2 = math.log(2.0)
LOG_2PI = math.log(2.0 * math.pi)


def _ce_select(logits: torch.Tensor, targets: torch.Tensor, select: torch.Tensor) -> torch.Tensor:
    picked_logits = logits[select]
    picked_targets = targets[select]
    if picked_targets.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(picked_logits.float(), picked_targets.clamp(min=0, max=257), ignore_index=PAD_ID)


def _per_sample_bpb(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Bits-per-target-byte for each row of a (N, C, S, V) logits tensor."""

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1).clamp(min=0, max=257),
        ignore_index=PAD_ID,
        reduction="none",
    ).reshape(targets.shape)
    w = weight.to(ce.dtype)
    return (ce * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1.0) / LN2


def grpo_forward(
    model,
    token_ids: torch.Tensor,
    tau_cut: float,
    cut_temperature: float,
    beta_sigma: float,
) -> dict:
    """Forward with sampled boundary cuts and perturbed beta gates.

    Returns logits (both decoder paths), summed action log-probs per sample
    (gradient-carrying), chunk packing, and drift auxiliaries.
    """

    config = model.config
    byte_states, confidence, valid, _logits = model._encode(token_ids)
    utf8_cont = token_ids.ge(129) & token_ids.le(192) & valid
    free = valid & ~utf8_cont
    bsz = token_ids.size(0)
    first = valid.float().argmax(dim=1)
    forced_first = torch.zeros_like(valid)
    forced_first[torch.arange(bsz, device=token_ids.device), first] = valid.any(dim=1)
    free = free & ~forced_first

    cut_prob = torch.sigmoid((confidence.float() - tau_cut) / cut_temperature).clamp(1.0e-4, 1.0 - 1.0e-4)
    sampled = torch.bernoulli(cut_prob)
    logp_cut = (sampled * cut_prob.log() + (1.0 - sampled) * (1.0 - cut_prob).log())
    free_count = free.float().sum(dim=1).clamp(min=1.0)
    logp_cut = (logp_cut * free.float()).sum(dim=1) / free_count
    # Expected cut count: smooth, differentiable relaxation of the deployable
    # hard count. Carries gradients for the direct rate loss (not the reward).
    expected_cuts = (cut_prob * free.float()).sum(dim=1)

    requested = sampled.gt(0.5) & free
    requested = requested | forced_first
    executable, overflow = FLUEDV34Probe._capacity_safe_cuts(
        requested, valid, config.max_chunks, config.max_span
    )
    # Deployable decision rule (eval): hard threshold on confidence. The rate
    # term must target this count, not the Bernoulli-sampled count -- the
    # temperature smear decouples the two (round-2 finding).
    hard_count = (confidence.gt(tau_cut) & free).float().sum(dim=1) + forced_first.float().sum(dim=1)

    chunks = model.chunk_builder(byte_states, valid, executable, confidence)
    memory = model.summarizer(chunks.span_embeddings, chunks.token_mask)
    gates = model.write_head(memory)

    base_beta = gates["beta"].float().clamp(1.0e-4, 1.0 - 1.0e-4)
    beta_mask = chunks.chunk_mask.float().unsqueeze(-1)
    beta_mean = (base_beta * beta_mask).sum() / beta_mask.sum().clamp(min=1.0)
    base_logit = torch.logit(base_beta)
    noise = torch.randn_like(base_logit)
    a_logit = base_logit + beta_sigma * noise
    beta_sampled = torch.sigmoid(a_logit)
    a_detached = a_logit.detach()
    logp_beta = (
        -0.5 * ((a_detached - base_logit) / beta_sigma).square()
        - math.log(beta_sigma)
        - 0.5 * LOG_2PI
    )
    beta_count = (chunks.chunk_mask.float().unsqueeze(-1) * base_logit.new_ones(1)).sum(dim=(1, 2)).clamp(min=1.0)
    logp_beta = (logp_beta * chunks.chunk_mask.float().unsqueeze(-1)).sum(dim=(1, 2)) / beta_count
    gates = {**gates, "beta": beta_sampled.detach().to(gates["beta"].dtype)}

    package, state_norm = model.state_machine(gates, chunks.chunk_mask)
    n_chunks = chunks.chunk_mask.size(1)
    pos = model.chunk_pos.weight.unsqueeze(0)[:, :n_chunks]
    if model.config.per_chunk_readout:
        content = package.mean(dim=2)
        backbone_out = model.backbone(content)
        cond_direct = model.decoder_in(content) + pos
        cond_backbone = backbone_out + pos
    else:
        backbone_out = model.backbone(package)
        cond_direct = model.decoder_in(package.mean(dim=1)).unsqueeze(1) + pos
        cond_backbone = backbone_out.mean(dim=1).unsqueeze(1) + pos
    logits_direct = model.decoder(cond_direct, chunks.token_mask)
    logits_backbone = model.decoder(cond_backbone, chunks.token_mask)

    return {
        "logits_direct": logits_direct,
        "logits_backbone": logits_backbone,
        "logp_cut": logp_cut,
        "logp_beta": logp_beta,
        "chunks": chunks,
        "cut_overflow": overflow,
        "hard_count": hard_count.detach(),
        "expected_cuts": expected_cuts,
        "beta_mean": beta_mean.detach(),
        "state_norm": state_norm,
        "cut_prob_mean": cut_prob[free].mean() if free.any() else cut_prob.mean(),
        "hard_cut_fraction": executable.float()[valid].mean() if valid.any() else executable.float().mean(),
    }


def grpo_step(model, batch, args, device, cbiu: CBIUState) -> tuple[torch.Tensor, dict]:
    clean = batch[0].to(device)
    valid = clean.ne(PAD_ID)
    byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    source = clean.masked_fill(byte_mask, MASK_ID)

    group = args.group_size
    wide_source = source.repeat_interleave(group, dim=0)
    wide_clean = clean.repeat_interleave(group, dim=0)
    wide_mask = byte_mask.repeat_interleave(group, dim=0)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        out = grpo_forward(model, wide_source, args.tau_cut, args.cut_temperature, args.beta_sigma)

    targets, slot_mask, masked_slot = make_targets(
        wide_clean, wide_mask, out["chunks"].chunk_ids, out["chunks"].offsets, args.max_chunks, args.max_span
    )
    unmasked_slot = slot_mask & ~masked_slot

    direct_loss = _ce_select(out["logits_direct"], targets, slot_mask)
    backbone_loss = _ce_select(out["logits_backbone"], targets, slot_mask)
    task_loss = args.task1_loss_weight * direct_loss + args.task2_loss_weight * backbone_loss

    risks = torch.stack(
        [
            _per_sample_bpb(out["logits_direct"], targets, slot_mask),
            _per_sample_bpb(out["logits_backbone"], targets, masked_slot),
            _per_sample_bpb(out["logits_backbone"], targets, unmasked_slot),
        ],
        dim=-1,
    )
    normalized = normalize_cbiu_risks(risks, cbiu.anchor_rich, cbiu.anchor_null)
    rho = robust_cbiu_risk(normalized)
    rewards = -rho
    reward_group = rewards.view(-1, group)
    advantage = reward_group - reward_group.mean(dim=1, keepdim=True)
    advantage = advantage / advantage.std(dim=1, keepdim=True).clamp(min=1.0e-6)
    advantage = advantage.reshape(-1).detach()

    logp = out["logp_cut"] + out["logp_beta"]
    pg_loss = -(advantage * logp).mean()
    chunks_ps = out["chunks"].chunk_mask.float().sum(dim=1)
    hard_ps = out["hard_count"].float()
    expected_ps = out["expected_cuts"].float()
    budget = max(float(args.chunk_budget), 1.0e-8)
    rate_penalty = torch.relu(expected_ps / budget - 1.0).mean()
    loss = task_loss + args.rl_weight * pg_loss + args.rate_weight * rate_penalty

    metrics = {
        "loss": float(loss.item()),
        "task_loss": float(task_loss.item()),
        "pg_loss": float(pg_loss.item()),
        "rate_penalty": float(rate_penalty.item()),
        "direct_loss": float(direct_loss.item()),
        "backbone_loss": float(backbone_loss.item()),
        "direct_acc": _safe_acc(out["logits_direct"].argmax(dim=-1), targets, slot_mask),
        "backbone_acc": _safe_acc(out["logits_backbone"].argmax(dim=-1), targets, slot_mask),
        "backbone_masked_acc": _safe_acc(out["logits_backbone"].argmax(dim=-1), targets, masked_slot),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std().item()),
        "rho_mean": float(rho.mean().item()),
        "truncated_tokens": float(out["chunks"].pack_info["truncated_tokens"].float().sum().item()),
        "cut_capacity_overflow": float(out["cut_overflow"].float().sum().item()),
        "chunks_per_sample": float(chunks_ps.mean().item()),
        "hard_cut_count": float(hard_ps.mean().item()),
        "expected_cut_count": float(expected_ps.mean().item()),
        "beta_mean": float(out["beta_mean"].item()),
        "hard_cut_fraction": float(out["hard_cut_fraction"].item()),
        "cut_prob_mean": float(out["cut_prob_mean"].item()),
        "state_norm": float(out["state_norm"].item()),
    }
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canonical_v36.json")
    parser.add_argument("--run-id", default="v36_grpo")
    parser.add_argument("--out-dir", default="checkpoints/v36_grpo")
    parser.add_argument("--checkpoint", required=True, help="S0.5 baseline snapshot (3K) to start from")
    parser.add_argument("--anchor-file", required=True, help="cbiu_v36_anchors.json from probe_v36_cbiu_anchors.py")
    parser.add_argument("--data-path", default="", help="overrides the placeholder data_path in --config")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2, help="prompts per step; wide batch is batch_size*group_size")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--cut-temperature", type=float, default=0.15)
    parser.add_argument("--beta-sigma", type=float, default=0.5)
    parser.add_argument("--rl-weight", type=float, default=1.0)
    parser.add_argument(
        "--chunk-budget",
        type=float,
        default=18.0,
        help="budget on expected cut count E[count] (direct rate loss); 18 on E[count] ~= 24 hard cuts (user-annotation granularity, median 21B/chunk) given the measured E[count]~0.73*hard slack",
    )
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-prefixes", default="byte_lookup,encoder_blocks")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    pre_args, _ = parser.parse_known_args()
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_args = Namespace(**train_config)
    if args.data_path:
        model_args.data_path = args.data_path
    if not hasattr(model_args, "data_manifest"):
        model_args.data_manifest = ""
    model_args.batch_size = args.batch_size
    model_args.max_eval_batches = args.max_eval_batches
    for key in (
        "mask_prob",
        "mask_span_min",
        "mask_span_max",
        "tau_cut",
        "max_chunks",
        "max_span",
        "task1_loss_weight",
        "task2_loss_weight",
    ):
        setattr(args, key, getattr(model_args, key))
    model = build_model(model_args).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    print(f"[grpo] init from {args.checkpoint}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    frozen = [p.strip() for p in args.freeze_prefixes.split(",") if p.strip()]
    n_frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen):
            param.requires_grad_(False)
            n_frozen += 1
    print(f"[grpo] frozen params={n_frozen} prefixes={frozen}", flush=True)

    cbiu = CBIUState.from_anchor_file(args.anchor_file, compute_budget=args.chunk_budget, device=device)
    print(
        f"[grpo] anchors rich={cbiu.anchor_rich.tolist()} null={cbiu.anchor_null.tolist()}",
        flush=True,
    )

    opt_args = Namespace(lr=args.lr, weight_decay=args.weight_decay, optimizer="fused_adamw")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer(opt_args, iter(trainable))
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    train_loader, eval_loader = make_dataloaders(model_args)

    log_path = out_dir / "train_log.jsonl"
    model.train()
    t0 = time.time()
    nan_skips = 0
    step = 0
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            loss, metrics = grpo_step(model, batch, args, device, cbiu)
            if not torch.isfinite(loss):
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad": float(grad.item()),
                    "steps_per_sec": step / max(time.time() - t0, 1e-9),
                    "nan_skips": nan_skips,
                    **metrics,
                }
                _append_jsonl(log_path, row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if step % args.ckpt_every == 0:
                torch.save(
                    {"step": step, "model": model.state_dict(), "args": vars(args), "cbiu_state": cbiu.state_dict()},
                    out_dir / "latest.pt",
                )
            if step % args.eval_every == 0:
                eval_stats = evaluate(model, eval_loader, model_args, device)
                eval_row = {"step": step, **eval_stats}
                _append_jsonl(log_path, eval_row)
                print(json.dumps(eval_row, ensure_ascii=False), flush=True)

    torch.save(
        {"step": step, "model": model.state_dict(), "args": vars(args), "cbiu_state": cbiu.state_dict()},
        out_dir / "latest.pt",
    )
    eval_stats = evaluate(model, eval_loader, model_args, device)
    summary = {
        "run_id": args.run_id,
        "steps": step,
        "elapsed_sec": time.time() - t0,
        "nan_skips": nan_skips,
        "args": vars(args),
        **eval_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
