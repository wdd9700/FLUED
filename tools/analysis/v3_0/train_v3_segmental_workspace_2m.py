"""Train the FLUED-v3.1 segmental latent workspace prototype.

This script is intentionally separate from ``train_v3_commit_controller_small``.
The older script tests the active/memory primitive. This one tests the fuller
v3.1 hypothesis:

  bytes -> causal hidden -> active segment + memory
        -> latent refinement
        -> lightweight autoregressive correction
        -> byte readout + commit/value diagnostics

The default size is kept near the previous 2M prototype so the full/ablation
matrix can run quickly on the local RTX 5080.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset, MASK_ID, PAD_ID, StreamingReconstructionDataset
from flued.e1_stage_a import corrupt_byte_inputs
from tools.analysis.v3_0.train_v3_commit_controller_small import (
    _apply_future_mask,
    _append_jsonl,
    _corr,
    _load_texts,
    _masked_acc,
    _masked_ce,
    _masked_stats,
    _prediction_target,
    _topk_enrichment,
    _cosine_with_warmup,
)


def _masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    if not valid.any():
        return (x - y).new_zeros(())
    diff = (x.float() - y.float()).pow(2)
    return diff.masked_select(valid).mean()


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & torch.isfinite(target)
    if not valid.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[valid].float(), target[valid].float())


def _normalized_target(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x.float())
    for b in range(x.size(0)):
        valid = mask[b] & torch.isfinite(x[b])
        if valid.sum() <= 2:
            continue
        vals = x[b, valid].float()
        vals = (vals - vals.mean()) / vals.std(unbiased=False).clamp(min=1e-6)
        out[b, valid] = torch.sigmoid(vals)
    return out


def _span_ce_target(ce: torch.Tensor, mask: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = ce.shape
    span = torch.zeros_like(ce.float())
    span_mask = mask.clone()
    for offset in range(max(1, horizon)):
        shifted = torch.zeros_like(ce.float())
        shifted_mask = torch.zeros_like(mask)
        if offset < seq_len:
            shifted[:, : seq_len - offset] = ce[:, offset:]
            shifted_mask[:, : seq_len - offset] = mask[:, offset:]
        span = span + shifted
        span_mask = span_mask & shifted_mask
    span_mask[:, 0] = False
    return span / float(max(1, horizon)), span_mask


class DepthResidualMixer(nn.Module):
    """AttenRes-style depth mixer over refinement states.

    Full layer-to-layer Attention Residuals is too expensive for this prototype.
    Here the same idea is applied only over a handful of latent refinement
    states, so the memory cost is tiny and the ablation is clean.
    """

    def __init__(self, hidden: int, max_states: int = 12, mode: str = "attn") -> None:
        super().__init__()
        if mode not in {"attn", "last", "mean"}:
            raise ValueError(f"unknown residual mixer: {mode}")
        self.mode = mode
        self.query = nn.Parameter(torch.zeros(max_states, hidden))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not states:
            raise RuntimeError("DepthResidualMixer requires at least one state")
        if self.mode == "last":
            alpha = states[-1].new_zeros(states[-1].size(0), states[-1].size(1), len(states))
            alpha[..., -1] = 1.0
            return states[-1], alpha
        stack = torch.stack(states, dim=2)  # [B,T,S,H]
        if self.mode == "mean":
            alpha = stack.new_full(stack.shape[:3], 1.0 / len(states))
            return stack.mean(dim=2), alpha
        q = self.query[: len(states)].to(stack.dtype)
        score = (stack * q.view(1, 1, len(states), -1)).sum(dim=-1) / math.sqrt(stack.size(-1))
        alpha = torch.softmax(score.float(), dim=-1).to(stack.dtype)
        mixed = (stack * alpha.unsqueeze(-1)).sum(dim=2)
        return mixed, alpha


class SegmentalLatentWorkspace2M(nn.Module):
    def __init__(
        self,
        vocab_size: int = 258,
        d_model: int = 192,
        hidden: int = 192,
        controller_hidden: int = 256,
        refine_steps: int = 4,
        student_refine_steps: int = 1,
        ar_correction_passes: int = 2,
        commit_stride: int = 1,
        residual_mixer: str = "attn",
        memory_enabled: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.refine_steps = max(0, int(refine_steps))
        self.student_refine_steps = max(0, int(student_refine_steps))
        self.ar_correction_passes = max(0, int(ar_correction_passes))
        self.commit_stride = max(1, int(commit_stride))
        self.memory_enabled = bool(memory_enabled)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.encoder = nn.GRU(d_model, hidden, num_layers=1, batch_first=True)
        self.active_update = nn.GRUCell(hidden, hidden)
        self.memory_update = nn.GRUCell(hidden, hidden)

        control_dim = hidden * 4 + 4
        self.controller = nn.Sequential(
            nn.LayerNorm(control_dim),
            nn.Linear(control_dim, controller_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(controller_hidden, 1),
        )
        self.commit_value_head = nn.Sequential(
            nn.LayerNorm(control_dim),
            nn.Linear(control_dim, controller_hidden),
            nn.SiLU(),
            nn.Linear(controller_hidden, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        refine_in = hidden * 4
        self.refine_cell = nn.GRUCell(refine_in, hidden)
        self.refine_norm = nn.LayerNorm(hidden)
        self.residual_mixer = DepthResidualMixer(hidden, max_states=max(2, self.refine_steps + 2), mode=residual_mixer)

        self.ar_cell = nn.GRUCell(hidden, hidden)
        self.ar_norm = nn.LayerNorm(hidden)

        self.memory_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self.memory_adapter = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.byte_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, vocab_size),
        )
        self.future_head = nn.Linear(hidden, vocab_size)

    def _roll_states(self, h: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len, hidden = h.shape
        active = h.new_zeros(bsz, hidden)
        memory = h.new_zeros(bsz, hidden)
        commit_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        ctrl_inputs: List[torch.Tensor] = []
        active_states: List[torch.Tensor] = []
        memory_states: List[torch.Tensor] = []
        future_logits: List[torch.Tensor] = []
        prev_p = h.new_zeros(bsz)

        for t in range(seq_len):
            ht = h[:, t]
            vt = valid[:, t].float().unsqueeze(-1)
            diff = ht - active
            budget = torch.stack([
                torch.stack(commit_probs, dim=1).mean(dim=1) if commit_probs else h.new_zeros(bsz),
                valid[:, : t + 1].float().mean(dim=1),
                h.new_full((bsz,), float(t) / max(1, seq_len - 1)),
                diff.norm(dim=-1) / math.sqrt(hidden),
            ], dim=-1)
            ctrl_in = torch.cat([ht, active, memory, diff, budget], dim=-1)
            if t % self.commit_stride == 0:
                p = torch.sigmoid(self.controller(ctrl_in)).squeeze(-1)
                prev_p = p
            else:
                p = prev_p
            if t == 0:
                p = torch.ones_like(p)
            p = p * valid[:, t].float()

            new_active = self.active_update(ht, active)
            new_memory = self.memory_update(new_active, memory)
            p_col = p.unsqueeze(-1)
            if self.memory_enabled:
                memory = torch.where(vt.bool(), (1.0 - p_col) * memory + p_col * new_memory, memory)
            active = torch.where(vt.bool(), (1.0 - p_col) * new_active + p_col * ht, active)

            commit_probs.append(p)
            values.append(self.commit_value_head(ctrl_in).squeeze(-1))
            ctrl_inputs.append(ctrl_in)
            active_states.append(active)
            memory_states.append(memory)
            future_logits.append(self.future_head(memory if self.memory_enabled else active))

        return {
            "commit_probs": torch.stack(commit_probs, dim=1),
            "commit_value_logits": torch.stack(values, dim=1),
            "control_inputs": torch.stack(ctrl_inputs, dim=1),
            "active": torch.stack(active_states, dim=1),
            "memory": torch.stack(memory_states, dim=1),
            "future_logits": torch.stack(future_logits, dim=1),
        }

    def _refine(self, h: torch.Tensor, active: torch.Tensor, memory: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = active
        states = [latent]
        mem = memory if self.memory_enabled else torch.zeros_like(memory)
        for _ in range(max(0, steps)):
            gate_in = torch.cat([latent, mem], dim=-1)
            gated_mem = self.memory_gate(gate_in) * self.memory_adapter(gate_in)
            inp = torch.cat([h, active, latent, gated_mem], dim=-1)
            latent = self.refine_norm(latent + self.refine_cell(inp.reshape(-1, inp.size(-1)), latent.reshape(-1, latent.size(-1))).view_as(latent))
            states.append(latent)
        return self.residual_mixer(states)

    def _ar_correct(self, latent: torch.Tensor, valid: torch.Tensor, passes: int) -> torch.Tensor:
        if passes <= 0:
            return latent
        out = latent
        bsz, seq_len, hidden = latent.shape
        for _ in range(passes):
            state = latent.new_zeros(bsz, hidden)
            corrected: List[torch.Tensor] = []
            for t in range(seq_len):
                vt = valid[:, t].float().unsqueeze(-1)
                state = self.ar_cell(out[:, t], state)
                z = self.ar_norm(out[:, t] + state)
                z = torch.where(vt.bool(), z, out[:, t])
                corrected.append(z)
            out = torch.stack(corrected, dim=1)
        return out

    def forward(self, src: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        emb = self.embedding(src.clamp(min=0, max=MASK_ID))
        h, _ = self.encoder(emb)
        rolled = self._roll_states(h, valid)
        teacher_latent, residual_alpha = self._refine(h, rolled["active"], rolled["memory"], self.refine_steps)
        student_latent, _ = self._refine(h, rolled["active"], rolled["memory"], self.student_refine_steps)
        teacher_corrected = self._ar_correct(teacher_latent, valid, self.ar_correction_passes)
        student_corrected = self._ar_correct(student_latent, valid, min(1, self.ar_correction_passes))
        logits = self.byte_head(teacher_corrected)
        student_logits = self.byte_head(student_corrected)
        confidence = self.confidence_head(torch.cat([teacher_corrected, rolled["memory"]], dim=-1)).squeeze(-1)
        rolled.update({
            "h": h,
            "latent": teacher_latent,
            "student_latent": student_latent,
            "corrected": teacher_corrected,
            "student_logits": student_logits,
            "confidence_logits": confidence,
            "residual_alpha": residual_alpha,
        })
        return logits, rolled


@torch.no_grad()
def evaluate(
    model: SegmentalLatentWorkspace2M,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    mode: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    vals: Dict[str, List[float]] = defaultdict(list)
    for i, (src, _) in enumerate(loader):
        if i >= max_batches:
            break
        src = src.to(device)
        valid = src != PAD_ID
        model_src = src
        loss_mask = valid
        if mode == "denoise":
            model_src = corrupt_byte_inputs(
                src, valid, mask_id=MASK_ID, corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob, span_min=args.span_min, span_max=args.span_max,
            )
        elif mode == "future_mask":
            model_src, loss_mask = _apply_future_mask(src, valid, args.future_min_prefix_ratio, args.future_max_prefix_ratio)
        logits, metrics = model(model_src, valid)
        target, target_mask = _prediction_target(src, valid, args.prediction_target)
        target_mask = target_mask & loss_mask
        ce = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(target)
        commit = metrics["commit_probs"].float()
        usable = valid.clone()
        usable[:, 0] = False
        density = commit[usable].mean().item() if usable.any() else 0.0
        vals["acc"].append(_masked_acc(logits, target, target_mask))
        vals["loss"].append(float(_masked_ce(logits, target, target_mask).item()))
        vals["student_loss"].append(float(_masked_ce(metrics["student_logits"], target, target_mask).item()))
        vals["future_loss"].append(float(_masked_ce(metrics["future_logits"], target, target_mask).item()))
        vals["commit_mn"].append(density)
        vals["commit_std"].append(commit[usable].std(unbiased=False).item() if usable.any() else 0.0)
        vals["commit_corr"].append(_corr(commit, ce, usable))
        vals["commit_enrich"].append(_topk_enrichment(commit, ce, usable, max(0.01, density)))
    prefix = "eval" if mode == "clean" else f"eval_{mode}"
    out: Dict[str, float] = {}
    for key, xs in vals.items():
        clean = [float(x) for x in xs if not math.isnan(float(x))]
        out[f"{prefix}_{key}"] = sum(clean) / max(1, len(clean))
    model.train()
    return out


def run(args: argparse.Namespace) -> Dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.streaming_train:
        train_ds = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=args.stream_samples_per_worker,
            seed=args.seed,
        )
        eval_texts = _load_texts(args.data_path, args.eval_max_lines)
        eval_ds = ByteReconstructionDataset(texts=eval_texts, seq_len=args.seq_len, stride=args.stride)
    else:
        texts = _load_texts(args.data_path, args.max_lines)
        ds = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
        n_eval = max(1, int(len(ds) * 0.1))
        n_train = max(1, len(ds) - n_eval)
        train_ds, eval_ds = torch.utils.data.random_split(
            ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.streaming_train,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SegmentalLatentWorkspace2M(
        d_model=args.d_model,
        hidden=args.hidden,
        controller_hidden=args.controller_hidden,
        refine_steps=args.refine_steps,
        student_refine_steps=args.student_refine_steps,
        ar_correction_passes=args.ar_correction_passes,
        commit_stride=args.commit_stride,
        residual_mixer=args.residual_mixer,
        memory_enabled=not args.no_memory,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"
    latest_path = out_dir / "latest.pt"
    log_path = out_dir / "run.log"
    train_iter = iter(train_loader)
    window: Dict[str, List[float]] = defaultdict(list)
    last_row: Dict[str, float] = {}
    rate_lambda = float(args.rate_lambda_init)

    def save_checkpoint(step: int, result: Optional[Dict] = None) -> None:
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rate_lambda": rate_lambda,
            "args": vars(args),
        }
        if result is not None:
            ckpt["summary"] = result
        tmp = latest_path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, latest_path)
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            step_path = out_dir / f"step{step:06d}.pt"
            tmp_step = step_path.with_suffix(".pt.tmp")
            torch.save(ckpt, tmp_step)
            os.replace(tmp_step, step_path)

    for step in range(1, args.max_steps + 1):
        try:
            src, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, _ = next(train_iter)
        src = src.to(device)
        valid = src != PAD_ID
        model_src = src
        loss_mask = valid
        task = "clean"
        if random.random() < args.future_mask_prob:
            model_src, loss_mask = _apply_future_mask(src, valid, args.future_min_prefix_ratio, args.future_max_prefix_ratio)
            task = "future_mask"
        elif (args.denoise_steps < 0 or step <= args.denoise_steps) and random.random() < args.denoise_prob:
            model_src = corrupt_byte_inputs(
                src, valid, mask_id=MASK_ID, corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob, span_min=args.span_min, span_max=args.span_max,
            )
            task = "denoise"

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits, metrics = model(model_src, valid)
            target, target_mask = _prediction_target(src, valid, args.prediction_target)
            target_mask = target_mask & loss_mask
            recon_loss = _masked_ce(logits, target, target_mask)
            student_loss = _masked_ce(metrics["student_logits"], target, target_mask)
            future_target, future_mask = _prediction_target(src, valid, "next_byte")
            future_mask = future_mask & loss_mask
            future_loss = _masked_ce(metrics["future_logits"], future_target, future_mask)

            ce = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).view_as(target)
            span_ce, span_mask = _span_ce_target(ce.detach(), target_mask, args.value_horizon)
            value_target = _normalized_target(span_ce, span_mask)
            value_loss = _masked_bce(metrics["commit_value_logits"], value_target, span_mask)
            confidence_target = (ce.detach() < ce.detach().masked_select(target_mask).float().median()).float()
            confidence_loss = _masked_bce(metrics["confidence_logits"], confidence_target, target_mask)
            distill_loss = _masked_mse(metrics["student_latent"], metrics["latent"].detach(), target_mask)

            commit = metrics["commit_probs"].float()
            usable = valid.clone()
            usable[:, 0] = False
            commit_rate = commit[usable].mean() if usable.any() else commit.new_zeros(())
            rate_excess = torch.relu(commit_rate - float(args.rate_target))
            vals = commit[usable]
            spread_loss = commit.new_zeros(())
            if vals.numel() > 4 and args.commit_min_std > 0:
                spread_loss = F.relu(float(args.commit_min_std) - vals.std(unbiased=False))
            loss = (
                recon_loss
                + args.student_loss_weight * student_loss
                + args.future_loss_weight * future_loss
                + args.value_loss_weight * value_loss
                + args.confidence_loss_weight * confidence_loss
                + args.distill_loss_weight * distill_loss
                + rate_lambda * rate_excess
                + args.commit_spread_weight * spread_loss
            )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        if args.rate_lambda_eta > 0:
            rate_lambda = max(0.0, min(args.rate_lambda_max, rate_lambda + args.rate_lambda_eta * float(rate_excess.detach().item())))

        with torch.no_grad():
            density = commit[usable].mean().item() if usable.any() else 0.0
            row = {
                "step": float(step),
                "loss": float(loss.item()),
                "recon": float(recon_loss.item()),
                "student_loss": float(student_loss.item()),
                "future_loss": float(future_loss.item()),
                "value_loss": float(value_loss.item()),
                "confidence_loss": float(confidence_loss.item()),
                "distill_loss": float(distill_loss.item()),
                "acc": float(_masked_acc(logits.detach(), target, target_mask)),
                "student_acc": float(_masked_acc(metrics["student_logits"].detach(), target, target_mask)),
                "task_clean": float(task == "clean"),
                "task_denoise": float(task == "denoise"),
                "task_future_mask": float(task == "future_mask"),
                "commit_mn": density,
                "commit_std": float(commit[usable].std(unbiased=False).item()) if usable.any() else 0.0,
                "commit_corr": float(_corr(commit, ce, usable)),
                "commit_enrich": float(_topk_enrichment(commit, ce, usable, max(0.01, density))),
                "value_corr": float(_corr(torch.sigmoid(metrics["commit_value_logits"]), span_ce, span_mask)),
                "rate_excess": float(rate_excess.detach().item()),
                "rate_lambda": float(rate_lambda),
                "spread": float(spread_loss.detach().item()),
                "grad_norm": float(grad),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            row.update(_masked_stats("commit", commit, usable))
            row.update(_masked_stats("ce", ce, target_mask))
            row.update(_masked_stats("residual_alpha_last", metrics["residual_alpha"][..., -1].float(), target_mask))
            last_row = row
        for k, v in row.items():
            if k != "step":
                window[k].append(float(v))
        if args.metrics_every > 0 and (step == 1 or step % args.metrics_every == 0 or step == args.max_steps):
            _append_jsonl(metrics_path, row)
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            def wavg(key: str) -> float:
                xs = [x for x in window.get(key, []) if not math.isnan(x)]
                return sum(xs) / max(1, len(xs))

            line = (
                f"step={step} loss={wavg('loss'):.4f} recon={wavg('recon'):.4f} "
                f"student={wavg('student_loss'):.4f} future={wavg('future_loss'):.4f} "
                f"value={wavg('value_loss'):.4f} distill={wavg('distill_loss'):.4f} "
                f"acc={wavg('acc'):.4f}/{wavg('student_acc'):.4f} "
                f"commit_m/n={wavg('commit_mn'):.3f} std={wavg('commit_std'):.3f} "
                f"corr={wavg('commit_corr'):.3f} enrich={wavg('commit_enrich'):.3f} "
                f"value_corr={wavg('value_corr'):.3f} "
                f"tasks=c{wavg('task_clean'):.2f}/d{wavg('task_denoise'):.2f}/f{wavg('task_future_mask'):.2f} "
                f"lambda={wavg('rate_lambda'):.3f} alpha_last={wavg('residual_alpha_last_mean'):.3f}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            window.clear()
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save_checkpoint(step)

    eval_metrics: Dict[str, float] = {}
    for mode in ("clean", "denoise", "future_mask"):
        eval_metrics.update(evaluate(model, eval_loader, device, args.max_eval_batches, mode, args))
    result = {
        "model": "v3_segmental_latent_workspace_2m",
        "steps": args.max_steps,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "refine_steps": args.refine_steps,
        "student_refine_steps": args.student_refine_steps,
        "ar_correction_passes": args.ar_correction_passes,
        "residual_mixer": args.residual_mixer,
        "memory_enabled": not args.no_memory,
        "commit_stride": args.commit_stride,
        "last": last_row,
        **eval_metrics,
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint(args.max_steps, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FLUED-v3.1 segmental latent workspace 2M")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=3000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--controller-hidden", type=int, default=256)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--student-refine-steps", type=int, default=1)
    parser.add_argument("--ar-correction-passes", type=int, default=2)
    parser.add_argument("--commit-stride", type=int, default=1)
    parser.add_argument("--residual-mixer", choices=["attn", "last", "mean"], default="attn")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--prediction-target", choices=["current", "next_byte"], default="next_byte")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--denoise-prob", type=float, default=0.5)
    parser.add_argument("--denoise-steps", type=int, default=7500)
    parser.add_argument("--corrupt-rate", type=float, default=0.15)
    parser.add_argument("--span-mask-prob", type=float, default=0.7)
    parser.add_argument("--span-min", type=int, default=1)
    parser.add_argument("--span-max", type=int, default=8)
    parser.add_argument("--future-mask-prob", type=float, default=0.20)
    parser.add_argument("--future-min-prefix-ratio", type=float, default=0.35)
    parser.add_argument("--future-max-prefix-ratio", type=float, default=0.75)
    parser.add_argument("--student-loss-weight", type=float, default=0.20)
    parser.add_argument("--future-loss-weight", type=float, default=0.10)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.02)
    parser.add_argument("--distill-loss-weight", type=float, default=0.10)
    parser.add_argument("--value-horizon", type=int, default=8)
    parser.add_argument("--rate-target", type=float, default=0.35)
    parser.add_argument("--rate-lambda-init", type=float, default=0.05)
    parser.add_argument("--rate-lambda-eta", type=float, default=0.002)
    parser.add_argument("--rate-lambda-max", type=float, default=2.0)
    parser.add_argument("--commit-spread-weight", type=float, default=0.10)
    parser.add_argument("--commit-min-std", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=3000)
    args = parser.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run(args)


if __name__ == "__main__":
    main()
