"""Train FLUED-v3.1 with a parallel latent-diffusion main path.

This is the corrected v3.1 prototype. The main model is not a byte-by-byte
recurrent controller:

  bytes -> parallel feature encoder
        -> boundary / memory / readout latent denoise blocks
        -> small AR correction heads
        -> commit, memory, readout

The recurrent parts are intentionally small GRU heads used only for correction.
They run through cuDNN's sequence kernel instead of a Python loop over bytes.
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
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset, MASK_ID, PAD_ID, StreamingReconstructionDataset
from flued.e1_stage_a import corrupt_byte_inputs
from tools.analysis.train_v3_commit_controller_small import (
    _apply_future_mask,
    _append_jsonl,
    _corr,
    _cosine_with_warmup,
    _load_texts,
    _masked_acc,
    _masked_ce,
    _masked_stats,
    _prediction_target,
    _topk_enrichment,
)


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & torch.isfinite(target)
    if not valid.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[valid].float(), target[valid].float())


def _span_ce_target(ce: torch.Tensor, mask: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = ce.shape
    out = torch.zeros_like(ce.float())
    out_mask = mask.clone()
    for offset in range(max(1, horizon)):
        shifted = torch.zeros_like(out)
        shifted_mask = torch.zeros_like(mask)
        if offset < seq_len:
            shifted[:, : seq_len - offset] = ce[:, offset:]
            shifted_mask[:, : seq_len - offset] = mask[:, offset:]
        out = out + shifted
        out_mask = out_mask & shifted_mask
    out_mask[:, 0] = False
    return out / float(max(1, horizon)), out_mask


def _normalized_target(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x.float())
    for b in range(x.size(0)):
        valid = mask[b] & torch.isfinite(x[b])
        if valid.sum() <= 4:
            continue
        vals = x[b, valid].float()
        vals = (vals - vals.mean()) / vals.std(unbiased=False).clamp(min=1e-6)
        out[b, valid] = torch.sigmoid(vals)
    return out


def _mask_latent(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return x * valid.unsqueeze(-1).to(dtype=x.dtype)


class ParallelDenoiseBlock(nn.Module):
    """One parallel latent-refinement step over [B,T,H]."""

    def __init__(self, hidden: int, ctx_dim: int, max_steps: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.step_embed = nn.Embedding(max_steps + 1, hidden)
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, ctx: torch.Tensor, step_id: int, valid: torch.Tensor) -> torch.Tensor:
        step = torch.full((z.size(0), z.size(1)), int(step_id), dtype=torch.long, device=z.device)
        cond = self.ctx_proj(ctx) + self.step_embed(step)
        x = self.norm(z + cond)
        x = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        delta = self.ff(x)
        z = z + self.gate(cond) * delta
        return _mask_latent(z, valid)


class SmallARCorrection(nn.Module):
    """Small causal correction head.

    This head is allowed to correct local order and boundary/readout details,
    but a delta penalty and small initialized gate prevent it from becoming the
    actual backbone.
    """

    def __init__(self, hidden: int, ar_hidden: int, gate_bias: float = -3.0, gate_scale: float = 0.10) -> None:
        super().__init__()
        self.gate_scale = float(gate_scale)
        self.gru = nn.GRU(hidden, ar_hidden, num_layers=1, batch_first=True)
        self.proj = nn.Linear(ar_hidden, hidden)
        self.gate = nn.Linear(hidden, 1)
        nn.init.constant_(self.gate.bias, gate_bias)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor, valid: torch.Tensor, passes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if passes <= 0:
            zero = z.new_zeros(())
            return z, zero, zero
        total_delta = []
        total_gate = []
        out = z
        for _ in range(passes):
            h, _ = self.gru(out)
            delta = torch.tanh(self.proj(h))
            gate = self.gate_scale * torch.sigmoid(self.gate(out))
            applied = gate * delta
            out = _mask_latent(out + applied, valid)
            valid_f = valid.unsqueeze(-1)
            ratio = applied.norm(dim=-1) / out.detach().norm(dim=-1).clamp(min=1e-6)
            total_delta.append(ratio[valid].mean() if valid.any() else ratio.new_zeros(()))
            gate_flat = gate.squeeze(-1)
            total_gate.append(gate_flat[valid].mean() if valid.any() else gate_flat.new_zeros(()))
        return out, torch.stack(total_delta).mean(), torch.stack(total_gate).mean()


class V31SegmentalDiffusion2M(nn.Module):
    def __init__(
        self,
        vocab_size: int = 258,
        d_model: int = 160,
        hidden: int = 160,
        nhead: int = 4,
        encoder_layers: int = 2,
        ffn_dim: int = 640,
        ar_hidden: int = 48,
        ar_gate_scale: float = 0.10,
        max_denoise_steps: int = 8,
        dropout: float = 0.0,
        use_memory: bool = True,
        use_ar: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.use_memory = use_memory
        self.use_ar = use_ar
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.input_proj = nn.Linear(d_model, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.boundary_init = nn.Linear(hidden, hidden)
        self.memory_init = nn.Linear(hidden, hidden)
        self.readout_init = nn.Linear(hidden, hidden)
        self.boundary_denoiser = ParallelDenoiseBlock(hidden, hidden, max_steps=max_denoise_steps, dropout=dropout)
        self.memory_denoiser = ParallelDenoiseBlock(hidden, hidden + 1, max_steps=max_denoise_steps, dropout=dropout)
        self.readout_denoiser = ParallelDenoiseBlock(hidden, hidden * 2, max_steps=max_denoise_steps, dropout=dropout)
        self.boundary_ar = SmallARCorrection(hidden, ar_hidden, gate_scale=ar_gate_scale)
        self.memory_ar = SmallARCorrection(hidden, ar_hidden, gate_scale=ar_gate_scale)
        self.readout_ar = SmallARCorrection(hidden, ar_hidden, gate_scale=ar_gate_scale)
        self.boundary_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.commit_value_head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.confidence_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.memory_write = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.readout_gate = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.byte_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, vocab_size))
        self.future_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, vocab_size))

    @staticmethod
    def _causal_memory(memory_z: torch.Tensor, commit: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        p = (commit * valid.float()).unsqueeze(-1)
        write = p * memory_z
        numer = torch.cumsum(write, dim=1)
        denom = torch.cumsum(p, dim=1).clamp(min=1e-4)
        hist = numer / denom
        zero = hist.new_zeros(hist.size(0), 1, hist.size(-1))
        return torch.cat([zero, hist[:, :-1]], dim=1)

    def _denoise(
        self,
        block: ParallelDenoiseBlock,
        z: torch.Tensor,
        ctx: torch.Tensor,
        steps: int,
        valid: torch.Tensor,
        noise_scale: float,
        training: bool,
    ) -> torch.Tensor:
        if training and noise_scale > 0:
            z = z + torch.randn_like(z) * float(noise_scale)
            z = _mask_latent(z, valid)
        for i in range(max(0, int(steps))):
            z = block(z, ctx, min(i, 8), valid)
        return z

    def forward(
        self,
        src: torch.Tensor,
        valid: torch.Tensor,
        boundary_steps: int,
        memory_steps: int,
        readout_steps: int,
        ar_passes: int,
        noise_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        emb = self.embedding(src.clamp(min=0, max=MASK_ID))
        h = self.input_proj(emb)
        h = self.encoder(h, src_key_padding_mask=~valid)
        h = _mask_latent(h, valid)

        boundary_z = torch.tanh(self.boundary_init(h))
        memory_z = torch.tanh(self.memory_init(h))
        readout_z = torch.tanh(self.readout_init(h))

        boundary_z = self._denoise(self.boundary_denoiser, boundary_z, h, boundary_steps, valid, noise_scale, self.training)
        if self.use_ar:
            boundary_z, boundary_delta, boundary_gate = self.boundary_ar(boundary_z, valid, ar_passes)
        else:
            boundary_delta = h.new_zeros(())
            boundary_gate = h.new_zeros(())
        commit_logits = self.boundary_head(boundary_z).squeeze(-1)
        commit = torch.sigmoid(commit_logits) * valid.float()
        if commit.size(1) > 0:
            commit[:, 0] = valid[:, 0].float()

        memory_ctx = torch.cat([h, commit.unsqueeze(-1)], dim=-1)
        memory_z = self._denoise(self.memory_denoiser, memory_z, memory_ctx, memory_steps, valid, noise_scale, self.training)
        if self.use_ar:
            memory_z, memory_delta, memory_gate = self.memory_ar(memory_z, valid, ar_passes)
        else:
            memory_delta = h.new_zeros(())
            memory_gate = h.new_zeros(())
        memory_write = self.memory_write(memory_z)
        if self.use_memory:
            hist_memory = self._causal_memory(memory_write, commit, valid)
        else:
            hist_memory = torch.zeros_like(memory_write)

        readout_ctx = torch.cat([h, hist_memory], dim=-1)
        readout_z = self._denoise(self.readout_denoiser, readout_z, readout_ctx, readout_steps, valid, noise_scale, self.training)
        gate = self.readout_gate(torch.cat([readout_z, hist_memory], dim=-1))
        readout_z = readout_z + gate * hist_memory
        if self.use_ar:
            readout_z, readout_delta, readout_ar_gate = self.readout_ar(readout_z, valid, ar_passes)
        else:
            readout_delta = h.new_zeros(())
            readout_ar_gate = h.new_zeros(())

        logits = self.byte_head(readout_z)
        future_logits = self.future_head(hist_memory)
        value_logits = self.commit_value_head(torch.cat([boundary_z, hist_memory], dim=-1)).squeeze(-1)
        confidence_logits = self.confidence_head(readout_z).squeeze(-1)
        metrics = {
            "h": h,
            "boundary_z": boundary_z,
            "memory_z": memory_z,
            "readout_z": readout_z,
            "hist_memory": hist_memory,
            "commit_probs": commit,
            "commit_logits": commit_logits,
            "future_logits": future_logits,
            "commit_value_logits": value_logits,
            "confidence_logits": confidence_logits,
            "boundary_delta": boundary_delta,
            "memory_delta": memory_delta,
            "readout_delta": readout_delta,
            "boundary_gate": boundary_gate,
            "memory_gate": memory_gate,
            "readout_ar_gate": readout_ar_gate,
            "readout_memory_gate_mean": gate.mean(dim=-1)[valid].mean() if valid.any() else h.new_zeros(()),
        }
        return logits, metrics


def schedule_steps(args: argparse.Namespace, step: int) -> Dict[str, float]:
    if args.step_schedule == "fixed_max":
        b, m, r, ar = args.max_boundary_steps, args.max_memory_steps, args.max_readout_steps, args.max_ar_passes
    elif args.step_schedule == "fixed_target":
        b, m, r, ar = args.target_boundary_steps, args.target_memory_steps, args.target_readout_steps, args.target_ar_passes
    else:
        pct = step / max(1, args.max_steps)
        if pct < args.stage_a_ratio:
            b, m, r, ar = args.max_boundary_steps, args.max_memory_steps, args.max_readout_steps, args.max_ar_passes
        elif pct < args.stage_b_ratio:
            span = max(1e-6, args.stage_b_ratio - args.stage_a_ratio)
            q = (pct - args.stage_a_ratio) / span
            b = round(args.max_boundary_steps + q * (args.mid_boundary_steps - args.max_boundary_steps))
            m = round(args.max_memory_steps + q * (args.mid_memory_steps - args.max_memory_steps))
            r = round(args.max_readout_steps + q * (args.mid_readout_steps - args.max_readout_steps))
            ar = round(args.max_ar_passes + q * (args.mid_ar_passes - args.max_ar_passes))
        else:
            b, m, r, ar = args.target_boundary_steps, args.target_memory_steps, args.target_readout_steps, args.target_ar_passes
    pct = step / max(1, args.max_steps)
    noise = max(float(args.target_noise_scale), float(args.noise_scale) * (1.0 - pct))
    return {
        "boundary_steps": max(0, int(b)),
        "memory_steps": max(0, int(m)),
        "readout_steps": max(0, int(r)),
        "ar_passes": max(0, int(ar)),
        "noise_scale": float(noise),
    }


@torch.no_grad()
def evaluate(
    model: V31SegmentalDiffusion2M,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    mode: str,
    args: argparse.Namespace,
    schedule: Dict[str, float],
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
                src,
                valid,
                mask_id=MASK_ID,
                corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob,
                span_min=args.span_min,
                span_max=args.span_max,
            )
        elif mode == "future_mask":
            model_src, loss_mask = _apply_future_mask(src, valid, args.future_min_prefix_ratio, args.future_max_prefix_ratio)
        logits, metrics = model(model_src, valid, **schedule)
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
        future_target, future_mask = _prediction_target(src, valid, args.future_target)
        future_mask = future_mask & loss_mask
        vals["loss"].append(float(_masked_ce(logits, target, target_mask).item()))
        vals["acc"].append(float(_masked_acc(logits, target, target_mask)))
        vals["future_loss"].append(float(_masked_ce(metrics["future_logits"], future_target, future_mask).item()))
        vals["commit_mn"].append(density)
        vals["commit_std"].append(commit[usable].std(unbiased=False).item() if usable.any() else 0.0)
        vals["commit_corr"].append(_corr(commit, ce, usable))
        vals["commit_enrich"].append(_topk_enrichment(commit, ce, usable, max(0.01, density)))
        vals["boundary_delta"].append(float(metrics["boundary_delta"].item()))
        vals["memory_delta"].append(float(metrics["memory_delta"].item()))
        vals["readout_delta"].append(float(metrics["readout_delta"].item()))
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
        train_ds, eval_ds = torch.utils.data.random_split(
            ds,
            [len(ds) - n_eval, n_eval],
            generator=torch.Generator().manual_seed(args.seed),
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

    model = V31SegmentalDiffusion2M(
        d_model=args.d_model,
        hidden=args.hidden,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        ffn_dim=args.ffn_dim,
        ar_hidden=args.ar_hidden,
        ar_gate_scale=args.ar_gate_scale,
        max_denoise_steps=max(args.max_boundary_steps, args.max_memory_steps, args.max_readout_steps, 8),
        dropout=args.dropout,
        use_memory=not args.no_memory,
        use_ar=not args.no_ar,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"
    latest_path = out_dir / "latest.pt"
    log_path = out_dir / "run.log"
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
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
        elif random.random() < args.denoise_prob:
            model_src = corrupt_byte_inputs(
                src,
                valid,
                mask_id=MASK_ID,
                corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob,
                span_min=args.span_min,
                span_max=args.span_max,
            )
            task = "denoise"

        current_schedule = schedule_steps(args, step)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits, metrics = model(model_src, valid, **current_schedule)
            target, target_mask = _prediction_target(src, valid, args.prediction_target)
            target_mask = target_mask & loss_mask
            recon_loss = _masked_ce(logits, target, target_mask)
            future_target, future_mask = _prediction_target(src, valid, args.future_target)
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
            boundary_value_loss = _masked_bce(metrics["commit_logits"], value_target, span_mask)
            if target_mask.any():
                med = ce.detach().masked_select(target_mask).float().median()
            else:
                med = ce.detach().float().median()
            conf_target = (ce.detach() < med).float()
            confidence_loss = _masked_bce(metrics["confidence_logits"], conf_target, target_mask)
            commit = metrics["commit_probs"].float()
            usable = valid.clone()
            usable[:, 0] = False
            commit_rate = commit[usable].mean() if usable.any() else commit.new_zeros(())
            rate_excess = torch.relu(commit_rate - float(args.rate_target))
            spread_loss = commit.new_zeros(())
            vals = commit[usable]
            if vals.numel() > 4 and args.commit_min_std > 0:
                spread_loss = F.relu(float(args.commit_min_std) - vals.std(unbiased=False))
            ar_delta_loss = metrics["boundary_delta"] + metrics["memory_delta"] + metrics["readout_delta"]
            loss = (
                args.recon_loss_weight * recon_loss
                + args.future_loss_weight * future_loss
                + args.value_loss_weight * value_loss
                + args.boundary_value_loss_weight * boundary_value_loss
                + args.confidence_loss_weight * confidence_loss
                + args.ar_delta_loss_weight * ar_delta_loss
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
                "future_loss": float(future_loss.item()),
                "value_loss": float(value_loss.item()),
                "boundary_value_loss": float(boundary_value_loss.item()),
                "confidence_loss": float(confidence_loss.item()),
                "ar_delta_loss": float(ar_delta_loss.item()),
                "acc": float(_masked_acc(logits.detach(), target, target_mask)),
                "task_clean": float(task == "clean"),
                "task_denoise": float(task == "denoise"),
                "task_future_mask": float(task == "future_mask"),
                "boundary_steps": float(current_schedule["boundary_steps"]),
                "memory_steps": float(current_schedule["memory_steps"]),
                "readout_steps": float(current_schedule["readout_steps"]),
                "ar_passes": float(current_schedule["ar_passes"]),
                "noise_scale": float(current_schedule["noise_scale"]),
                "commit_mn": density,
                "commit_std": float(commit[usable].std(unbiased=False).item()) if usable.any() else 0.0,
                "commit_corr": float(_corr(commit, ce, usable)),
                "commit_enrich": float(_topk_enrichment(commit, ce, usable, max(0.01, density))),
                "value_corr": float(_corr(torch.sigmoid(metrics["commit_value_logits"]), span_ce, span_mask)),
                "boundary_delta": float(metrics["boundary_delta"].item()),
                "memory_delta": float(metrics["memory_delta"].item()),
                "readout_delta": float(metrics["readout_delta"].item()),
                "boundary_gate": float(metrics["boundary_gate"].item()),
                "memory_gate": float(metrics["memory_gate"].item()),
                "readout_ar_gate": float(metrics["readout_ar_gate"].item()),
                "readout_memory_gate": float(metrics["readout_memory_gate_mean"].item()),
                "rate_excess": float(rate_excess.detach().item()),
                "rate_lambda": float(rate_lambda),
                "spread": float(spread_loss.detach().item()),
                "grad_norm": float(grad),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            row.update(_masked_stats("commit", commit, usable))
            row.update(_masked_stats("ce", ce, target_mask))
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
                f"future={wavg('future_loss'):.4f} acc={wavg('acc'):.4f} "
                f"commit_m/n={wavg('commit_mn'):.3f} std={wavg('commit_std'):.3f} "
                f"corr={wavg('commit_corr'):.3f} enrich={wavg('commit_enrich'):.3f} "
                f"value_corr={wavg('value_corr'):.3f} "
                f"steps=b{wavg('boundary_steps'):.1f}/m{wavg('memory_steps'):.1f}/r{wavg('readout_steps'):.1f}/ar{wavg('ar_passes'):.1f} "
                f"delta=b{wavg('boundary_delta'):.4f}/m{wavg('memory_delta'):.4f}/r{wavg('readout_delta'):.4f} "
                f"lambda={wavg('rate_lambda'):.3f}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            window.clear()
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save_checkpoint(step)

    deploy_schedule = {
        "boundary_steps": args.target_boundary_steps,
        "memory_steps": args.target_memory_steps,
        "readout_steps": args.target_readout_steps,
        "ar_passes": args.target_ar_passes,
        "noise_scale": 0.0,
    }
    max_schedule = {
        "boundary_steps": args.max_boundary_steps,
        "memory_steps": args.max_memory_steps,
        "readout_steps": args.max_readout_steps,
        "ar_passes": args.max_ar_passes,
        "noise_scale": 0.0,
    }
    eval_metrics: Dict[str, float] = {}
    for label, schedule in (("deploy", deploy_schedule), ("multi", max_schedule)):
        for mode in ("clean", "denoise", "future_mask"):
            got = evaluate(model, eval_loader, device, args.max_eval_batches, mode, args, schedule)
            eval_metrics.update({f"{label}_{k}": v for k, v in got.items()})
    result = {
        "model": "v31_segmental_diffusion_2m",
        "steps": args.max_steps,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "step_schedule": args.step_schedule,
        "deploy_schedule": deploy_schedule,
        "max_schedule": max_schedule,
        "last": last_row,
        **eval_metrics,
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint(args.max_steps, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FLUED-v3.1 parallel segmental diffusion 2M")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=3000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=640)
    parser.add_argument("--ar-hidden", type=int, default=48)
    parser.add_argument("--ar-gate-scale", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-ar", action="store_true")
    parser.add_argument("--prediction-target", choices=["current", "next_byte"], default="current")
    parser.add_argument("--future-target", choices=["current", "next_byte"], default="current")
    parser.add_argument("--step-schedule", choices=["anneal", "fixed_max", "fixed_target"], default="anneal")
    parser.add_argument("--max-boundary-steps", type=int, default=4)
    parser.add_argument("--max-memory-steps", type=int, default=6)
    parser.add_argument("--max-readout-steps", type=int, default=6)
    parser.add_argument("--max-ar-passes", type=int, default=2)
    parser.add_argument("--mid-boundary-steps", type=int, default=2)
    parser.add_argument("--mid-memory-steps", type=int, default=3)
    parser.add_argument("--mid-readout-steps", type=int, default=3)
    parser.add_argument("--mid-ar-passes", type=int, default=1)
    parser.add_argument("--target-boundary-steps", type=int, default=1)
    parser.add_argument("--target-memory-steps", type=int, default=1)
    parser.add_argument("--target-readout-steps", type=int, default=1)
    parser.add_argument("--target-ar-passes", type=int, default=1)
    parser.add_argument("--stage-a-ratio", type=float, default=0.40)
    parser.add_argument("--stage-b-ratio", type=float, default=0.75)
    parser.add_argument("--noise-scale", type=float, default=0.04)
    parser.add_argument("--target-noise-scale", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--denoise-prob", type=float, default=0.5)
    parser.add_argument("--corrupt-rate", type=float, default=0.15)
    parser.add_argument("--span-mask-prob", type=float, default=0.7)
    parser.add_argument("--span-min", type=int, default=1)
    parser.add_argument("--span-max", type=int, default=8)
    parser.add_argument("--future-mask-prob", type=float, default=0.20)
    parser.add_argument("--future-min-prefix-ratio", type=float, default=0.35)
    parser.add_argument("--future-max-prefix-ratio", type=float, default=0.75)
    parser.add_argument("--recon-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-loss-weight", type=float, default=0.15)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--boundary-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.02)
    parser.add_argument("--ar-delta-loss-weight", type=float, default=0.05)
    parser.add_argument("--value-horizon", type=int, default=8)
    parser.add_argument("--rate-target", type=float, default=0.35)
    parser.add_argument("--rate-lambda-init", type=float, default=0.05)
    parser.add_argument("--rate-lambda-eta", type=float, default=0.002)
    parser.add_argument("--rate-lambda-max", type=float, default=2.0)
    parser.add_argument("--commit-spread-weight", type=float, default=0.10)
    parser.add_argument("--commit-min-std", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    args = parser.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run(args)


if __name__ == "__main__":
    main()
