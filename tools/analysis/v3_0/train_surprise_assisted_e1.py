"""Small FLUED E1 experiment with surprise-assisted boundary training.

This is an experimental v3 validation runner. It intentionally stays outside
``flued.e1_stage_a`` so the v2 reproduction path remains unchanged.

Modes:
  native          — FLUED E1 loss only
  teacher_oracle  — boundary is encouraged to align with stopgrad residual
  learned_causal  — a tiny causal probe predicts residual from past bytes;
                    boundary aligns to the probe signal
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset, MASK_ID, PAD_ID, StreamingReconstructionDataset, safe_train_eval_split
from flued.e1_stage_a import corrupt_byte_inputs, reconstruction_accuracy
from flued.model import FLUEDAutoencoder
from tools.analysis.v3_0.train_causal_surprise_probe import CausalSurpriseProbe


def _cosine_with_warmup(optimizer: optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_texts(path: str, max_lines: Optional[int]) -> List[str]:
    texts: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            line = line.rstrip("\n")
            if line:
                texts.append(line)
    if not texts:
        raise RuntimeError(f"no non-empty text loaded from {path}")
    return texts


def _standardize_valid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    vals = x[valid].float()
    if vals.numel() == 0:
        return torch.zeros_like(x).float()
    return ((x.float() - vals.mean()) / vals.std(unbiased=False).clamp(min=1e-6)).masked_fill(~valid, 0.0)


def _corr(score: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    s = score[valid].float()
    t = target[valid].float()
    if s.numel() <= 4:
        return float("nan")
    s = (s - s.mean()) / s.std(unbiased=False).clamp(min=1e-6)
    t = (t - t.mean()) / t.std(unbiased=False).clamp(min=1e-6)
    return (s * t).mean().item()


def _topk_enrichment(score: torch.Tensor, residual: torch.Tensor, valid: torch.Tensor, density: float) -> float:
    s = score[valid].float()
    r = residual[valid].float()
    n = s.numel()
    if n < 8:
        return float("nan")
    k = max(1, min(n, int(round(n * density))))
    top_s = torch.topk(s, k=k).indices
    top_r = torch.topk(r, k=k).indices
    sel = torch.zeros(n, dtype=torch.bool, device=score.device)
    high = torch.zeros(n, dtype=torch.bool, device=score.device)
    sel[top_s] = True
    high[top_r] = True
    overlap = (sel & high).float().mean().item()
    return overlap / max(k / n, 1e-6)


def _boundary_corr_loss(bp: torch.Tensor, signal: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    p = bp[valid].float()
    s = signal[valid].float()
    if p.numel() <= 4:
        return bp.new_zeros(())
    p = (p - p.mean()) / p.std(unbiased=False).clamp(min=1e-6)
    s = (s - s.mean()) / s.std(unbiased=False).clamp(min=1e-6)
    return -(p * s).mean()


def _boundary_ranking_loss(
    bp: torch.Tensor,
    signal: torch.Tensor,
    valid: torch.Tensor,
    high_quantile: float,
    low_quantile: float,
    temperature: float,
) -> torch.Tensor:
    """Soft ranking: high-surprise positions should have higher bp than low-surprise ones.

    This is not hard segmentation supervision. It only shapes relative boundary
    preference and keeps boundary_probs continuous.
    """
    losses: List[torch.Tensor] = []
    for b in range(bp.size(0)):
        mask = valid[b]
        p = bp[b][mask].float()
        s = signal[b][mask].float()
        if p.numel() < 16:
            continue
        hi_thr = torch.quantile(s.detach(), float(high_quantile))
        lo_thr = torch.quantile(s.detach(), float(low_quantile))
        high = p[s >= hi_thr]
        low = p[s <= lo_thr]
        if high.numel() == 0 or low.numel() == 0:
            continue
        # Compare distribution means instead of hard pair labels. This is cheap
        # and avoids forcing exact 0/1 boundaries.
        margin = (high.mean() - low.mean()) / max(float(temperature), 1e-4)
        losses.append(F.softplus(-margin))
    if not losses:
        return bp.new_zeros(())
    return torch.stack(losses).mean()


def _boundary_align_loss(
    bp: torch.Tensor,
    signal: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
    high_quantile: float,
    low_quantile: float,
    temperature: float,
) -> torch.Tensor:
    if mode == "none":
        return bp.new_zeros(())
    if mode == "corr":
        return _boundary_corr_loss(bp, signal, valid)
    if mode == "ranking":
        return _boundary_ranking_loss(bp, signal, valid, high_quantile, low_quantile, temperature)
    raise ValueError(f"unknown align loss mode: {mode}")


def _external_budget_loss(
    bp: torch.Tensor,
    valid: torch.Tensor,
    target: float,
    mode: str,
) -> torch.Tensor:
    valid_f = valid.float()
    valid_n = valid_f.sum(dim=1).clamp(min=1.0)
    density = (bp.float() * valid_f).sum(dim=1) / valid_n
    target_t = torch.full_like(density, float(target))
    if mode == "none":
        return bp.new_zeros(())
    if mode == "mse":
        return (density - target_t).pow(2).mean()
    if mode == "l1":
        return (density - target_t).abs().mean()
    if mode == "hinge_high":
        return torch.relu(density - target_t).mean()
    raise ValueError(f"unknown budget loss mode: {mode}")


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return float(sum(vals) / max(len(vals), 1))


def _safe_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def _masked_stats(prefix: str, x: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    vals = x[valid].detach().float()
    if vals.numel() == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "min", "p10", "p50", "p90", "p99", "max")}
    qs = torch.quantile(vals, torch.tensor([0.10, 0.50, 0.90, 0.99], device=vals.device))
    return {
        f"{prefix}_mean": vals.mean().item(),
        f"{prefix}_std": vals.std(unbiased=False).item(),
        f"{prefix}_min": vals.min().item(),
        f"{prefix}_p10": qs[0].item(),
        f"{prefix}_p50": qs[1].item(),
        f"{prefix}_p90": qs[2].item(),
        f"{prefix}_p99": qs[3].item(),
        f"{prefix}_max": vals.max().item(),
    }


def _masked_reconstruction_loss(logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        target.view(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(target)
    mask = loss_mask & (target != PAD_ID)
    if not mask.any():
        return ce.new_zeros(())
    return ce[mask].mean()


def _masked_reconstruction_accuracy(logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> float:
    mask = loss_mask & (target != PAD_ID)
    total = mask.sum().item()
    if total == 0:
        return float("nan")
    preds = logits.argmax(dim=-1)
    return ((preds == target) & mask).sum().item() / total


def _apply_future_mask(
    src: torch.Tensor,
    valid: torch.Tensor,
    mask_id: int,
    min_prefix_ratio: float,
    max_prefix_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask future suffix while scoring only the visible prefix.

    This approximates a prefill/decoding boundary condition without converting
    the whole FLUED encoder into a causal Transformer. Future bytes are hidden
    from the input, and reconstruction loss is computed only on the prefix
    before the cut.
    """
    out = src.clone()
    loss_mask = torch.zeros_like(valid)
    cut_mask = torch.zeros_like(valid)
    lo = max(0.05, min(float(min_prefix_ratio), 0.95))
    hi = max(lo, min(float(max_prefix_ratio), 0.98))
    for b in range(src.size(0)):
        n = int(valid[b].sum().item())
        if n <= 4:
            loss_mask[b] = valid[b]
            continue
        low = max(1, min(n - 1, int(round(n * lo))))
        high = max(low, min(n - 1, int(round(n * hi))))
        cut = random.randint(low, high)
        out[b, cut:n] = mask_id
        loss_mask[b, :cut] = valid[b, :cut]
        cut_mask[b, cut:n] = valid[b, cut:n]
    return out, loss_mask, cut_mask


def _coding_rate_metrics(
    bp: torch.Tensor,
    valid: torch.Tensor,
    z: torch.Tensor,
    proj: Optional[torch.Tensor],
    alpha: float,
) -> Dict[str, float]:
    with torch.no_grad():
        vf = valid.float()
        n = vf.sum().clamp(min=1.0)
        p = bp.float().clamp(1e-4, 1.0 - 1e-4)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        out = {
            "rate_boundary": float((bp.float() * vf).sum().item() / n.item()),
            "rate_boundary_entropy": float((entropy * vf).sum().item() / n.item()),
            "rate_latent": float("nan"),
        }
        if proj is None:
            return out
        vals = z.detach().float()[valid]
        if vals.size(0) < 4:
            return out
        y = vals @ proj
        y = y - y.mean(dim=0, keepdim=True)
        cov = (y.T @ y) / max(1, y.size(0) - 1)
        eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        sign, logabsdet = torch.linalg.slogdet(eye + float(alpha) * cov)
        out["rate_latent"] = float(logabsdet.item() if sign.item() > 0 else float("nan"))
        return out


def _coding_rate_tensor(
    bp: torch.Tensor,
    valid: torch.Tensor,
    z: torch.Tensor,
    proj: Optional[torch.Tensor],
    alpha: float,
    latent_weight: float,
) -> torch.Tensor:
    valid_f = valid.float()
    valid_n = valid_f.sum().clamp(min=1.0)
    rate_boundary = (bp.float() * valid_f).sum() / valid_n
    if proj is None or latent_weight <= 0:
        return rate_boundary
    vals = z.float()[valid]
    if vals.size(0) < 4:
        return rate_boundary
    y = vals @ proj
    y = y - y.mean(dim=0, keepdim=True)
    cov = (y.T @ y) / max(1, y.size(0) - 1)
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    sign, logabsdet = torch.linalg.slogdet(eye + float(alpha) * cov)
    rate_latent = torch.where(sign > 0, logabsdet / max(1, proj.size(1)), rate_boundary.new_zeros(()))
    return rate_boundary + float(latent_weight) * rate_latent


def _boundary_contrast_loss(
    bp: torch.Tensor,
    valid: torch.Tensor,
    min_std: float,
    min_iqr: float,
) -> torch.Tensor:
    """Prevent the soft boundary distribution from collapsing into a 0.5 band."""
    losses: List[torch.Tensor] = []
    for b in range(bp.size(0)):
        vals = bp[b][valid[b]].float()
        if vals.numel() < 16:
            continue
        if min_std > 0:
            losses.append(F.relu(float(min_std) - vals.std(unbiased=False)))
        if min_iqr > 0:
            qs = torch.quantile(vals, torch.tensor([0.10, 0.90], device=vals.device))
            losses.append(F.relu(float(min_iqr) - (qs[1] - qs[0])))
    if not losses:
        return bp.new_zeros(())
    return torch.stack(losses).mean()


def _grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += g.pow(2).sum().item()
    return math.sqrt(total)


def _append_jsonl(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(
    model: FLUEDAutoencoder,
    surprise: Optional[CausalSurpriseProbe],
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    amp: bool,
    mode: str = "clean",
    corrupt_rate: float = 0.15,
    span_mask_prob: float = 0.7,
    span_min: int = 1,
    span_max: int = 8,
    future_min_prefix_ratio: float = 0.35,
    future_max_prefix_ratio: float = 0.75,
) -> Dict[str, float]:
    model.eval()
    if surprise is not None:
        surprise.eval()
    accs: List[float] = []
    soft_mn: List[float] = []
    bp_std: List[float] = []
    bp_corr: List[float] = []
    signal_corr: List[float] = []
    bp_enrich: List[float] = []
    signal_enrich: List[float] = []
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
                corrupt_rate=corrupt_rate,
                span_mask_prob=span_mask_prob,
                span_min=span_min,
                span_max=span_max,
            )
        elif mode == "future_mask":
            model_src, loss_mask, _ = _apply_future_mask(
                src,
                valid,
                mask_id=MASK_ID,
                min_prefix_ratio=future_min_prefix_ratio,
                max_prefix_ratio=future_max_prefix_ratio,
            )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits, metrics = model(model_src, src_key_padding_mask=(src == PAD_ID), boundary_src=src, skip_hard=True)
        ce = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            src.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(src)
        bp = metrics["boundary_probs"].float()
        if surprise is not None:
            pred, _ = surprise(src)
            signal = pred.float()
        else:
            signal = ce.float()
        density = bp[valid].mean().item()
        accs.append(_masked_reconstruction_accuracy(logits.detach(), src, loss_mask))
        soft_mn.append(metrics["soft_m_over_n"].item())
        bp_std.append(bp[valid].std(unbiased=False).item())
        bp_corr.append(_corr(bp, ce, valid))
        signal_corr.append(_corr(signal, ce, valid))
        bp_enrich.append(_topk_enrichment(bp, ce, valid, density))
        signal_enrich.append(_topk_enrichment(signal, ce, valid, density))
    model.train()
    if surprise is not None:
        surprise.train()
    prefix = "eval" if mode == "clean" else f"eval_{mode}"
    return {
        f"{prefix}_acc": _mean(accs),
        f"{prefix}_soft_mn": _mean(soft_mn),
        f"{prefix}_bp_std": _mean(bp_std),
        f"{prefix}_bp_residual_corr": _mean(bp_corr),
        f"{prefix}_signal_residual_corr": _mean(signal_corr),
        f"{prefix}_bp_residual_enrichment": _mean(bp_enrich),
        f"{prefix}_signal_residual_enrichment": _mean(signal_enrich),
    }


def run(args: argparse.Namespace) -> Dict[str, float]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"

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
        dataset = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
        train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.1, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.streaming_train,
        drop_last=True,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )

    model = FLUEDAutoencoder(
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        swiglu_hidden=args.swiglu_hidden,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        assignment_window=args.assignment_window,
        dropout=0.0,
        target_compression=args.target_compression,
        compression_weight=args.compression_weight,
        lambda_utf8=args.lambda_utf8,
        lambda_cjk=args.lambda_cjk,
        lambda_type=args.lambda_type,
    ).to(device)

    surprise: Optional[CausalSurpriseProbe] = None
    boundary_params = list(model.boundary_head.parameters())
    boundary_param_ids = {id(p) for p in boundary_params}
    main_params = [p for p in model.parameters() if id(p) not in boundary_param_ids]
    params: List[nn.Parameter] = main_params + boundary_params
    param_groups = [
        {"params": main_params, "lr": args.lr, "name": "main"},
        {"params": boundary_params, "lr": args.lr, "name": "boundary"},
    ]
    if args.mode == "learned_causal":
        surprise = CausalSurpriseProbe(d_model=args.surprise_d_model, hidden=args.surprise_hidden).to(device)
        surprise_params = list(surprise.parameters())
        params += surprise_params
        param_groups.append({"params": surprise_params, "lr": args.lr, "name": "surprise"})

    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _cosine_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    coding_proj: Optional[torch.Tensor] = None
    if args.coding_rate_proj_dim > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 1009)
        coding_proj = torch.randn(
            args.d_model,
            args.coding_rate_proj_dim,
            generator=gen,
            device=device,
        ) / math.sqrt(max(1, args.d_model))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"
    latest_path = out_dir / "latest.pt"

    train_iter = iter(train_loader)
    rows: List[Dict[str, float]] = []
    window: Dict[str, List[float]] = defaultdict(list)
    start_step = 0
    adaptive_rate_lambda = float(args.adaptive_rate_lambda_init)
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and args.auto_resume and latest_path.exists():
        resume_path = latest_path
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if surprise is not None and "surprise" in ckpt:
            surprise.load_state_dict(ckpt["surprise"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", ckpt.get("summary", {}).get("steps", 0)))
        print(f"resumed from {resume_path} at step={start_step}", flush=True)

    def _save_checkpoint(step: int, partial_summary: Optional[Dict] = None) -> None:
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }
        if surprise is not None:
            ckpt["surprise"] = surprise.state_dict()
        if partial_summary is not None:
            ckpt["summary"] = partial_summary
        tmp = latest_path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, latest_path)
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            step_path = out_dir / f"step{step:06d}.pt"
            tmp_step = step_path.with_suffix(".pt.tmp")
            torch.save(ckpt, tmp_step)
            os.replace(tmp_step, step_path)

    for step in range(start_step + 1, args.max_steps + 1):
        try:
            src, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, _ = next(train_iter)
        src = src.to(device)
        pad = src == PAD_ID
        valid = ~pad
        task_rand = random.random()
        use_future_mask = task_rand < args.future_mask_prob
        denoise_active = args.denoise_steps < 0 or step <= args.denoise_steps
        use_denoise = (
            not use_future_mask
            and denoise_active
            and random.random() < args.denoise_prob
        )
        loss_mask = valid
        future_cut_mask = torch.zeros_like(valid)
        if use_future_mask:
            model_src, loss_mask, future_cut_mask = _apply_future_mask(
                src,
                valid,
                mask_id=MASK_ID,
                min_prefix_ratio=args.future_min_prefix_ratio,
                max_prefix_ratio=args.future_max_prefix_ratio,
            )
            task_mode = "future_mask"
        elif use_denoise:
            model_src = corrupt_byte_inputs(
                src,
                valid,
                mask_id=MASK_ID,
                corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob,
                span_min=args.span_min,
                span_max=args.span_max,
            )
            task_mode = "denoise"
        else:
            model_src = src
            task_mode = "clean"

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, metrics = model(model_src, src_key_padding_mask=pad, boundary_src=src, skip_hard=True)
            recon_loss = _masked_reconstruction_loss(logits, src, loss_mask)
            comp_loss = metrics["compression_loss"]

        ce = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            src.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(src)
        residual_target = _standardize_valid(ce.detach(), valid).clamp(-5.0, 5.0)
        bp = metrics["boundary_probs"].float()

        surprise_loss = logits.new_zeros(())
        surprise_mse_loss = logits.new_zeros(())
        surprise_byte_loss = logits.new_zeros(())
        align_loss = logits.new_zeros(())
        signal = residual_target
        if args.mode == "teacher_oracle":
            signal = residual_target
            if step >= args.align_warmup_steps:
                align_loss = _boundary_align_loss(
                    bp,
                    signal.detach(),
                    valid,
                    mode=args.align_loss,
                    high_quantile=args.ranking_high_quantile,
                    low_quantile=args.ranking_low_quantile,
                    temperature=args.ranking_temperature,
                )
        elif args.mode == "learned_causal":
            assert surprise is not None
            pred, byte_logits = surprise(src)
            pred_z = _standardize_valid(pred, valid).clamp(-5.0, 5.0)
            surprise_mse_loss = F.mse_loss(pred_z[valid].float(), residual_target[valid].float())
            surprise_byte_loss = F.cross_entropy(byte_logits.view(-1, byte_logits.size(-1)), src.view(-1), ignore_index=PAD_ID)
            surprise_loss = surprise_mse_loss + args.surprise_byte_weight * surprise_byte_loss
            signal = pred_z
            if step >= args.align_warmup_steps:
                align_loss = _boundary_align_loss(
                    bp,
                    signal.detach(),
                    valid,
                    mode=args.align_loss,
                    high_quantile=args.ranking_high_quantile,
                    low_quantile=args.ranking_low_quantile,
                    temperature=args.ranking_temperature,
                )

        align_weight = args.align_weight
        if args.align_warmup_steps > 0:
            align_weight *= min(1.0, max(0.0, (step - args.align_warmup_steps) / max(1, args.align_ramp_steps)))
        budget_loss = _external_budget_loss(
            bp=bp,
            valid=valid,
            target=args.external_budget_target,
            mode=args.external_budget_loss,
        )
        budget_weight = args.external_budget_weight
        if step < args.external_budget_start_steps:
            budget_weight = 0.0
        elif args.external_budget_warmup_steps > 0:
            progress = (step - args.external_budget_start_steps) / max(1, args.external_budget_warmup_steps)
            budget_weight *= min(1.0, max(0.0, progress))
        coding_rate_value = _coding_rate_tensor(
            bp=bp,
            valid=valid,
            z=metrics["z"].float(),
            proj=coding_proj,
            alpha=args.coding_rate_alpha,
            latent_weight=args.coding_rate_latent_weight,
        )
        adaptive_rate_loss = coding_rate_value - float(args.adaptive_rate_target)
        adaptive_rate_weight = adaptive_rate_lambda if step >= args.adaptive_rate_start_steps else 0.0
        contrast_loss = _boundary_contrast_loss(
            bp=bp,
            valid=valid,
            min_std=args.anti_collapse_min_std,
            min_iqr=args.anti_collapse_min_iqr,
        )
        total_loss = (
            recon_loss
            + comp_loss
            + args.surprise_weight * surprise_loss
            + align_weight * align_loss
            + budget_weight * budget_loss
            + adaptive_rate_weight * adaptive_rate_loss
            + args.anti_collapse_weight * contrast_loss
        )
        total_loss.backward()
        total_grad_norm = _grad_norm(params)
        boundary_grad_norm = _grad_norm(boundary_params)
        surprise_grad_norm = _grad_norm(surprise.parameters()) if surprise is not None else 0.0
        clipped_grad_norm = nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if args.adaptive_rate_eta > 0 and step >= args.adaptive_rate_start_steps:
            with torch.no_grad():
                adaptive_rate_lambda = max(
                    0.0,
                    min(
                        float(args.adaptive_rate_lambda_max),
                        adaptive_rate_lambda + float(args.adaptive_rate_eta) * float(adaptive_rate_loss.detach().item()),
                    ),
                )
        if args.budget_stage_lr_restart and step >= args.external_budget_start_steps:
            for group in optimizer.param_groups:
                name = group.get("name")
                if name == "main" and args.budget_stage_main_lr > 0:
                    group["lr"] = args.budget_stage_main_lr
                elif name == "boundary" and args.budget_stage_boundary_lr > 0:
                    group["lr"] = args.budget_stage_boundary_lr
                elif name == "surprise" and args.budget_stage_surprise_lr > 0:
                    group["lr"] = args.budget_stage_surprise_lr

        with torch.no_grad():
            density = bp[valid].mean().item()
            coding = _coding_rate_metrics(
                bp=bp,
                valid=valid,
                z=metrics["z"].float(),
                proj=coding_proj,
                alpha=args.coding_rate_alpha,
            )
            row = {
                    "step": float(step),
                    "use_denoise": float(use_denoise),
                    "use_future_mask": float(use_future_mask),
                    "task_clean": float(task_mode == "clean"),
                    "task_denoise": float(task_mode == "denoise"),
                    "task_future_mask": float(task_mode == "future_mask"),
                    "loss_tokens": float(loss_mask.sum().item()),
                    "future_mask_tokens": float(future_cut_mask.sum().item()),
                    "valid_tokens": float(valid.sum().item()),
                    "loss": float(total_loss.item()),
                    "recon": float(recon_loss.item()),
                    "comp": float(comp_loss.item()),
                    "surprise": float(surprise_loss.item()),
                    "surprise_mse": float(surprise_mse_loss.item()),
                    "surprise_byte": float(surprise_byte_loss.item()),
                    "align": float(align_loss.item()),
                    "align_weight": float(align_weight),
                    "budget": float(budget_loss.item()),
                    "budget_weight": float(budget_weight),
                    "coding_rate_loss": float(adaptive_rate_loss.detach().item()),
                    "coding_rate_value": float(coding_rate_value.detach().item()),
                    "adaptive_rate_lambda": float(adaptive_rate_lambda),
                    "adaptive_rate_weight": float(adaptive_rate_weight),
                    "contrast": float(contrast_loss.detach().item()),
                    "contrast_weight": float(args.anti_collapse_weight),
                    "acc": float(_masked_reconstruction_accuracy(logits.detach(), src, loss_mask)),
                    "soft_mn": float(metrics["soft_m_over_n"].item()),
                    "bp_std": float(bp[valid].std(unbiased=False).item()),
                    "bp_residual_corr": float(_corr(bp, ce, valid)),
                    "signal_residual_corr": float(_corr(signal, ce, valid)),
                    "bp_residual_enrichment": float(_topk_enrichment(bp, ce, valid, density)),
                    "signal_residual_enrichment": float(_topk_enrichment(signal, ce, valid, density)),
                    "total_grad_norm": float(total_grad_norm),
                    "boundary_grad_norm": float(boundary_grad_norm),
                    "surprise_grad_norm": float(surprise_grad_norm),
                    "clipped_grad_norm": _safe_float(clipped_grad_norm),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "boundary_lr": float(next(g["lr"] for g in optimizer.param_groups if g.get("name") == "boundary")),
            }
            row.update(_masked_stats("ce", ce, valid))
            row.update(_masked_stats("bp", bp, valid))
            row.update(_masked_stats("signal", signal, valid))
            row.update(coding)
            row.update({
                "bp_gt_03": (bp[valid] > 0.3).float().mean().item(),
                "bp_gt_05": (bp[valid] > 0.5).float().mean().item(),
                "bp_gt_07": (bp[valid] > 0.7).float().mean().item(),
                "bp_lt_03": (bp[valid] < 0.3).float().mean().item(),
                "gpu_mem_alloc_mb": torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0,
                "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2) if device.type == "cuda" else 0.0,
            })
            for key in ("utf8_cont", "ascii", "cjk", "op", "digit"):
                val = metrics.get(f"{key}_bp_mean", float("nan"))
                row[f"type_{key}_bp"] = float(val) if val == val else float("nan")

        for key, value in row.items():
            if key != "step":
                window[key].append(float(value))

        if args.metrics_every > 0 and (step == 1 or step % args.metrics_every == 0 or step == args.max_steps):
            _append_jsonl(metrics_path, row)

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            def wavg(key: str) -> float:
                vals = [v for v in window.get(key, []) if not math.isnan(v)]
                return sum(vals) / max(1, len(vals))
            log_row = dict(row)
            for key in (
                "loss", "recon", "comp", "surprise", "surprise_mse", "surprise_byte",
                "align", "coding_rate_loss", "coding_rate_value", "adaptive_rate_lambda",
                "contrast", "acc", "soft_mn", "bp_std", "bp_residual_corr",
                "signal_residual_corr", "bp_residual_enrichment", "signal_residual_enrichment",
                "use_denoise", "use_future_mask", "task_clean", "task_denoise", "task_future_mask",
                "total_grad_norm", "boundary_grad_norm", "surprise_grad_norm",
                "ce_mean", "ce_p90", "bp_mean", "bp_p10", "bp_p50", "bp_p90",
                "rate_boundary", "rate_boundary_entropy", "rate_latent",
            ):
                log_row[f"{key}_win"] = wavg(key)
            rows.append(row)
            line = (
                "step={step:.0f} loss={loss_win:.4f} recon={recon_win:.4f} comp={comp_win:.4f} "
                "surprise={surprise_win:.4f} smse={surprise_mse_win:.4f} sbyte={surprise_byte_win:.4f} "
                "align={align_win:.4f} aw={align_weight:.3f} "
                "budget={budget:.4f} bw={budget_weight:.3f} "
                "crate={coding_rate_value_win:.3f}/{coding_rate_loss_win:.3f} cl={adaptive_rate_lambda_win:.3f} "
                "contrast={contrast_win:.4f} cw={contrast_weight:.3f} "
                "acc={acc_win:.4f} soft_m/n={soft_mn_win:.3f} bp_std={bp_std_win:.3f} "
                "bp_corr={bp_residual_corr_win:.3f} signal_corr={signal_residual_corr_win:.3f} "
                "bp_enrich={bp_residual_enrichment_win:.3f} signal_enrich={signal_residual_enrichment_win:.3f}"
                " tasks=c{task_clean_win:.2f}/d{task_denoise_win:.2f}/f{task_future_mask_win:.2f}"
                " denoise={use_denoise_win:.2f} future={use_future_mask_win:.2f}"
                " ce={ce_mean_win:.3f}/p90={ce_p90_win:.3f}"
                " bp={bp_mean_win:.3f}/p10={bp_p10_win:.3f}/p50={bp_p50_win:.3f}/p90={bp_p90_win:.3f}"
                " rate=b{rate_boundary_win:.3f}/h{rate_boundary_entropy_win:.3f}/z{rate_latent_win:.3f}"
                " g={total_grad_norm_win:.2f} bg={boundary_grad_norm_win:.4f} sg={surprise_grad_norm_win:.2f}"
                " blr={boundary_lr:.2e}"
            ).format(**log_row)
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            window.clear()
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            partial = {
                "mode": args.mode,
                "steps": step,
                "streaming_train": bool(args.streaming_train),
                "data_path": args.data_path,
                "model_params": model.count_parameters(),
                "surprise_params": sum(p.numel() for p in surprise.parameters()) if surprise is not None else 0,
                "last": rows[-1] if rows else {},
            }
            _save_checkpoint(step, partial)

    eval_metrics = {}
    eval_metrics.update(evaluate(model, surprise, eval_loader, device, args.max_eval_batches, amp, mode="clean"))
    eval_metrics.update(evaluate(
        model,
        surprise,
        eval_loader,
        device,
        args.max_eval_batches,
        amp,
        mode="denoise",
        corrupt_rate=args.corrupt_rate,
        span_mask_prob=args.span_mask_prob,
        span_min=args.span_min,
        span_max=args.span_max,
    ))
    eval_metrics.update(evaluate(
        model,
        surprise,
        eval_loader,
        device,
        args.max_eval_batches,
        amp,
        mode="future_mask",
        future_min_prefix_ratio=args.future_min_prefix_ratio,
        future_max_prefix_ratio=args.future_max_prefix_ratio,
    ))
    result = {
        "mode": args.mode,
        "steps": args.max_steps,
        "streaming_train": bool(args.streaming_train),
        "data_path": args.data_path,
        "model_params": model.count_parameters(),
        "surprise_params": sum(p.numel() for p in surprise.parameters()) if surprise is not None else 0,
        "last": rows[-1] if rows else {},
        **eval_metrics,
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _save_checkpoint(args.max_steps, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Surprise-assisted FLUED-small E1 experiment")
    parser.add_argument("--mode", choices=["native", "teacher_oracle", "learned_causal"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=2500)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--swiglu-hidden", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--assignment-window", type=int, default=128)
    parser.add_argument("--target-compression", type=float, default=0.3)
    parser.add_argument("--compression-weight", type=float, default=0.1)
    parser.add_argument("--lambda-utf8", type=float, default=0.05)
    parser.add_argument("--lambda-cjk", type=float, default=0.05)
    parser.add_argument("--lambda-type", type=float, default=0.02)
    parser.add_argument("--denoise-prob", type=float, default=0.5)
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=20000,
        help="Use denoising only up to this step; set -1 to keep it active for the whole run.",
    )
    parser.add_argument("--corrupt-rate", type=float, default=0.15)
    parser.add_argument("--span-mask-prob", type=float, default=0.7)
    parser.add_argument("--span-min", type=int, default=1)
    parser.add_argument("--span-max", type=int, default=8)
    parser.add_argument("--future-mask-prob", type=float, default=0.20)
    parser.add_argument("--future-min-prefix-ratio", type=float, default=0.35)
    parser.add_argument("--future-max-prefix-ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--surprise-d-model", type=int, default=96)
    parser.add_argument("--surprise-hidden", type=int, default=160)
    parser.add_argument("--surprise-weight", type=float, default=0.10)
    parser.add_argument("--surprise-byte-weight", type=float, default=0.20)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--align-loss", choices=["none", "corr", "ranking"], default="ranking")
    parser.add_argument("--align-warmup-steps", type=int, default=100)
    parser.add_argument("--align-ramp-steps", type=int, default=200)
    parser.add_argument("--ranking-high-quantile", type=float, default=0.80)
    parser.add_argument("--ranking-low-quantile", type=float, default=0.20)
    parser.add_argument("--ranking-temperature", type=float, default=0.10)
    parser.add_argument("--coding-rate-proj-dim", type=int, default=64)
    parser.add_argument("--coding-rate-alpha", type=float, default=1.0)
    parser.add_argument("--coding-rate-latent-weight", type=float, default=0.0)
    parser.add_argument("--adaptive-rate-target", type=float, default=0.45)
    parser.add_argument("--adaptive-rate-start-steps", type=int, default=0)
    parser.add_argument("--adaptive-rate-eta", type=float, default=0.0)
    parser.add_argument("--adaptive-rate-lambda-init", type=float, default=0.0)
    parser.add_argument("--adaptive-rate-lambda-max", type=float, default=5.0)
    parser.add_argument("--anti-collapse-weight", type=float, default=0.0)
    parser.add_argument("--anti-collapse-min-std", type=float, default=0.0)
    parser.add_argument("--anti-collapse-min-iqr", type=float, default=0.0)
    parser.add_argument("--external-budget-loss", choices=["none", "mse", "l1", "hinge_high"], default="none")
    parser.add_argument("--external-budget-weight", type=float, default=0.0)
    parser.add_argument("--external-budget-target", type=float, default=0.3)
    parser.add_argument("--external-budget-start-steps", type=int, default=0)
    parser.add_argument("--external-budget-warmup-steps", type=int, default=200)
    parser.add_argument("--budget-stage-lr-restart", action="store_true")
    parser.add_argument("--budget-stage-main-lr", type=float, default=0.0)
    parser.add_argument("--budget-stage-boundary-lr", type=float, default=0.0)
    parser.add_argument("--budget-stage-surprise-lr", type=float, default=0.0)
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run(args)


if __name__ == "__main__":
    main()
