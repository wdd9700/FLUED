"""Train a minimal FLUED-v3 commit-controller prototype.

This is not the v2 FLUED autoencoder. It is a small diagnostic model for the
v3 route:

  byte stream -> causal hidden h_t
              -> active segment state a_t
              -> committed memory summary m_t
              -> soft commit probability p_t

The purpose is to test whether commit decisions improve when the boundary
controller can see the current active segment and committed history summary.
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


def _cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        target.view(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(target)
    valid = mask & (target != PAD_ID)
    if not valid.any():
        return ce.new_zeros(())
    return ce[valid].mean()


def _masked_acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask & (target != PAD_ID)
    total = valid.sum().item()
    if total == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return ((pred == target) & valid).sum().item() / total


def _prediction_target(src: torch.Tensor, valid: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "current":
        return src, valid
    if mode == "next_byte":
        target = torch.full_like(src, PAD_ID)
        target[:, :-1] = src[:, 1:]
        mask = valid.clone()
        mask[:, :-1] = valid[:, :-1] & valid[:, 1:]
        mask[:, -1] = False
        return target, mask
    raise ValueError(f"unknown prediction target: {mode}")


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
    return ((sel & high).float().mean().item()) / max(k / n, 1e-6)


def _masked_stats(prefix: str, x: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    vals = x[valid].detach().float()
    if vals.numel() == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "p10", "p50", "p90", "p99")}
    qs = torch.quantile(vals, torch.tensor([0.10, 0.50, 0.90, 0.99], device=vals.device))
    return {
        f"{prefix}_mean": vals.mean().item(),
        f"{prefix}_std": vals.std(unbiased=False).item(),
        f"{prefix}_p10": qs[0].item(),
        f"{prefix}_p50": qs[1].item(),
        f"{prefix}_p90": qs[2].item(),
        f"{prefix}_p99": qs[3].item(),
    }


def _append_jsonl(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _apply_future_mask(
    src: torch.Tensor,
    valid: torch.Tensor,
    min_prefix_ratio: float,
    max_prefix_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = src.clone()
    loss_mask = torch.zeros_like(valid)
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
        out[b, cut:n] = MASK_ID
        loss_mask[b, :cut] = valid[b, :cut]
    return out, loss_mask


class MLPStateUpdate(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([x, state], dim=-1)))


class V3CommitControllerSmall(nn.Module):
    def __init__(
        self,
        vocab_size: int = 258,
        d_model: int = 192,
        hidden: int = 192,
        controller_hidden: int = 256,
        decoder_input: str = "active_memory",
        controller_memory_mode: str = "raw",
        update_cell: str = "gru",
        commit_stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if decoder_input not in {"hidden_active_memory", "active_memory", "gated_active_memory", "active", "memory"}:
            raise ValueError(f"unknown decoder_input: {decoder_input}")
        if controller_memory_mode not in {"raw", "features"}:
            raise ValueError(f"unknown controller_memory_mode: {controller_memory_mode}")
        if update_cell not in {"gru", "mlp"}:
            raise ValueError(f"unknown update_cell: {update_cell}")
        self.decoder_input = decoder_input
        self.controller_memory_mode = controller_memory_mode
        self.update_cell = update_cell
        self.commit_stride = max(1, int(commit_stride))
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.encoder = nn.GRU(d_model, hidden, num_layers=1, batch_first=True)
        if update_cell == "gru":
            self.active_update = nn.GRUCell(hidden, hidden)
            self.memory_update = nn.GRUCell(hidden, hidden)
        else:
            self.active_update = MLPStateUpdate(hidden)
            self.memory_update = MLPStateUpdate(hidden)
        if controller_memory_mode == "raw":
            control_dim = hidden * 4 + 4
        else:
            control_dim = hidden * 3 + 8
        self.controller = nn.Sequential(
            nn.LayerNorm(control_dim),
            nn.Linear(control_dim, controller_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(controller_hidden, 1),
        )
        if decoder_input == "hidden_active_memory":
            decoder_dim = hidden * 3
        elif decoder_input == "active_memory":
            decoder_dim = hidden * 2
        elif decoder_input == "gated_active_memory":
            decoder_dim = hidden
            gate_dim = hidden * 2
            self.memory_gate = nn.Sequential(
                nn.LayerNorm(gate_dim),
                nn.Linear(gate_dim, hidden),
                nn.Sigmoid(),
            )
            self.memory_adapter = nn.Sequential(
                nn.LayerNorm(gate_dim),
                nn.Linear(gate_dim, hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )
        elif decoder_input in {"active", "memory"}:
            decoder_dim = hidden
        self.byte_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, vocab_size),
        )
        self.future_head = nn.Linear(hidden, vocab_size)

    def forward(self, src: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        emb = self.embedding(src.clamp(min=0, max=MASK_ID))
        h, _ = self.encoder(emb)
        bsz, seq_len, hidden = h.shape
        active = h.new_zeros(bsz, hidden)
        memory = h.new_zeros(bsz, hidden)
        commit_probs: List[torch.Tensor] = []
        active_states: List[torch.Tensor] = []
        memory_states: List[torch.Tensor] = []
        future_logits: List[torch.Tensor] = []
        memory_gate: Optional[torch.Tensor] = None
        prev_p = h.new_zeros(bsz)

        for t in range(seq_len):
            ht = h[:, t]
            vt = valid[:, t].float().unsqueeze(-1)
            diff = ht - active
            mem_delta = active - memory
            mem_match = (active * memory).sum(dim=-1, keepdim=True) / math.sqrt(hidden)
            mem_delta_norm = mem_delta.norm(dim=-1, keepdim=True) / math.sqrt(hidden)
            mem_norm = memory.norm(dim=-1, keepdim=True) / math.sqrt(hidden)
            active_norm = active.norm(dim=-1, keepdim=True) / math.sqrt(hidden)
            age = torch.full((bsz, 1), float(t) / max(1, seq_len - 1), device=h.device, dtype=h.dtype)
            budget = torch.stack([
                torch.stack(commit_probs, dim=1).mean(dim=1) if commit_probs else h.new_zeros(bsz),
                valid[:, : t + 1].float().mean(dim=1),
                age.squeeze(-1),
                diff.norm(dim=-1) / math.sqrt(hidden),
            ], dim=-1)
            if t % self.commit_stride == 0:
                if self.controller_memory_mode == "raw":
                    ctrl_in = torch.cat([ht, active, memory, diff, budget], dim=-1)
                else:
                    memory_features = torch.cat([mem_match, mem_delta_norm, mem_norm, active_norm], dim=-1)
                    ctrl_in = torch.cat([ht, active, diff, budget, memory_features], dim=-1)
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
            memory = torch.where(vt.bool(), (1.0 - p_col) * memory + p_col * new_memory, memory)
            active = torch.where(vt.bool(), (1.0 - p_col) * new_active + p_col * ht, active)

            commit_probs.append(p)
            active_states.append(active)
            memory_states.append(memory)
            future_logits.append(self.future_head(memory))

        commit = torch.stack(commit_probs, dim=1)
        active_seq = torch.stack(active_states, dim=1)
        memory_seq = torch.stack(memory_states, dim=1)
        if self.decoder_input == "hidden_active_memory":
            decoder_state = torch.cat([h, active_seq, memory_seq], dim=-1)
        elif self.decoder_input == "active_memory":
            decoder_state = torch.cat([active_seq, memory_seq], dim=-1)
        elif self.decoder_input == "gated_active_memory":
            gate_in = torch.cat([active_seq, memory_seq], dim=-1)
            gate = self.memory_gate(gate_in)
            decoder_state = active_seq + gate * self.memory_adapter(gate_in)
            memory_gate = gate.mean(dim=-1)
        elif self.decoder_input == "active":
            decoder_state = active_seq
        else:
            decoder_state = memory_seq
        logits = self.byte_head(decoder_state)
        future = torch.stack(future_logits, dim=1)
        out = {
            "commit_probs": commit,
            "h": h,
            "active": active_seq,
            "memory": memory_seq,
            "future_logits": future,
        }
        if memory_gate is not None:
            out["memory_gate"] = memory_gate
        return logits, out


@torch.no_grad()
def evaluate(
    model: V3CommitControllerSmall,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    mode: str,
    corrupt_rate: float,
    span_mask_prob: float,
    future_min_prefix_ratio: float,
    future_max_prefix_ratio: float,
    prediction_target: str,
    hybrid_existing_diffusion: bool,
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
                corrupt_rate=corrupt_rate,
                span_mask_prob=span_mask_prob,
                span_min=1,
                span_max=8,
            )
        elif mode == "future_mask":
            model_src, loss_mask = _apply_future_mask(src, valid, future_min_prefix_ratio, future_max_prefix_ratio)
        logits, metrics = model(model_src, valid)
        target_mode = "current" if hybrid_existing_diffusion and mode == "denoise" else prediction_target
        target, target_mask = _prediction_target(src, valid, target_mode)
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
        vals["commit_mn"].append(density)
        vals["commit_std"].append(commit[usable].std(unbiased=False).item() if usable.any() else 0.0)
        vals["commit_corr"].append(_corr(commit, ce, usable))
        vals["commit_enrich"].append(_topk_enrichment(commit, ce, usable, max(0.01, density)))
    prefix = "eval" if mode == "clean" else f"eval_{mode}"
    out = {}
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
            ds,
            [n_train, len(ds) - n_train],
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

    model = V3CommitControllerSmall(
        d_model=args.d_model,
        hidden=args.hidden,
        controller_hidden=args.controller_hidden,
        decoder_input=args.decoder_input,
        controller_memory_mode=args.controller_memory_mode,
        update_cell=args.update_cell,
        commit_stride=args.commit_stride,
    ).to(device)
    if args.torch_compile:
        try:
            import triton  # type: ignore  # noqa: F401

            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled mode={args.compile_mode}", flush=True)
        except Exception as exc:
            print(f"torch.compile failed, continuing eager: {exc}", flush=True)
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
    start_step = int(args.start_step)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if args.resume_training_state:
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                sched.load_state_dict(ckpt["scheduler"])
            rate_lambda = float(ckpt.get("rate_lambda", rate_lambda))
            start_step = int(ckpt.get("step", start_step))
        print(f"loaded init checkpoint {args.init_ckpt} at start_step={start_step}", flush=True)

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

    for step in range(start_step + 1, args.max_steps + 1):
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
                src,
                valid,
                mask_id=MASK_ID,
                corrupt_rate=args.corrupt_rate,
                span_mask_prob=args.span_mask_prob,
                span_min=args.span_min,
                span_max=args.span_max,
            )
            task = "denoise"

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits, metrics = model(model_src, valid)
            target_mode = "current" if args.hybrid_existing_diffusion and task == "denoise" else args.prediction_target
            target, target_mask = _prediction_target(src, valid, target_mode)
            target_mask = target_mask & loss_mask
            recon_loss = _masked_ce(logits, target, target_mask)
            future_target, future_mask = _prediction_target(src, valid, "next_byte")
            future_mask = future_mask & loss_mask
            future_loss = _masked_ce(metrics["future_logits"], future_target, future_mask)
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
                + args.future_loss_weight * future_loss
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
            ce = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).view_as(target)
            density = commit[usable].mean().item() if usable.any() else 0.0
            row = {
                "step": float(step),
                "loss": float(loss.item()),
                "recon": float(recon_loss.item()),
                "future_loss": float(future_loss.item()),
                "acc": float(_masked_acc(logits.detach(), target, target_mask)),
                "task_clean": float(task == "clean"),
                "task_denoise": float(task == "denoise"),
                "task_future_mask": float(task == "future_mask"),
                "target_current": float(target_mode == "current"),
                "target_next": float(target_mode == "next_byte"),
                "commit_mn": density,
                "commit_std": float(commit[usable].std(unbiased=False).item()) if usable.any() else 0.0,
                "commit_corr": float(_corr(commit, ce, usable)),
                "commit_enrich": float(_topk_enrichment(commit, ce, usable, max(0.01, density))),
                "rate_excess": float(rate_excess.detach().item()),
                "rate_lambda": float(rate_lambda),
                "spread": float(spread_loss.detach().item()),
                "grad_norm": float(grad),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            row.update(_masked_stats("commit", commit, usable))
            if "memory_gate" in metrics:
                row.update(_masked_stats("memory_gate", metrics["memory_gate"].float(), valid))
            row.update(_masked_stats("ce", ce, valid))
            last_row = row
        for k, v in row.items():
            if k != "step":
                window[k].append(float(v))
        if args.metrics_every > 0 and (step == 1 or step % args.metrics_every == 0 or step == args.max_steps):
            _append_jsonl(metrics_path, row)
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            def wavg(key: str) -> float:
                vals2 = [x for x in window.get(key, []) if not math.isnan(x)]
                return sum(vals2) / max(1, len(vals2))

            line = (
                f"step={step} loss={wavg('loss'):.4f} recon={wavg('recon'):.4f} "
                f"future={wavg('future_loss'):.4f} acc={wavg('acc'):.4f} "
                f"commit_m/n={wavg('commit_mn'):.3f} std={wavg('commit_std'):.3f} "
                f"p10={wavg('commit_p10'):.3f} p50={wavg('commit_p50'):.3f} p90={wavg('commit_p90'):.3f} "
                f"corr={wavg('commit_corr'):.3f} enrich={wavg('commit_enrich'):.3f} "
                f"tasks=c{wavg('task_clean'):.2f}/d{wavg('task_denoise'):.2f}/f{wavg('task_future_mask'):.2f} "
                f"rate_excess={wavg('rate_excess'):.3f} lambda={wavg('rate_lambda'):.3f} spread={wavg('spread'):.3f}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            window.clear()
        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save_checkpoint(step)

    eval_metrics = {}
    for mode in ("clean", "denoise", "future_mask"):
        eval_metrics.update(evaluate(
            model,
            eval_loader,
            device,
            args.max_eval_batches,
            mode=mode,
            corrupt_rate=args.corrupt_rate,
            span_mask_prob=args.span_mask_prob,
            future_min_prefix_ratio=args.future_min_prefix_ratio,
            future_max_prefix_ratio=args.future_max_prefix_ratio,
            prediction_target=args.prediction_target,
            hybrid_existing_diffusion=args.hybrid_existing_diffusion,
        ))
    result = {
        "model": "v3_commit_controller_small",
        "prediction_target": args.prediction_target,
        "hybrid_existing_diffusion": args.hybrid_existing_diffusion,
        "decoder_input": args.decoder_input,
        "controller_memory_mode": args.controller_memory_mode,
        "update_cell": args.update_cell,
        "commit_stride": args.commit_stride,
        "steps": args.max_steps,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "last": last_row,
        **eval_metrics,
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint(args.max_steps, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal FLUED-v3 commit controller")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=3000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--controller-hidden", type=int, default=256)
    parser.add_argument(
        "--decoder-input",
        choices=["hidden_active_memory", "active_memory", "gated_active_memory", "active", "memory"],
        default="active_memory",
    )
    parser.add_argument("--controller-memory-mode", choices=["raw", "features"], default="raw")
    parser.add_argument("--update-cell", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--commit-stride", type=int, default=1)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--prediction-target", choices=["current", "next_byte"], default="next_byte")
    parser.add_argument("--hybrid-existing-diffusion", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--denoise-prob", type=float, default=0.5)
    parser.add_argument("--denoise-steps", type=int, default=1500)
    parser.add_argument("--corrupt-rate", type=float, default=0.15)
    parser.add_argument("--span-mask-prob", type=float, default=0.7)
    parser.add_argument("--span-min", type=int, default=1)
    parser.add_argument("--span-max", type=int, default=8)
    parser.add_argument("--future-mask-prob", type=float, default=0.20)
    parser.add_argument("--future-min-prefix-ratio", type=float, default=0.35)
    parser.add_argument("--future-max-prefix-ratio", type=float, default=0.75)
    parser.add_argument("--future-loss-weight", type=float, default=0.10)
    parser.add_argument("--rate-target", type=float, default=0.35)
    parser.add_argument("--rate-lambda-init", type=float, default=0.05)
    parser.add_argument("--rate-lambda-eta", type=float, default=0.002)
    parser.add_argument("--rate-lambda-max", type=float, default=2.0)
    parser.add_argument("--commit-spread-weight", type=float, default=0.10)
    parser.add_argument("--commit-min-std", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--resume-training-state", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run(args)


if __name__ == "__main__":
    main()
