"""Train FLUED-v3.2 codec with strict masked-source denoising.

This is the v3.2.1 repair path after strict Stage 4 showed that the backbone
was not the main bottleneck.  The task masks raw byte/span positions before
FLUED sees the sample:

  clean bytes -> byte/span mask -> masked bytes
  masked bytes -> FLUED segmentation + encoder + memory-conditioned readout
  readout -> FLUED decoder -> clean bytes at the originally masked positions

Mask sampling is byte/span-level, not segment-level.  Clean segmentation,
clean readout, and clean memory are never used as inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID, MASK_ID, STUB_CORPUS, ByteReconstructionDataset, StreamingReconstructionDataset  # noqa: E402
from tools.analysis.train_v3_commit_controller_small import _append_jsonl, _cosine_with_warmup, _load_texts  # noqa: E402
from tools.analysis.train_v32_language_codec_2m import (  # noqa: E402
    V32LanguageCodec2M,
    build_segments,
    complete_utf8_edge_valid,
    weak_boundary_starts,
)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / max(len(vals), 1)


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def make_byte_mask(valid: torch.Tensor, mask_prob: float, span_min: int, span_max: int) -> torch.Tensor:
    """Sample byte/span masks without using segmentation."""

    bsz, seq_len = valid.shape
    mask = torch.zeros_like(valid, dtype=torch.bool)
    span_min = max(1, int(span_min))
    span_max = max(span_min, int(span_max))
    for b in range(bsz):
        positions = valid[b].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        target = max(1, int(round(float(mask_prob) * int(positions.numel()))))
        attempts = 0
        while int(mask[b].sum().item()) < target and attempts < target * 8:
            attempts += 1
            start_pos = positions[torch.randint(positions.numel(), (1,), device=valid.device)].item()
            span = int(torch.randint(span_min, span_max + 1, (1,), device=valid.device).item())
            end_pos = min(seq_len, int(start_pos) + span)
            mask[b, int(start_pos):end_pos] |= valid[b, int(start_pos):end_pos]
        if not bool(mask[b].any()):
            mask[b, positions[torch.randint(positions.numel(), (1,), device=valid.device)]] = True
    return mask & valid


def targets_from_masked_segments(
    clean_src: torch.Tensor,
    byte_mask: torch.Tensor,
    seg_ids: torch.Tensor,
    seg_mask: torch.Tensor,
    max_span: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack clean byte targets into masked-source internal segment slots."""

    bsz, seq_len = clean_src.shape
    max_units = seg_mask.size(1)
    targets = torch.full((bsz, max_units, max_span), PAD_ID, dtype=torch.long, device=clean_src.device)
    loss_mask = torch.zeros((bsz, max_units, max_span), dtype=torch.bool, device=clean_src.device)
    lengths = torch.zeros((bsz, max_units), dtype=torch.long, device=clean_src.device)
    cursor = torch.zeros((bsz, max_units), dtype=torch.long, device=clean_src.device)
    valid = seg_ids.ge(0)
    for b in range(bsz):
        for t in range(seq_len):
            if not bool(valid[b, t]):
                continue
            unit = int(seg_ids[b, t].item())
            if unit < 0 or unit >= max_units:
                continue
            slot = int(cursor[b, unit].item())
            if slot >= max_span:
                continue
            targets[b, unit, slot] = clean_src[b, t]
            loss_mask[b, unit, slot] = bool(byte_mask[b, t])
            cursor[b, unit] += 1
            lengths[b, unit] = max(int(lengths[b, unit].item()), slot + 1)
    return targets, loss_mask, lengths


class MaskedCodecCollator:
    """Build strict masked-source codec batches in DataLoader workers."""

    def __init__(
        self,
        min_span: int,
        max_span: int,
        max_units: int,
        mask_prob: float,
        mask_span_min: int,
        mask_span_max: int,
    ) -> None:
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        self.max_units = int(max_units)
        self.mask_prob = float(mask_prob)
        self.mask_span_min = int(mask_span_min)
        self.mask_span_max = int(mask_span_max)

    def __call__(self, batch):
        clean_src = torch.stack([item[0] for item in batch], dim=0).long()
        clean_valid = complete_utf8_edge_valid(clean_src, clean_src.ne(PAD_ID))
        byte_mask = make_byte_mask(clean_valid, self.mask_prob, self.mask_span_min, self.mask_span_max)
        masked_src = clean_src.masked_fill(byte_mask, MASK_ID)
        valid = complete_utf8_edge_valid(masked_src, masked_src.ne(PAD_ID))
        starts = weak_boundary_starts(masked_src, valid, self.min_span, self.max_span)
        max_units = min(self.max_units, clean_src.size(1))
        seg_ids, masked_targets, masked_lengths, seg_mask = build_segments(masked_src, valid, starts, max_units, self.max_span)
        clean_targets, loss_mask, clean_lengths = targets_from_masked_segments(clean_src, byte_mask, seg_ids, seg_mask, self.max_span)
        lengths = torch.where(clean_lengths.gt(0), clean_lengths, masked_lengths)
        unit_mask = loss_mask.any(dim=-1) & seg_mask
        return clean_src, masked_src, valid, starts, seg_ids, clean_targets, lengths, seg_mask, loss_mask, unit_mask, masked_targets


def move_masked_batch(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    return tuple(x.to(device, non_blocking=non_blocking) for x in batch)


def make_dataloaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    collate_fn = MaskedCodecCollator(
        args.min_span,
        args.max_span,
        args.max_units,
        args.mask_prob,
        args.mask_span_min,
        args.mask_span_max,
    )
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if args.streaming_train:
        train_ds = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=args.stream_samples_per_worker,
            seed=args.seed,
        )
        if args.streaming_eval and args.data_path:
            eval_ds = StreamingReconstructionDataset(
                file_path=args.data_path,
                seq_len=args.seq_len,
                samples_per_worker=max(args.batch_size * args.max_eval_batches, 1024),
                seed=args.seed + 9999,
            )
        else:
            eval_texts = _load_texts(args.data_path, args.eval_max_lines) if args.data_path else STUB_CORPUS
            eval_ds = ByteReconstructionDataset(texts=eval_texts, seq_len=args.seq_len, stride=args.stride)
        shuffle = False
    else:
        texts = _load_texts(args.data_path, args.max_lines) if args.data_path else STUB_CORPUS * 64
        ds = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
        train_ds = ds
        eval_ds = ds
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return train_loader, eval_loader


def masked_codec_step(model: V32LanguageCodec2M, batch, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    _clean_src, masked_src, valid, starts, seg_ids, targets, lengths, seg_mask, loss_mask, unit_mask, masked_targets = move_masked_batch(batch, device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        byte_logits, length_logits, metrics = model(masked_src, valid, seg_ids, seg_mask)
        ce = F.cross_entropy(
            byte_logits.float().reshape(-1, byte_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(targets)
        slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
        keep_mask = slot_mask & (~loss_mask)
        masked_recon_loss = ce[loss_mask].mean() if loss_mask.any() else ce.new_zeros(())
        keep_recon_loss = ce[keep_mask].mean() if keep_mask.any() else ce.new_zeros(())
        recon_loss = masked_recon_loss + args.keep_recon_weight * keep_recon_loss
        length_target = (lengths.clamp(min=1, max=args.max_span) - 1).clamp(min=0)
        length_loss = F.cross_entropy(length_logits[seg_mask].float(), length_target[seg_mask]) if seg_mask.any() else recon_loss.new_zeros(())
        boundary_loss = F.binary_cross_entropy_with_logits(metrics["boundary_logits"][valid].float(), starts[valid].float()) if valid.any() else recon_loss.new_zeros(())
        loss = recon_loss + args.length_loss_weight * length_loss + args.boundary_loss_weight * boundary_loss

    with torch.no_grad():
        pred = byte_logits.argmax(dim=-1)
        bpred = metrics["boundary_logits"].sigmoid().ge(0.5)
        units = seg_mask.float().sum().item()
        bytes_n = valid.float().sum().item()
        memory_slots_per_byte = float(units * args.memory_slots_per_chunk / max(bytes_n, 1.0))
        metric_row = {
            "loss": float(loss.item()),
            "masked_recon_loss": float(masked_recon_loss.item()),
            "keep_recon_loss": float(keep_recon_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "length_loss": float(length_loss.item()),
            "boundary_loss": float(boundary_loss.item()),
            "masked_recon_acc": _safe_acc(pred, targets, loss_mask),
            "keep_recon_acc": _safe_acc(pred, targets, keep_mask),
            "length_acc": _safe_acc(length_logits.argmax(dim=-1), length_target, seg_mask),
            "masked_length_acc": _safe_acc(length_logits.argmax(dim=-1), length_target, unit_mask),
            "boundary_acc": _safe_acc(bpred, starts, valid),
            "masked_bytes": float(loss_mask.sum().item()),
            "valid_bytes": float(slot_mask.sum().item()),
            "masked_units": float(unit_mask.sum().item()),
            "active_units": float(seg_mask.sum().item()),
            "masked_byte_fraction": float(loss_mask.sum().item() / max(slot_mask.sum().item(), 1)),
            "units_per_byte": float(units / max(bytes_n, 1.0)),
            "readout_units_per_byte": float(units / max(bytes_n, 1.0)),
            "memory_slots_per_byte": memory_slots_per_byte,
            "retrieval_entropy": float(metrics.get("retrieval_entropy", loss.new_zeros(())).float().item()),
            "retrieval_valid_frac": float(metrics.get("retrieval_valid_frac", loss.new_zeros(())).float().item()),
            "retrieval_no_past_frac": float(metrics.get("retrieval_no_past_frac", loss.new_zeros(())).float().item()),
            "retrieval_active_units": float(metrics.get("retrieval_active_units", loss.new_zeros(())).float().item()),
            "retrieval_past_only_violation_count": float(metrics.get("retrieval_past_only_violation_count", loss.new_zeros(())).float().item()),
            "retrieval_max_selected_unit_delta": float(metrics.get("retrieval_max_selected_unit_delta", loss.new_full((), -1.0)).float().item()),
            "memory_context_norm": float(metrics.get("memory_context_norm", loss.new_zeros(())).float().item()),
            "memory_slot_norm": float(metrics.get("memory_slot_norm", loss.new_zeros(())).float().item()),
            "summary_norm": float(metrics.get("summary_norm", loss.new_zeros(())).float().item()),
            "first_unit_memory_norm": float(metrics.get("first_unit_memory_norm", loss.new_zeros(())).float().item()),
        }
    return loss, metric_row


@torch.no_grad()
def evaluate(model: V32LanguageCodec2M, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    # PyTorch's TransformerEncoder can hit a CPU eval/no_grad fast path that
    # returns NaNs for this prototype stack. Dropout is 0 in the fair runs, so
    # train-mode/no-grad gives the same deterministic layers while avoiding
    # that backend path.
    was_training = model.training
    model.train()
    rows: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        _loss, metrics = masked_codec_step(model, batch, args, device)
        rows.append(metrics)
    if not was_training:
        model.eval()
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


def run(args: argparse.Namespace) -> Dict[str, float]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, eval_loader = make_dataloaders(args, device)

    model = V32LanguageCodec2M(
        d_model=args.d_model,
        hidden=args.hidden,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        ffn_dim=args.ffn_dim,
        max_span=args.max_span,
        refine_steps=args.refine_steps,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
        memory_slots_per_chunk=args.memory_slots_per_chunk,
        memory_topk=args.memory_topk,
        memory_retrieval_mode=args.memory_retrieval_mode,
        causal_byte_encoder=args.causal_byte_encoder,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"

    step = 0
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = 0
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss, metrics = masked_codec_step(model, batch, args, device)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                now = time.perf_counter()
                elapsed = now - train_start_time
                recent_elapsed = now - last_log_time
                recent_steps = max(step - last_log_step, 1)
                total_steps_per_sec = (step + 1) / max(elapsed, 1e-9)
                recent_steps_per_sec = recent_steps / max(recent_elapsed, 1e-9)
                max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
                row = {
                    "step": step,
                    "task": "strict_masked_source_codec",
                    "memory_enabled": bool(args.memory_slots_per_chunk > 0),
                    "memory_path": "past_slot_retrieval" if args.memory_slots_per_chunk > 0 else "none",
                    "memory_retrieval_mode": str(args.memory_retrieval_mode),
                    "byte_encoder_causal": bool(args.causal_byte_encoder),
                    "memory_read_scope": "past_only",
                    "boundary_reads_memory": False,
                    "interpreter_reads_memory": bool(args.memory_slots_per_chunk > 0),
                    "decoder_reads_memory": False,
                    "mask_granularity": "byte_span",
                    "clean_segment_used": False,
                    "clean_readout_used": False,
                    **metrics,
                    "grad": float(grad.item() if hasattr(grad, "item") else grad),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "elapsed_sec": float(elapsed),
                    "steps_per_sec": float(total_steps_per_sec),
                    "recent_steps_per_sec": float(recent_steps_per_sec),
                    "samples_per_sec": float(total_steps_per_sec * args.batch_size),
                    "recent_samples_per_sec": float(recent_steps_per_sec * args.batch_size),
                    "bytes_per_sec": float(total_steps_per_sec * args.batch_size * args.seq_len),
                    "max_memory_allocated_mb": float(max_mem_mb),
                }
                _append_jsonl(log_path, row)
                print(
                    f"step={step} loss={row['loss']:.4f} mask={row['masked_recon_acc']:.3f} "
                    f"keep={row['keep_recon_acc']:.3f} len={row['length_acc']:.3f} "
                    f"boundary={row['boundary_acc']:.3f} u/b={row['units_per_byte']:.3f} "
                    f"samples/s={row['recent_samples_per_sec']:.0f} mem={row['max_memory_allocated_mb']:.0f}MB",
                    flush=True,
                )
                last_log_time = now
                last_log_step = step

            if step > 0 and step % args.ckpt_every == 0:
                payload = {"model": model.state_dict(), "args": vars(args), "step": step, "params": params}
                torch.save(payload, out_dir / f"step{step}.pt")
                torch.save(payload, out_dir / "latest.pt")
            step += 1

    eval_stats = evaluate(model, eval_loader, args, device)
    elapsed_sec = time.perf_counter() - train_start_time
    result = {
        "params": params,
        "steps": step,
        "model_version": "v3.2.1-strict-masked-source-codec",
        "task": "strict_masked_source_codec",
        "mask_granularity": "byte_span",
        "clean_segment_used": False,
        "clean_readout_used": False,
        "memory_enabled": bool(args.memory_slots_per_chunk > 0),
        "memory_path": "past_slot_retrieval" if args.memory_slots_per_chunk > 0 else "none",
        "memory_retrieval_mode": str(args.memory_retrieval_mode),
        "byte_encoder_causal": bool(args.causal_byte_encoder),
        "memory_read_scope": "past_only",
        "boundary_reads_memory": False,
        "interpreter_reads_memory": bool(args.memory_slots_per_chunk > 0),
        "decoder_reads_memory": False,
        "eval_mode": "streaming" if args.streaming_train and args.streaming_eval else "fixed_text",
        "train_elapsed_sec": elapsed_sec,
        "train_steps_per_sec": step / max(elapsed_sec, 1e-9),
        "train_samples_per_sec": (step * args.batch_size) / max(elapsed_sec, 1e-9),
        "train_bytes_per_sec": (step * args.batch_size * args.seq_len) / max(elapsed_sec, 1e-9),
        **{f"eval_{k}": v for k, v in eval_stats.items()},
    }
    if device.type == "cuda":
        result["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step, "summary": result}, out_dir / "latest.pt")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FLUED-v3.2.1 strict masked-source codec")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=3000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--refine-steps", type=int, default=1)
    parser.add_argument("--pool-mode", choices=["mean", "mean_first_last"], default="mean_first_last")
    parser.add_argument("--memory-slots-per-chunk", type=int, default=2)
    parser.add_argument("--memory-topk", type=int, default=4)
    parser.add_argument("--memory-retrieval-mode", choices=["topk", "random"], default="topk")
    parser.set_defaults(causal_byte_encoder=True)
    parser.add_argument("--causal-byte-encoder", dest="causal_byte_encoder", action="store_true")
    parser.add_argument("--no-causal-byte-encoder", dest="causal_byte_encoder", action="store_false")
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=128)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--keep-recon-weight", type=float, default=0.25)
    parser.add_argument("--length-loss-weight", type=float, default=0.10)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
