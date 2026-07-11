"""Strict masked-source backbone for FLUED-v3.2.

This is the leakage-safe Stage 4 task.  Masking is byte/span-level, not
segment-level:

  clean bytes -> sample byte/span mask
  masked bytes -> FLUED-v3.2 segmentation + encoder -> readout latents
  readout latents -> small backbone
  predicted readout -> frozen FLUED decoder -> original masked bytes

The backbone never sees clean readout, clean segmentation, FLUED summary, or
FLUED memory.  FLUED segmentation is recomputed from masked bytes only, and is
kept as an internal execution detail.  Clean bytes are used only as loss
targets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_VOCAB_SIZE, MASK_ID, PAD_ID, STUB_CORPUS, ByteReconstructionDataset, StreamingReconstructionDataset  # noqa: E402
from tools.analysis.v3_0.train_v3_commit_controller_small import _append_jsonl, _cosine_with_warmup, _load_texts  # noqa: E402
from tools.analysis.v3_2.train_v32_language_codec_2m import (  # noqa: E402
    build_segments,
    complete_utf8_edge_valid,
    weak_boundary_starts,
)
from tools.analysis.v3_2.train_v32_min_backbone import (  # noqa: E402
    ByteInfillBackbone,
    LatentInfillBackbone,
    decode_readout,
    load_frozen_codec,
)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / max(len(vals), 1)


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


class StrictMaskedCollator:
    """Build masked-source segmentation in DataLoader workers."""

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
        masked_src, valid, seg_ids, seg_mask, targets, loss_mask, lengths, unit_mask = prepare_masked_codec_inputs(clean_src, self)
        return clean_src, masked_src, valid, seg_ids, seg_mask, targets, loss_mask, lengths, unit_mask


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    collate = StrictMaskedCollator(
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
        eval_ds = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=max(args.batch_size * args.max_eval_batches, 1024),
            seed=args.seed + 9999,
        ) if args.streaming_eval and args.data_path else ByteReconstructionDataset(
            texts=_load_texts(args.data_path, args.eval_max_lines) if args.data_path else STUB_CORPUS,
            seq_len=args.seq_len,
            stride=args.stride,
        )
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
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate,
        **loader_kwargs,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    return train_loader, eval_loader


def make_byte_mask(valid: torch.Tensor, mask_prob: float, span_min: int, span_max: int) -> torch.Tensor:
    """Sample byte/span masks without using clean segmentation."""

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
    """Build clean byte targets aligned to masked-source segment ids."""

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


def prepare_masked_codec_inputs(
    clean_src: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_valid = complete_utf8_edge_valid(clean_src, clean_src.ne(PAD_ID))
    byte_mask = make_byte_mask(clean_valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    masked_src = clean_src.masked_fill(byte_mask, MASK_ID)
    masked_valid = complete_utf8_edge_valid(masked_src, masked_src.ne(PAD_ID))
    starts = weak_boundary_starts(masked_src, masked_valid, args.min_span, args.max_span)
    max_units = min(args.max_units, clean_src.size(1))
    seg_ids, _masked_targets, masked_lengths, seg_mask = build_segments(masked_src, masked_valid, starts, max_units, args.max_span)
    clean_targets, loss_mask, clean_lengths = targets_from_masked_segments(clean_src, byte_mask, seg_ids, seg_mask, args.max_span)
    lengths = torch.where(clean_lengths.gt(0), clean_lengths, masked_lengths)
    unit_mask = loss_mask.any(dim=-1) & seg_mask
    return masked_src, masked_valid, seg_ids, seg_mask, clean_targets, loss_mask, lengths, unit_mask


@torch.no_grad()
def encode_masked_readout(codec, masked_src: torch.Tensor, valid: torch.Tensor, seg_ids: torch.Tensor, seg_mask: torch.Tensor, amp: bool) -> torch.Tensor:
    with torch.amp.autocast(device_type=masked_src.device.type, dtype=torch.bfloat16, enabled=amp and masked_src.device.type == "cuda"):
        _byte_logits, _length_logits, metrics = codec(masked_src, valid, seg_ids, seg_mask)
    return metrics["readout"].detach()


def _move_prepared_batch(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    return tuple(x.to(device, non_blocking=non_blocking) for x in batch)


def latent_step(backbone: LatentInfillBackbone, codec, batch, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    _clean_src, masked_src, valid, seg_ids, seg_mask, targets, loss_mask, lengths, unit_mask = _move_prepared_batch(batch, device)
    readout = encode_masked_readout(codec, masked_src, valid, seg_ids, seg_mask, args.amp)
    zero_unit_mask = torch.zeros_like(seg_mask)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        pred_readout = backbone(readout, seg_mask, zero_unit_mask)
        mixed = torch.where(unit_mask.unsqueeze(-1), pred_readout, readout)
        byte_logits, length_logits = decode_readout(codec, mixed, seg_mask)
        ce = F.cross_entropy(
            byte_logits.float().reshape(-1, byte_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(targets)
        byte_loss = ce[loss_mask].mean() if loss_mask.any() else ce.new_zeros(())
        if args.length_loss_weight > 0 and unit_mask.any():
            length_target = (lengths.clamp(min=1, max=codec.max_span) - 1).clamp(min=0)
            length_loss = F.cross_entropy(length_logits[unit_mask].float(), length_target[unit_mask])
            loss = byte_loss + args.length_loss_weight * length_loss
        else:
            length_loss = byte_loss.new_zeros(())
            loss = byte_loss

    with torch.no_grad():
        pred = byte_logits.argmax(dim=-1)
        slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
        keep_mask = slot_mask & (~loss_mask)
        length_target = (lengths.clamp(min=1, max=codec.max_span) - 1).clamp(min=0)
        len_pred = length_logits.argmax(dim=-1)
        metrics = {
            "loss": float(loss.item()),
            "byte_loss": float(byte_loss.item()),
            "length_loss": float(length_loss.item()),
            "mask_byte_acc": _safe_acc(pred, targets, loss_mask),
            "keep_byte_acc": _safe_acc(pred, targets, keep_mask),
            "mask_length_acc": _safe_acc(len_pred, length_target, unit_mask),
            "keep_length_acc": _safe_acc(len_pred, length_target, seg_mask & (~unit_mask)),
            "masked_bytes": float(loss_mask.sum().item()),
            "valid_bytes": float(slot_mask.sum().item()),
            "masked_units": float(unit_mask.sum().item()),
            "active_units": float(seg_mask.sum().item()),
            "masked_byte_fraction": float(loss_mask.sum().item() / max(slot_mask.sum().item(), 1)),
            "units_per_byte": float(seg_mask.sum().item() / max(valid.sum().item(), 1)),
        }
    return loss, metrics


def byte_step(backbone: ByteInfillBackbone, batch, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    clean_src, masked_src, valid, _seg_ids, _seg_mask, _targets, loss_mask, _lengths, _unit_mask = _move_prepared_batch(batch, device)
    # Reconstruct byte-level mask from masked source directly. The loss_mask is
    # segment-slot aligned and not position aligned.
    byte_mask = masked_src.eq(MASK_ID) & valid
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        logits = backbone(masked_src, valid, torch.zeros_like(valid))
        per_pos = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            clean_src.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(clean_src)
        loss = per_pos[byte_mask].mean() if byte_mask.any() else per_pos.new_zeros(())
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        keep_mask = valid & (~byte_mask)
        metrics = {
            "loss": float(loss.item()),
            "mask_byte_acc": _safe_acc(pred, clean_src, byte_mask),
            "keep_byte_acc_model": _safe_acc(pred, clean_src, keep_mask),
            "keep_byte_acc_copy": 1.0,
            "masked_bytes": float(byte_mask.sum().item()),
            "valid_bytes": float(valid.sum().item()),
            "masked_byte_fraction": float(byte_mask.sum().item() / max(valid.sum().item(), 1)),
        }
    return loss, metrics


def average_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


@torch.no_grad()
def evaluate(backbone: nn.Module, codec, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    backbone.eval()
    rows: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        if args.mode == "latent":
            assert codec is not None
            _loss, metrics = latent_step(backbone, codec, batch, args, device)
        else:
            _loss, metrics = byte_step(backbone, batch, args, device)
        rows.append(metrics)
    backbone.train()
    return average_metrics(rows)


def run(args: argparse.Namespace) -> Dict[str, float]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = None
    codec_args: Dict[str, object] = {}
    codec_path = ""
    if args.mode == "latent":
        codec, codec_args, resolved = load_frozen_codec(args.codec_ckpt, device)
        codec_path = str(resolved)
        args.max_span = int(codec_args.get("max_span", args.max_span))

    train_loader, eval_loader = make_dataloaders(args)
    if args.mode == "latent":
        assert codec is not None
        latent_dim = int(codec_args.get("hidden", args.hidden))
        backbone = LatentInfillBackbone(latent_dim, args.hidden, args.layers, args.nhead, args.ffn_dim, args.max_units, args.dropout).to(device)
    else:
        backbone = ByteInfillBackbone(args.hidden, args.layers, args.nhead, args.ffn_dim, args.seq_len, args.dropout, BYTE_VOCAB_SIZE).to(device)

    params = sum(p.numel() for p in backbone.parameters())
    opt = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"
    start = time.perf_counter()

    step = 0
    backbone.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            opt.zero_grad(set_to_none=True)
            if args.mode == "latent":
                assert codec is not None
                loss, metrics = latent_step(backbone, codec, batch, args, device)
            else:
                loss, metrics = byte_step(backbone, batch, args, device)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(backbone.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if step % args.log_every == 0:
                row = {"step": step, "mode": args.mode, "lr": float(opt.param_groups[0]["lr"]), "grad": float(grad.item() if hasattr(grad, "item") else grad), **metrics}
                _append_jsonl(log_path, row)
                print(
                    f"step={step} mode={args.mode} loss={row['loss']:.4f} "
                    f"mask_acc={row.get('mask_byte_acc', 0.0):.3f} "
                    f"keep_acc={row.get('keep_byte_acc', row.get('keep_byte_acc_model', 0.0)):.3f} "
                    f"mask_frac={row.get('masked_byte_fraction', 0.0):.3f}",
                    flush=True,
                )
            if step > 0 and step % args.ckpt_every == 0:
                payload = {"model": backbone.state_dict(), "args": vars(args), "step": step, "params": params}
                torch.save(payload, out_dir / f"step{step}.pt")
                torch.save(payload, out_dir / "latest.pt")
            step += 1

    eval_stats = evaluate(backbone, codec, eval_loader, args, device)
    elapsed = time.perf_counter() - start
    result = {
        "mode": args.mode,
        "task": "strict_masked_source",
        "params": params,
        "steps": step,
        "codec_ckpt": codec_path,
        "codec_memory_enabled": bool(codec_args.get("memory_slots_per_chunk", 0)) if codec_args else False,
        "codec_memory_retrieval_mode": (
            str(codec_args.get("memory_retrieval_mode", "topk"))
            if codec_args and int(codec_args.get("memory_slots_per_chunk", 0) or 0) > 0
            else "none"
        ),
        "codec_pool_mode": str(codec_args.get("pool_mode", "none")) if codec_args else "none",
        "eval_mode": "streaming" if args.streaming_train and args.streaming_eval else "fixed_text",
        "train_elapsed_sec": elapsed,
        "train_steps_per_sec": step / max(elapsed, 1e-9),
        **{f"eval_{k}": v for k, v in eval_stats.items()},
    }
    torch.save({"model": backbone.state_dict(), "args": vars(args), "step": step, "summary": result}, out_dir / "latest.pt")
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict masked-source FLUED-v3.2 backbone")
    parser.add_argument("--mode", choices=["latent", "byte"], required=True)
    parser.add_argument("--codec-ckpt", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=100000)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--eval-max-lines", type=int, default=5000)
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
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=64)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "latent" and not args.codec_ckpt:
        parser.error("--mode latent requires --codec-ckpt")
    run(args)


if __name__ == "__main__":
    main()
