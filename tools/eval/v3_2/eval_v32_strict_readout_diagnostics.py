"""Readout predictability diagnostics for strict FLUED-v3.2 masked-source task.

The clean-source readout computed here is an offline oracle target only.  It is
not a valid backbone input, training input, or deployable path for the strict
masked-source task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID, STUB_CORPUS, ByteReconstructionDataset, StreamingReconstructionDataset  # noqa: E402
from tools.analysis.v3_0.train_v3_commit_controller_small import _load_texts  # noqa: E402
from tools.analysis.v3_2.train_v32_min_backbone import decode_readout, load_frozen_codec  # noqa: E402
from tools.analysis.v3_2.train_v32_strict_masked_backbone import StrictMaskedCollator  # noqa: E402


def _mean(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {key: sum(float(row[key]) for row in rows if key in row) / max(sum(1 for row in rows if key in row), 1) for key in keys}


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def _make_loader(args: argparse.Namespace) -> DataLoader:
    collate = StrictMaskedCollator(args.min_span, args.max_span, args.max_units, args.mask_prob, args.mask_span_min, args.mask_span_max)
    if args.streaming_eval:
        dataset = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=max(args.batch_size * args.max_eval_batches, 1024),
            seed=args.seed + 9999,
        )
    else:
        texts = _load_texts(args.data_path, args.eval_max_lines) if args.data_path else STUB_CORPUS * 32
        dataset = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)


@torch.no_grad()
def _codec_readout(codec, src, valid, seg_ids, seg_mask, amp: bool):
    with torch.amp.autocast(device_type=src.device.type, dtype=torch.bfloat16, enabled=amp and src.device.type == "cuda"):
        byte_logits, length_logits, metrics = codec(src, valid, seg_ids, seg_mask)
    return byte_logits, length_logits, metrics


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    codec, codec_args, codec_path = load_frozen_codec(args.codec_ckpt, device)
    loader = _make_loader(args)
    rows: List[Dict[str, float]] = []
    non_blocking = device.type == "cuda"
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        clean_src, masked_src, valid, seg_ids, seg_mask, targets, loss_mask, lengths, unit_mask = tuple(
            x.to(device, non_blocking=non_blocking) for x in batch
        )
        masked_byte_logits, masked_len_logits, masked_metrics = _codec_readout(codec, masked_src, valid, seg_ids, seg_mask, args.amp)
        _clean_byte_logits, _clean_len_logits, clean_metrics = _codec_readout(codec, clean_src, valid, seg_ids, seg_mask, args.amp)
        masked_readout = masked_metrics["readout"].float()
        clean_readout = clean_metrics["readout"].float()
        l2 = (masked_readout - clean_readout).pow(2).mean(dim=-1).sqrt()
        cos = F.cosine_similarity(masked_readout, clean_readout, dim=-1)
        pred = masked_byte_logits.argmax(dim=-1)
        slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
        keep_mask = slot_mask & (~loss_mask)
        len_target = (lengths.clamp(min=1, max=codec.max_span) - 1).clamp(min=0)
        len_pred = masked_len_logits.argmax(dim=-1)
        rows.append(
            {
                "direct_mask_byte_acc": _safe_acc(pred, targets, loss_mask),
                "direct_keep_byte_acc": _safe_acc(pred, targets, keep_mask),
                "direct_mask_length_acc": _safe_acc(len_pred, len_target, unit_mask),
                "masked_unit_readout_l2": float(l2[unit_mask].mean().item()) if unit_mask.any() else 0.0,
                "keep_unit_readout_l2": float(l2[seg_mask & (~unit_mask)].mean().item()) if (seg_mask & (~unit_mask)).any() else 0.0,
                "masked_unit_readout_cos": float(cos[unit_mask].mean().item()) if unit_mask.any() else 0.0,
                "keep_unit_readout_cos": float(cos[seg_mask & (~unit_mask)].mean().item()) if (seg_mask & (~unit_mask)).any() else 0.0,
                "memory_context_norm": float(masked_metrics.get("memory_context_norm", torch.zeros((), device=device)).float().item()),
                "retrieval_entropy": float(masked_metrics.get("retrieval_entropy", torch.zeros((), device=device)).float().item()),
                "masked_bytes": float(loss_mask.sum().item()),
                "active_units": float(seg_mask.sum().item()),
                "masked_units": float(unit_mask.sum().item()),
            }
        )
    summary = _mean(rows)
    report = {
        "codec_ckpt": str(codec_path),
        "codec_memory_enabled": bool(codec_args.get("memory_slots_per_chunk", 0)),
        "codec_memory_retrieval_mode": (
            str(codec_args.get("memory_retrieval_mode", "topk"))
            if int(codec_args.get("memory_slots_per_chunk", 0) or 0) > 0
            else "none"
        ),
        "codec_pool_mode": str(codec_args.get("pool_mode", "unknown")),
        "summary": summary,
    }
    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict masked-source readout diagnostics")
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--out-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int, default=32)
    parser.add_argument("--eval-max-lines", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=64)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
