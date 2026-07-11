"""Decoder diagnostics for the FLUED-v3.1 language codec prototype.

This evaluator checks whether the explicit length head and slot decoder are
learning the byte span reconstruction task, rather than only exploiting a
fixed-length shortcut.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID, STUB_CORPUS, ByteReconstructionDataset, StreamingReconstructionDataset
from tools.analysis.v3_1.train_v31_language_codec_2m import (
    CodecCollator,
    V31LanguageCodec2M,
    move_codec_batch,
)
from tools.analysis.v3_0.train_v3_commit_controller_small import _load_texts


def resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "latest.pt"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def model_kwargs(saved_args: Dict[str, object]) -> Dict[str, object]:
    keys = ["d_model", "hidden", "nhead", "encoder_layers", "ffn_dim", "max_span", "refine_steps", "dropout", "pool_mode"]
    return {k: saved_args[k] for k in keys if k in saved_args}


def token_to_raw_byte(token_id: int) -> int | None:
    if BYTE_OFFSET <= token_id < MASK_ID:
        return token_id - BYTE_OFFSET
    return None


def decode_pred_span(tokens: Iterable[int], length: int) -> Tuple[bytes, bool, int]:
    raw: List[int] = []
    bad_ids = 0
    for token in list(tokens)[: max(0, int(length))]:
        b = token_to_raw_byte(int(token))
        if b is None:
            bad_ids += 1
        else:
            raw.append(b)
    data = bytes(raw)
    try:
        data.decode("utf-8")
        valid_utf8 = bad_ids == 0
    except UnicodeDecodeError:
        valid_utf8 = False
    return data, valid_utf8, bad_ids


def make_loader(args: argparse.Namespace, saved_args: Dict[str, object]) -> DataLoader:
    seq_len = int(args.seq_len or saved_args.get("seq_len", 128))
    stride = int(args.stride or saved_args.get("stride", max(1, seq_len // 2)))
    batch_size = int(args.batch_size or saved_args.get("batch_size", 32))
    max_span = int(saved_args.get("max_span", 16))
    max_units = int(saved_args.get("max_units", seq_len))
    min_span = int(saved_args.get("min_span", 2))

    if args.streaming_eval:
        if not args.data_path:
            raise ValueError("--streaming-eval requires --data-path")
        ds = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=seq_len,
            samples_per_worker=max(batch_size * args.max_batches, 1024),
            seed=args.seed,
        )
    elif args.data_path:
        texts = _load_texts(args.data_path, args.eval_max_lines)
        ds = ByteReconstructionDataset(texts=texts, seq_len=seq_len, stride=stride)
    else:
        ds = ByteReconstructionDataset(texts=STUB_CORPUS * 64, seq_len=seq_len, stride=stride)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CodecCollator(min_span=min_span, max_span=max_span, max_units=max_units),
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    ckpt_path = resolve_checkpoint(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args: Dict[str, object] = dict(ckpt.get("args", {}))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = V31LanguageCodec2M(**model_kwargs(saved_args)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = make_loader(args, saved_args)
    max_span = int(saved_args.get("max_span", model.max_span))

    totals = Counter()
    length_buckets: Dict[int, Counter] = defaultdict(Counter)
    pred_len_dist = Counter()
    target_len_dist = Counter()
    invalid_pred_utf8 = 0
    invalid_target_utf8 = 0
    invalid_pred_ids = 0
    total_segments = 0
    long_correct = 0
    long_slots_total = 0
    loss_values: List[float] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        src, starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
        valid = src.ne(PAD_ID)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            byte_logits, length_logits, metrics = model(src, valid, seg_ids, seg_mask)
            recon_loss = F.cross_entropy(
                byte_logits.float().view(-1, byte_logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_ID,
            )
        loss_values.append(float(recon_loss.item()))

        pred = byte_logits.argmax(dim=-1)
        pred_len = length_logits.argmax(dim=-1) + 1
        target_len = lengths.clamp(min=1, max=max_span)
        active = seg_mask.nonzero(as_tuple=False)

        for bi, ui in active.tolist():
            tl = int(target_len[bi, ui].item())
            pl = int(pred_len[bi, ui].item())
            total_segments += 1
            pred_len_dist[pl] += 1
            target_len_dist[tl] += 1

            slot_mask = targets[bi, ui].ne(PAD_ID)
            correct_slots = int((pred[bi, ui][slot_mask] == targets[bi, ui][slot_mask]).sum().item())
            slot_total = int(slot_mask.sum().item())
            exact_len = int(pl == tl)
            exact_span = int(exact_len and correct_slots == slot_total)

            totals["slots_correct"] += correct_slots
            totals["slots_total"] += slot_total
            totals["length_correct"] += exact_len
            totals["exact_span_correct"] += exact_span
            length_buckets[tl]["segments"] += 1
            length_buckets[tl]["slots_correct"] += correct_slots
            length_buckets[tl]["slots_total"] += slot_total
            length_buckets[tl]["length_correct"] += exact_len
            length_buckets[tl]["exact_span_correct"] += exact_span

            if tl >= max(2, math.ceil(max_span * 0.75)):
                long_correct += correct_slots
                long_slots_total += slot_total

            _, pred_valid_utf8, bad_ids = decode_pred_span(pred[bi, ui].detach().cpu().tolist(), pl)
            _, target_valid_utf8, _ = decode_pred_span(targets[bi, ui].detach().cpu().tolist(), tl)
            invalid_pred_ids += bad_ids
            invalid_pred_utf8 += int(not pred_valid_utf8)
            invalid_target_utf8 += int(not target_valid_utf8)

    bucket_rows = []
    for length in sorted(length_buckets):
        c = length_buckets[length]
        bucket_rows.append(
            {
                "length": length,
                "segments": int(c["segments"]),
                "recon_acc": c["slots_correct"] / max(c["slots_total"], 1),
                "length_acc": c["length_correct"] / max(c["segments"], 1),
                "exact_span_acc": c["exact_span_correct"] / max(c["segments"], 1),
            }
        )

    return {
        "checkpoint": str(ckpt_path),
        "eval_mode": "streaming" if args.streaming_eval else "fixed_text",
        "batches": args.max_batches,
        "segments": total_segments,
        "recon_loss": sum(loss_values) / max(len(loss_values), 1),
        "recon_acc": totals["slots_correct"] / max(totals["slots_total"], 1),
        "length_acc": totals["length_correct"] / max(total_segments, 1),
        "exact_span_acc": totals["exact_span_correct"] / max(total_segments, 1),
        "long_span_recon_acc": long_correct / max(long_slots_total, 1),
        "invalid_pred_utf8_ratio": invalid_pred_utf8 / max(total_segments, 1),
        "invalid_target_utf8_ratio": invalid_target_utf8 / max(total_segments, 1),
        "invalid_pred_ids": invalid_pred_ids,
        "target_len_dist": dict(sorted(target_len_dist.items())),
        "pred_len_dist": dict(sorted(pred_len_dist.items())),
        "length_buckets": bucket_rows,
    }


def format_counter_table(title: str, values: Dict[int, int]) -> str:
    rows = [f"### {title}", "", "| length | count |", "| ---: | ---: |"]
    rows.extend(f"| {k} | {v} |" for k, v in sorted(values.items()))
    return "\n".join(rows)


def write_markdown(stats: Dict[str, object]) -> str:
    lines = [
        "# FLUED v3.1 Decoder Diagnostics",
        "",
        f"- checkpoint: `{stats['checkpoint']}`",
        f"- eval_mode: `{stats['eval_mode']}`",
        f"- batches: `{stats['batches']}`",
        f"- segments: `{stats['segments']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in [
        "recon_loss",
        "recon_acc",
        "length_acc",
        "exact_span_acc",
        "long_span_recon_acc",
        "invalid_pred_utf8_ratio",
        "invalid_target_utf8_ratio",
        "invalid_pred_ids",
    ]:
        value = stats[key]
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines += [
        "",
        format_counter_table("Target Length Distribution", stats["target_len_dist"]),
        "",
        format_counter_table("Predicted Length Distribution", stats["pred_len_dist"]),
        "",
        "## By Target Length",
        "",
        "| length | segments | recon_acc | length_acc | exact_span_acc |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats["length_buckets"]:
        lines.append(
            f"| {row['length']} | {row['segments']} | {row['recon_acc']:.6f} | "
            f"{row['length_acc']:.6f} | {row['exact_span_acc']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FLUED-v3.1 codec decoder length behavior")
    parser.add_argument("--ckpt", required=True, help="Path to latest.pt or a run directory")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--out-path", default="")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--eval-max-lines", type=int, default=5000)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    stats = evaluate(args)
    text = write_markdown(stats)
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(json.dumps({"out_path": str(out_path), "segments": stats["segments"]}, ensure_ascii=False))
    else:
        print(text)


if __name__ == "__main__":
    main()
