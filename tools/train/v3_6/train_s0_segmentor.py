"""FLUED v3.6 S0: standalone segmentor pretraining on teacher-labeled boundaries.

Data: filtered teacher labels (text + boundary byte offsets). Samples are packed
into fixed windows split only at labeled boundaries; per-byte BCE (positive =
segment start) trains byte encoder + segmentor blocks end to end. All other v3.6
components stay frozen out of scope.

Metrics on a held-out split: precision/recall/F1 at a rate-calibrated threshold,
cut rate, and segment-length distribution vs the labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID  # noqa: E402
from flued.v36 import FLUEDV36, V36Config  # noqa: E402


def load_labeled_samples(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"text": r["text"], "boundaries": r["boundaries_bytes"]})
    return rows


def pack_windows(samples: list[dict], window: int) -> list[dict]:
    windows = []
    cur_bytes = bytearray()
    cur_labels = []
    for sample in samples:
        raw = sample["text"].encode("utf-8")
        segs = []
        prev = 0
        for b in sample["boundaries"] + [len(raw)]:
            segs.append(raw[prev:b])
            prev = b
        for seg in segs:
            if len(seg) == 0:
                continue
            if len(seg) > window:
                for off in range(0, len(seg), window):
                    piece = seg[off : off + window]
                    if len(cur_bytes) + len(piece) > window:
                        if cur_bytes:
                            windows.append({"ids": bytes(cur_bytes), "labels": cur_labels})
                        cur_bytes, cur_labels = bytearray(), []
                    cur_bytes += piece
                    cur_labels += [1] + [0] * (len(piece) - 1)
                continue
            if len(cur_bytes) + len(seg) > window:
                windows.append({"ids": bytes(cur_bytes), "labels": cur_labels})
                cur_bytes, cur_labels = bytearray(), []
            cur_bytes += seg
            cur_labels += [1] + [0] * (len(seg) - 1)
    if cur_bytes:
        windows.append({"ids": bytes(cur_bytes), "labels": cur_labels})
    return windows


def to_batch(windows: list[dict], window: int, mask_prob: float, device, generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = len(windows)
    ids = torch.zeros(bsz, window, dtype=torch.long)
    labels = torch.zeros(bsz, window, dtype=torch.float32)
    for i, w in enumerate(windows):
        n = len(w["ids"])
        ids[i, :n] = torch.tensor([b + 1 for b in w["ids"]], dtype=torch.long)
        labels[i, :n] = torch.tensor(w["labels"], dtype=torch.float32)
    valid = ids.ne(PAD_ID)
    if mask_prob > 0:
        mask = (torch.rand(ids.shape, generator=generator) < mask_prob) & valid
        ids = ids.masked_fill(mask, MASK_ID)
    return ids.to(device), labels.to(device), valid.to(device)


def calibrate_threshold(probs: torch.Tensor, labels: torch.Tensor, target_rate: float) -> float:
    flat_p = probs[labels >= 0]
    k = max(1, int(flat_p.numel() * target_rate))
    threshold = flat_p.sort(descending=True).values[k - 1].item()
    return float(threshold)


def evaluate(model, windows, args, device) -> dict:
    model.eval()
    all_probs = []
    all_labels = []
    all_valid = []
    with torch.no_grad():
        for i in range(0, len(windows), args.batch_size):
            chunk = windows[i : i + args.batch_size]
            ids, labels, valid = to_batch(chunk, args.window, 0.0, device, None)
            logits, _ = model.encode_boundary_logits(ids)
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(labels.cpu())
            all_valid.append(valid.cpu())
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    valid = torch.cat(all_valid)
    probs = probs[valid]
    labels = labels[valid]
    target_rate = float(labels.mean().item())
    threshold = calibrate_threshold(probs, labels, target_rate)
    pred = probs >= threshold
    pos = labels > 0.5
    tp = (pred & pos).sum().item()
    precision = tp / max(pred.sum().item(), 1)
    recall = tp / max(pos.sum().item(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    seg_lens = []
    count = 0
    for w in windows:
        positions = [i for i, v in enumerate(w["labels"]) if v > 0.5]
        positions.append(len(w["ids"]))
        seg_lens.extend(positions[j + 1] - positions[j] for j in range(len(positions) - 1))
        count += 1
    seg_lens.sort()
    n = len(seg_lens)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_cut_rate": target_rate,
        "label_seg_bytes_median": seg_lens[n // 2] if n else 0,
        "label_seg_bytes_p90": seg_lens[int(n * 0.9)] if n else 0,
        "windows": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="s0_segmentor")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--pos-weight", type=float, default=12.0)
    parser.add_argument("--mask-prob", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--d-byte", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--segmentor-layers", type=int, default=9)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1152)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    samples = load_labeled_samples(Path(args.labels))
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(samples), generator=g).tolist()
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = [samples[i] for i in order[:n_val]]
    train_samples = [samples[i] for i in order[n_val:]]
    train_windows = pack_windows(train_samples, args.window)
    val_windows = pack_windows(val_samples, args.window)
    print(f"[s0] train_windows={len(train_windows)} val_windows={len(val_windows)}", flush=True)

    model = FLUEDV36(
        V36Config(
            d_byte=args.d_byte,
            encoder_layers=args.encoder_layers,
            segmentor_layers=args.segmentor_layers,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
        )
    ).to(device)
    params = [p for n, p in model.named_parameters() if n.startswith(("byte_lookup", "encoder_blocks", "segmentor"))]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps or int(len(train_windows) / args.batch_size * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / max(args.warmup_steps, 1)
        if s < args.warmup_steps
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - args.warmup_steps) / max(total_steps - args.warmup_steps, 1), 1.0))),
    )

    log_path = out_dir / "train_log.jsonl"
    gen = torch.Generator().manual_seed(args.seed + 1)
    step = 0
    t0 = time.time()
    model.train()
    while step < total_steps:
        order_w = torch.randperm(len(train_windows), generator=g).tolist()
        for start in range(0, len(order_w) - args.batch_size + 1, args.batch_size):
            if step >= total_steps:
                break
            chunk = [train_windows[i] for i in order_w[start : start + args.batch_size]]
            ids, labels, valid = to_batch(chunk, args.window, args.mask_prob, device, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                logits, _ = model.encode_boundary_logits(ids)
            loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                labels,
                reduction="none",
                pos_weight=torch.tensor(args.pos_weight, device=device),
            )
            loss = (loss * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "loss": float(loss.item()),
                    "grad": float(grad.item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "steps_per_sec": step / max(time.time() - t0, 1e-9),
                }
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")

    eval_stats = evaluate(model, val_windows, args, device)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "args": vars(args),
        "trained_prefixes": ["byte_lookup", "encoder_blocks", "segmentor"],
    }
    torch.save(payload, out_dir / "latest.pt")
    summary = {"run_id": args.run_id, "steps": step, "elapsed_sec": time.time() - t0, **eval_stats}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
