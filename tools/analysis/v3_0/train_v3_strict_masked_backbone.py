"""Strict masked-source backbone sweep for FLUED v3-family codecs.

This fills the Backbone row of the v3 checkpoint re-evaluation table.  The
task is intentionally leakage-safe:

  clean bytes -> sample byte/span mask
  masked bytes -> frozen FLUED codec -> readout latents
  readout latents -> small infill backbone
  predicted readout -> frozen FLUED decoder -> original masked bytes

The codec never receives clean bytes at masked positions.  Clean bytes are used
only as training targets.  This script supports archived v3.1, v3.2, and
v3.2.1 checkpoints through one common task.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_VOCAB_SIZE, PAD_ID  # noqa: E402
from tools.analysis.v3_0.train_v3_commit_controller_small import _append_jsonl, _cosine_with_warmup  # noqa: E402
from tools.analysis.v3_1.train_v31_language_codec_2m import V31LanguageCodec2M  # noqa: E402
from tools.analysis.v3_1.train_v31_min_backbone import decode_readout as decode_v31_readout  # noqa: E402
from tools.analysis.v3_2.train_v32_language_codec_2m import V32LanguageCodec2M  # noqa: E402
from tools.analysis.v3_2.train_v32_min_backbone import decode_readout as decode_v32_readout  # noqa: E402
from tools.analysis.v3_2.train_v32_strict_masked_backbone import (  # noqa: E402
    ByteInfillBackbone,
    LatentInfillBackbone,
    byte_step,
    make_dataloaders,
    _move_prepared_batch,
    _safe_acc,
)


@dataclass
class FrozenCodec:
    model: nn.Module
    family: str
    saved_args: Dict[str, object]
    ckpt_path: Path


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / max(len(vals), 1)


def _resolve_ckpt(path: str) -> Path:
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "latest.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"codec checkpoint not found: {ckpt}")
    return ckpt


def _infer_family(ckpt_path: Path, saved_args: Dict[str, object], state: Dict[str, torch.Tensor]) -> str:
    text = str(ckpt_path).lower()
    if "v31" in text or "codec_40k" in text or "codec_10k_pool" in text:
        return "v31"
    if "v32" in text or "v321" in text or "stage2_" in text or "stage3_" in text:
        return "v32"
    if any(key.startswith("byte_seed.") for key in state):
        return "v32"
    if any(key.startswith("embedding.") for key in state):
        return "v31"
    if "memory_slots_per_chunk" in saved_args or "causal_byte_encoder" in saved_args:
        return "v32"
    return "v31"


def _v31_kwargs(saved_args: Dict[str, object]) -> Dict[str, object]:
    keys = ["d_model", "hidden", "nhead", "encoder_layers", "ffn_dim", "max_span", "refine_steps", "dropout", "pool_mode"]
    return {key: saved_args[key] for key in keys if key in saved_args}


def _infer_memory_slots(state: Dict[str, torch.Tensor], hidden: int) -> int:
    for key, value in state.items():
        if key.endswith("memory_slot_head.3.weight") and value.ndim == 2:
            return int(value.shape[0] // max(hidden, 1))
    return 0


def _v32_kwargs(saved_args: Dict[str, object], state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    keys = [
        "d_model",
        "hidden",
        "nhead",
        "encoder_layers",
        "ffn_dim",
        "max_span",
        "refine_steps",
        "dropout",
        "pool_mode",
        "memory_slots_per_chunk",
        "memory_topk",
        "memory_retrieval_mode",
        "causal_byte_encoder",
    ]
    kwargs = {key: saved_args[key] for key in keys if key in saved_args}
    hidden = int(kwargs.get("hidden", saved_args.get("hidden", 192)))
    has_memory_weight = any("memory_slot_head" in key for key in state)
    if has_memory_weight:
        kwargs["memory_slots_per_chunk"] = int(kwargs.get("memory_slots_per_chunk", _infer_memory_slots(state, hidden)))
    else:
        kwargs["memory_slots_per_chunk"] = 0
    return kwargs


def load_frozen_codec(path: str, device: torch.device) -> FrozenCodec:
    ckpt_path = _resolve_ckpt(path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = dict(ckpt.get("args", {}))
    state = ckpt["model"]
    family = _infer_family(ckpt_path, saved_args, state)
    if family == "v32":
        model = V32LanguageCodec2M(**_v32_kwargs(saved_args, state)).to(device)
    else:
        model = V31LanguageCodec2M(**_v31_kwargs(saved_args)).to(device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return FrozenCodec(model=model, family=family, saved_args=saved_args, ckpt_path=ckpt_path)


@torch.no_grad()
def encode_masked_readout(bundle: FrozenCodec, masked_src: torch.Tensor, valid: torch.Tensor, seg_ids: torch.Tensor, seg_mask: torch.Tensor, amp: bool) -> torch.Tensor:
    with torch.amp.autocast(device_type=masked_src.device.type, dtype=torch.bfloat16, enabled=amp and masked_src.device.type == "cuda"):
        _byte_logits, _length_logits, metrics = bundle.model(masked_src, valid, seg_ids, seg_mask)
    return metrics["readout"].detach()


def decode_readout(bundle: FrozenCodec, readout: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if bundle.family == "v32":
        return decode_v32_readout(bundle.model, readout, seg_mask)
    return decode_v31_readout(bundle.model, readout, seg_mask)


def latent_step(backbone: LatentInfillBackbone, bundle: FrozenCodec, batch, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    _clean_src, masked_src, valid, seg_ids, seg_mask, targets, loss_mask, lengths, unit_mask = _move_prepared_batch(batch, device)
    readout = encode_masked_readout(bundle, masked_src, valid, seg_ids, seg_mask, args.amp)
    zero_unit_mask = torch.zeros_like(seg_mask)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        pred_readout = backbone(readout, seg_mask, zero_unit_mask)
        mixed = torch.where(unit_mask.unsqueeze(-1), pred_readout, readout)
        byte_logits, length_logits = decode_readout(bundle, mixed, seg_mask)
        ce = F.cross_entropy(
            byte_logits.float().reshape(-1, byte_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(targets)
        byte_loss = ce[loss_mask].mean() if loss_mask.any() else ce.new_zeros(())
        if args.length_loss_weight > 0 and unit_mask.any():
            length_target = (lengths.clamp(min=1, max=bundle.model.max_span) - 1).clamp(min=0)
            length_loss = F.cross_entropy(length_logits[unit_mask].float(), length_target[unit_mask])
            loss = byte_loss + args.length_loss_weight * length_loss
        else:
            length_loss = byte_loss.new_zeros(())
            loss = byte_loss

    with torch.no_grad():
        pred = byte_logits.argmax(dim=-1)
        slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
        keep_mask = slot_mask & (~loss_mask)
        length_target = (lengths.clamp(min=1, max=bundle.model.max_span) - 1).clamp(min=0)
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


def average_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


@torch.no_grad()
def evaluate(backbone: nn.Module, bundle: FrozenCodec | None, loader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    backbone.eval()
    rows: list[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        if args.mode == "latent":
            assert bundle is not None
            _loss, metrics = latent_step(backbone, bundle, batch, args, device)
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

    bundle: FrozenCodec | None = None
    codec_args: Dict[str, object] = {}
    if args.mode == "latent":
        bundle = load_frozen_codec(args.codec_ckpt, device)
        codec_args = bundle.saved_args
        args.max_span = int(codec_args.get("max_span", args.max_span))

    train_loader, eval_loader = make_dataloaders(args)
    if args.mode == "latent":
        assert bundle is not None
        latent_dim = int(codec_args.get("hidden", args.hidden))
        backbone = LatentInfillBackbone(latent_dim, args.hidden, args.layers, args.nhead, args.ffn_dim, args.max_units, args.dropout).to(device)
    else:
        backbone = ByteInfillBackbone(args.hidden, args.layers, args.nhead, args.ffn_dim, args.seq_len, args.dropout, BYTE_VOCAB_SIZE).to(device)

    params = sum(param.numel() for param in backbone.parameters())
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
                assert bundle is not None
                loss, metrics = latent_step(backbone, bundle, batch, args, device)
            else:
                loss, metrics = byte_step(backbone, batch, args, device)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(backbone.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if step % args.log_every == 0:
                row = {
                    "step": step,
                    "mode": args.mode,
                    "codec_family": bundle.family if bundle is not None else "byte",
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad": float(grad.item() if hasattr(grad, "item") else grad),
                    **metrics,
                }
                _append_jsonl(log_path, row)
                print(
                    f"step={step} mode={args.mode} family={row['codec_family']} "
                    f"loss={row['loss']:.4f} mask_acc={row.get('mask_byte_acc', 0.0):.3f} "
                    f"keep_acc={row.get('keep_byte_acc', row.get('keep_byte_acc_model', 0.0)):.3f} "
                    f"mask_frac={row.get('masked_byte_fraction', 0.0):.3f}",
                    flush=True,
                )
            if step > 0 and step % args.ckpt_every == 0:
                payload = {"model": backbone.state_dict(), "args": vars(args), "step": step, "params": params}
                torch.save(payload, out_dir / f"step{step}.pt")
                torch.save(payload, out_dir / "latest.pt")
            step += 1

    eval_stats = evaluate(backbone, bundle, eval_loader, args, device)
    elapsed = time.perf_counter() - start
    result = {
        "mode": args.mode,
        "task": "strict_masked_source",
        "params": params,
        "steps": step,
        "codec_family": bundle.family if bundle is not None else "byte",
        "codec_ckpt": str(bundle.ckpt_path) if bundle is not None else "",
        "codec_name": bundle.ckpt_path.parent.name if bundle is not None else "byte_baseline",
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
        **{f"eval_{key}": value for key, value in eval_stats.items()},
    }
    torch.save({"model": backbone.state_dict(), "args": vars(args), "step": step, "summary": result}, out_dir / "latest.pt")
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict masked-source backbone over FLUED v3-family codecs")
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
