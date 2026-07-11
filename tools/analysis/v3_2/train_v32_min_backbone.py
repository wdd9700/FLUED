"""Train a minimal backbone over FLUED-v3.2 readout latents or raw bytes.

This is the first Goal-6 experiment.  It deliberately keeps the interface
boundary strict:

  latent mode:
    bytes -> frozen FLUED-v3.2 encoder -> readout latent sequence
          -> small masked-infill backbone
          -> frozen FLUED decoder -> byte-span metrics

  byte mode:
    bytes -> small masked-infill backbone -> byte metrics

The external backbone never receives FLUED summary / memory tensors.  By
default, latent mode trains only a latent reconstruction loss; decoded byte
accuracy is measured through the frozen decoder instead of directly training
the FLUED encoder with byte cross entropy.
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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import (
    BYTE_VOCAB_SIZE,
    ByteReconstructionDataset,
    MASK_ID,
    PAD_ID,
    STUB_CORPUS,
    StreamingReconstructionDataset,
)
from tools.analysis.v3_0.train_v3_commit_controller_small import _append_jsonl, _cosine_with_warmup, _load_texts
from tools.analysis.v3_2.train_v32_language_codec_2m import (
    CodecCollator,
    V32LanguageCodec2M,
    move_codec_batch,
)


def _mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values]
    return sum(vals) / max(len(vals), 1)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return x.new_zeros(())
    return x[mask].float().mean()


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def _resolve_ckpt(path: str) -> Path:
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "latest.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"codec checkpoint not found: {ckpt}")
    return ckpt


def _codec_kwargs(saved_args: Dict[str, object]) -> Dict[str, object]:
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
    return {k: saved_args[k] for k in keys if k in saved_args}


def load_frozen_codec(path: str, device: torch.device) -> Tuple[V32LanguageCodec2M, Dict[str, object], Path]:
    ckpt_path = _resolve_ckpt(path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = dict(ckpt.get("args", {}))
    model = V32LanguageCodec2M(**_codec_kwargs(saved_args)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, saved_args, ckpt_path


class LatentInfillBackbone(nn.Module):
    """Small Transformer over FLUED readout latents."""

    def __init__(
        self,
        latent_dim: int,
        hidden: int,
        layers: int,
        nhead: int,
        ffn_dim: int,
        max_units: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.pos = nn.Embedding(max_units, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent_dim))

    def forward(self, readout: torch.Tensor, active: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, units, _ = readout.shape
        h = self.in_proj(readout)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        pos = torch.arange(units, device=readout.device).view(1, units)
        h = h + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=~active)
        return self.out(h) * active.unsqueeze(-1).to(h.dtype)


class ByteInfillBackbone(nn.Module):
    """Small Transformer byte baseline for the same masked-infill task."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        nhead: int,
        ffn_dim: int,
        seq_len: int,
        dropout: float = 0.0,
        vocab_size: int = BYTE_VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=PAD_ID)
        self.pos = nn.Embedding(seq_len, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, vocab_size))

    def forward(self, src: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = src.masked_fill(mask, MASK_ID).clamp(min=0, max=MASK_ID)
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).view(1, seq_len)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=~valid)
        return self.out(h)


def decode_readout(codec: V32LanguageCodec2M, readout: torch.Tensor, seg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run only the frozen length / slot decoder over readout latents."""

    length_logits = codec.length_head(readout)
    slots = torch.arange(codec.max_span, device=readout.device)
    slot_h = readout.unsqueeze(2) + codec.slot_embed(slots).view(1, 1, codec.max_span, -1)
    slot_h = slot_h.view(-1, codec.max_span, readout.size(-1))
    slot_mask = seg_mask.unsqueeze(-1).expand(-1, -1, codec.max_span).reshape(-1, codec.max_span)
    slot_h = codec.slot_decoder(slot_h, slot_mask)
    slot_h = slot_h.view(readout.size(0), readout.size(1), codec.max_span, readout.size(-1))
    byte_logits = codec.byte_head(slot_h)
    return byte_logits, length_logits


def make_random_mask(active: torch.Tensor, prob: float) -> torch.Tensor:
    mask = (torch.rand(active.shape, device=active.device) < float(prob)) & active
    if active.ndim != 2:
        return mask
    for b in range(active.size(0)):
        if bool(active[b].any()) and not bool(mask[b].any()):
            idx = active[b].nonzero(as_tuple=False).flatten()
            chosen = idx[torch.randint(idx.numel(), (1,), device=active.device)]
            mask[b, chosen] = True
    return mask


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    collate = CodecCollator(args.min_span, args.max_span, args.max_units)
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
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate,
        **loader_kwargs,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    return train_loader, eval_loader


@torch.no_grad()
def encode_readout(
    codec: V32LanguageCodec2M,
    src: torch.Tensor,
    seg_ids: torch.Tensor,
    seg_mask: torch.Tensor,
    amp: bool,
) -> torch.Tensor:
    # The codec sees only bytes that participate in clean segments.  Summary /
    # memory are used internally by the encoder but are not returned to the
    # backbone.
    valid = seg_ids.ge(0)
    with torch.amp.autocast(device_type=src.device.type, dtype=torch.bfloat16, enabled=amp and src.device.type == "cuda"):
        _, _, metrics = codec(src, valid, seg_ids, seg_mask)
    return metrics["readout"].detach()


def latent_step(
    backbone: LatentInfillBackbone,
    codec: V32LanguageCodec2M,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    src, _starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
    readout = encode_readout(codec, src, seg_ids, seg_mask, args.amp)
    unit_mask = make_random_mask(seg_mask, args.mask_prob)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        pred_readout = backbone(readout, seg_mask, unit_mask)
        diff = (pred_readout.float() - readout.float()).pow(2).mean(dim=-1)
        latent_loss = _masked_mean(diff, unit_mask)
        loss = latent_loss
        if args.latent_byte_loss_weight > 0:
            mixed = torch.where(unit_mask.unsqueeze(-1), pred_readout, readout)
            byte_logits, _ = decode_readout(codec, mixed, seg_mask)
            slot_mask = targets.ne(PAD_ID) & unit_mask.unsqueeze(-1)
            ce = F.cross_entropy(
                byte_logits.float().view(-1, byte_logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).view_as(targets)
            byte_loss = _masked_mean(ce, slot_mask)
            loss = loss + args.latent_byte_loss_weight * byte_loss
        else:
            byte_loss = latent_loss.new_zeros(())

    with torch.no_grad():
        mixed = torch.where(unit_mask.unsqueeze(-1), pred_readout.detach(), readout)
        byte_logits, length_logits = decode_readout(codec, mixed, seg_mask)
        pred = byte_logits.argmax(dim=-1)
        slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
        mask_slots = slot_mask & unit_mask.unsqueeze(-1)
        keep_slots = slot_mask & (~unit_mask).unsqueeze(-1)
        length_target = (lengths.clamp(min=1, max=codec.max_span) - 1).clamp(min=0)
        len_pred = length_logits.argmax(dim=-1)
        masked_units = int(unit_mask.sum().item())
        metrics = {
            "loss": float(loss.item()),
            "latent_mse": float(latent_loss.item()),
            "byte_loss_aux": float(byte_loss.item()),
            "mask_byte_acc": _safe_acc(pred, targets, mask_slots),
            "keep_byte_acc": _safe_acc(pred, targets, keep_slots),
            "mask_length_acc": _safe_acc(len_pred, length_target, unit_mask),
            "keep_length_acc": _safe_acc(len_pred, length_target, seg_mask & (~unit_mask)),
            "masked_units": float(masked_units),
            "active_units": float(seg_mask.sum().item()),
            "masked_byte_fraction": float(mask_slots.sum().item() / max(slot_mask.sum().item(), 1)),
            "units_per_byte": float(seg_mask.sum().item() / max(seg_ids.ge(0).sum().item(), 1)),
        }
    return loss, metrics


def byte_step(
    backbone: ByteInfillBackbone,
    batch,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    src, _starts, seg_ids, _targets, _lengths, _seg_mask = move_codec_batch(batch, device)
    valid = seg_ids.ge(0)
    if args.byte_mask_mode == "segment":
        active_units = _seg_mask
        unit_mask = make_random_mask(active_units, args.mask_prob)
        gathered = unit_mask.gather(1, seg_ids.clamp(min=0))
        byte_mask = gathered & valid
        masked_units = float(unit_mask.sum().item())
        active_units_n = float(active_units.sum().item())
    else:
        byte_mask = make_random_mask(valid, args.mask_prob)
        masked_units = 0.0
        active_units_n = 0.0

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        logits = backbone(src, valid, byte_mask)
        per_pos = F.cross_entropy(logits.float().view(-1, logits.size(-1)), src.view(-1), ignore_index=PAD_ID, reduction="none").view_as(src)
        loss = _masked_mean(per_pos, byte_mask)

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        keep_mask = valid & (~byte_mask)
        metrics = {
            "loss": float(loss.item()),
            "mask_byte_acc": _safe_acc(pred, src, byte_mask),
            "keep_byte_acc_model": _safe_acc(pred, src, keep_mask),
            "keep_byte_acc_copy": 1.0,
            "masked_bytes": float(byte_mask.sum().item()),
            "valid_bytes": float(valid.sum().item()),
            "masked_byte_fraction": float(byte_mask.sum().item() / max(valid.sum().item(), 1)),
            "masked_units": masked_units,
            "active_units": active_units_n,
        }
    return loss, metrics


def average_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {key: _mean(row[key] for row in rows if key in row) for key in keys}


@torch.no_grad()
def evaluate(backbone: nn.Module, codec: V32LanguageCodec2M | None, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    backbone.eval()
    rows: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        if args.mode == "latent":
            assert codec is not None
            _loss, metrics = latent_step(backbone, codec, batch, args, device, train=False)
        else:
            _loss, metrics = byte_step(backbone, batch, args, device, train=False)
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

    codec: V32LanguageCodec2M | None = None
    codec_args: Dict[str, object] = {}
    codec_path = ""
    if args.mode == "latent":
        codec, codec_args, resolved = load_frozen_codec(args.codec_ckpt, device)
        codec_path = str(resolved)
        # The latent backbone is evaluated through the frozen codec decoder, so
        # span packing must match the checkpoint's decoder slots.
        args.max_span = int(codec_args.get("max_span", args.max_span))

    train_loader, eval_loader = make_dataloaders(args)

    if args.mode == "latent":
        assert codec is not None
        latent_dim = int(codec_args.get("hidden", args.hidden))
        backbone = LatentInfillBackbone(
            latent_dim=latent_dim,
            hidden=args.hidden,
            layers=args.layers,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
            max_units=args.max_units,
            dropout=args.dropout,
        ).to(device)
    else:
        backbone = ByteInfillBackbone(
            hidden=args.hidden,
            layers=args.layers,
            nhead=args.nhead,
            ffn_dim=args.ffn_dim,
            seq_len=args.seq_len,
            dropout=args.dropout,
        ).to(device)

    params = sum(p.numel() for p in backbone.parameters())
    opt = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)
    log_path = out_dir / "train_log.jsonl"
    start_time = time.perf_counter()

    step = 0
    backbone.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            opt.zero_grad(set_to_none=True)
            if args.mode == "latent":
                assert codec is not None
                loss, metrics = latent_step(backbone, codec, batch, args, device, train=True)
            else:
                loss, metrics = byte_step(backbone, batch, args, device, train=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(backbone.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                row = {
                    "step": step,
                    "mode": args.mode,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad": float(grad.item() if hasattr(grad, "item") else grad),
                    **metrics,
                }
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
    elapsed = time.perf_counter() - start_time
    result = {
        "mode": args.mode,
        "params": params,
        "steps": step,
        "codec_ckpt": codec_path,
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
    parser = argparse.ArgumentParser(description="Train minimal FLUED-latent or byte masked-infill backbone")
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
    parser.add_argument("--byte-mask-mode", choices=["segment", "random"], default="segment")
    parser.add_argument("--latent-byte-loss-weight", type=float, default=0.0)
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
