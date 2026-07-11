"""Memory ablation for FLUED-v3.2 language codec.

This evaluates the v3.2 boundary contract directly:
boundary does not read memory, decoder does not read memory, and the
interpreter reads only retrieved past memory.  Ablations replace the retrieved
memory context before ``readout_head``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID, STUB_CORPUS, ByteReconstructionDataset, StreamingReconstructionDataset  # noqa: E402
from tools.analysis.train_v32_language_codec_2m import (  # noqa: E402
    CodecCollator,
    V32LanguageCodec2M,
    _load_texts,
    move_codec_batch,
    segment_edge_pool,
    segment_mean_pool,
)


DEFAULT_MODES = ("full", "zero", "shuffled", "stale")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _resolve_checkpoint(path: Path) -> Path:
    candidates = [path]
    if path.is_dir():
        candidates.insert(0, path / "latest.pt")
    if path.suffix != ".pt":
        candidates.append(path / "latest.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"checkpoint not found: {path}")


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_model(path: Path, device: torch.device) -> Tuple[V32LanguageCodec2M, Dict[str, Any]]:
    ckpt = _torch_load(path)
    args = _dict(ckpt.get("args", {}) if isinstance(ckpt, Mapping) else {})
    state = ckpt.get("model", ckpt) if isinstance(ckpt, Mapping) else ckpt
    model = V32LanguageCodec2M(
        d_model=int(args.get("d_model", 192) or 192),
        hidden=int(args.get("hidden", 192) or 192),
        nhead=int(args.get("nhead", 4) or 4),
        encoder_layers=int(args.get("encoder_layers", 2) or 2),
        ffn_dim=int(args.get("ffn_dim", 768) or 768),
        max_span=int(args.get("max_span", 16) or 16),
        refine_steps=int(args.get("refine_steps", 1) or 1),
        dropout=float(args.get("dropout", 0.0) or 0.0),
        pool_mode=str(args.get("pool_mode", "mean") or "mean"),
        memory_slots_per_chunk=int(args.get("memory_slots_per_chunk", 0) or 0),
        memory_topk=int(args.get("memory_topk", 4) or 4),
        memory_retrieval_mode=str(args.get("memory_retrieval_mode", "topk") or "topk"),
        causal_byte_encoder=bool(args.get("causal_byte_encoder", True)),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, {"args": args, "step": ckpt.get("step") if isinstance(ckpt, Mapping) else None}


def _arg(args: argparse.Namespace, ckpt_args: Mapping[str, Any], key: str, default: Any) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return ckpt_args.get(key, default)


def _make_loader(args: argparse.Namespace, ckpt_args: Mapping[str, Any]) -> Tuple[DataLoader, Dict[str, Any]]:
    seq_len = int(_arg(args, ckpt_args, "seq_len", 128))
    stride = int(_arg(args, ckpt_args, "stride", 64))
    batch_size = int(_arg(args, ckpt_args, "batch_size", 32))
    max_eval_batches = int(_arg(args, ckpt_args, "max_eval_batches", 8))
    min_span = int(_arg(args, ckpt_args, "min_span", 2))
    max_span = int(_arg(args, ckpt_args, "max_span", 16))
    max_units = int(_arg(args, ckpt_args, "max_units", seq_len))
    seed = int(_arg(args, ckpt_args, "seed", 1234))

    if args.streaming_eval:
        if not args.data_path:
            raise ValueError("--streaming-eval requires --data-path")
        samples = args.stream_samples_per_worker or max(batch_size * max_eval_batches, 1024)
        dataset = StreamingReconstructionDataset(args.data_path, seq_len=seq_len, samples_per_worker=int(samples), seed=seed + 9999)
        data_desc = f"streaming:{args.data_path}"
    else:
        texts = _load_texts(args.data_path, int(args.eval_max_lines or 20000)) if args.data_path else STUB_CORPUS * int(args.stub_repeats)
        dataset = ByteReconstructionDataset(texts=texts, seq_len=seq_len, stride=stride)
        data_desc = f"fixed:{args.data_path or 'stub'}"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=CodecCollator(min_span, max_span, max_units),
    )
    return loader, {
        "seq_len": seq_len,
        "stride": stride,
        "batch_size": batch_size,
        "max_eval_batches": max_eval_batches,
        "min_span": min_span,
        "max_span": max_span,
        "max_units": max_units,
        "data": data_desc,
    }


def _shuffle_memory(memory: torch.Tensor, seg_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    out = torch.zeros_like(memory)
    active = seg_mask.bool()
    if not active.any():
        return out
    flat = memory[active]
    if flat.size(0) <= 1:
        out[active] = flat
        return out
    perm = torch.randperm(flat.size(0), generator=generator).to(memory.device)
    out[active] = flat[perm]
    return out


def _fit_stale(memory: torch.Tensor, previous: Optional[torch.Tensor], seg_mask: torch.Tensor) -> torch.Tensor:
    stale = torch.zeros_like(memory)
    if previous is None:
        return stale
    prev = previous.to(device=memory.device, dtype=memory.dtype)
    b = min(stale.size(0), prev.size(0))
    u = min(stale.size(1), prev.size(1))
    h = min(stale.size(2), prev.size(2))
    stale[:b, :u, :h] = prev[:b, :u, :h]
    return stale * seg_mask.unsqueeze(-1).to(stale.dtype)


@torch.no_grad()
def forward_with_mode(
    model: V32LanguageCodec2M,
    src: torch.Tensor,
    valid: torch.Tensor,
    seg_ids: torch.Tensor,
    seg_mask: torch.Tensor,
    mode: str,
    *,
    previous_memory: Optional[torch.Tensor],
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    emb = model.byte_seed(src)
    h = model.input_proj(emb)
    if model.causal_byte_encoder:
        seq_len = h.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool), diagonal=1)
    else:
        causal_mask = None
    h = model.encoder(h, mask=causal_mask, src_key_padding_mask=~valid)
    h = h * valid.unsqueeze(-1).to(h.dtype)

    pooled = segment_mean_pool(h, seg_ids, seg_mask)
    if model.pool_mode == "mean_first_last":
        first, last = segment_edge_pool(h, seg_ids, seg_mask)
        pooled = torch.cat([pooled, first, last], dim=-1)
    segment = model.segment_proj(pooled) * seg_mask.unsqueeze(-1).to(h.dtype)
    summary = model.summary_head(segment) * seg_mask.unsqueeze(-1).to(h.dtype)
    if model.memory_slots_per_chunk > 0:
        memory_slots = model.memory_slot_head(summary).view(summary.size(0), summary.size(1), model.memory_slots_per_chunk, summary.size(2))
        memory_slots = memory_slots * seg_mask.unsqueeze(-1).unsqueeze(-1).to(memory_slots.dtype)
    else:
        memory_slots = summary.new_zeros((summary.size(0), summary.size(1), 0, summary.size(2)))
    full_memory, retrieval = model.retrieve_past_memory(segment, memory_slots, seg_mask)

    if mode == "full":
        memory = full_memory
    elif mode == "zero":
        memory = torch.zeros_like(full_memory)
    elif mode == "shuffled":
        memory = _shuffle_memory(full_memory, seg_mask, generator)
    elif mode == "stale":
        memory = _fit_stale(full_memory, previous_memory, seg_mask)
    else:
        raise ValueError(f"unknown mode: {mode}")

    readout = model.readout_head(torch.cat([segment, memory], dim=-1))
    readout = readout * seg_mask.unsqueeze(-1).to(readout.dtype)
    for block in model.refiners:
        readout = block(readout, seg_mask)

    length_logits = model.length_head(readout)
    slots = torch.arange(model.max_span, device=src.device)
    slot_h = readout.unsqueeze(2) + model.slot_embed(slots).view(1, 1, model.max_span, -1)
    slot_h = slot_h.reshape(-1, model.max_span, readout.size(-1))
    slot_mask = seg_mask.unsqueeze(-1).expand(-1, -1, model.max_span).reshape(-1, model.max_span)
    slot_h = model.slot_decoder(slot_h, slot_mask)
    slot_h = slot_h.reshape(readout.size(0), readout.size(1), model.max_span, readout.size(-1))
    byte_logits = model.byte_head(slot_h)
    return byte_logits, length_logits, {"full_memory": full_memory, "memory": memory, **retrieval}


def _empty() -> Dict[str, float]:
    return {"loss_sum": 0.0, "slots": 0.0, "correct": 0.0, "units": 0.0, "length_correct": 0.0}


def _update(stats: Dict[str, float], byte_logits: torch.Tensor, length_logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor, seg_mask: torch.Tensor, max_span: int) -> None:
    slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
    loss = F.cross_entropy(byte_logits.float().reshape(-1, byte_logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID, reduction="none").view_as(targets)
    if slot_mask.any():
        stats["loss_sum"] += float(loss[slot_mask].sum().item())
        pred = byte_logits.argmax(dim=-1)
        stats["correct"] += float(((pred == targets) & slot_mask).sum().item())
        stats["slots"] += float(slot_mask.sum().item())
    if seg_mask.any():
        target = (lengths.clamp(min=1, max=max_span) - 1).clamp(min=0)
        pred_len = length_logits.argmax(dim=-1)
        stats["length_correct"] += float(((pred_len == target) & seg_mask).sum().item())
        stats["units"] += float(seg_mask.sum().item())


def _final(stats: Mapping[str, float]) -> Dict[str, float]:
    slots = float(stats["slots"])
    units = float(stats["units"])
    return {
        "recon_loss": float(stats["loss_sum"]) / slots if slots else float("nan"),
        "recon_acc": float(stats["correct"]) / slots if slots else float("nan"),
        "length_acc": float(stats["length_correct"]) / units if units else float("nan"),
        "slots": slots,
        "units": units,
    }


def _fmt(x: float, digits: int = 6) -> str:
    return "n/a" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def _fmt_delta(x: float, digits: int = 6) -> str:
    return "n/a" if not math.isfinite(float(x)) else f"{float(x):+.{digits}f}"


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> str:
    ckpt_path = _resolve_checkpoint(Path(args.checkpoint))
    ckpt = _torch_load(ckpt_path)
    ckpt_args = _dict(ckpt.get("args", {}) if isinstance(ckpt, Mapping) else {})
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model, meta = _load_model(ckpt_path, device)
    loader, cfg = _make_loader(args, ckpt_args)
    modes = [m for m in args.modes.replace(",", " ").split() if m]
    if "full" not in modes:
        modes.insert(0, "full")
    for mode in modes:
        if mode not in DEFAULT_MODES:
            raise ValueError(f"unknown mode {mode}; choices: {DEFAULT_MODES}")

    totals = {mode: _empty() for mode in modes}
    previous_full_memory: Optional[torch.Tensor] = None
    generator = torch.Generator()
    generator.manual_seed(int(ckpt_args.get("seed", 1234)) + 711)
    max_span = int(cfg["max_span"])
    amp = bool(args.amp)
    for i, batch in enumerate(loader):
        if i >= int(cfg["max_eval_batches"]):
            break
        src, _starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
        valid = src.ne(PAD_ID)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            full_logits, full_lengths, full_metrics = forward_with_mode(
                model, src, valid, seg_ids, seg_mask, "full", previous_memory=previous_full_memory, generator=generator
            )
            full_memory = full_metrics["full_memory"].detach()
            for mode in modes:
                if mode == "full":
                    byte_logits, length_logits = full_logits, full_lengths
                else:
                    byte_logits, length_logits, _ = forward_with_mode(
                        model, src, valid, seg_ids, seg_mask, mode, previous_memory=previous_full_memory, generator=generator
                    )
                _update(totals[mode], byte_logits, length_logits, targets, lengths, seg_mask, max_span)
        previous_full_memory = full_memory

    results = {mode: _final(totals[mode]) for mode in modes}
    full = results["full"]
    lines = [
        "# FLUED v3.2 Memory Ablation",
        "",
        f"- checkpoint: `{ckpt_path}`",
        f"- checkpoint_step: `{meta.get('step')}`",
        f"- device: `{device}`",
        f"- model_version: `{ckpt.get('summary', {}).get('model_version', 'n/a') if isinstance(ckpt, Mapping) else 'n/a'}`",
        f"- memory_enabled: `{getattr(model, 'memory_slots_per_chunk', 0) > 0}`",
        f"- byte_encoder_causal: `{getattr(model, 'causal_byte_encoder', False)}`",
        f"- memory_read_scope: `past_only`",
        f"- decoder_reads_memory: `False`",
        "",
        "| mode | recon_loss | delta_loss | delta_loss_% | recon_acc | delta_acc | length_acc | slots | units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        row = results[mode]
        d_loss = row["recon_loss"] - full["recon_loss"]
        pct = 100.0 * d_loss / full["recon_loss"] if math.isfinite(row["recon_loss"]) and full["recon_loss"] else float("nan")
        d_acc = row["recon_acc"] - full["recon_acc"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    _fmt(row["recon_loss"]),
                    _fmt_delta(d_loss),
                    _fmt_delta(pct, 3),
                    _fmt(row["recon_acc"]),
                    _fmt_delta(d_acc),
                    _fmt(row["length_acc"]),
                    str(int(row["slots"])),
                    str(int(row["units"])),
                ]
            )
            + " |"
        )
    markdown = "\n".join(lines)
    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FLUED-v3.2 retrieved-memory ablations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=None)
    parser.add_argument("--out-path", default="")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--eval-max-lines", type=int, default=20000)
    parser.add_argument("--min-span", type=int, default=None)
    parser.add_argument("--max-span", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stub-repeats", type=int, default=4)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
