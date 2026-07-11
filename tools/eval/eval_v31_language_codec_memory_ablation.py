"""Evaluate causal summary-memory ablations for FLUED-v3.1 language codec.

The v3.1 interface boundary is intentionally kept explicit here:
``readout`` is the external latent interface, while ``summary`` and
``memory`` are internal FLUED encoder mechanisms.  This script does not change
the model class; it replays the model's forward path and swaps only the causal
summary memory before ``readout_head``.
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

from flued.data import (  # noqa: E402
    MASK_ID,
    PAD_ID,
    STUB_CORPUS,
    ByteReconstructionDataset,
    StreamingReconstructionDataset,
)
from tools.analysis.train_v31_language_codec_2m import (  # noqa: E402
    CodecCollator,
    V31LanguageCodec2M,
    _load_texts,
    move_codec_batch,
    segment_edge_pool,
    segment_mean_pool,
)


DEFAULT_MODES = ("full", "zero", "shuffled", "stale", "summary_detached")


def _resolve_checkpoint(path: Path) -> Path:
    expanded = path.expanduser()
    bases = [expanded]
    if not expanded.is_absolute():
        bases.append(REPO_ROOT / expanded)

    candidates: List[Path] = []
    for base in bases:
        if base.is_dir():
            candidates.append(base / "latest.pt")
        candidates.append(base)
        if base.suffix != ".pt":
            candidates.append(base / "latest.pt")

    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(f"checkpoint not found; tried: {tried}")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _arg_value(cli_value: Any, ckpt_args: Mapping[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    value = ckpt_args.get(key, default)
    return default if value is None else value


def _arg_int(cli_value: Any, ckpt_args: Mapping[str, Any], key: str, default: int) -> int:
    return int(_arg_value(cli_value, ckpt_args, key, default))


def _arg_float(cli_value: Any, ckpt_args: Mapping[str, Any], key: str, default: float) -> float:
    return float(_arg_value(cli_value, ckpt_args, key, default))


def _load_model(checkpoint_path: Path, device: torch.device) -> Tuple[V31LanguageCodec2M, Dict[str, Any]]:
    ckpt = _torch_load(checkpoint_path)
    ckpt_args = _dict_or_empty(ckpt.get("args", {}) if isinstance(ckpt, Mapping) else {})
    state = ckpt.get("model", ckpt) if isinstance(ckpt, Mapping) else ckpt
    if not isinstance(state, Mapping):
        raise RuntimeError(f"checkpoint does not contain a state dict: {checkpoint_path}")

    model = V31LanguageCodec2M(
        d_model=int(ckpt_args.get("d_model", 192) or 192),
        hidden=int(ckpt_args.get("hidden", 192) or 192),
        nhead=int(ckpt_args.get("nhead", 4) or 4),
        encoder_layers=int(ckpt_args.get("encoder_layers", 2) or 2),
        ffn_dim=int(ckpt_args.get("ffn_dim", 768) or 768),
        max_span=int(ckpt_args.get("max_span", 16) or 16),
        refine_steps=int(ckpt_args.get("refine_steps", 1) or 1),
        dropout=float(ckpt_args.get("dropout", 0.0) or 0.0),
        pool_mode=str(ckpt_args.get("pool_mode", "mean") or "mean"),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    meta = {
        "args": ckpt_args,
        "step": ckpt.get("step") if isinstance(ckpt, Mapping) else None,
        "summary": ckpt.get("summary", {}) if isinstance(ckpt, Mapping) else {},
    }
    return model, meta


def _select_device(requested: Optional[str], ckpt_args: Mapping[str, Any]) -> torch.device:
    raw = str(_arg_value(requested, ckpt_args, "device", "cuda"))
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _shuffle_active_memory(
    memory: torch.Tensor,
    seg_mask: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    shuffled = memory.new_zeros(memory.shape)
    active = seg_mask.bool()
    if not active.any():
        return shuffled
    flat = memory[active]
    if flat.size(0) <= 1:
        shuffled[active] = flat
        return shuffled
    perm = torch.randperm(flat.size(0), generator=generator).to(memory.device)
    shuffled[active] = flat[perm]
    return shuffled


def _fit_stale_memory(
    current_memory: torch.Tensor,
    previous_memory: Optional[torch.Tensor],
    seg_mask: torch.Tensor,
) -> torch.Tensor:
    stale = current_memory.new_zeros(current_memory.shape)
    if previous_memory is None:
        return stale

    prev = previous_memory.to(device=current_memory.device, dtype=current_memory.dtype)
    bsz = min(stale.size(0), prev.size(0))
    units = min(stale.size(1), prev.size(1))
    hidden = min(stale.size(2), prev.size(2))
    if bsz > 0 and units > 0 and hidden > 0:
        stale[:bsz, :units, :hidden] = prev[:bsz, :units, :hidden]
    return stale * seg_mask.unsqueeze(-1).to(stale.dtype)


def forward_with_memory_mode(
    model: V31LanguageCodec2M,
    src: torch.Tensor,
    valid: torch.Tensor,
    seg_ids: torch.Tensor,
    seg_mask: torch.Tensor,
    mode: str,
    *,
    stale_memory: Optional[torch.Tensor] = None,
    shuffle_generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Replay ``V31LanguageCodec2M.forward`` while replacing only memory."""

    if mode not in DEFAULT_MODES:
        raise ValueError(f"unknown memory mode: {mode}")

    emb = model.embedding(src.clamp(min=0, max=MASK_ID))
    h = model.input_proj(emb)
    h = model.encoder(h, src_key_padding_mask=~valid)
    h = h * valid.unsqueeze(-1).to(h.dtype)

    pooled = segment_mean_pool(h, seg_ids, seg_mask)
    if getattr(model, "pool_mode", "mean") == "mean_first_last":
        first, last = segment_edge_pool(h, seg_ids, seg_mask)
        pooled = torch.cat([pooled, first, last], dim=-1)
    segment = model.segment_proj(pooled) * seg_mask.unsqueeze(-1).to(h.dtype)
    summary = model.summary_head(segment) * seg_mask.unsqueeze(-1).to(h.dtype)
    full_memory = model.causal_summary_memory(summary, seg_mask)

    if mode == "full":
        memory = full_memory
    elif mode == "zero":
        memory = torch.zeros_like(full_memory)
    elif mode == "shuffled":
        memory = _shuffle_active_memory(full_memory, seg_mask, shuffle_generator)
    elif mode == "stale":
        memory = _fit_stale_memory(full_memory, stale_memory, seg_mask)
    elif mode == "summary_detached":
        memory = model.causal_summary_memory(summary.detach(), seg_mask)
    else:
        raise AssertionError(f"unhandled mode: {mode}")

    readout = model.readout_head(torch.cat([segment, memory], dim=-1))
    readout = readout * seg_mask.unsqueeze(-1).to(h.dtype)
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
    boundary_logits = model.boundary_head(h).squeeze(-1)
    metrics = {
        "h": h,
        "segment": segment,
        "summary": summary,
        "full_memory": full_memory,
        "memory": memory,
        "readout": readout,
        "boundary_logits": boundary_logits,
    }
    return byte_logits, length_logits, metrics


def _parse_modes(raw: str) -> List[str]:
    modes: List[str] = []
    for item in raw.replace(",", " ").split():
        if item not in DEFAULT_MODES:
            raise ValueError(f"unknown mode {item!r}; choices: {', '.join(DEFAULT_MODES)}")
        if item not in modes:
            modes.append(item)
    if "full" not in modes:
        modes.insert(0, "full")
    return modes


def _empty_totals() -> Dict[str, float]:
    return {
        "recon_loss_sum": 0.0,
        "target_slots": 0.0,
        "recon_correct": 0.0,
        "length_units": 0.0,
        "length_correct": 0.0,
        "bytes": 0.0,
        "batches": 0.0,
    }


def _update_totals(
    totals: Dict[str, float],
    byte_logits: torch.Tensor,
    length_logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    seg_mask: torch.Tensor,
    valid: torch.Tensor,
    max_span: int,
) -> None:
    slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
    slot_count = int(slot_mask.sum().item())
    loss_sum = F.cross_entropy(
        byte_logits.float().reshape(-1, byte_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="sum",
    )
    pred = byte_logits.argmax(dim=-1)
    recon_correct = int(((pred == targets) & slot_mask).sum().item())

    length_target = (lengths.clamp(min=1, max=max_span) - 1).clamp(min=0)
    unit_count = int(seg_mask.sum().item())
    if unit_count:
        length_pred = length_logits.argmax(dim=-1)
        length_correct = int(((length_pred == length_target) & seg_mask).sum().item())
    else:
        length_correct = 0

    totals["recon_loss_sum"] += float(loss_sum.item())
    totals["target_slots"] += float(slot_count)
    totals["recon_correct"] += float(recon_correct)
    totals["length_units"] += float(unit_count)
    totals["length_correct"] += float(length_correct)
    totals["bytes"] += float(valid.sum().item())
    totals["batches"] += 1.0


def _finalize_totals(totals: Mapping[str, float]) -> Dict[str, float]:
    slots = totals.get("target_slots", 0.0)
    units = totals.get("length_units", 0.0)
    return {
        "recon_loss": totals["recon_loss_sum"] / slots if slots else float("nan"),
        "recon_acc": totals["recon_correct"] / slots if slots else float("nan"),
        "length_acc": totals["length_correct"] / units if units else float("nan"),
        "target_slots": slots,
        "length_units": units,
        "bytes": totals.get("bytes", 0.0),
        "batches": totals.get("batches", 0.0),
    }


def _build_loader(
    cfg: argparse.Namespace,
    ckpt_args: Mapping[str, Any],
    device: torch.device,
) -> Tuple[DataLoader, Dict[str, Any]]:
    seq_len = _arg_int(cfg.seq_len, ckpt_args, "seq_len", 128)
    stride = _arg_int(cfg.stride, ckpt_args, "stride", 64)
    batch_size = _arg_int(cfg.batch_size, ckpt_args, "batch_size", 32)
    max_eval_batches = _arg_int(cfg.max_eval_batches, ckpt_args, "max_eval_batches", 8)
    eval_max_lines = _arg_int(cfg.eval_max_lines, ckpt_args, "eval_max_lines", 20000)
    min_span = _arg_int(cfg.min_span, ckpt_args, "min_span", 2)
    max_span = _arg_int(cfg.max_span, ckpt_args, "max_span", 16)
    max_units = _arg_int(cfg.max_units, ckpt_args, "max_units", 128)
    num_workers = _arg_int(cfg.num_workers, ckpt_args, "num_workers", 0)
    seed = _arg_int(cfg.seed, ckpt_args, "seed", 1234)

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be positive")

    data_path = cfg.data_path or ""
    if cfg.streaming_eval:
        if not data_path:
            raise ValueError("--streaming-eval requires --data-path")
        stream_samples = cfg.stream_samples_per_worker
        if stream_samples is None:
            stream_samples = max(batch_size * max_eval_batches, 64)
        dataset = StreamingReconstructionDataset(
            file_path=data_path,
            seq_len=seq_len,
            samples_per_worker=int(stream_samples),
            seed=seed + 9999,
        )
        data_desc = f"streaming:{data_path}"
    else:
        if data_path:
            texts = _load_texts(data_path, eval_max_lines)
            data_desc = f"fixed:{data_path}"
        else:
            texts = STUB_CORPUS * max(1, int(cfg.stub_repeats))
            data_desc = f"builtin_stub:{len(texts)} texts"
        dataset = ByteReconstructionDataset(texts=texts, seq_len=seq_len, stride=stride)

    collate_fn = CodecCollator(min_span=min_span, max_span=max_span, max_units=max_units)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    resolved = {
        "seq_len": seq_len,
        "stride": stride,
        "batch_size": batch_size,
        "max_eval_batches": max_eval_batches,
        "eval_max_lines": eval_max_lines,
        "min_span": min_span,
        "max_span": max_span,
        "max_units": max_units,
        "num_workers": num_workers,
        "seed": seed,
        "data": data_desc,
        "streaming_eval": bool(cfg.streaming_eval),
    }
    return loader, resolved


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _fmt(value: float, digits: int = 6) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_int(value: float) -> str:
    if not _is_finite(value):
        return "n/a"
    return str(int(value))


def _fmt_delta(value: float, digits: int = 6) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _memory_effect(
    results: Mapping[str, Mapping[str, float]],
    loss_eps: float,
    acc_eps: float,
) -> Tuple[str, Dict[str, float]]:
    full = results["full"]
    probes = [mode for mode in ("zero", "shuffled", "stale") if mode in results]
    max_loss_delta = 0.0
    max_acc_delta = 0.0
    for mode in probes:
        row = results[mode]
        if _is_finite(row["recon_loss"]) and _is_finite(full["recon_loss"]):
            max_loss_delta = max(max_loss_delta, abs(row["recon_loss"] - full["recon_loss"]))
        if _is_finite(row["recon_acc"]) and _is_finite(full["recon_acc"]):
            max_acc_delta = max(max_acc_delta, abs(row["recon_acc"] - full["recon_acc"]))
        if _is_finite(row["length_acc"]) and _is_finite(full["length_acc"]):
            max_acc_delta = max(max_acc_delta, abs(row["length_acc"] - full["length_acc"]))
    weak = len(probes) == 3 and max_loss_delta <= loss_eps and max_acc_delta <= acc_eps
    return "weak" if weak else "visible", {
        "max_probe_recon_loss_delta": max_loss_delta,
        "max_probe_acc_delta": max_acc_delta,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    results = report["results"]
    full = results["full"]
    effect = report["memory_effect"]
    effect_stats = report["memory_effect_stats"]
    lines: List[str] = []
    lines.append("# FLUED v3.1 Language Codec Memory Ablation")
    lines.append("")
    lines.append(f"- checkpoint: `{report['checkpoint_path']}`")
    lines.append(f"- checkpoint_step: `{report['checkpoint_step']}`")
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- eval_data: `{report['eval_config']['data']}`")
    lines.append(f"- batches: `{_fmt_int(full['batches'])}`")
    lines.append(
        "- interface note: `readout` is the external latent interface; "
        "`summary` and `memory` are internal FLUED encoder mechanisms, and "
        "the backbone is not evaluated as directly reading memory."
    )
    lines.append(f"- memory_effect: `{effect}`")
    lines.append(
        "- weak_rule: `zero`, `shuffled`, and `stale` are treated as weak if "
        f"max |delta recon_loss| <= {report['weak_loss_eps']} and max "
        f"|delta acc| <= {report['weak_acc_eps']}."
    )
    lines.append(
        f"- observed_probe_max: recon_loss_delta="
        f"`{_fmt(effect_stats['max_probe_recon_loss_delta'])}`, "
        f"acc_delta=`{_fmt(effect_stats['max_probe_acc_delta'])}`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| mode | recon_loss | delta_loss | delta_loss_% | recon_acc | "
        "delta_recon_acc | length_acc | delta_length_acc | slots | units |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in report["modes"]:
        row = results[mode]
        loss_delta = row["recon_loss"] - full["recon_loss"]
        if _is_finite(row["recon_loss"]) and _is_finite(full["recon_loss"]) and full["recon_loss"] != 0:
            loss_pct = 100.0 * loss_delta / full["recon_loss"]
        else:
            loss_pct = float("nan")
        recon_acc_delta = row["recon_acc"] - full["recon_acc"]
        length_acc_delta = row["length_acc"] - full["length_acc"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    _fmt(row["recon_loss"]),
                    _fmt_delta(loss_delta),
                    _fmt_delta(loss_pct),
                    _fmt(row["recon_acc"]),
                    _fmt_delta(recon_acc_delta),
                    _fmt(row["length_acc"]),
                    _fmt_delta(length_acc_delta),
                    _fmt_int(row["target_slots"]),
                    _fmt_int(row["length_units"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Mode Definitions")
    lines.append("")
    lines.append("- `full`: original causal summary memory.")
    lines.append("- `zero`: causal summary memory replaced with zeros before `readout_head`.")
    lines.append("- `shuffled`: active memory vectors randomly permuted across the same eval batch.")
    lines.append("- `stale`: memory copied from the previous eval batch; the first batch uses zeros.")
    lines.append(
        "- `summary_detached`: memory is built from `summary.detach()`. In no-grad eval "
        "this should match `full` up to numerical noise; it mainly documents the "
        "training-gradient interface."
    )
    return "\n".join(lines)


@torch.no_grad()
def evaluate_memory_modes(
    model: V31LanguageCodec2M,
    loader: DataLoader,
    modes: Sequence[str],
    eval_config: Mapping[str, Any],
    device: torch.device,
    *,
    amp: bool,
    shuffle_generator: torch.Generator,
) -> Dict[str, Dict[str, float]]:
    totals = {mode: _empty_totals() for mode in modes}
    previous_full_memory: Optional[torch.Tensor] = None
    max_batches = int(eval_config["max_eval_batches"])
    max_span = int(eval_config["max_span"])

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        src, starts, seg_ids, targets, lengths, seg_mask = move_codec_batch(batch, device)
        del starts
        valid = src.ne(PAD_ID)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            full_outputs = forward_with_memory_mode(model, src, valid, seg_ids, seg_mask, "full")
            full_memory = full_outputs[2]["full_memory"].detach()

            for mode in modes:
                if mode == "full":
                    byte_logits, length_logits, _ = full_outputs
                else:
                    byte_logits, length_logits, _ = forward_with_memory_mode(
                        model,
                        src,
                        valid,
                        seg_ids,
                        seg_mask,
                        mode,
                        stale_memory=previous_full_memory,
                        shuffle_generator=shuffle_generator,
                    )
                _update_totals(
                    totals[mode],
                    byte_logits,
                    length_logits,
                    targets,
                    lengths,
                    seg_mask,
                    valid,
                    max_span,
                )

        previous_full_memory = full_memory

    return {mode: _finalize_totals(totals[mode]) for mode in modes}


def run(args: argparse.Namespace) -> str:
    checkpoint_path = _resolve_checkpoint(Path(args.checkpoint))
    ckpt_for_device = _torch_load(checkpoint_path)
    ckpt_args = _dict_or_empty(ckpt_for_device.get("args", {}) if isinstance(ckpt_for_device, Mapping) else {})
    device = _select_device(args.device, ckpt_args)
    model, meta = _load_model(checkpoint_path, device)
    ckpt_args = meta["args"]

    seed = _arg_int(args.seed, ckpt_args, "seed", 1234)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    modes = _parse_modes(args.modes)
    loader, eval_config = _build_loader(args, ckpt_args, device)
    if int(eval_config["max_span"]) != int(model.max_span):
        raise ValueError(
            "--max-span must match the checkpoint model max_span "
            f"({model.max_span}); rebuild the checkpoint to evaluate a different span size"
        )
    amp = bool(_arg_value(args.amp, ckpt_args, "amp", False))
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed + 3101)

    results = evaluate_memory_modes(
        model,
        loader,
        modes,
        eval_config,
        device,
        amp=amp,
        shuffle_generator=shuffle_generator,
    )
    effect, effect_stats = _memory_effect(results, args.weak_loss_eps, args.weak_acc_eps)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": meta.get("step"),
        "device": str(device),
        "eval_config": eval_config,
        "modes": modes,
        "results": results,
        "memory_effect": effect,
        "memory_effect_stats": effect_stats,
        "weak_loss_eps": args.weak_loss_eps,
        "weak_acc_eps": args.weak_acc_eps,
    }
    markdown = _render_markdown(report)
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v3.1 codec causal memory ablations")
    parser.add_argument("--checkpoint", default="checkpoint/latest.pt")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--stream-samples-per-worker", type=int, default=None)
    parser.add_argument("--out-path", default="")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--device", default=None)
    parser.set_defaults(amp=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--eval-max-lines", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-span", type=int, default=None)
    parser.add_argument("--max-span", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--stub-repeats", type=int, default=4)
    parser.add_argument("--weak-loss-eps", type=float, default=0.01)
    parser.add_argument("--weak-acc-eps", type=float, default=0.005)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
