"""Read-only FLUED v3.4 position-generalization evaluation.

The checkpoint and corpus are never modified. Quality metrics reuse the v3.4
training evaluation path, while profiling measures the same identity plus
masked-completion inference performed by ``step_model``.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID  # noqa: E402
from flued.v34.model import load_v34_state_dict_compatible  # noqa: E402
from tools.train.v3_3.train_v33 import LatentInfillBackbone, make_dataloaders  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    _ce,
    _mean_masked,
    apply_boundary_curriculum,
    build_model,
    evaluate,
    step_model,
)
from tools.train.v3_3.train_v33 import make_targets  # noqa: E402


PREFIX_CASES = (
    {
        "id": "english",
        "prefix": "Inserted context: the following paragraph is unchanged.\n",
        "body": (
            "The compiler reuses cached latent states when the function name and argument types remain unchanged. "
            "Position-aware decoding should preserve boundaries throughout the repeated paragraph.\n"
        ),
    },
    {
        "id": "chinese",
        "prefix": "插入前缀：下方正文保持不变，用于测试字节偏移后的边界稳定性。\n",
        "body": "语言编码器需要保留字节细节，同时稳定地划分中文标点、术语和重复句段。位置变化不应破坏正文边界。\n",
    },
    {
        "id": "code",
        "prefix": "# inserted preface: implementation below is unchanged\n",
        "body": (
            "def update_cache(key: str, value: Tensor) -> None:\n"
            "    if key in memory_pool:\n"
            "        memory_pool[key] = value.detach()\n"
            "    return memory_pool.get(key)\n\n"
        ),
    },
    {
        "id": "entity_repetition",
        "prefix": "Entity registry preface: ALPHA-42 remains the target entity.\n",
        "body": (
            "ALPHA-42 maps to entity_alpha_42; ALPHA-42 references entity_alpha_42 again. "
            "实体 ALPHA-42 与 entity_alpha_42 重复出现，但每次都指向同一对象。\n"
        ),
    },
)


def _resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_dir():
        path = path / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path.resolve()


def _parse_lengths(text: str) -> list[int]:
    lengths = []
    for item in text.split(","):
        item = item.strip()
        if item:
            lengths.append(int(item))
    lengths = list(dict.fromkeys(lengths))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must be a comma-separated list of positive integers")
    return lengths


def _seed_everything(seed: int, device: torch.device, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _restore_boundary_state(model, args: Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("runtime_boundary_state")
    if state:
        model.config.boundary_mode = str(state["mode"])
        model.config.coding_rate_mode = str(state["coding_rate_mode"])
        model.config.boundary_blend_alpha = float(state.get("blend_alpha", 1.0))
        model.coding_rate_selector.mode = model.config.coding_rate_mode
        source = "checkpoint"
    else:
        apply_boundary_curriculum(model, args, int(payload.get("step", 0)))
        source = "curriculum_step_fallback"
    # step_model branches on args while the encoder branches on model.config.
    args.boundary_mode = model.config.boundary_mode
    args.boundary_coding_rate_mode = model.coding_rate_selector.mode
    args.boundary_blend_alpha = float(model.config.boundary_blend_alpha)
    return {
        "mode": model.config.boundary_mode,
        "coding_rate_mode": model.coding_rate_selector.mode,
        "blend_alpha": float(model.config.boundary_blend_alpha),
        "restored_from": source,
    }


def _load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[Any, Any, Namespace, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "args" not in payload or "model" not in payload or "backbone" not in payload:
        raise KeyError("v3.4 checkpoint must contain args, model, and backbone")
    args = Namespace(**dict(payload["args"]))
    if not hasattr(args, "emit_threshold"):
        args.emit_threshold = 0.5

    model = build_model(args)
    compatibility = load_v34_state_dict_compatible(model, payload["model"])
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    )
    backbone.load_state_dict(payload["backbone"])
    boundary_state = _restore_boundary_state(model, args, payload)
    model.to(device).eval().requires_grad_(False)
    backbone.to(device).eval().requires_grad_(False)
    metadata = {
        "path": str(checkpoint),
        "step": int(payload.get("step", 0)),
        "boundary_state": boundary_state,
        "checkpoint_compatibility": compatibility,
    }
    return model, backbone, args, metadata


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _actual_chunk_starts(out, valid: torch.Tensor) -> torch.Tensor:
    return out.chunks.chunk_ids.ge(0) & out.chunks.offsets.eq(0) & valid


def _utf8_counts(clean: torch.Tensor, out) -> dict[str, int]:
    valid = clean.ne(PAD_ID)
    raw = (clean - 1).clamp(min=0, max=255)
    continuation = raw.ge(0x80) & raw.le(0xBF) & valid
    not_sequence_start = torch.arange(clean.size(1), device=clean.device).view(1, -1).ne(0)
    policy_violation = out.policy.hard_cut & continuation
    chunk_violation = _actual_chunk_starts(out, valid) & continuation & not_sequence_start
    return {
        "valid_bytes": int(valid.sum().item()),
        "utf8_continuation_bytes": int(continuation.sum().item()),
        "utf8_policy_violation_count": int(policy_violation.sum().item()),
        "utf8_chunk_boundary_violation_count": int(chunk_violation.sum().item()),
        "actual_chunk_count": int(_actual_chunk_starts(out, valid).sum().item()),
    }


def _sum_counts(rows: Iterable[dict[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            result[key] = result.get(key, 0) + int(value)
    return result


def _set_codec_capacity(model, seq_len: int) -> int:
    target = math.ceil(seq_len / max(int(model.config.bytes_per_chunk_budget), 1))
    forced_margin = math.ceil(seq_len / max(int(model.config.max_span), 1))
    max_chunks = max(int(model.config.max_chunks), target + forced_margin + 1)
    model.config.max_chunks = max_chunks
    model.chunk_builder.max_chunks = max_chunks
    model.boundary_bridge.max_chunks = max_chunks
    return max_chunks


@torch.inference_mode()
def _evaluate_long_codec(model, loader, args: Namespace, device: torch.device, amp: bool) -> dict[str, Any]:
    rows = []
    elapsed = 0.0
    valid_bytes = 0
    counts = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for index, batch in enumerate(loader):
        if index >= args.max_eval_batches:
            break
        clean = batch[0].to(device, non_blocking=device.type == "cuda")
        valid = clean.ne(PAD_ID)
        _sync(device)
        started = time.perf_counter()
        with _autocast(device, amp):
            out = model(clean)
        _sync(device)
        elapsed += time.perf_counter() - started
        zero_mask = torch.zeros_like(valid)
        targets, slot_mask, _ = make_targets(
            clean,
            zero_mask,
            out.chunks.chunk_ids,
            out.chunks.offsets,
            model.config.max_chunks,
            model.config.max_span,
        )
        ce = _ce(out.byte_logits, targets)
        pred = out.byte_logits.argmax(dim=-1)
        active = out.emit_hard if args.use_emit_controller else out.chunks.chunk_mask.unsqueeze(-1).expand_as(out.emit_hard)
        rows.append(
            {
                "identity_loss": float(_mean_masked(ce, slot_mask).item()),
                "identity_acc": float((pred[slot_mask] == targets[slot_mask]).float().mean().item()),
                "actual_latents_per_byte": float(active.float().sum().item() / valid.sum().clamp(min=1).item()),
                "soft_latents_per_byte": float((out.emit_soft * out.chunks.chunk_mask.unsqueeze(-1)).sum().item() / valid.sum().clamp(min=1).item()),
                "chunks_per_byte": float(out.chunks.chunk_mask.float().sum().item() / valid.sum().clamp(min=1).item()),
                "truncated_tokens": float(out.chunks.pack_info["truncated_tokens"].float().mean().item()),
            }
        )
        valid_bytes += int(valid.sum().item())
        counts.append(_utf8_counts(clean, out))
    count_totals = _sum_counts(counts)
    mean = lambda key: sum(row[key] for row in rows) / max(len(rows), 1)
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "codec_identity_accuracy": mean("identity_acc"),
        "codec_identity_loss": mean("identity_loss"),
        "actual_latents_per_byte": mean("actual_latents_per_byte"),
        "soft_latents_per_byte": mean("soft_latents_per_byte"),
        "chunks_per_byte": mean("chunks_per_byte"),
        "truncated_tokens": mean("truncated_tokens"),
        "profile": {
            "timed_batches": len(rows),
            "timed_valid_bytes": valid_bytes,
            "elapsed_sec": elapsed,
            "throughput_bytes_per_sec": valid_bytes / max(elapsed, 1.0e-12),
            "peak_vram_bytes": int(peak_bytes),
            "peak_vram_gib": peak_bytes / (1024**3),
            **count_totals,
        },
    }


@torch.inference_mode()
def _profile_loader(
    model,
    backbone,
    loader,
    args: Namespace,
    device: torch.device,
    seed: int,
    warmup_batches: int,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    backbone.eval()
    if warmup_batches > 0:
        _seed_everything(seed, device, bool(args.deterministic))
        for index, batch in enumerate(loader):
            if index >= warmup_batches:
                break
            with _autocast(device, amp):
                step_model(model, backbone, batch, args, device, collect_metrics=False, global_step=-1)
        _sync(device)

    _seed_everything(seed, device, bool(args.deterministic))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    count_rows = []
    timed_bytes = 0
    timed_sequences = 0
    timed_batches = 0
    elapsed = 0.0
    for index, batch in enumerate(loader):
        if index >= args.max_eval_batches:
            break
        clean = batch[0].to(device, non_blocking=device.type == "cuda")
        _sync(device)
        started = time.perf_counter()
        with _autocast(device, amp):
            step_model(model, backbone, batch, args, device, collect_metrics=False, global_step=-1)
        _sync(device)
        elapsed += time.perf_counter() - started
        timed_bytes += int(clean.ne(PAD_ID).sum().item())
        timed_sequences += int(clean.size(0))
        timed_batches += 1
        with _autocast(device, amp):
            count_rows.append(_utf8_counts(clean, model(clean)))

    counts = _sum_counts(count_rows)
    violations = counts.get("utf8_chunk_boundary_violation_count", 0)
    continuation = counts.get("utf8_continuation_bytes", 0)
    chunks = counts.get("actual_chunk_count", 0)
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "timed_batches": timed_batches,
        "timed_sequences": timed_sequences,
        "timed_valid_bytes": timed_bytes,
        "elapsed_sec": elapsed,
        "throughput_bytes_per_sec": timed_bytes / max(elapsed, 1.0e-12),
        "throughput_sequences_per_sec": timed_sequences / max(elapsed, 1.0e-12),
        "peak_vram_bytes": int(peak_bytes),
        "peak_vram_gib": peak_bytes / (1024**3),
        "peak_vram_supported": device.type == "cuda",
        **counts,
        "utf8_violation_count": violations,
        "utf8_violation_per_continuation_byte": violations / max(continuation, 1),
        "utf8_violation_per_chunk": violations / max(chunks, 1),
    }


def _repeat_bytes(text: str, minimum_length: int) -> bytes:
    encoded = text.encode("utf-8")
    return (encoded * (minimum_length // len(encoded) + 1))[:minimum_length]


def _tokens(raw: bytes, seq_len: int, device: torch.device) -> torch.Tensor:
    used = raw[:seq_len]
    ids = [byte + 1 for byte in used]
    return torch.tensor([ids + [PAD_ID] * (seq_len - len(ids))], dtype=torch.long, device=device)


def _boundary_offsets(out, valid_length: int) -> set[int]:
    valid = torch.arange(out.chunks.chunk_ids.size(1), device=out.chunks.chunk_ids.device).view(1, -1) < valid_length
    return set(torch.where(_actual_chunk_starts(out, valid)[0])[0].cpu().tolist())


def _set_scores(left: set[int], right: set[int]) -> dict[str, float | int]:
    intersection = len(left & right)
    union = len(left | right)
    precision = intersection / max(len(right), 1)
    recall = intersection / max(len(left), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    if not left and not right:
        precision = recall = f1 = 1.0
    return {
        "reference_boundary_count": len(left),
        "prefixed_boundary_count": len(right),
        "intersection_count": intersection,
        "f1": f1,
        "jaccard": intersection / union if union else 1.0,
    }


@torch.inference_mode()
def _prefix_stability(model, lengths: list[int], device: torch.device, amp: bool) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for seq_len in lengths:
        _set_codec_capacity(model, seq_len)
        for case in PREFIX_CASES:
            prefix = str(case["prefix"]).encode("utf-8")
            if len(prefix) >= seq_len:
                raise ValueError(f"prefix for {case['id']} is not shorter than sequence length {seq_len}")
            body = _repeat_bytes(str(case["body"]), seq_len)
            overlap = min(len(body), seq_len - len(prefix))
            with _autocast(device, amp):
                base_out = model(_tokens(body, seq_len, device))
                prefixed_out = model(_tokens(prefix + body, seq_len, device))
            base = {offset for offset in _boundary_offsets(base_out, seq_len) if 0 < offset < overlap}
            shifted = {
                offset - len(prefix)
                for offset in _boundary_offsets(prefixed_out, seq_len)
                if len(prefix) < offset < len(prefix) + overlap
            }
            rows.append(
                {
                    "case": case["id"],
                    "seq_len": seq_len,
                    "prefix_bytes": len(prefix),
                    "overlap_body_bytes": overlap,
                    "alignment": "prefixed boundary byte offset minus prefix byte length",
                    "excluded_overlap_offsets": [0],
                    **_set_scores(base, shifted),
                }
            )
    return rows


def _args_for_length(base: Namespace, cli: argparse.Namespace, seq_len: int) -> Namespace:
    values = dict(vars(base))
    values.update(
        {
            "seq_len": seq_len,
            "stride": max(1, seq_len // 2),
            "max_eval_batches": cli.max_eval_batches,
            "num_workers": 0,
            "seed": cli.eval_seed,
            "device": cli.device,
            "deterministic": cli.deterministic,
        }
    )
    if cli.data_path:
        values["data_path"] = cli.data_path
        values["data_manifest"] = ""
    if cli.data_manifest:
        values["data_manifest"] = cli.data_manifest
        values["data_path"] = ""
    configured_batch = cli.batch_size or int(values.get("batch_size", 1))
    values["batch_size"] = cli.long_batch_size if seq_len > cli.long_length_threshold else configured_batch
    return Namespace(**values)


def run(cli: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    if cli.max_eval_batches <= 0 or cli.long_batch_size <= 0 or cli.batch_size < 0:
        raise ValueError("batch sizes and max-eval-batches must be positive (batch-size may be 0 for checkpoint default)")
    device = torch.device(cli.device)
    lengths = _parse_lengths(cli.lengths)
    _seed_everything(cli.eval_seed, device, cli.deterministic)
    checkpoint = _resolve_checkpoint(cli.checkpoint)
    model, backbone, checkpoint_args, checkpoint_meta = _load_checkpoint(checkpoint, device)

    rows = []
    for seq_len in lengths:
        args = _args_for_length(checkpoint_args, cli, seq_len)
        runtime_max_chunks = _set_codec_capacity(model, seq_len)
        _seed_everything(cli.eval_seed, device, cli.deterministic)
        _, eval_loader = make_dataloaders(args)
        if seq_len <= cli.long_length_threshold:
            with torch.inference_mode(), _autocast(device, cli.amp):
                quality = evaluate(model, backbone, eval_loader, args, device)
            model.eval()
            backbone.eval()
            _seed_everything(cli.eval_seed, device, cli.deterministic)
            _, profile_loader = make_dataloaders(args)
            profile = _profile_loader(
                model, backbone, profile_loader, args, device, cli.eval_seed, cli.warmup_batches, cli.amp
            )
            row = {
                "seq_len": seq_len,
                "evaluation_scope": "codec_and_backbone",
                "runtime_max_chunks": runtime_max_chunks,
                "batch_size": args.batch_size,
                "max_eval_batches": args.max_eval_batches,
                "codec_identity_accuracy": quality["identity_acc"],
                "codec_identity_loss": quality["identity_loss"],
                "masked_completion_accuracy": quality["completion_mask_acc"],
                "masked_completion_loss": quality["completion_masked_loss"],
                "masked_completion_ppl": math.exp(min(20.0, quality["completion_masked_loss"])),
                "masked_completion_preserve_accuracy": quality["completion_preserve_acc"],
                "actual_latents_per_byte": quality["actual_backbone_units_per_byte"],
                "soft_latents_per_byte": quality["soft_readout_units_per_byte"],
                "chunks_per_byte": quality["chunks_per_byte"],
                "truncated_tokens": quality.get("truncated_tokens", 0.0),
                "cut_capacity_overflow": quality.get("cut_capacity_overflow", 0.0),
                "profile": profile,
            }
        else:
            quality = _evaluate_long_codec(model, eval_loader, args, device, cli.amp)
            row = {
                "seq_len": seq_len,
                "evaluation_scope": "segmentor_and_codec_only",
                "runtime_max_chunks": runtime_max_chunks,
                "batch_size": args.batch_size,
                "max_eval_batches": args.max_eval_batches,
                **quality,
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    _seed_everything(cli.eval_seed, device, cli.deterministic)
    stability = _prefix_stability(model, lengths, device, cli.amp)
    aggregate = {
        "mean_f1": sum(float(row["f1"]) for row in stability) / max(len(stability), 1),
        "mean_jaccard": sum(float(row["jaccard"]) for row in stability) / max(len(stability), 1),
    }
    result = {
        "protocol": "FLUED v3.4 read-only position generalization",
        "inference_only": True,
        "checkpoint": checkpoint_meta,
        "device": str(device),
        "eval_seed": cli.eval_seed,
        "deterministic": cli.deterministic,
        "amp": cli.amp and device.type == "cuda",
        "lengths": lengths,
        "quality_and_efficiency": rows,
        "prefix_insertion_boundary_stability": {
            "boundary_definition": "actual chunk starts in overlapping body bytes",
            "aggregate": aggregate,
            "rows": stability,
        },
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "lengths": lengths}, ensure_ascii=False), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="v3.4 latest.pt, step_*.pt, or checkpoint directory")
    parser.add_argument("--output", default="outputs/v34_position_generalization.json")
    parser.add_argument("--lengths", default="512,2048,4096")
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260712)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=0, help="0 keeps checkpoint batch size for short lengths")
    parser.add_argument("--long-batch-size", type=int, default=1)
    parser.add_argument("--long-length-threshold", type=int, default=512)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
