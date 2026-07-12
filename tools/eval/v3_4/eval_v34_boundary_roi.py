"""CPU boundary/ROI audit for the FLUED v3.4 probe.

This is an inference-only diagnostic.  It never trains, mutates a checkpoint,
or writes outside the requested output directory.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID, text_to_byte_ids  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import apply_boundary_curriculum, build_model  # noqa: E402


DEFAULT_MODEL = {
    "d_model": 64,
    "nhead": 4,
    "ffn_dim": 128,
    "segmentor_layers": 1,
    "interpreter_layers": 1,
    "memory_rank": 2,
    "readout_vectors": 4,
    "ar_hidden": 16,
    "use_position": True,
    "position_strategy": "layered_rope",
    "prompt_position_scale": 0.1,
    "use_ar": True,
    "use_structured_lookup": True,
    "use_memory": True,
    "use_boundary_bridge": False,
    "memory_use_position": True,
    "memory_residual_scale": 0.1,
    "boundary_mode": "threshold",
    "boundary_coding_rate_dim": 16,
    "boundary_coding_rate_epsilon": 1.0,
    "boundary_coding_rate_temperature": 0.15,
    "boundary_coding_rate_mode": "exact",
    "fixed_chunk_budget": 0,
    "bytes_per_chunk_budget": 16,
    "use_emit_controller": True,
    "emit_forward_mode": "hard_st",
    "emit_initial_probability": 0.1,
    "emit_threshold": 0.5,
    "max_chunks": 32,
    "max_span": 32,
    "tau_cut": 0.9,
    "tau_trans": 0.75,
    "boundary_temperature": 0.15,
    "noise_scale": 0.0,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_dir():
        path = path / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_model(checkpoint_text: str, model_args: dict[str, Any], device: torch.device):
    args = dict(DEFAULT_MODEL)
    args.update(model_args)
    metadata: dict[str, Any] = {"source": "deterministic_random_init", "checkpoint": None}
    runtime_boundary_state = None
    checkpoint_step = 0
    if checkpoint_text:
        checkpoint = _resolve_checkpoint(checkpoint_text)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_args = dict(payload.get("args", {})) if isinstance(payload, dict) else {}
        args.update(saved_args)
        metadata = {
            "source": "checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_step": payload.get("step") if isinstance(payload, dict) else None,
        }
        checkpoint_step = int(payload.get("step", 0)) if isinstance(payload, dict) else 0
        runtime_boundary_state = payload.get("runtime_boundary_state") if isinstance(payload, dict) else None
        state = payload.get("model", payload.get("model_state_dict")) if isinstance(payload, dict) else None
        if state is None and isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
            state = payload
        if state is None:
            raise KeyError("checkpoint must contain 'model' or 'model_state_dict'")
    model = build_model(Namespace(**args)).to(device).eval()
    if checkpoint_text:
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [name for name in unexpected if name != "logic_transition_prior"]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
        if runtime_boundary_state:
            model.config.boundary_mode = str(runtime_boundary_state["mode"])
            model.config.coding_rate_mode = str(runtime_boundary_state["coding_rate_mode"])
            model.config.boundary_blend_alpha = float(runtime_boundary_state.get("blend_alpha", 1.0))
            model.coding_rate_selector.mode = model.config.coding_rate_mode
        else:
            apply_boundary_curriculum(model, Namespace(**args), checkpoint_step)
        metadata["runtime_boundary_state"] = {
            "mode": model.config.boundary_mode,
            "coding_rate_mode": model.coding_rate_selector.mode,
            "blend_alpha": float(model.config.boundary_blend_alpha),
            "restored_from": "checkpoint" if runtime_boundary_state else "curriculum_step_fallback",
        }
    return model, args, metadata


def _byte_char_labels(text: str) -> list[str]:
    """Map original UTF-8 byte offsets to display labels without re-decoding.

    The source string is encoded once, then each character owns its exact
    original byte span.  This remains aligned when a caller later slices the
    raw byte array in the middle of a multi-byte character.
    """
    encoded_text = text.encode("utf-8")
    labels = ["↳"] * len(encoded_text)
    offset = 0
    for char in text:
        encoded = char.encode("utf-8")
        labels[offset] = char
        offset += len(encoded)
    return labels


def _safe_float(value: torch.Tensor, index: int) -> float:
    return float(value[index].detach().float().cpu().item())


@torch.no_grad()
def inspect_case(model, case: dict[str, Any], seq_len: int, device: torch.device) -> dict[str, Any]:
    text = str(case["text"])
    raw_bytes = list(text.encode("utf-8"))
    original_length = len(raw_bytes)
    used_bytes = raw_bytes[:seq_len]
    truncated = original_length > len(used_bytes)
    used_text = bytes(used_bytes).decode("utf-8", errors="replace")
    ids = [byte + 1 for byte in used_bytes]
    padded = ids + [PAD_ID] * (seq_len - len(ids))
    token_ids = torch.tensor([padded], dtype=torch.long, device=device)
    out = model(token_ids)

    valid_count = len(ids)
    valid = token_ids[0, :valid_count].ne(PAD_ID)
    confidence = out.segmentor.confidence[0, :valid_count]
    raw = token_ids[0, :valid_count] - 1
    utf8_cont = raw.ge(0x80) & raw.le(0xBF) & valid
    requested = out.aux["requested_hard_cut"][0, :valid_count].bool()
    model_hard = out.policy.hard_cut[0, :valid_count].bool()
    logic = out.policy.soft_transition[0, :valid_count].bool()
    chunk_ids = out.chunks.chunk_ids[0, :valid_count]
    offsets = out.chunks.offsets[0, :valid_count]
    chunk_start = chunk_ids.ge(0) & offsets.eq(0)
    first = chunk_start.nonzero(as_tuple=False).flatten()[0].item() if chunk_start.any() else 0
    forced_max_span = chunk_start & ~model_hard & torch.arange(valid_count, device=device).ne(first)
    force_continue = out.policy.force_continue[0, :valid_count].bool()
    # Slice labels generated from the original text's byte offsets.  Do not
    # decode the truncated prefix and encode it again: errors=replace would
    # change one partial code point into a different byte span.
    labels = _byte_char_labels(text)[:valid_count]

    rates = out.aux["marginal_coding_rate"][0, :valid_count]
    bytes_out = []
    for index, byte in enumerate(used_bytes):
        bytes_out.append(
            {
                "index": index,
                "raw_byte": byte,
                "hex": f"0x{byte:02x}",
                "char": labels[index],
                "signed_confidence": _safe_float(confidence, index),
                "requested_model_boundary": bool(requested[index].item()),
                "model_hard_boundary": bool(model_hard[index].item()),
                "hard_chunk_boundary": bool(chunk_start[index].item()),
                "logic_transition": bool(logic[index].item()),
                "utf8_continuation": bool(utf8_cont[index].item()),
                "force_continue": bool(force_continue[index].item()),
                "forced_max_span_boundary": bool(forced_max_span[index].item()),
                "chunk_id": int(chunk_ids[index].item()),
                "chunk_offset": int(offsets[index].item()),
                "marginal_coding_rate": _safe_float(rates, index),
            }
        )

    chunk_mask = out.chunks.chunk_mask[0]
    active_chunks = torch.where(chunk_mask)[0].tolist()
    per_chunk = []
    for chunk_id in active_chunks:
        token_mask = out.chunks.token_mask[0, chunk_id]
        length = int(token_mask.sum().item())
        emit_soft = out.emit_soft[0, chunk_id]
        emit_hard = out.emit_hard[0, chunk_id]
        per_chunk.append(
            {
                "chunk_id": int(chunk_id),
                "byte_length": length,
                "readout_candidates": int(out.readout_candidates.size(2)),
                "readout_slots_soft": float(emit_soft.float().sum().item()),
                "readout_slots_hard": int(emit_hard.sum().item()),
                "readout_slot_mask": [bool(v) for v in out.emit_hard[0, chunk_id].cpu().tolist()],
            }
        )

    active_readout = int(out.emit_hard[0, chunk_mask].sum().item()) if chunk_mask.any() else 0
    soft_readout = float(out.emit_soft[0, chunk_mask].float().sum().item()) if chunk_mask.any() else 0.0
    target_chunks = out.aux.get("target_chunks", chunk_mask.sum().view(1))[0]
    config = model.config
    result = {
        "id": case.get("id", "unnamed"),
        "title": case.get("title", case.get("id", "unnamed")),
        "category": case.get("category", "uncategorized"),
        "pair_id": case.get("pair_id"),
        "variant": case.get("variant"),
        "tags": case.get("tags", []),
        "audit_targets": case.get("audit_targets", []),
        "text": used_text,
        "original_text": text if not truncated else None,
        "byte_length": valid_count,
        "original_byte_length": original_length,
        "truncated_to_seq_len": truncated,
        "bytes": bytes_out,
        "chunks": per_chunk,
        "summary": {
            "model_boundary_count": int(model_hard.sum().item()),
            "requested_model_boundary_count": int(requested.sum().item()),
            "hard_chunk_boundary_count": int(chunk_start.sum().item()),
            "logic_transition_count": int(logic.sum().item()),
            "utf8_continuation_count": int(utf8_cont.sum().item()),
            "forced_max_span_boundary_count": int(forced_max_span.sum().item()),
            "force_continue_count": int(force_continue.sum().item()),
            "active_chunk_count": len(active_chunks),
            "mean_chunk_bytes": valid_count / max(len(active_chunks), 1),
            "active_readout_slots_hard": active_readout,
            "active_readout_slots_soft": soft_readout,
            "readout_slots_per_byte_hard": active_readout / max(valid_count, 1),
            "readout_slots_per_byte_soft": soft_readout / max(valid_count, 1),
            "mean_signed_confidence": float(confidence.float().mean().item()) if valid_count else 0.0,
            "target_chunks": int(target_chunks.item()) if torch.is_tensor(target_chunks) else int(target_chunks),
        },
        "budget": {
            "max_chunks": int(config.max_chunks),
            "max_span": int(config.max_span),
            "fixed_chunk_budget": int(config.fixed_chunk_budget),
            "bytes_per_chunk_budget": int(config.bytes_per_chunk_budget),
            "configured_readout_vectors": int(config.readout_vectors),
            "max_readout_slots": int(config.max_chunks * config.readout_vectors),
            "actual_readout_slots_hard": active_readout,
            "actual_readout_slots_soft": soft_readout,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(REPO_ROOT / "configs/v3_4/v34_boundary_roi_cases.json"))
    parser.add_argument("--checkpoint", default="", help="v3.4 latest.pt, step_*.pt, or checkpoint directory")
    parser.add_argument("--output-dir", default="outputs/v34_boundary_roi")
    parser.add_argument("--device", default="cpu", help="cpu by default; no training is performed")
    parser.add_argument("--seq-len", type=int, default=0, help="override fixed-case sequence length")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable; use --device cpu")
    device = torch.device(args.device)
    config = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    eval_config = dict(config.get("evaluation", {}))
    seq_len = int(args.seq_len or eval_config.get("seq_len", 512))
    cases = list(config.get("cases", []))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("case config contains no cases")
    torch.manual_seed(int(eval_config.get("seed", 42)))
    model, model_args, checkpoint_meta = load_model(args.checkpoint, config.get("model", {}), device)
    rows = [inspect_case(model, case, seq_len, device) for case in cases]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "FLUED v3.4 boundary ROI / split behavior CPU audit",
        "inference_only": True,
        "device": str(device),
        "cases_file": str(Path(args.cases).resolve()),
        "checkpoint": checkpoint_meta,
        "model_args": model_args,
        "evaluation": {**eval_config, "seq_len": seq_len, "case_count": len(rows)},
        "cases": rows,
    }
    destination = output_dir / "v34_boundary_roi.json"
    destination.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "case_count": len(rows), "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
