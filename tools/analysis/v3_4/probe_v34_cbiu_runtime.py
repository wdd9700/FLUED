"""Measure hard-policy v3.4 encoder and compact-backbone runtime."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID  # noqa: E402
from tools.analysis.v3_4.probe_v34_cbiu import _restore_runtime_state  # noqa: E402
from tools.train.v3_3.train_v33 import (  # noqa: E402
    LatentInfillBackbone,
    _compact_active_readout,
    _scatter_compact_readout,
    make_byte_mask,
    make_dataloaders,
    make_targets,
)
from tools.train.v3_4.train_v34_pos_ar_probe import (  # noqa: E402
    _run_completion,
    _strict_affected_readouts,
    build_model,
)


def _cuda_time(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("runtime probe requires CUDA")
    device = torch.device("cuda")
    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(device="cuda", num_workers=0, max_eval_batches=1)
    args = Namespace(**config)
    torch.manual_seed(args.eval_mask_seed)
    torch.cuda.manual_seed_all(args.eval_mask_seed)
    torch.set_float32_matmul_precision("high")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    _restore_runtime_state(model, payload)
    backbone = LatentInfillBackbone(
        args.d_model,
        args.backbone_hidden,
        args.backbone_layers,
        args.backbone_nhead,
        args.backbone_ffn_dim,
        args.max_chunks * args.readout_vectors,
        0.0,
    ).to(device)
    backbone.load_state_dict(payload["backbone"], strict=True)
    model.eval()
    backbone.eval()
    _, loader = make_dataloaders(args)
    clean = next(iter(loader))[0].to(device)
    valid = clean.ne(PAD_ID)
    byte_mask = make_byte_mask(valid, args.mask_prob, args.mask_span_min, args.mask_span_max)
    source = clean.masked_fill(byte_mask, MASK_ID)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
        out = model(source)
        _, _, masked_slot = make_targets(
            clean,
            byte_mask,
            out.chunks.chunk_ids,
            out.chunks.offsets,
            args.max_chunks,
            args.max_span,
        )
        active = out.emit_hard.clone()
        active[..., 0] |= out.chunks.chunk_mask
        affected = _strict_affected_readouts(masked_slot, out.chunks.chunk_mask, active)
        bsz, chunks, readouts, dim = out.readout_z.shape
        flat_z = out.readout_z.reshape(bsz, chunks * readouts, dim)
        flat_active = active.reshape(bsz, -1)
        flat_affected = affected.reshape(bsz, -1) & flat_active
        positions = torch.arange(flat_z.size(1), device=device).view(1, -1).expand(bsz, -1)
        compact_z, compact_active, compact_affected, compact_pos = _compact_active_readout(
            flat_z,
            flat_active,
            flat_affected,
            positions,
        )
        predicted_compact = backbone(
            compact_z,
            compact_active,
            compact_affected,
            position_ids=compact_pos,
        )
        predicted_flat = _scatter_compact_readout(
            predicted_compact,
            compact_pos,
            flat_z.size(1),
        )
        predicted = predicted_flat.reshape_as(out.readout_z)
        completed = torch.where((affected & active).unsqueeze(-1), predicted, out.readout_z)

        def encode_only():
            return model(source)

        def compact_backbone_decode():
            return _run_completion(
                model,
                backbone,
                out.readout_z,
                active,
                affected,
                out.chunks.chunk_mask,
            )

        def compact_only():
            return _compact_active_readout(
                flat_z,
                flat_active,
                flat_affected,
                positions,
            )

        def backbone_only():
            return backbone(
                compact_z,
                compact_active,
                compact_affected,
                position_ids=compact_pos,
            )

        def decode_only():
            return model.decode(completed, out.chunks.chunk_mask, active)

        def end_to_end():
            current = model(source)
            current_active = current.emit_hard.clone()
            current_active[..., 0] |= current.chunks.chunk_mask
            current_affected = _strict_affected_readouts(
                masked_slot,
                current.chunks.chunk_mask,
                current_active,
            )
            return _run_completion(
                model,
                backbone,
                current.readout_z,
                current_active,
                current_affected,
                current.chunks.chunk_mask,
            )

        encode_ms = _cuda_time(encode_only, cli.warmup, cli.iterations)
        compact_ms = _cuda_time(compact_only, cli.warmup, cli.iterations)
        backbone_ms = _cuda_time(backbone_only, cli.warmup, cli.iterations)
        decode_ms = _cuda_time(decode_only, cli.warmup, cli.iterations)
        backbone_decode_ms = _cuda_time(compact_backbone_decode, cli.warmup, cli.iterations)
        end_to_end_ms = _cuda_time(end_to_end, cli.warmup, cli.iterations)

    valid_bytes = float(valid.sum().item())
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "batch_size": int(clean.size(0)),
        "sequence_length": int(clean.size(1)),
        "valid_bytes": valid_bytes,
        "policy_readouts_per_byte": float(active.float().sum().item() / valid_bytes),
        "max_active_readouts_per_sample": int(active.reshape(active.size(0), -1).sum(dim=1).max().item()),
        "encode_ms": encode_ms,
        "compact_ms": compact_ms,
        "backbone_ms": backbone_ms,
        "decode_ms": decode_ms,
        "compact_backbone_decode_ms": backbone_decode_ms,
        "end_to_end_ms": end_to_end_ms,
        "bytes_per_second_end_to_end": valid_bytes / (end_to_end_ms / 1000.0),
        "iterations": cli.iterations,
    }
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
