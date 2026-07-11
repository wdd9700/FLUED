"""Benchmark FLUED v3.3 training hot path.

This script intentionally reuses ``tools.train.v3_3.train_v33`` model, loss, optimizer,
and dataloader code.  It measures the optimizer-step path without final eval or
checkpoint I/O so speed numbers are not polluted by archival work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID  # noqa: E402
from tools.train.v3_3.train_v33 import (  # noqa: E402
    LatentInfillBackbone,
    _cosine_with_warmup,
    build_model,
    build_optimizer,
    make_dataloaders,
    step_model,
)


DEFAULTS: Dict[str, object] = {
    "experiment_name": "v33_benchmark",
    "run_id": "",
    "data_path": "",
    "data_manifest": "",
    "streaming_train": False,
    "streaming_eval": False,
    "stream_samples_per_worker": 100000,
    "max_lines": 20000,
    "eval_max_lines": 5000,
    "seq_len": 128,
    "stride": 64,
    "batch_size": 32,
    "max_steps": 200,
    "max_eval_batches": 8,
    "num_workers": 0,
    "prefetch_factor": 4,
    "seed": 42,
    "device": "cuda",
    "amp": False,
    "d_model": 256,
    "d_z": 256,
    "d_mem": 256,
    "hidden": 512,
    "max_chunks": 128,
    "max_span": 16,
    "max_readout_vectors": 1,
    "chunk_mixer": "mean",
    "dropout": 0.0,
    "tau_cut": 0.90,
    "tau_trans": 0.75,
    "tau_keep": 0.65,
    "use_memory": False,
    "memory_rank": 0,
    "memory_top_k": 4,
    "memory_build_mode": "parallel_local",
    "memory_visibility": "bidirectional_no_self",
    "strict_masked_source": True,
    "mask_prob": 0.15,
    "mask_span_min": 1,
    "mask_span_max": 8,
    "visible_loss_weight": 0.25,
    "masked_loss_weight": 1.0,
    "length_loss_weight": 0.05,
    "boundary_loss_weight": 0.02,
    "boundary_credit_loss_weight": 0.0,
    "boundary_credit_backbone_weight": 1.0,
    "rate_loss_weight": 0.02,
    "rate_loss": "upper",
    "target_rate": 0.50,
    "coding_rate_loss_weight": 0.0,
    "memory_vector_overlap_loss_weight": 0.0,
    "memory_gate_loss_weight": 0.0,
    "use_backbone": False,
    "backbone_loss_weight": 0.0,
    "active_only_backbone": True,
    "emit_compute_mode": "all",
    "emit_threshold": 0.5,
    "backbone_hidden": 192,
    "backbone_layers": 2,
    "backbone_nhead": 4,
    "backbone_ffn_dim": 768,
    "detach_backbone_input": True,
    "detach_backbone_keep": True,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "warmup_steps": 50,
    "grad_clip": 1.0,
    "log_every": 20,
    "ckpt_every": 500,
    "grad_accum_steps": 1,
    "resume": False,
    "dry_run": False,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(start: float, device: torch.device) -> float:
    _sync(device)
    return time.perf_counter() - start


def _load_args(path: str) -> SimpleNamespace:
    values = dict(DEFAULTS)
    if path:
        values.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return SimpleNamespace(**values)


def _override(args: SimpleNamespace, cli: argparse.Namespace) -> SimpleNamespace:
    for key in (
        "optimizer",
        "batch_size",
        "grad_accum_steps",
        "seq_len",
        "max_chunks",
        "max_span",
        "max_readout_vectors",
        "chunk_mixer",
        "memory_build_mode",
        "memory_visibility",
        "emit_compute_mode",
        "emit_threshold",
        "num_workers",
        "prefetch_factor",
    ):
        value = getattr(cli, key)
        if value is not None:
            setattr(args, key, value)
    if cli.no_backbone:
        args.use_backbone = False
        args.backbone_loss_weight = 0.0
    if cli.full_backbone:
        args.active_only_backbone = False
    if cli.backbone_layers is not None:
        args.backbone_layers = cli.backbone_layers
    if cli.backbone_hidden is not None:
        args.backbone_hidden = cli.backbone_hidden
    if cli.amp is not None:
        args.amp = cli.amp
    return args


def _make_synthetic_batch(args: SimpleNamespace) -> tuple[torch.Tensor]:
    # Keep this on CPU; step_model performs the same non_blocking host->device
    # transfer as the real dataloader path.
    ids = torch.randint(1, 257, (int(args.batch_size), int(args.seq_len)), dtype=torch.long)
    ids[:, -1] = PAD_ID
    return (ids,)


def _mean(rows: Iterable[float]) -> float:
    vals = list(rows)
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FLUED v3.3 training step cost")
    parser.add_argument("--config", default="configs/v3_3/v33_full_300m_100m_corpus_v4.json")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer", choices=["adamw", "fused_adamw", "foreach_adamw"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-span", type=int, default=None)
    parser.add_argument("--max-readout-vectors", type=int, default=None)
    parser.add_argument("--chunk-mixer", choices=["mean", "delta_lite"], default=None)
    parser.add_argument("--memory-build-mode", choices=["causal_current", "parallel_local"], default=None)
    parser.add_argument("--memory-visibility", choices=["past_only", "bidirectional_no_self", "all_visible"], default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--backbone-layers", type=int, default=None)
    parser.add_argument("--backbone-hidden", type=int, default=None)
    parser.add_argument("--no-backbone", action="store_true")
    parser.add_argument("--full-backbone", action="store_true", help="Disable active-only packing and run all padded readout positions.")
    parser.add_argument("--emit-compute-mode", choices=["all", "emitted", "masked_or_emitted"], default=None)
    parser.add_argument("--emit-threshold", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--out-json", default="")
    cli = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = _override(_load_args(cli.config), cli)
    args.max_steps = max(args.max_steps, cli.steps + cli.warmup + 1)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    backbone = None
    if args.use_backbone:
        backbone = LatentInfillBackbone(
            args.d_z,
            args.backbone_hidden,
            args.backbone_layers,
            args.backbone_nhead,
            args.backbone_ffn_dim,
            args.max_chunks * args.max_readout_vectors,
            args.dropout,
        ).to(device)
    model.train()
    if backbone is not None:
        backbone.train()

    train_loader = None
    train_iter = None
    if not cli.synthetic:
        train_loader, _eval_loader = make_dataloaders(args)
        train_iter = iter(train_loader)
    opt_params = list(model.parameters()) + (list(backbone.parameters()) if backbone is not None else [])
    opt = build_optimizer(args, opt_params)
    sched = _cosine_with_warmup(opt, args.warmup_steps, args.max_steps)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    totals = {name: [] for name in ("data", "forward_loss", "backward", "clip", "optimizer", "step_total")}
    measured = 0
    total_iters = cli.warmup + cli.steps
    for idx in range(total_iters):
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        data_time = 0.0
        fw_time = 0.0
        bw_time = 0.0
        for _micro in range(max(1, int(args.grad_accum_steps))):
            start = time.perf_counter()
            if cli.synthetic:
                batch = _make_synthetic_batch(args)
            else:
                assert train_iter is not None
                try:
                    batch = next(train_iter)
                except StopIteration:
                    assert train_loader is not None
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
            data_time += _elapsed(start, device)

            start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.amp) and device.type == "cuda"):
                loss, _metrics = step_model(model, backbone, batch, args, device, collect_metrics=False)
            fw_time += _elapsed(start, device)

            start = time.perf_counter()
            (loss / max(1, int(args.grad_accum_steps))).backward()
            bw_time += _elapsed(start, device)

        start = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(opt_params, args.grad_clip)
        clip_time = _elapsed(start, device)

        start = time.perf_counter()
        opt.step()
        sched.step()
        opt_time = _elapsed(start, device)
        total_time = _elapsed(step_start, device)

        if idx >= cli.warmup:
            measured += 1
            totals["data"].append(data_time)
            totals["forward_loss"].append(fw_time)
            totals["backward"].append(bw_time)
            totals["clip"].append(clip_time)
            totals["optimizer"].append(opt_time)
            totals["step_total"].append(total_time)
            print(
                f"bench_step={measured} total={total_time:.3f}s "
                f"data={data_time:.3f}s fw={fw_time:.3f}s bw={bw_time:.3f}s "
                f"clip={clip_time:.3f}s opt={opt_time:.3f}s",
                flush=True,
            )

    mean_total = _mean(totals["step_total"])
    result = {
        "config": cli.config,
        "synthetic": bool(cli.synthetic),
        "optimizer": args.optimizer,
        "amp": bool(args.amp),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "seq_len": args.seq_len,
        "max_chunks": args.max_chunks,
        "max_span": args.max_span,
        "max_readout_vectors": args.max_readout_vectors,
        "chunk_mixer": args.chunk_mixer,
        "memory_build_mode": args.memory_build_mode,
        "memory_visibility": args.memory_visibility,
        "use_memory": bool(args.use_memory),
        "use_backbone": bool(args.use_backbone),
        "active_only_backbone": bool(args.active_only_backbone),
        "emit_compute_mode": args.emit_compute_mode,
        "emit_threshold": args.emit_threshold,
        "backbone_layers": args.backbone_layers,
        "backbone_hidden": args.backbone_hidden,
        "measured_steps": measured,
        "mean_step_sec": mean_total,
        "steps_per_sec": 1.0 / mean_total if mean_total > 0 else 0.0,
        "mean_data_sec": _mean(totals["data"]),
        "mean_forward_loss_sec": _mean(totals["forward_loss"]),
        "mean_backward_sec": _mean(totals["backward"]),
        "mean_clip_sec": _mean(totals["clip"]),
        "mean_optimizer_sec": _mean(totals["optimizer"]),
        "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if cli.out_json:
        Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
