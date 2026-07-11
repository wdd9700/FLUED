"""Audit archived FLUED v3-family checkpoints and evidence.

The goal is not to run a new training/evaluation pass.  This script reads the
archived summaries/logs/checkpoints, normalizes the metric names, and rebuilds
the comparison tables with explicit evidence scopes.  It is intentionally
conservative about cross-version comparisons: metrics from different tasks are
kept in separate buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_ARCHIVE = Path(r"K:\FLUED_archive")
DEFAULT_OUT = DEFAULT_ARCHIVE / "v3_checkpoint_audit_20260703"
V3_ARCHIVE_MARKERS = (
    "v3_",
    "v31_",
    "v32_",
    "smoke_v31",
)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"_load_error": repr(exc)}
    return data if isinstance(data, dict) else {"_json_value": data}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    return str(value)


def _first_number(name: str) -> Optional[int]:
    match = re.search(r"(\d+)k", name.lower())
    if match:
        return int(match.group(1)) * 1000
    match = re.search(r"step0*(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return None


def _family(path: Path) -> str:
    text = str(path).replace("/", "\\").lower()
    if "v32_masked_codec" in text or "v321_" in text:
        return "v3.2.1"
    if "v32_" in text:
        return "v3.2"
    if "v31_" in text or "smoke_v31" in text:
        return "v3.1"
    if "v3_" in text:
        return "v3"
    return "unknown"


def _task_scope(path: Path, summary: Mapping[str, Any]) -> str:
    text = str(path).replace("/", "\\").lower()
    model = str(summary.get("model", summary.get("model_version", ""))).lower()
    task = str(summary.get("task", "")).lower()
    if "strict_masked_source_codec" in task or "v32_masked_codec" in text:
        return "strict_masked_codec"
    if "v32_strict_backbone" in text:
        return "strict_masked_backbone"
    if "v31_backbone" in text or "v32_backbone" in text:
        return "legacy_backbone"
    if "language_codec" in text:
        return "clean_codec"
    if "segmental_diffusion" in text or "segmental_diffusion" in model:
        return "segmental_diffusion"
    if "segmental_workspace" in text or "segmental_latent_workspace" in model:
        return "segmental_workspace"
    if "commit_controller" in text or "commit_controller" in model:
        return "commit_controller"
    return "other"


def _run_name(summary_path: Path, archive: Path) -> str:
    try:
        parent = summary_path.parent.relative_to(archive)
    except ValueError:
        parent = summary_path.parent
    return str(parent).replace("\\", "/")


def _metric(summary: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    last = summary.get("last")
    if isinstance(last, Mapping):
        for key in keys:
            if key in last:
                return last[key]
    return None


def _checkpoint_stats(run_dir: Path) -> Tuple[int, float, str]:
    checkpoints = sorted(run_dir.glob("*.pt"))
    steps: List[str] = []
    total = 0
    for ckpt in checkpoints:
        total += ckpt.stat().st_size
        if ckpt.name == "latest.pt":
            steps.append("latest")
        else:
            step = _first_number(ckpt.name)
            steps.append(str(step) if step is not None else ckpt.stem)
    return len(checkpoints), total / (1024 * 1024), ",".join(steps)


def _summarize_log(log_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not log_rows:
        return {}
    out: Dict[str, Any] = {}
    last = log_rows[-1]
    for key in (
        "loss",
        "eval_loss",
        "masked_recon_acc",
        "eval_masked_recon_acc",
        "mask_byte_acc",
        "eval_mask_byte_acc",
        "recon_acc",
        "eval_recon_acc",
        "boundary_acc",
        "eval_boundary_acc",
        "recent_samples_per_sec",
    ):
        if key in last:
            out[f"log_last_{key}"] = last[key]
    for key in (
        "eval_masked_recon_acc",
        "masked_recon_acc",
        "eval_mask_byte_acc",
        "mask_byte_acc",
        "eval_recon_acc",
        "acc",
    ):
        values = [row[key] for row in log_rows if isinstance(row.get(key), (int, float))]
        if values:
            out[f"log_best_{key}"] = max(values)
    return out


def _normalize_summary(summary_path: Path, archive: Path) -> Dict[str, Any]:
    summary = _load_json(summary_path)
    run_dir = summary_path.parent
    log_summary = _summarize_log(_load_jsonl(run_dir / "train_log.jsonl"))
    ckpt_count, ckpt_mb, ckpt_steps = _checkpoint_stats(run_dir)
    family = _family(summary_path)
    scope = _task_scope(summary_path, summary)
    run = _run_name(summary_path, archive)
    steps = _metric(summary, "steps", "step") or _first_number(run) or log_summary.get("log_last_step")

    row: Dict[str, Any] = {
        "run": run,
        "family": family,
        "scope": scope,
        "model": summary.get("model", summary.get("model_version", "")),
        "task": summary.get("task", ""),
        "steps": steps,
        "params": summary.get("params"),
        "memory_enabled": summary.get("memory_enabled", summary.get("codec_memory_enabled")),
        "memory_mode": summary.get("memory_retrieval_mode", summary.get("codec_memory_retrieval_mode", summary.get("memory_path", ""))),
        "pool_mode": summary.get("codec_pool_mode", summary.get("pool_mode", "")),
        "checkpoint_count": ckpt_count,
        "checkpoint_mb": ckpt_mb,
        "checkpoint_steps": ckpt_steps,
        "summary_path": str(summary_path),
    }
    metric_keys = {
        "codec_recon_acc": ("eval_recon_acc", "deploy_eval_acc", "multi_eval_acc", "eval_acc"),
        "codec_recon_loss": ("eval_recon_loss", "eval_loss", "deploy_eval_loss"),
        "masked_recon_acc": ("eval_masked_recon_acc",),
        "masked_recon_loss": ("eval_masked_recon_loss",),
        "keep_recon_acc": ("eval_keep_recon_acc",),
        "backbone_mask_acc": ("eval_mask_byte_acc",),
        "backbone_mask_loss": ("eval_byte_loss", "eval_loss"),
        "backbone_keep_acc": ("eval_keep_byte_acc",),
        "length_acc": ("eval_length_acc", "deploy_eval_length_acc"),
        "masked_length_acc": ("eval_masked_length_acc", "eval_mask_length_acc"),
        "boundary_acc": ("eval_boundary_acc",),
        "boundary_loss": ("eval_boundary_loss",),
        "units_per_byte": ("eval_units_per_byte",),
        "readout_units_per_byte": ("eval_readout_units_per_byte",),
        "memory_slots_per_byte": ("eval_memory_slots_per_byte",),
        "retrieval_entropy": ("eval_retrieval_entropy",),
        "retrieval_valid_frac": ("eval_retrieval_valid_frac",),
        "memory_context_norm": ("eval_memory_context_norm",),
        "memory_slot_norm": ("eval_memory_slot_norm",),
        "commit_mn": ("eval_commit_mn", "deploy_eval_commit_mn"),
        "commit_std": ("eval_commit_std", "deploy_eval_commit_std"),
        "commit_corr": ("eval_commit_corr", "deploy_eval_commit_corr"),
        "commit_enrich": ("eval_commit_enrich", "deploy_eval_commit_enrich"),
        "future_mask_acc": ("eval_future_mask_acc", "deploy_eval_future_mask_acc"),
        "denoise_acc": ("eval_denoise_acc", "deploy_eval_denoise_acc"),
        "deploy_eval_acc": ("deploy_eval_acc",),
        "multi_eval_acc": ("multi_eval_acc",),
        "train_steps_per_sec": ("train_steps_per_sec",),
        "train_samples_per_sec": ("train_samples_per_sec",),
        "max_memory_mb": ("max_memory_allocated_mb",),
    }
    for out_key, keys in metric_keys.items():
        row[out_key] = _metric(summary, *keys)
    row.update(log_summary)
    return row


def _discover_summaries(archive: Path) -> List[Path]:
    paths = []
    for path in archive.rglob("summary.json"):
        lower = str(path).lower()
        if any(marker in lower for marker in V3_ARCHIVE_MARKERS):
            paths.append(path)
    return sorted(paths)


def _read_memory_stress(archive: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in archive.rglob("memory_stress_15k\\*.json"):
        data = _load_json(path)
        summary = data.get("summary", {})
        if not isinstance(summary, Mapping):
            continue
        full = summary.get("full", {})
        zero = summary.get("zero", {})
        shuffled = summary.get("shuffled", {})
        stale = summary.get("stale", {})
        if not isinstance(full, Mapping):
            full = {}
        row = {
            "file": str(path),
            "memory_enabled": data.get("memory_enabled"),
            "retrieval": data.get("memory_retrieval_mode"),
            "full_acc": _metric(full, "masked_acc"),
            "full_loss": _metric(full, "masked_loss"),
            "zero_loss_delta": _metric(zero if isinstance(zero, Mapping) else {}, "delta_loss_vs_full"),
            "shuffled_loss_delta": _metric(shuffled if isinstance(shuffled, Mapping) else {}, "delta_loss_vs_full"),
            "stale_loss_delta": _metric(stale if isinstance(stale, Mapping) else {}, "delta_loss_vs_full"),
            "zero_acc_delta": _metric(zero if isinstance(zero, Mapping) else {}, "delta_acc_vs_full"),
            "shuffled_acc_delta": _metric(shuffled if isinstance(shuffled, Mapping) else {}, "delta_acc_vs_full"),
            "stale_acc_delta": _metric(stale if isinstance(stale, Mapping) else {}, "delta_acc_vs_full"),
            "case_count": len(data.get("cases", [])) if isinstance(data.get("cases"), list) else 0,
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _best_by(rows: Sequence[Mapping[str, Any]], scope: str, metric: str, reverse: bool = True, limit: int = 8) -> List[Mapping[str, Any]]:
    scoped = [r for r in rows if r.get("scope") == scope and isinstance(r.get(metric), (int, float))]
    return sorted(scoped, key=lambda r: float(r[metric]), reverse=reverse)[:limit]


def _find_run(rows: Sequence[Mapping[str, Any]], needle: str) -> Optional[Mapping[str, Any]]:
    needle_l = needle.lower()
    for row in rows:
        if needle_l in str(row.get("run", "")).lower():
            return row
    return None


def _delta(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]], metric: str) -> Optional[float]:
    if not a or not b:
        return None
    av = a.get(metric)
    bv = b.get(metric)
    if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
        return float(av) - float(bv)
    return None


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(x) for x in row) + " |")
    return "\n".join(out)


def _report(rows: Sequence[Mapping[str, Any]], stress_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> str:
    by_scope: Dict[str, int] = defaultdict(int)
    by_family: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_scope[str(row.get("scope"))] += 1
        by_family[str(row.get("family"))] += 1

    strict_byte_15k = _find_run(rows, "v32_strict_backbone_20260703_masked_codec_15k/byte_3k_strict_mask_recheck")
    strict_nomem_15k = _find_run(rows, "latent_v321_mfl_nomemory_maskedcodec15k_3k")
    strict_mem_15k = _find_run(rows, "latent_v321_mfl_memory_maskedcodec15k_3k")
    strict_byte_old = _find_run(rows, "v32_strict_backbone_20260703/byte_3k_strict_mask")
    strict_nomem_old = _find_run(rows, "latent_v32_mfl_nomemory_3k_strict_fast")
    strict_mem_old = _find_run(rows, "latent_v32_mfl_memory_3k_strict_fast")

    masked_mem_15k = _find_run(rows, "v321_mfl_memory_masked_15k")
    masked_nomem_15k = _find_run(rows, "v321_mfl_nomemory_masked_15k")
    masked_mem_3k = _find_run(rows, "v321_mfl_memory_masked_3k")
    masked_nomem_3k = _find_run(rows, "v321_mfl_nomemory_masked_3k")
    masked_random_3k = _find_run(rows, "v321_mfl_random_masked_3k")

    clean_v31 = _find_run(rows, "v31_language_codec_2m_20260702/codec_40k_utf8clean")
    clean_v31_mfl = _find_run(rows, "v31_language_codec_2m_20260702/codec_10k_pool_mfl")
    clean_v32_mem = _find_run(rows, "stage3_v32_mfl_memory_10k")
    clean_v32_nomem = _find_run(rows, "stage3_v32_mfl_nomemory_10k")
    clean_v32_random = _find_run(rows, "stage3_v32_mfl_random_10k")

    parts: List[str] = []
    parts.append("# FLUED v3 checkpoint evidence audit")
    parts.append("")
    parts.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    parts.append(f"- Output directory: `{out_dir}`")
    parts.append(f"- Runs with `summary.json`: `{len(rows)}`")
    parts.append(f"- Memory stress files: `{len(stress_rows)}`")
    parts.append("")
    parts.append("## Coverage")
    parts.append("")
    parts.append(_table(["family", "run_count"], sorted(by_family.items())))
    parts.append("")
    parts.append(_table(["scope", "run_count"], sorted(by_scope.items())))
    parts.append("")

    parts.append("## Same-scope comparisons")
    parts.append("")
    parts.append("### Strict masked-source backbone")
    parts.append("")
    parts.append(_table(
        ["run", "mask_acc", "mask_loss", "keep_acc", "codec_memory", "codec_pool"],
        [
            ["old byte baseline", strict_byte_old and strict_byte_old.get("backbone_mask_acc"), strict_byte_old and strict_byte_old.get("backbone_mask_loss"), strict_byte_old and strict_byte_old.get("backbone_keep_acc"), "-", "-"],
            ["old v3.2 no-memory", strict_nomem_old and strict_nomem_old.get("backbone_mask_acc"), strict_nomem_old and strict_nomem_old.get("backbone_mask_loss"), strict_nomem_old and strict_nomem_old.get("backbone_keep_acc"), strict_nomem_old and strict_nomem_old.get("memory_enabled"), strict_nomem_old and strict_nomem_old.get("pool_mode")],
            ["old v3.2 top-k memory", strict_mem_old and strict_mem_old.get("backbone_mask_acc"), strict_mem_old and strict_mem_old.get("backbone_mask_loss"), strict_mem_old and strict_mem_old.get("backbone_keep_acc"), strict_mem_old and strict_mem_old.get("memory_enabled"), strict_mem_old and strict_mem_old.get("pool_mode")],
            ["v3.2.1 byte baseline", strict_byte_15k and strict_byte_15k.get("backbone_mask_acc"), strict_byte_15k and strict_byte_15k.get("backbone_mask_loss"), strict_byte_15k and strict_byte_15k.get("backbone_keep_acc"), "-", "-"],
            ["v3.2.1 no-memory", strict_nomem_15k and strict_nomem_15k.get("backbone_mask_acc"), strict_nomem_15k and strict_nomem_15k.get("backbone_mask_loss"), strict_nomem_15k and strict_nomem_15k.get("backbone_keep_acc"), strict_nomem_15k and strict_nomem_15k.get("memory_enabled"), strict_nomem_15k and strict_nomem_15k.get("pool_mode")],
            ["v3.2.1 top-k memory", strict_mem_15k and strict_mem_15k.get("backbone_mask_acc"), strict_mem_15k and strict_mem_15k.get("backbone_mask_loss"), strict_mem_15k and strict_mem_15k.get("backbone_keep_acc"), strict_mem_15k and strict_mem_15k.get("memory_enabled"), strict_mem_15k and strict_mem_15k.get("pool_mode")],
        ],
    ))
    parts.append("")
    parts.append("- v3.2.1 no-memory vs byte `mask_acc` delta: " + _fmt(_delta(strict_nomem_15k, strict_byte_15k, "backbone_mask_acc")))
    parts.append("- v3.2.1 top-k memory vs no-memory `mask_acc` delta: " + _fmt(_delta(strict_mem_15k, strict_nomem_15k, "backbone_mask_acc")))
    parts.append("")

    parts.append("### Strict masked-source codec")
    parts.append("")
    parts.append(_table(
        ["run", "steps", "masked_recon_acc", "keep_recon_acc", "length_acc", "boundary_acc", "retrieval_entropy", "mem_slots/B"],
        [
            ["v3.2.1 memory 3k", masked_mem_3k and masked_mem_3k.get("steps"), masked_mem_3k and masked_mem_3k.get("masked_recon_acc"), masked_mem_3k and masked_mem_3k.get("keep_recon_acc"), masked_mem_3k and masked_mem_3k.get("length_acc"), masked_mem_3k and masked_mem_3k.get("boundary_acc"), masked_mem_3k and masked_mem_3k.get("retrieval_entropy"), masked_mem_3k and masked_mem_3k.get("memory_slots_per_byte")],
            ["v3.2.1 no-memory 3k", masked_nomem_3k and masked_nomem_3k.get("steps"), masked_nomem_3k and masked_nomem_3k.get("masked_recon_acc"), masked_nomem_3k and masked_nomem_3k.get("keep_recon_acc"), masked_nomem_3k and masked_nomem_3k.get("length_acc"), masked_nomem_3k and masked_nomem_3k.get("boundary_acc"), masked_nomem_3k and masked_nomem_3k.get("retrieval_entropy"), masked_nomem_3k and masked_nomem_3k.get("memory_slots_per_byte")],
            ["v3.2.1 random 3k", masked_random_3k and masked_random_3k.get("steps"), masked_random_3k and masked_random_3k.get("masked_recon_acc"), masked_random_3k and masked_random_3k.get("keep_recon_acc"), masked_random_3k and masked_random_3k.get("length_acc"), masked_random_3k and masked_random_3k.get("boundary_acc"), masked_random_3k and masked_random_3k.get("retrieval_entropy"), masked_random_3k and masked_random_3k.get("memory_slots_per_byte")],
            ["v3.2.1 memory 15k", masked_mem_15k and masked_mem_15k.get("steps"), masked_mem_15k and masked_mem_15k.get("masked_recon_acc"), masked_mem_15k and masked_mem_15k.get("keep_recon_acc"), masked_mem_15k and masked_mem_15k.get("length_acc"), masked_mem_15k and masked_mem_15k.get("boundary_acc"), masked_mem_15k and masked_mem_15k.get("retrieval_entropy"), masked_mem_15k and masked_mem_15k.get("memory_slots_per_byte")],
            ["v3.2.1 no-memory 15k", masked_nomem_15k and masked_nomem_15k.get("steps"), masked_nomem_15k and masked_nomem_15k.get("masked_recon_acc"), masked_nomem_15k and masked_nomem_15k.get("keep_recon_acc"), masked_nomem_15k and masked_nomem_15k.get("length_acc"), masked_nomem_15k and masked_nomem_15k.get("boundary_acc"), masked_nomem_15k and masked_nomem_15k.get("retrieval_entropy"), masked_nomem_15k and masked_nomem_15k.get("memory_slots_per_byte")],
        ],
    ))
    parts.append("")
    parts.append("- 15k top-k memory vs no-memory `masked_recon_acc` delta: " + _fmt(_delta(masked_mem_15k, masked_nomem_15k, "masked_recon_acc")))
    parts.append("- 15k top-k memory vs no-memory `keep_recon_acc` delta: " + _fmt(_delta(masked_mem_15k, masked_nomem_15k, "keep_recon_acc")))
    parts.append("")

    parts.append("### Clean codec legacy scope")
    parts.append("")
    parts.append(_table(
        ["run", "steps", "recon_acc", "length_acc", "boundary_acc", "units/B", "memory"],
        [
            ["v3.1 utf8clean 40k", clean_v31 and clean_v31.get("steps"), clean_v31 and clean_v31.get("codec_recon_acc"), clean_v31 and clean_v31.get("length_acc"), clean_v31 and clean_v31.get("boundary_acc"), clean_v31 and clean_v31.get("units_per_byte"), clean_v31 and clean_v31.get("memory_enabled")],
            ["v3.1 mfl 10k", clean_v31_mfl and clean_v31_mfl.get("steps"), clean_v31_mfl and clean_v31_mfl.get("codec_recon_acc"), clean_v31_mfl and clean_v31_mfl.get("length_acc"), clean_v31_mfl and clean_v31_mfl.get("boundary_acc"), clean_v31_mfl and clean_v31_mfl.get("units_per_byte"), clean_v31_mfl and clean_v31_mfl.get("memory_enabled")],
            ["v3.2 mfl memory 10k", clean_v32_mem and clean_v32_mem.get("steps"), clean_v32_mem and clean_v32_mem.get("codec_recon_acc"), clean_v32_mem and clean_v32_mem.get("length_acc"), clean_v32_mem and clean_v32_mem.get("boundary_acc"), clean_v32_mem and clean_v32_mem.get("units_per_byte"), clean_v32_mem and clean_v32_mem.get("memory_enabled")],
            ["v3.2 mfl no-memory 10k", clean_v32_nomem and clean_v32_nomem.get("steps"), clean_v32_nomem and clean_v32_nomem.get("codec_recon_acc"), clean_v32_nomem and clean_v32_nomem.get("length_acc"), clean_v32_nomem and clean_v32_nomem.get("boundary_acc"), clean_v32_nomem and clean_v32_nomem.get("units_per_byte"), clean_v32_nomem and clean_v32_nomem.get("memory_enabled")],
            ["v3.2 mfl random 10k", clean_v32_random and clean_v32_random.get("steps"), clean_v32_random and clean_v32_random.get("codec_recon_acc"), clean_v32_random and clean_v32_random.get("length_acc"), clean_v32_random and clean_v32_random.get("boundary_acc"), clean_v32_random and clean_v32_random.get("units_per_byte"), clean_v32_random and clean_v32_random.get("memory_enabled")],
        ],
    ))
    parts.append("")
    parts.append("- Clean-codec metrics are not comparable to strict masked-source metrics. They only show historical mechanism behavior.")
    parts.append("")

    if stress_rows:
        parts.append("### Strict memory stress")
        parts.append("")
        parts.append(_table(
            ["file", "memory", "retrieval", "full_acc", "full_loss", "zero_loss_delta", "shuffled_loss_delta", "stale_loss_delta"],
            [
                [
                    Path(str(row.get("file"))).stem,
                    row.get("memory_enabled"),
                    row.get("retrieval"),
                    row.get("full_acc"),
                    row.get("full_loss"),
                    row.get("zero_loss_delta"),
                    row.get("shuffled_loss_delta"),
                    row.get("stale_loss_delta"),
                ]
                for row in stress_rows
            ],
        ))
        parts.append("")

    parts.append("## Best runs inside each scope")
    parts.append("")
    for scope, metric in (
        ("strict_masked_codec", "masked_recon_acc"),
        ("strict_masked_backbone", "backbone_mask_acc"),
        ("clean_codec", "codec_recon_acc"),
        ("commit_controller", "future_mask_acc"),
        ("segmental_diffusion", "deploy_eval_acc"),
    ):
        best = _best_by(rows, scope, metric, True, 6)
        if not best:
            continue
        parts.append(f"### {scope} by `{metric}`")
        parts.append("")
        parts.append(_table(["run", "family", "steps", metric, "memory", "params"], [[r.get("run"), r.get("family"), r.get("steps"), r.get(metric), r.get("memory_enabled"), r.get("params")] for r in best]))
        parts.append("")

    parts.append("## Rebuilt conclusions")
    parts.append("")
    parts.append("1. `masked-source codec training` is the only currently validated direction that survives strict leakage control. It improves strict backbone masked-byte accuracy over the byte baseline.")
    parts.append("2. Current top-k memory is active but not a default-mainline win. The fair strict masked-source backbone result has no-memory slightly above top-k memory, while memory stress only proves local/path activity.")
    parts.append("3. The older v3/v3.1 memory-positive conclusions are weaker than they looked because they were mostly clean reconstruction, same-checkpoint ablation, or legacy backbone objectives. They support mechanism exploration, not final architecture selection.")
    parts.append("4. Boundary is numerically stable against weak labels, but the archive still lacks calibration/utility evaluation. `boundary_acc` alone is not enough to claim semantic segmentation.")
    parts.append("5. Latent quality is currently validated indirectly through decoder accuracy and strict backbone gains. The archive lacks probe/geometry/causal tests for latent semantics.")
    parts.append("6. Memory quality is not directly supervised or directly measured. The next step should add evaluation-only probes and patching before adding memory losses to training.")
    parts.append("")
    parts.append("## Next evaluation gaps before training-supervision changes")
    parts.append("")
    parts.append("- Latent: MDL/control probes for byte type, relative position, span length, entity/code markers; CKA/RSA against clean oracle and backbone hidden states.")
    parts.append("- Memory: memory-slot probes, readout-memory retrieval Recall@k/InfoNCE, and causal patching from correct/wrong memory slots.")
    parts.append("- Boundary: tolerance F1, WindowDiff/Pk, ECE/Brier calibration, perturbation stability, learned-vs-random segmentation utility gap.")
    parts.append("- Backbone: repeat strict masked-source evaluation at longer sequence length and entity/code-heavy splits before treating memory as unnecessary.")
    parts.append("")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit archived FLUED v3-family checkpoint evidence")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    archive = args.archive.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_normalize_summary(path, archive) for path in _discover_summaries(archive)]
    stress_rows = _read_memory_stress(archive)

    rows_json = out_dir / "runs.json"
    rows_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "runs.csv", rows)
    (out_dir / "memory_stress.json").write_text(json.dumps(stress_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "memory_stress.csv", stress_rows)
    report = _report(rows, stress_rows, out_dir)
    (out_dir / "checkpoint_evidence_audit.md").write_text(report, encoding="utf-8")
    print(f"Wrote {len(rows)} runs and {len(stress_rows)} stress files to {out_dir}")


if __name__ == "__main__":
    main()
