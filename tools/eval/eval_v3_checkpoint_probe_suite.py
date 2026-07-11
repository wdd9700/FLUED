"""Evaluation-only probes for archived FLUED v3.1/v3.2 codec checkpoints.

This script measures properties that reconstruction summaries do not cover:

* strict masked-source direct decoding and clean-oracle readout gap
* boundary confidence calibration against weak boundary labels
* readout geometry statistics
* memory/readout association and masked-source memory ablation for v3.2

The metrics are diagnostics, not training targets.  They should be run before
turning any of these signals into losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import MASK_ID, PAD_ID, STUB_CORPUS, ByteReconstructionDataset, StreamingReconstructionDataset  # noqa: E402
from tools.analysis.train_v3_commit_controller_small import _load_texts  # noqa: E402
from tools.analysis.train_v31_language_codec_2m import V31LanguageCodec2M  # noqa: E402
from tools.analysis.train_v32_language_codec_2m import V32LanguageCodec2M, build_segments  # noqa: E402
from tools.analysis.train_v32_strict_masked_backbone import StrictMaskedCollator, targets_from_masked_segments  # noqa: E402
from tools.eval.eval_v32_language_codec_memory_ablation import forward_with_mode as v32_forward_with_memory_mode  # noqa: E402


DEFAULT_CHECKPOINTS = [
    r"K:\FLUED_archive\v31_language_codec_2m_20260702\codec_40k_utf8clean\latest.pt",
    r"K:\FLUED_archive\v31_language_codec_2m_20260702\codec_10k_pool_mfl\latest.pt",
    r"K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_nomemory_10k\latest.pt",
    r"K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k\latest.pt",
    r"K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_random_10k\latest.pt",
    r"K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_nomemory_masked_15k\latest.pt",
    r"K:\FLUED_archive\v32_masked_codec_2m_20260703\v321_mfl_memory_masked_15k\latest.pt",
]


def _discover_codec_checkpoints(archive: Path, include_steps: bool) -> List[Path]:
    roots = [
        archive / "v31_language_codec_2m_20260702",
        archive / "v32_language_codec_2m_20260703",
        archive / "v32_masked_codec_2m_20260703",
    ]
    out: List[Path] = []
    seen = set()
    pattern = "*.pt" if include_steps else "latest.pt"
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            if "smoke_cpu" in str(path).lower() and not include_steps:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(path.resolve())
    return sorted(out)


def _torch_load(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_ckpt(path: str | Path) -> Path:
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "latest.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt


def _codec_kwargs(args: Mapping[str, Any], family: str) -> Dict[str, Any]:
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
    ]
    if family == "v32":
        keys.extend(["memory_slots_per_chunk", "memory_topk", "memory_retrieval_mode", "causal_byte_encoder"])
    return {k: args[k] for k in keys if k in args}


def _infer_family(path: Path, ckpt: Mapping[str, Any]) -> str:
    args = ckpt.get("args", {})
    if isinstance(args, Mapping) and ("memory_slots_per_chunk" in args or "causal_byte_encoder" in args):
        return "v32"
    lower = str(path).lower()
    if "v32" in lower or "v321" in lower:
        return "v32"
    return "v31"


def _load_codec(path: Path, device: torch.device):
    ckpt = _torch_load(path)
    args = dict(ckpt.get("args", {}))
    family = _infer_family(path, ckpt)
    model_cls = V32LanguageCodec2M if family == "v32" else V31LanguageCodec2M
    model = model_cls(**_codec_kwargs(args, family)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, args, family


def _mean_dict(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        out[key] = sum(vals) / max(len(vals), 1)
    return out


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def _binary_f1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float, float]:
    pred = pred[mask].bool()
    target = target[mask].bool()
    if target.numel() == 0:
        return 0.0, 0.0, 0.0
    tp = float((pred & target).sum().item())
    fp = float((pred & ~target).sum().item())
    fn = float((~pred & target).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _ece(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, bins: int = 10) -> float:
    p = prob[mask].float().flatten()
    y = target[mask].float().flatten()
    if p.numel() == 0:
        return 0.0
    total = float(p.numel())
    err = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        in_bin = (p >= lo) & (p < hi if i + 1 < bins else p <= hi)
        if not in_bin.any():
            continue
        conf = float(p[in_bin].mean().item())
        acc = float(y[in_bin].mean().item())
        err += float(in_bin.float().sum().item()) / total * abs(acc - conf)
    return err


def _rank_auc(score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    s = score[mask].float().flatten()
    y = target[mask].bool().flatten()
    pos = int(y.sum().item())
    neg = int((~y).sum().item())
    if pos == 0 or neg == 0:
        return 0.0
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float)
    pos_rank_sum = ranks[y].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1)
    return float(auc.item())


def _effective_rank(x: torch.Tensor) -> float:
    if x.numel() == 0 or x.size(0) < 2:
        return 0.0
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    # Use singular values of centered activations as a stable small-sample proxy.
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-8]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()).item())


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0 or x.size(0) < 2 or y.size(0) < 2:
        return 0.0
    n = min(x.size(0), y.size(0))
    x = x[:n].float()
    y = y[:n].float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x.T @ y).pow(2).sum()
    xnorm = (x.T @ x).pow(2).sum().sqrt()
    ynorm = (y.T @ y).pow(2).sum().sqrt()
    return float((hsic / (xnorm * ynorm).clamp(min=1e-12)).item())


def _byte_type_labels(first_tokens: torch.Tensor) -> torch.Tensor:
    # Token IDs are byte+1, PAD=0, MASK=257.  The labels are intentionally
    # coarse: the probe should test easily extractable byte-class information,
    # not memorize exact bytes.
    labels = torch.zeros_like(first_tokens, dtype=torch.long)
    raw = first_tokens - 1
    labels[(raw >= ord("0")) & (raw <= ord("9"))] = 1
    labels[((raw >= ord("A")) & (raw <= ord("Z"))) | ((raw >= ord("a")) & (raw <= ord("z")))] = 2
    labels[(raw == ord(" ")) | (raw == 9) | (raw == 10) | (raw == 13)] = 3
    punct = torch.zeros_like(labels, dtype=torch.bool)
    for ch in b".,;:!?()[]{}<>\"'`~@#$%^&*-+=_/\\|":
        punct |= raw == int(ch)
    labels[punct] = 4
    labels[(raw >= 0x80) & (raw <= 0xBF)] = 5
    labels[(raw >= 0xC0) & (raw <= 0xFF)] = 6
    return labels


def _probe_mdl_bits(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    classes: int,
    steps: int,
    seed: int,
    control: bool = False,
) -> Tuple[float, float]:
    """Small online-coding MDL proxy with a linear probe.

    It is not a full probing paper implementation, but it follows the online
    coding idea: labels in each block are encoded by a probe trained only on
    earlier examples.  Lower bits/label means the information is easier to
    extract from the representation.
    """

    if x.numel() == 0 or y.numel() == 0 or classes <= 1:
        return 0.0, 0.0
    x = x.float().detach().cpu()
    y = y.long().detach().cpu()
    keep = (y >= 0) & (y < classes)
    x = x[keep]
    y = y[keep]
    if x.size(0) < max(classes * 2, 32):
        return 0.0, 0.0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=gen)
    x = x[perm]
    y = y[perm]
    if control:
        y = y[torch.randperm(y.numel(), generator=gen)]
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp(min=1e-5)

    blocks = [max(8, x.size(0) // 10), max(16, x.size(0) // 5), max(24, x.size(0) // 2), x.size(0)]
    blocks = sorted(set(min(max(b, 1), x.size(0)) for b in blocks))
    total_nll = 0.0
    total_count = 0
    prev = 0
    uniform_nll = math.log(classes)
    for end in blocks:
        if end <= prev:
            continue
        if prev < max(classes, 4):
            total_nll += uniform_nll * (end - prev)
            total_count += end - prev
            prev = end
            continue
        model = torch.nn.Linear(x.size(1), classes)
        opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-3)
        train_x = x[:prev]
        train_y = y[:prev]
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x), train_y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            nll = F.cross_entropy(model(x[prev:end]), y[prev:end], reduction="sum").item()
            pred = model(x[prev:end]).argmax(dim=-1)
        total_nll += nll
        total_count += end - prev
        prev = end
    if total_count == 0:
        return 0.0, 0.0
    # Final block accuracy from the last trained split, as a coarse sanity
    # check for the MDL number.
    final_train = max(classes, int(x.size(0) * 0.8))
    final_train = min(final_train, x.size(0) - 1)
    model = torch.nn.Linear(x.size(1), classes)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[:final_train]), y[:final_train])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x[final_train:]).argmax(dim=-1)
        acc = float((pred == y[final_train:]).float().mean().item()) if final_train < x.size(0) else 0.0
    return total_nll / max(total_count, 1) / math.log(2.0), acc


def _recall_at_k(query: torch.Tensor, key: torch.Tensor, k: int) -> Tuple[float, float]:
    if query.numel() == 0 or key.numel() == 0 or query.size(0) < 2 or key.size(0) < 2:
        return 0.0, 0.0
    n = min(query.size(0), key.size(0))
    query = F.normalize(query[:n].float(), dim=-1)
    key = F.normalize(key[:n].float(), dim=-1)
    logits = query @ key.T
    labels = torch.arange(n, device=logits.device)
    topk = logits.topk(min(k, n), dim=-1).indices
    recall = float((topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())
    info_nce = float(F.cross_entropy(logits, labels).item())
    return recall, info_nce


def _ensure_first_starts(starts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = starts.clone()
    for b in range(out.size(0)):
        idx = valid[b].nonzero(as_tuple=False).flatten()
        if idx.numel():
            out[b, idx[0]] = True
    return out & valid


def _tolerance_f1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, tol: int) -> Tuple[float, float, float]:
    tp = fp = fn = 0.0
    for b in range(pred.size(0)):
        valid_idx = set(int(i) for i in valid[b].nonzero(as_tuple=False).flatten().tolist())
        p = [int(i) for i in pred[b].nonzero(as_tuple=False).flatten().tolist() if int(i) in valid_idx]
        t = [int(i) for i in target[b].nonzero(as_tuple=False).flatten().tolist() if int(i) in valid_idx]
        used = set()
        for pi in p:
            match = None
            best = tol + 1
            for j, ti in enumerate(t):
                if j in used:
                    continue
                dist = abs(pi - ti)
                if dist <= tol and dist < best:
                    best = dist
                    match = j
            if match is None:
                fp += 1.0
            else:
                tp += 1.0
                used.add(match)
        fn += float(len(t) - len(used))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _segment_ids_from_starts(starts_1d: torch.Tensor, valid_1d: torch.Tensor) -> torch.Tensor:
    ids = torch.full_like(starts_1d.long(), -1)
    cur = -1
    for i in range(starts_1d.numel()):
        if not bool(valid_1d[i]):
            continue
        if bool(starts_1d[i]) or cur < 0:
            cur += 1
        ids[i] = cur
    return ids


def _pk_windowdiff(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, window: int) -> Tuple[float, float]:
    pk_err = wd_err = count = 0.0
    window = max(1, int(window))
    for b in range(pred.size(0)):
        idx = valid[b].nonzero(as_tuple=False).flatten()
        if idx.numel() <= window:
            continue
        p = pred[b, idx]
        t = target[b, idx]
        p_ids = _segment_ids_from_starts(p, torch.ones_like(p, dtype=torch.bool))
        t_ids = _segment_ids_from_starts(t, torch.ones_like(t, dtype=torch.bool))
        for i in range(0, idx.numel() - window):
            p_same = p_ids[i] == p_ids[i + window]
            t_same = t_ids[i] == t_ids[i + window]
            pk_err += float(p_same != t_same)
            p_count = int(p[i : i + window + 1].sum().item())
            t_count = int(t[i : i + window + 1].sum().item())
            wd_err += float(p_count != t_count)
            count += 1.0
    return pk_err / max(count, 1.0), wd_err / max(count, 1.0)


def _random_starts_like(target: torch.Tensor, valid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    out = torch.zeros_like(target, dtype=torch.bool)
    for b in range(target.size(0)):
        idx = valid[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        n = int(target[b, idx].sum().item())
        n = max(1, min(n, idx.numel()))
        out[b, idx[0]] = True
        if n > 1 and idx.numel() > 1:
            perm = torch.randperm(idx.numel() - 1, generator=generator)[: n - 1].to(idx.device) + 1
            out[b, idx[perm]] = True
    return out & valid


def _perturb_source(src: torch.Tensor, valid: torch.Tensor, seed: int, rate: float) -> torch.Tensor:
    gen = torch.Generator(device=src.device).manual_seed(seed)
    out = src.clone()
    random_mask = torch.rand(src.shape, generator=gen, device=src.device) < rate
    mask = random_mask & valid & src.gt(PAD_ID)
    # Punctuation/space perturbations keep text roughly plausible and avoid
    # turning this into a second masking task.
    replacement = torch.full_like(out, ord(" ") + 1)
    out = torch.where(mask, replacement, out)
    return out


def _make_loader(args: argparse.Namespace) -> DataLoader:
    collate = StrictMaskedCollator(args.min_span, args.max_span, args.max_units, args.mask_prob, args.mask_span_min, args.mask_span_max)
    if args.streaming_eval and args.data_path:
        dataset = StreamingReconstructionDataset(
            file_path=args.data_path,
            seq_len=args.seq_len,
            samples_per_worker=max(args.batch_size * args.max_eval_batches, 1024),
            seed=args.seed + 9999,
        )
    else:
        texts = _load_texts(args.data_path, args.eval_max_lines) if args.data_path else STUB_CORPUS * 64
        dataset = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.stride)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)


@torch.no_grad()
def _codec_forward(model, src, valid, seg_ids, seg_mask, amp: bool):
    with torch.amp.autocast(device_type=src.device.type, dtype=torch.bfloat16, enabled=amp and src.device.type == "cuda"):
        return model(src, valid, seg_ids, seg_mask)


def _ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none").view_as(targets)
    if not mask.any():
        return 0.0
    return float(ce[mask].mean().item())


def _eval_one(path: Path, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    model, ckpt_args, family = _load_codec(path, device)
    loader = _make_loader(args)
    rows: List[Dict[str, float]] = []
    memory_rows: List[Dict[str, float]] = []
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    non_blocking = device.type == "cuda"
    readout_x: List[torch.Tensor] = []
    clean_readout_x: List[torch.Tensor] = []
    perturb_readout_x: List[torch.Tensor] = []
    length_y: List[torch.Tensor] = []
    type_y: List[torch.Tensor] = []
    memory_x: List[torch.Tensor] = []
    memory_length_y: List[torch.Tensor] = []
    memory_type_y: List[torch.Tensor] = []
    memory_slot_x_all: List[torch.Tensor] = []
    readout_for_slot_all: List[torch.Tensor] = []

    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        clean_src, masked_src, valid, seg_ids, seg_mask, targets, loss_mask, lengths, unit_mask = tuple(
            x.to(device, non_blocking=non_blocking) for x in batch
        )
        byte_logits, length_logits, metrics = _codec_forward(model, masked_src, valid, seg_ids, seg_mask, args.amp)
        clean_logits, clean_length_logits, clean_metrics = _codec_forward(model, clean_src, valid, seg_ids, seg_mask, args.amp)
        del clean_logits, clean_length_logits

        pred = byte_logits.argmax(dim=-1)
        slot_mask = targets.ne(PAD_ID) & seg_mask.unsqueeze(-1)
        keep_mask = slot_mask & (~loss_mask)
        len_target = (lengths.clamp(min=1, max=model.max_span) - 1).clamp(min=0)
        len_pred = length_logits.argmax(dim=-1)

        readout = metrics["readout"].float()
        clean_readout = clean_metrics["readout"].float()
        active_readout = readout[seg_mask].detach()
        l2 = (readout - clean_readout).pow(2).mean(dim=-1).sqrt()
        cos = F.cosine_similarity(readout, clean_readout, dim=-1)
        perturb_src = _perturb_source(masked_src, valid, args.seed + i + 1000, args.perturb_rate)
        _perturb_logits, _perturb_len, perturb_metrics = _codec_forward(model, perturb_src, valid, seg_ids, seg_mask, args.amp)
        del _perturb_logits, _perturb_len
        perturb_readout = perturb_metrics["readout"].float()
        perturb_cos = F.cosine_similarity(readout, perturb_readout, dim=-1)

        active_lengths = (lengths[seg_mask].clamp(min=1, max=model.max_span) - 1).detach().cpu()
        first_tokens = targets[:, :, 0][seg_mask].detach().cpu()
        active_types = _byte_type_labels(first_tokens).detach().cpu()
        active_readout_cpu = active_readout.detach().cpu()
        clean_active_readout_cpu = clean_readout[seg_mask].detach().cpu()
        perturb_active_readout_cpu = perturb_readout[seg_mask].detach().cpu()
        readout_x.append(active_readout_cpu)
        clean_readout_x.append(clean_active_readout_cpu)
        perturb_readout_x.append(perturb_active_readout_cpu)
        length_y.append(active_lengths)
        type_y.append(active_types)

        boundary_prob = metrics["boundary_logits"].sigmoid()
        boundary_pred = _ensure_first_starts(boundary_prob.ge(0.5), valid)
        starts = batch_to_device_starts(masked_src, valid, args)
        starts = _ensure_first_starts(starts, valid)
        b_prec, b_rec, b_f1 = _binary_f1(boundary_pred, starts, valid)
        _tbp, _tbr, b_tol_f1 = _tolerance_f1(boundary_pred, starts, valid, args.boundary_tolerance)
        pk, windowdiff = _pk_windowdiff(boundary_pred, starts, valid, args.pk_window)
        active_boundary_prob = boundary_prob[valid]
        active_starts = starts[valid].float()

        row = {
            "direct_mask_acc": _safe_acc(pred, targets, loss_mask),
            "direct_keep_acc": _safe_acc(pred, targets, keep_mask),
            "direct_mask_loss": _ce_loss(byte_logits, targets, loss_mask),
            "direct_keep_loss": _ce_loss(byte_logits, targets, keep_mask),
            "mask_length_acc": _safe_acc(len_pred, len_target, unit_mask),
            "keep_length_acc": _safe_acc(len_pred, len_target, seg_mask & (~unit_mask)),
            "masked_readout_cos": float(cos[unit_mask].mean().item()) if unit_mask.any() else 0.0,
            "keep_readout_cos": float(cos[seg_mask & (~unit_mask)].mean().item()) if (seg_mask & (~unit_mask)).any() else 0.0,
            "masked_readout_l2": float(l2[unit_mask].mean().item()) if unit_mask.any() else 0.0,
            "keep_readout_l2": float(l2[seg_mask & (~unit_mask)].mean().item()) if (seg_mask & (~unit_mask)).any() else 0.0,
            "readout_norm": float(active_readout.norm(dim=-1).mean().item()) if active_readout.numel() else 0.0,
            "readout_eff_rank": _effective_rank(active_readout[: args.max_rank_samples]),
            "clean_oracle_cka": _linear_cka(active_readout[: args.max_rank_samples], clean_readout[seg_mask].float()[: args.max_rank_samples]),
            "perturb_readout_cos": float(perturb_cos[seg_mask].mean().item()) if seg_mask.any() else 0.0,
            "perturb_readout_cka": _linear_cka(active_readout[: args.max_rank_samples], perturb_readout[seg_mask].float()[: args.max_rank_samples]),
            "boundary_acc": _safe_acc(boundary_pred, starts, valid),
            "boundary_precision": b_prec,
            "boundary_recall": b_rec,
            "boundary_f1": b_f1,
            "boundary_tolerance_f1": b_tol_f1,
            "boundary_pk": pk,
            "boundary_windowdiff": windowdiff,
            "boundary_ece": _ece(boundary_prob, starts, valid, bins=args.ece_bins),
            "boundary_brier": float((boundary_prob[valid].float() - starts[valid].float()).pow(2).mean().item()) if valid.any() else 0.0,
            "boundary_auc": _rank_auc(boundary_prob, starts, valid),
            "boundary_pos_conf": float(active_boundary_prob[active_starts.bool()].mean().item()) if active_starts.bool().any() else 0.0,
            "boundary_neg_conf": float(active_boundary_prob[~active_starts.bool()].mean().item()) if (~active_starts.bool()).any() else 0.0,
            "active_units": float(seg_mask.sum().item()),
            "masked_units": float(unit_mask.sum().item()),
            "masked_bytes": float(loss_mask.sum().item()),
            "valid_bytes": float(valid.sum().item()),
        }

        byte_mask = masked_src.eq(MASK_ID) & valid
        for prefix, candidate_starts in (
            ("boundary_pred", boundary_pred),
            ("boundary_random", _random_starts_like(starts, valid, gen)),
        ):
            cand_seg_ids, _cand_masked_targets, _cand_masked_lengths, cand_seg_mask = build_segments(
                masked_src,
                valid,
                candidate_starts,
                args.max_units,
                args.max_span,
            )
            cand_targets, cand_loss_mask, _cand_lengths = targets_from_masked_segments(
                clean_src,
                byte_mask,
                cand_seg_ids,
                cand_seg_mask,
                args.max_span,
            )
            cand_logits, _cand_len_logits, _cand_metrics = _codec_forward(model, masked_src, valid, cand_seg_ids, cand_seg_mask, args.amp)
            row[f"{prefix}_mask_loss"] = _ce_loss(cand_logits, cand_targets, cand_loss_mask)
            row[f"{prefix}_mask_acc"] = _safe_acc(cand_logits.argmax(dim=-1), cand_targets, cand_loss_mask)
        row["boundary_pred_vs_random_loss_delta"] = row["boundary_random_mask_loss"] - row["boundary_pred_mask_loss"]
        row["boundary_weak_vs_random_loss_delta"] = row["boundary_random_mask_loss"] - row["direct_mask_loss"]

        memory = metrics.get("memory")
        memory_slots = metrics.get("memory_slots")
        if isinstance(memory, torch.Tensor) and memory.numel() and memory.shape[-1] == readout.shape[-1]:
            active_memory = memory[seg_mask].float()
            row["memory_context_norm"] = float(active_memory.norm(dim=-1).mean().item()) if active_memory.numel() else 0.0
            row["memory_readout_cos"] = float(F.cosine_similarity(active_memory, active_readout, dim=-1).mean().item()) if active_memory.numel() and active_readout.numel() else 0.0
        if isinstance(memory_slots, torch.Tensor) and memory_slots.numel() and memory_slots.size(2) > 0:
            slot_mean = memory_slots.mean(dim=2).float()
            active_slot = slot_mean[seg_mask]
            row["memory_slot_norm"] = float(active_slot.norm(dim=-1).mean().item()) if active_slot.numel() else 0.0
            row["memory_slot_readout_cos"] = float(F.cosine_similarity(active_slot, active_readout, dim=-1).mean().item()) if active_slot.numel() and active_readout.numel() else 0.0
            memory_x.append(active_slot.detach().cpu())
            memory_length_y.append(active_lengths)
            memory_type_y.append(active_types)
            memory_slot_x_all.append(active_slot.detach().cpu())
            readout_for_slot_all.append(active_readout_cpu)
        row["retrieval_entropy"] = float(metrics.get("retrieval_entropy", torch.zeros((), device=device)).float().item()) if isinstance(metrics, Mapping) else 0.0
        rows.append(row)

        if family == "v32" and int(ckpt_args.get("memory_slots_per_chunk", 0) or 0) > 0:
            mode_losses: Dict[str, float] = {}
            mode_accs: Dict[str, float] = {}
            for mode in ("full", "zero", "shuffled", "stale"):
                mode_logits, _mode_len, _mode_metrics = v32_forward_with_memory_mode(
                    model,
                    masked_src,
                    valid,
                    seg_ids,
                    seg_mask,
                    mode,
                    previous_memory=None,
                    generator=gen,
                )
                mode_losses[mode] = _ce_loss(mode_logits, targets, loss_mask)
                mode_accs[mode] = _safe_acc(mode_logits.argmax(dim=-1), targets, loss_mask)
            memory_rows.append(
                {
                    "memory_zero_loss_delta": mode_losses["zero"] - mode_losses["full"],
                    "memory_shuffled_loss_delta": mode_losses["shuffled"] - mode_losses["full"],
                    "memory_stale_loss_delta": mode_losses["stale"] - mode_losses["full"],
                    "memory_zero_acc_delta": mode_accs["zero"] - mode_accs["full"],
                    "memory_shuffled_acc_delta": mode_accs["shuffled"] - mode_accs["full"],
                    "memory_stale_acc_delta": mode_accs["stale"] - mode_accs["full"],
                }
            )

    summary = _mean_dict(rows)
    summary.update({f"ablation_{k}": v for k, v in _mean_dict(memory_rows).items()})
    if readout_x:
        rx = torch.cat(readout_x, dim=0)[: args.max_probe_samples]
        crx = torch.cat(clean_readout_x, dim=0)[: args.max_probe_samples]
        prx = torch.cat(perturb_readout_x, dim=0)[: args.max_probe_samples]
        ly = torch.cat(length_y, dim=0)[: args.max_probe_samples]
        ty = torch.cat(type_y, dim=0)[: args.max_probe_samples]
        summary["latent_clean_oracle_cka"] = _linear_cka(rx, crx)
        summary["latent_perturb_cka"] = _linear_cka(rx, prx)
        summary["latent_length_mdl_bits"], summary["latent_length_probe_acc"] = _probe_mdl_bits(
            rx,
            ly,
            classes=int(model.max_span),
            steps=args.probe_steps,
            seed=args.seed + 101,
        )
        summary["latent_length_control_mdl_bits"], _ = _probe_mdl_bits(
            rx,
            ly,
            classes=int(model.max_span),
            steps=args.probe_steps,
            seed=args.seed + 102,
            control=True,
        )
        summary["latent_length_selectivity_bits"] = summary["latent_length_control_mdl_bits"] - summary["latent_length_mdl_bits"]
        summary["latent_type_mdl_bits"], summary["latent_type_probe_acc"] = _probe_mdl_bits(
            rx,
            ty,
            classes=7,
            steps=args.probe_steps,
            seed=args.seed + 103,
        )
        summary["latent_type_control_mdl_bits"], _ = _probe_mdl_bits(
            rx,
            ty,
            classes=7,
            steps=args.probe_steps,
            seed=args.seed + 104,
            control=True,
        )
        summary["latent_type_selectivity_bits"] = summary["latent_type_control_mdl_bits"] - summary["latent_type_mdl_bits"]

    if memory_x:
        mx = torch.cat(memory_x, dim=0)[: args.max_probe_samples]
        mly = torch.cat(memory_length_y, dim=0)[: args.max_probe_samples]
        mty = torch.cat(memory_type_y, dim=0)[: args.max_probe_samples]
        summary["memory_length_mdl_bits"], summary["memory_length_probe_acc"] = _probe_mdl_bits(
            mx,
            mly,
            classes=int(model.max_span),
            steps=args.probe_steps,
            seed=args.seed + 201,
        )
        summary["memory_length_control_mdl_bits"], _ = _probe_mdl_bits(
            mx,
            mly,
            classes=int(model.max_span),
            steps=args.probe_steps,
            seed=args.seed + 202,
            control=True,
        )
        summary["memory_length_selectivity_bits"] = summary["memory_length_control_mdl_bits"] - summary["memory_length_mdl_bits"]
        summary["memory_type_mdl_bits"], summary["memory_type_probe_acc"] = _probe_mdl_bits(
            mx,
            mty,
            classes=7,
            steps=args.probe_steps,
            seed=args.seed + 203,
        )
        summary["memory_type_control_mdl_bits"], _ = _probe_mdl_bits(
            mx,
            mty,
            classes=7,
            steps=args.probe_steps,
            seed=args.seed + 204,
            control=True,
        )
        summary["memory_type_selectivity_bits"] = summary["memory_type_control_mdl_bits"] - summary["memory_type_mdl_bits"]

    if memory_slot_x_all and readout_for_slot_all:
        ms = torch.cat(memory_slot_x_all, dim=0)[: args.max_probe_samples]
        rr = torch.cat(readout_for_slot_all, dim=0)[: args.max_probe_samples]
        summary["memory_readout_recall_at_1"], summary["memory_readout_infonce"] = _recall_at_k(ms, rr, 1)
        summary["memory_readout_recall_at_5"], _ = _recall_at_k(ms, rr, 5)
    return {
        "checkpoint": str(path),
        "family": family,
        "steps": int(_torch_load(path).get("step", ckpt_args.get("max_steps", 0)) or 0),
        "args": {
            "pool_mode": ckpt_args.get("pool_mode"),
            "memory_slots_per_chunk": int(ckpt_args.get("memory_slots_per_chunk", 0) or 0),
            "memory_retrieval_mode": ckpt_args.get("memory_retrieval_mode", "none") if int(ckpt_args.get("memory_slots_per_chunk", 0) or 0) else "none",
            "causal_byte_encoder": ckpt_args.get("causal_byte_encoder", None),
        },
        "summary": summary,
    }


def batch_to_device_starts(masked_src: torch.Tensor, valid: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    # Import lazily to avoid one more global dependency name in stack traces.
    from tools.analysis.train_v32_language_codec_2m import weak_boundary_starts

    return weak_boundary_starts(masked_src, valid, args.min_span, args.max_span)


def _flatten_report(reports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in reports:
        args = report.get("args", {})
        summary = report.get("summary", {})
        row: Dict[str, Any] = {
            "checkpoint": report.get("checkpoint"),
            "family": report.get("family"),
            "steps": report.get("steps"),
        }
        if isinstance(args, Mapping):
            row.update({f"arg_{k}": v for k, v in args.items()})
        if isinstance(summary, Mapping):
            row.update(summary)
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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "-"
        return f"{float(value):.4f}"
    return str(value)


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "checkpoint",
        "family",
        "mem_slots",
        "codec_mask_acc",
        "codec_mask_CE",
        "codec_keep_acc",
        "length_acc",
        "latent_type_sel",
        "latent_len_sel",
        "clean_CKA",
        "perturb_CKA",
        "mem_type_sel",
        "mem_R@5",
        "mem_patch_CE",
        "boundary_F1",
        "tol_F1",
        "Pk",
        "WinDiff",
        "ECE",
        "utility_gap",
    ]
    lines = ["# FLUED v3 full metric table", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        name = Path(str(row.get("checkpoint"))).parent.name
        values = [
            name,
            row.get("family"),
            row.get("arg_memory_slots_per_chunk"),
            row.get("direct_mask_acc"),
            row.get("direct_mask_loss"),
            row.get("direct_keep_acc"),
            row.get("mask_length_acc"),
            row.get("latent_type_selectivity_bits"),
            row.get("latent_length_selectivity_bits"),
            row.get("latent_clean_oracle_cka"),
            row.get("latent_perturb_cka"),
            row.get("memory_type_selectivity_bits"),
            row.get("memory_readout_recall_at_5"),
            row.get("ablation_memory_zero_loss_delta"),
            row.get("boundary_f1"),
            row.get("boundary_tolerance_f1"),
            row.get("boundary_pk"),
            row.get("boundary_windowdiff"),
            row.get("boundary_ece"),
            row.get("boundary_pred_vs_random_loss_delta"),
        ]
        lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- Codec: `codec_mask_acc/CE`, `codec_keep_acc`, `length_acc` are strict masked-source direct decode metrics.")
    lines.append("- Backbone: not retrained in this script; use the paired strict backbone audit for `byte baseline` and `latent` deltas.")
    lines.append("- Latent: `*_sel` is online-MDL selectivity in bits, `clean_CKA` is masked-source vs clean-oracle CKA, `perturb_CKA` is stability under light byte perturbation.")
    lines.append("- Memory: `mem_R@5` is same-unit memory-slot/readout retrieval, `mem_patch_CE` is zero-memory CE delta; positive means memory helps.")
    lines.append("- Boundary: `F1/tol_F1/Pk/WinDiff/ECE` measure hard boundary and confidence quality against weak labels; `utility_gap` is predicted-boundary CE improvement over random equal-count segmentation.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Full metric table for archived FLUED v3.1/v3.2 codec checkpoints")
    parser.add_argument("--codec-ckpt", action="append", default=[])
    parser.add_argument("--archive", type=Path, default=Path(r"K:\FLUED_archive"))
    parser.add_argument("--discover", action="store_true", help="discover v3.1/v3.2/v3.2.1 codec checkpoints under --archive")
    parser.add_argument("--include-step-checkpoints", action="store_true", help="with --discover, evaluate step*.pt as well as latest.pt")
    parser.add_argument("--out-dir", type=Path, default=Path(r"K:\FLUED_archive\v3_checkpoint_audit_20260703\probe_suite"))
    parser.add_argument("--data-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--streaming-eval", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--eval-max-lines", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--max-span", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=64)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mask-span-min", type=int, default=1)
    parser.add_argument("--mask-span-max", type=int, default=8)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-rank-samples", type=int, default=2048)
    parser.add_argument("--max-probe-samples", type=int, default=4096)
    parser.add_argument("--probe-steps", type=int, default=80)
    parser.add_argument("--perturb-rate", type=float, default=0.05)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--pk-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.codec_ckpt:
        ckpts = [Path(x) for x in args.codec_ckpt]
    elif args.discover:
        ckpts = _discover_codec_checkpoints(args.archive, args.include_step_checkpoints)
    else:
        ckpts = [Path(x) for x in DEFAULT_CHECKPOINTS]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for ckpt in ckpts:
        resolved = _resolve_ckpt(ckpt)
        print(f"Evaluating {resolved}")
        try:
            reports.append(_eval_one(resolved, args, device))
        except Exception as exc:
            reports.append({"checkpoint": str(resolved), "family": "error", "steps": 0, "args": {}, "summary": {"error": repr(exc)}})
            print(f"ERROR {resolved}: {exc!r}")

    flat = _flatten_report(reports)
    (args.out_dir / "probe_suite.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.out_dir / "probe_suite.csv", flat)
    (args.out_dir / "probe_suite.md").write_text(_markdown(flat), encoding="utf-8")
    print(f"Wrote probe suite for {len(reports)} checkpoints to {args.out_dir}")


if __name__ == "__main__":
    main()
