"""Diagnose whether FLUED-v3 commit writes help future prediction.

This is an evaluation-only tool. It loads trained v3 commit-controller
checkpoints and intervenes on memory write weights derived from the original
commit probabilities:

  original       keep the learned writes
  no_memory      disable writes after byte 0
  drop_high      disable the top commit-probability writes
  drop_low       disable the lowest commit-probability writes
  top_only       keep only the top commit-probability writes
  drop_qXX_YY    disable one commit-probability quantile bucket

The primary metric is next-byte CE from future_head(memory), because it asks
whether committed memory itself predicts future bytes. Decoder next-byte CE is
also reported as a secondary metric.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset, PAD_ID
from tools.analysis.v3_0.train_v3_commit_controller_small import (
    V3CommitControllerSmall,
    _load_texts,
    _prediction_target,
)


SAMPLES: Dict[str, str] = {
    "zh_long": "昨天晚上，研究团队重新检查了实验日志，发现模型在处理专有名词和长距离指代时更倾向于保留边界，而在常见虚词附近会自动合并。",
    "en_long": "The compression module should preserve rare entity names while aggressively merging predictable function words and repeated local patterns.",
    "code": "def normalize_rate(values):\n    total = sum(values)\n    return [v / total for v in values if total > 0]\n",
    "math_digits": "The final score was 94.7%, p < 0.001, with loss=1.482 and budget_lambda=0.037 after 40,000 steps.",
    "mixed": "FLUED-v3 需要同时处理 ByteFlow-style coding rate、中文语义段、APINameLikeThis 和 2026-06-26 这样的结构。",
    "template": "订单编号 A1029 已确认。订单编号 A1030 已确认。订单编号 A1031 已确认。订单编号 A1032 已确认。",
    "entities": "OpenAI, NVIDIA, AutoDL, ByteFlow, SOMBRERO, FLEXITOKENS, and Fast BLT all imply different compression tradeoffs.",
}


def _load_model(path: Path, device: torch.device) -> Tuple[V3CommitControllerSmall, Dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = V3CommitControllerSmall(
        d_model=int(args.get("d_model", 192)),
        hidden=int(args.get("hidden", 192)),
        controller_hidden=int(args.get("controller_hidden", 256)),
        decoder_input=str(args.get("decoder_input", "active_memory")),
        controller_memory_mode=str(args.get("controller_memory_mode", "raw")),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, args


def _masked_ce_per_pos(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).reshape_as(target)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask & torch.isfinite(x)
    if not valid.any():
        return float("nan")
    return float(x[valid].mean().item())


def _quantile_mask(values: torch.Tensor, valid: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    out = torch.zeros_like(valid)
    for b in range(values.size(0)):
        idx = valid[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        vals = values[b, idx]
        order = torch.argsort(vals)
        start = int(math.floor(idx.numel() * lo))
        end = int(math.ceil(idx.numel() * hi))
        if end <= start:
            continue
        chosen = idx[order[start:end]]
        out[b, chosen] = True
    return out


def _top_budget_mask(values: torch.Tensor, valid: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(valid)
    for b in range(values.size(0)):
        idx = valid[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        k = int(round(float(budget[b].item())))
        k = max(1, min(int(idx.numel()), k))
        chosen = idx[torch.topk(values[b, idx], k=k).indices]
        out[b, chosen] = True
    return out


def _policy_write_probs(
    original_p: torch.Tensor,
    usable: torch.Tensor,
    policy: str,
    ablate_fraction: float,
) -> torch.Tensor:
    p = original_p.clone()
    p[:, 0] = original_p[:, 0]
    budget = original_p.masked_fill(~usable, 0.0).sum(dim=1)
    if policy == "original":
        return p
    if policy == "no_memory":
        p[:, 1:] = 0.0
        return p
    if policy == "drop_high":
        p[_quantile_mask(original_p, usable, 1.0 - ablate_fraction, 1.0)] = 0.0
        return p
    if policy == "drop_low":
        p[_quantile_mask(original_p, usable, 0.0, ablate_fraction)] = 0.0
        return p
    if policy == "top_only":
        keep = _top_budget_mask(original_p, usable, budget)
        p[~keep & usable] = 0.0
        return p
    if policy.startswith("drop_q"):
        body = policy.removeprefix("drop_q")
        lo_s, hi_s = body.split("_", 1)
        lo = float(lo_s) / 100.0
        hi = float(hi_s) / 100.0
        p[_quantile_mask(original_p, usable, lo, hi)] = 0.0
        return p
    raise ValueError(f"unknown policy: {policy}")


@torch.no_grad()
def _forward_with_memory_policy(
    model: V3CommitControllerSmall,
    src: torch.Tensor,
    valid: torch.Tensor,
    policy: str,
    ablate_fraction: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    logits, original = model(src, valid)
    original_p = original["commit_probs"].float()
    usable = valid.clone()
    usable[:, 0] = False
    write_p = _policy_write_probs(original_p, usable, policy, ablate_fraction)
    if policy == "original":
        out = dict(original)
        out["write_probs"] = write_p
        return logits, out

    emb = model.embedding(src.clamp(min=0))
    h, _ = model.encoder(emb)
    bsz, seq_len, hidden = h.shape
    active = h.new_zeros(bsz, hidden)
    memory = h.new_zeros(bsz, hidden)
    active_states: List[torch.Tensor] = []
    memory_states: List[torch.Tensor] = []
    future_logits: List[torch.Tensor] = []

    for t in range(seq_len):
        ht = h[:, t]
        vt = valid[:, t].float().unsqueeze(-1)
        p_active = original_p[:, t].unsqueeze(-1)
        p_memory = write_p[:, t].unsqueeze(-1)
        new_active = model.active_update(ht, active)
        new_memory = model.memory_update(new_active, memory)
        memory = torch.where(vt.bool(), (1.0 - p_memory) * memory + p_memory * new_memory, memory)
        active = torch.where(vt.bool(), (1.0 - p_active) * new_active + p_active * ht, active)
        active_states.append(active)
        memory_states.append(memory)
        future_logits.append(model.future_head(memory))

    active_seq = torch.stack(active_states, dim=1)
    memory_seq = torch.stack(memory_states, dim=1)
    if model.decoder_input == "hidden_active_memory":
        decoder_state = torch.cat([h, active_seq, memory_seq], dim=-1)
        memory_gate = None
    elif model.decoder_input == "active_memory":
        decoder_state = torch.cat([active_seq, memory_seq], dim=-1)
        memory_gate = None
    elif model.decoder_input == "gated_active_memory":
        gate_in = torch.cat([active_seq, memory_seq], dim=-1)
        gate = model.memory_gate(gate_in)
        decoder_state = active_seq + gate * model.memory_adapter(gate_in)
        memory_gate = gate.mean(dim=-1)
    elif model.decoder_input == "active":
        decoder_state = active_seq
        memory_gate = None
    else:
        decoder_state = memory_seq
        memory_gate = None
    new_logits = model.byte_head(decoder_state)
    out = {
        "commit_probs": original_p,
        "write_probs": write_p,
        "h": h,
        "active": active_seq,
        "memory": memory_seq,
        "future_logits": torch.stack(future_logits, dim=1),
    }
    if memory_gate is not None:
        out["memory_gate"] = memory_gate
    return new_logits, out


def _summarize_values(values: Sequence[float]) -> Dict[str, float]:
    clean = np.array([x for x in values if math.isfinite(float(x))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "p10": float(np.quantile(clean, 0.10)),
        "p50": float(np.quantile(clean, 0.50)),
        "p90": float(np.quantile(clean, 0.90)),
    }


@torch.no_grad()
def evaluate_loader(
    model: V3CommitControllerSmall,
    loader: DataLoader,
    device: torch.device,
    policies: Sequence[str],
    ablate_fraction: float,
    max_batches: int,
) -> Tuple[List[Dict], Dict[str, List[float]]]:
    rows: List[Dict] = []
    bucket_values: Dict[str, List[float]] = defaultdict(list)
    for batch_idx, (src, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        src = src.to(device)
        valid = src != PAD_ID
        target, target_mask = _prediction_target(src, valid, "next_byte")
        usable = valid.clone()
        usable[:, 0] = False
        target_mask = target_mask & usable

        original_logits, original_metrics = _forward_with_memory_policy(model, src, valid, "original", ablate_fraction)
        original_future_ce = _masked_ce_per_pos(original_metrics["future_logits"], target)
        original_decoder_ce = _masked_ce_per_pos(original_logits, target)
        original_future_loss = _masked_mean(original_future_ce, target_mask)
        original_decoder_loss = _masked_mean(original_decoder_ce, target_mask)
        commit = original_metrics["commit_probs"].float()

        for policy in policies:
            logits, metrics = _forward_with_memory_policy(model, src, valid, policy, ablate_fraction)
            future_ce = _masked_ce_per_pos(metrics["future_logits"], target)
            decoder_ce = _masked_ce_per_pos(logits, target)
            future_loss = _masked_mean(future_ce, target_mask)
            decoder_loss = _masked_mean(decoder_ce, target_mask)
            write = metrics["write_probs"].float()
            row = {
                "batch": batch_idx,
                "policy": policy,
                "future_loss": future_loss,
                "decoder_loss": decoder_loss,
                "future_delta_vs_original": future_loss - original_future_loss,
                "decoder_delta_vs_original": decoder_loss - original_decoder_loss,
                "commit_mean": _masked_mean(commit, usable),
                "write_mean": _masked_mean(write, usable),
                "write_kept_fraction": _masked_mean((write > 0).float(), usable),
                "future_original": original_future_loss,
                "decoder_original": original_decoder_loss,
            }
            rows.append(row)
            if policy.startswith("drop_q"):
                bucket_values[policy].append(row["future_delta_vs_original"])
    return rows, bucket_values


def _aggregate_rows(rows: Sequence[Dict], model_name: str, ckpt: Path, args: Dict) -> List[Dict]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        policy = row["policy"]
        for key, value in row.items():
            if key in {"batch", "policy"}:
                continue
            grouped[policy][key].append(float(value))
    out = []
    for policy, cols in grouped.items():
        item = {
            "model": model_name,
            "ckpt": str(ckpt),
            "policy": policy,
            "decoder_input": str(args.get("decoder_input", "")),
            "controller_memory_mode": str(args.get("controller_memory_mode", "")),
        }
        for key, vals in cols.items():
            item[key] = _summarize_values(vals)["mean"]
        out.append(item)
    return out


def _encode_text(text: str, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, bytes]:
    raw = text.encode("utf-8")[:seq_len]
    ids = torch.tensor([b + 1 for b in raw], dtype=torch.long, device=device).unsqueeze(0)
    return ids, raw


def _byte_type(byte: int) -> str:
    if 0x80 <= byte <= 0xBF:
        return "utf8_cont"
    if 0xE4 <= byte <= 0xE9:
        return "cjk_lead"
    ch = chr(byte) if byte < 128 else ""
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    if ch in "+-*/%=<>!&|^~@#$\\:;.,?()[]{}_'\"`":
        return "op"
    if ch.isalpha():
        return "ascii_alpha"
    if byte < 128:
        return "ascii_other"
    return "utf8_other"


@torch.no_grad()
def _sample_report(
    model: V3CommitControllerSmall,
    model_name: str,
    out_dir: Path,
    seq_len: int,
    policies: Sequence[str],
    ablate_fraction: float,
    device: torch.device,
) -> List[Dict]:
    rows: List[Dict] = []
    sample_dir = out_dir / model_name / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample_name, text in SAMPLES.items():
        ids, raw = _encode_text(text, seq_len, device)
        valid = ids != PAD_ID
        target, target_mask = _prediction_target(ids, valid, "next_byte")
        target_mask[:, 0] = False
        original_logits, original_metrics = _forward_with_memory_policy(model, ids, valid, "original", ablate_fraction)
        original_future_ce = _masked_ce_per_pos(original_metrics["future_logits"], target)
        original_decoder_ce = _masked_ce_per_pos(original_logits, target)
        commit = original_metrics["commit_probs"].float().squeeze(0).cpu().numpy()[: len(raw)]
        original_future_loss = _masked_mean(original_future_ce, target_mask)
        original_decoder_loss = _masked_mean(original_decoder_ce, target_mask)
        policy_losses = {}
        for policy in policies:
            logits, metrics = _forward_with_memory_policy(model, ids, valid, policy, ablate_fraction)
            future_ce = _masked_ce_per_pos(metrics["future_logits"], target)
            decoder_ce = _masked_ce_per_pos(logits, target)
            future_loss = _masked_mean(future_ce, target_mask)
            decoder_loss = _masked_mean(decoder_ce, target_mask)
            policy_losses[policy] = {
                "future_loss": future_loss,
                "decoder_loss": decoder_loss,
                "future_delta": future_loss - original_future_loss,
                "decoder_delta": decoder_loss - original_decoder_loss,
            }
            rows.append({
                "model": model_name,
                "sample": sample_name,
                "policy": policy,
                **policy_losses[policy],
                "commit_mean": float(np.nanmean(commit[1:])) if len(commit) > 1 else float("nan"),
                "commit_p90": float(np.nanquantile(commit[1:], 0.90)) if len(commit) > 1 else float("nan"),
            })
        drop_high_logits, drop_high_metrics = _forward_with_memory_policy(model, ids, valid, "drop_high", ablate_fraction)
        drop_high_ce = _masked_ce_per_pos(drop_high_metrics["future_logits"], target).squeeze(0).cpu().numpy()[: len(raw)]
        base_ce = original_future_ce.squeeze(0).cpu().numpy()[: len(raw)]
        value = drop_high_ce - base_ce
        _render_html(sample_dir / f"{sample_name}.html", model_name, sample_name, text, raw, commit, value, policy_losses)
    return rows


def _render_html(
    out: Path,
    model_name: str,
    sample_name: str,
    text: str,
    raw: bytes,
    commit: np.ndarray,
    value: np.ndarray,
    policy_losses: Dict[str, Dict[str, float]],
) -> None:
    spans = []
    byte_i = 0
    for ch in text:
        ch_bytes = ch.encode("utf-8")
        if byte_i >= len(commit):
            break
        p = float(commit[byte_i])
        v = float(value[byte_i]) if byte_i < len(value) and math.isfinite(float(value[byte_i])) else 0.0
        hue = 210 if v <= 0 else 15
        light = max(35, min(86, 78 - int(min(abs(v), 2.0) * 18)))
        border = max(0, min(4, int(round(p * 4))))
        title = f"byte={byte_i} commit={p:.3f} drop_high_future_ce_delta={v:.3f} type={_byte_type(raw[byte_i])}"
        spans.append(
            f'<span class="ch" title="{html.escape(title)}" '
            f'style="background:hsl({hue},80%,{light}%);border-bottom:{border}px solid #111">{html.escape(ch)}</span>'
        )
        byte_i += len(ch_bytes)
    rows = []
    for policy, vals in policy_losses.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(policy)}</td>"
            f"<td>{vals['future_loss']:.4f}</td><td>{vals['future_delta']:+.4f}</td>"
            f"<td>{vals['decoder_loss']:.4f}</td><td>{vals['decoder_delta']:+.4f}</td>"
            "</tr>"
        )
    top = np.argsort(-commit)[: min(18, len(commit))]
    top_rows = []
    for idx in sorted(int(i) for i in top):
        left = max(0, idx - 18)
        right = min(len(raw), idx + 19)
        top_rows.append(
            "<tr>"
            f"<td>{idx}</td><td>{commit[idx]:.3f}</td><td>{value[idx]:+.3f}</td>"
            f"<td>{_byte_type(raw[idx])}</td><td>{html.escape(raw[left:right].decode('utf-8', errors='replace'))}</td>"
            "</tr>"
        )
    out.write_text(
        "\n".join([
            "<!doctype html><meta charset='utf-8'>",
            "<style>",
            "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f7f7f7;color:#111}",
            ".panel{background:white;border:1px solid #d0d0d0;border-radius:6px;padding:16px;margin:14px 0}",
            ".heat{font-family:Consolas,monospace;font-size:18px;line-height:2.1;word-break:break-all}",
            ".ch{padding:2px 1px;margin:0 1px;color:#111;border-radius:2px}",
            "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:5px;vertical-align:top}",
            "</style>",
            f"<h1>{html.escape(model_name)} commit value: {html.escape(sample_name)}</h1>",
            "<p>Orange means future CE rises when high-commit memory writes are removed. Blue means the removal helped or was neutral. Dark underline marks larger original commit probability.</p>",
            "<div class='panel heat'>" + "".join(spans) + "</div>",
            "<div class='panel'><h2>Policy Losses</h2><table><tr><th>policy</th><th>future loss</th><th>future delta</th><th>decoder loss</th><th>decoder delta</th></tr>" + "".join(rows) + "</table></div>",
            "<div class='panel'><h2>Top Original Commits</h2><table><tr><th>byte</th><th>commit</th><th>drop-high future delta</th><th>type</th><th>context</th></tr>" + "".join(top_rows) + "</table></div>",
        ]),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_key(src: Dict, keys: Iterable[str]) -> Dict:
    return {key: src.get(key) for key in keys if key in src}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v3 commit memory write value")
    parser.add_argument("--raw-ckpt", required=True)
    parser.add_argument("--gated-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--eval-max-lines", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--ablate-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "raw": Path(args.raw_ckpt),
        "gated": Path(args.gated_ckpt),
    }
    models: Dict[str, Tuple[V3CommitControllerSmall, Dict]] = {}
    for name, ckpt in checkpoints.items():
        models[name] = _load_model(ckpt, device)

    first_args = next(iter(models.values()))[1]
    data_path = args.data_path or str(first_args.get("data_path"))
    if not data_path:
        raise RuntimeError("data path missing; pass --data-path")
    seq_len = int(args.seq_len or first_args.get("seq_len", 128))
    stride = int(args.stride or first_args.get("stride", max(1, seq_len // 2)))
    texts = _load_texts(data_path, args.eval_max_lines)
    eval_ds = ByteReconstructionDataset(texts=texts, seq_len=seq_len, stride=stride)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    policies = ["original", "no_memory", "drop_high", "drop_low", "top_only"]
    bucket_policies = [f"drop_q{i:02d}_{i + 20:02d}" for i in range(0, 100, 20)]
    all_policies = policies + bucket_policies

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []
    sample_rows: List[Dict] = []
    metadata = {
        "data_path": data_path,
        "eval_max_lines": args.eval_max_lines,
        "seq_len": seq_len,
        "stride": stride,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "ablate_fraction": args.ablate_fraction,
        "device": str(device),
        "policies": all_policies,
    }

    for name, (model, model_args) in models.items():
        rows, _ = evaluate_loader(model, loader, device, all_policies, args.ablate_fraction, args.max_batches)
        for row in rows:
            row = {"model": name, **row}
            detail_rows.append(row)
        summary_rows.extend(_aggregate_rows(rows, name, checkpoints[name], model_args))
        sample_rows.extend(_sample_report(model, name, out_dir, seq_len, all_policies, args.ablate_fraction, device))

    summary = {
        "metadata": metadata,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "model_args": {
            name: _copy_key(model_args, [
                "decoder_input",
                "controller_memory_mode",
                "d_model",
                "hidden",
                "controller_hidden",
                "prediction_target",
                "max_steps",
            ])
            for name, (_, model_args) in models.items()
        },
        "summary": summary_rows,
        "samples": sample_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "details.csv", detail_rows)
    _write_csv(out_dir / "sample_summary.csv", sample_rows)
    print(json.dumps({"out_dir": str(out_dir), "summary_rows": len(summary_rows), "detail_rows": len(detail_rows)}, indent=2))


if __name__ == "__main__":
    main()
