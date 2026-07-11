"""FMC boundary probe for FLUED checkpoints.

This is a diagnostic, not a new model. It compares the native FLUED boundary
probabilities against a fixed low-dimensional control-space boundary score.

The probe is intentionally lightweight:
  - load a checkpoint
  - run a small eval sample
  - collect hidden/proxy statistics through model outputs
  - compare density and enrichment against reconstruction residual

The goal is to decide whether FLUED-v3-FMC is worth implementing in model code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset
from flued.model import FLUEDAutoencoder, MASK_ID, PAD_ID


def _infer_model_from_ckpt(ckpt_path: str, device: torch.device) -> FLUEDAutoencoder:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    cfg = ckpt.get("model_config", {})

    d_model = state["embedding.weight"].shape[1]
    nhead = cfg.get("nhead", max(1, d_model // 64))
    if "blocks.0.ff_gate.weight" in state:
        swiglu_hidden = state["blocks.0.ff_gate.weight"].shape[0]
        dim_ff = cfg.get("dim_feedforward", swiglu_hidden)
    else:
        swiglu_hidden = None
        dim_ff = state["blocks.0.ff1.weight"].shape[0]
    num_layers = cfg.get(
        "num_layers",
        len({k.split(".")[1] for k in state if k.startswith("blocks.")}),
    )

    model = FLUEDAutoencoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        swiglu_hidden=swiglu_hidden,
        num_layers=num_layers,
        max_seq_len=cfg.get("max_seq_len", 512),
        assignment_window=cfg.get("assignment_window", 128),
        dropout=0.0,
        target_compression=cfg.get("target_compression", 0.3),
        compression_weight=cfg.get("compression_weight", 0.1),
        min_boundary_units=cfg.get("min_boundary_units", 1.0),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _standardize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = x[mask]
    if valid.numel() == 0:
        return torch.zeros_like(x)
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp(min=1e-6)
    return (x - mean) / std


def _topk_enrichment(score: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor, density: float) -> float:
    valid_score = score[mask]
    valid_res = residual[mask]
    n = valid_score.numel()
    if n < 8:
        return float("nan")
    k = max(1, min(n, int(round(n * density))))
    top_score_idx = torch.topk(valid_score, k=k).indices
    top_res_idx = torch.topk(valid_res, k=k).indices
    selected = torch.zeros(n, dtype=torch.bool, device=score.device)
    selected[top_score_idx] = True
    high_res = torch.zeros(n, dtype=torch.bool, device=score.device)
    high_res[top_res_idx] = True
    overlap = (selected & high_res).float().mean().item()
    baseline = k / n
    return overlap / max(baseline, 1e-6)


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[:, 1:] = x[:, :-1]
    return y


def _causal_rolling_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of previous valid positions only; current position is excluded."""
    prev_x = _shift_right(x * mask.float())
    prev_m = _shift_right(mask.float())
    denom = prev_m.cumsum(dim=1).clamp(min=1.0)
    return prev_x.cumsum(dim=1) / denom


def _bigram_surprise(src: torch.Tensor, valid: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Causal byte-transition surprise estimated from the current probe batch.

    The score at position t uses only byte[t-1] -> byte[t] transition counts
    aggregated over the sampled batch. This is not a trained language model, but
    it is a cheap causal proxy for local byte-level surprise.
    """
    device = src.device
    vocab = int(MASK_ID + 1)
    counts = torch.full((vocab, vocab), smoothing, device=device)
    src_ids = src.clamp(min=0, max=vocab - 1).long()

    for row, row_valid in zip(src_ids, valid):
        usable = row_valid[:-1] & row_valid[1:]
        if not usable.any():
            continue
        prev = row[:-1][usable]
        cur = row[1:][usable]
        flat = prev * vocab + cur
        binc = torch.bincount(flat, minlength=vocab * vocab).float().to(device)
        counts += binc.view(vocab, vocab)

    probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-6)
    score = torch.zeros_like(src.float())
    prev = src_ids[:, :-1]
    cur = src_ids[:, 1:]
    score[:, 1:] = -torch.log(probs[prev, cur].clamp(min=1e-9))
    score = score.masked_fill(~valid, 0.0)
    return score


def _build_fmc_scores(
    bp: torch.Tensor,
    ce: torch.Tensor,
    delta: torch.Tensor,
    src: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    raw = src.float()
    pos = torch.linspace(0, 1, src.size(1), device=src.device).view(1, -1).expand_as(raw)
    ce_shift = _shift_right(ce)
    ce_roll = _causal_rolling_mean(ce, valid)
    surprise = _bigram_surprise(src, valid)

    def score(weighted_features: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        total = torch.zeros_like(raw)
        for feature, weight in weighted_features:
            total = total + _standardize(feature, valid) * weight
        total = total + pos * 0.05
        return torch.sigmoid(total).masked_fill(~valid, 0.0)

    return {
        # Diagnostic upper bound. It sees the same-position residual and cannot
        # be used directly at generation time.
        "oracle": score([(bp, 0.35), (ce, 0.45), (delta, 0.15)]),
        # Causal proxy: previous residual only.
        "shifted_residual": score([(bp, 0.35), (ce_shift, 0.45), (delta, 0.15)]),
        # Causal proxy: past running residual baseline.
        "rolling_residual": score([(bp, 0.35), (ce_roll, 0.45), (delta, 0.15)]),
        # No reconstruction residual. This approximates a zero-extra-model
        # runtime controller.
        "no_residual": score([(bp, 0.55), (delta, 0.35)]),
        # Cheap causal surprise from byte transition frequency.
        "bigram_surprise": score([(bp, 0.30), (surprise, 0.50), (delta, 0.15)]),
    }


def _tensor_stats(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    y = x.float()
    if mask is not None:
        while mask.dim() < y.dim():
            mask = mask.unsqueeze(-1)
        y = y.masked_select(mask.expand_as(y))
    if y.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "abs_mean": float("nan"), "max": float("nan")}
    return {
        "mean": y.mean().item(),
        "std": y.std(unbiased=False).item(),
        "abs_mean": y.abs().mean().item(),
        "max": y.max().item(),
    }


@torch.no_grad()
def capture_activations(
    model: FLUEDAutoencoder,
    src: torch.Tensor,
    fmc_score: torch.Tensor,
    residual: torch.Tensor,
    max_positions: int = 128,
) -> Dict:
    """Capture compact activation diagnostics for one sample.

    This intentionally stores per-position scalar summaries, not full hidden
    vectors, to keep files small and safe for repeated experiments.
    """
    src = src[:1]
    valid = (src != PAD_ID) & (src != MASK_ID)
    pad_mask = src == PAD_ID
    fmc_score = fmc_score[:1]
    residual = residual[:1]

    emb = model.embedding(src)
    h = model.pos_enc(emb)
    enc_layers: List[torch.Tensor] = [h.detach().float()]
    enc_stats = [{"name": "embedding_pos", **_tensor_stats(h, valid)}]
    for i, block in enumerate(model.blocks):
        h = block.forward_block(h, key_padding_mask=pad_mask)
        enc_layers.append(h.detach().float())
        enc_stats.append({"name": f"encoder_{i:02d}", **_tensor_stats(h, valid)})

    delta = torch.zeros_like(h)
    delta[:, 1:] = h[:, 1:] - h[:, :-1]
    boundary_scores = model.boundary_head(delta).squeeze(-1)
    byte_types = model._classify_bytes(src)
    expanded, metrics = model._compile_semantic_units(
        h,
        boundary_scores,
        pad_mask,
        byte_types,
        skip_hard=True,
    )
    bp = metrics["boundary_probs"].float()

    x = expanded
    dec_stats = [{"name": "expanded", **_tensor_stats(x, valid)}]
    dec_layers: List[torch.Tensor] = [x.detach().float()]
    for i, block in enumerate(reversed(model.blocks)):
        x = block.inverse_block(x, key_padding_mask=pad_mask)
        dec_layers.append(x.detach().float())
        dec_stats.append({"name": f"decoder_{i:02d}", **_tensor_stats(x, valid)})

    limit = min(src.size(1), max_positions)
    pos = torch.arange(limit, device=src.device)
    sample = {
        "token_ids": src[0, :limit].detach().cpu().tolist(),
        "valid": valid[0, :limit].detach().cpu().int().tolist(),
        "boundary_prob": bp[0, :limit].detach().cpu().tolist(),
        "boundary_score": boundary_scores[0, :limit].float().detach().cpu().tolist(),
        "fmc_score": fmc_score[0, :limit].float().detach().cpu().tolist(),
        "residual_ce": residual[0, :limit].float().detach().cpu().tolist(),
        "delta_norm": delta[0, :limit].float().norm(dim=-1).detach().cpu().tolist(),
        "expanded_norm": expanded[0, :limit].float().norm(dim=-1).detach().cpu().tolist(),
        "encoder_first_norm": enc_layers[0][0, :limit].norm(dim=-1).detach().cpu().tolist(),
        "encoder_last_norm": enc_layers[-1][0, :limit].norm(dim=-1).detach().cpu().tolist(),
        "decoder_last_norm": dec_layers[-1][0, :limit].norm(dim=-1).detach().cpu().tolist(),
        "position": pos.detach().cpu().tolist(),
    }

    return {
        "seq_len": int(src.size(1)),
        "captured_positions": int(limit),
        "encoder_layer_stats": enc_stats,
        "decoder_layer_stats": dec_stats,
        "sample": sample,
    }


def write_activation_plot(activation: Dict, output_png: str) -> None:
    import matplotlib.pyplot as plt

    sample = activation["sample"]
    x = sample["position"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), dpi=140, sharex=True)

    axes[0].plot(x, sample["boundary_prob"], label="native boundary")
    axes[0].plot(x, sample["fmc_score"], label="FMC score", alpha=0.8)
    axes[0].set_ylabel("boundary")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, sample["residual_ce"], label="CE residual", color="tab:red")
    axes[1].plot(x, sample["delta_norm"], label="delta hidden norm", color="tab:purple", alpha=0.8)
    axes[1].set_ylabel("difficulty")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(x, sample["encoder_first_norm"], label="encoder first norm")
    axes[2].plot(x, sample["encoder_last_norm"], label="encoder last norm")
    axes[2].plot(x, sample["expanded_norm"], label="expanded norm")
    axes[2].plot(x, sample["decoder_last_norm"], label="decoder last norm")
    axes[2].set_ylabel("activation norm")
    axes[2].set_xlabel("byte position")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


@torch.no_grad()
def probe_checkpoint(
    ckpt_path: str,
    data_path: str,
    seq_len: int,
    max_lines: int,
    max_batches: int,
    batch_size: int,
    device: torch.device,
    activation_json: Optional[str] = None,
    activation_plot: Optional[str] = None,
) -> Dict:
    model = _infer_model_from_ckpt(ckpt_path, device)
    texts: List[str] = []
    with open(data_path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            if line:
                texts.append(line)
    dataset = ByteReconstructionDataset(texts=texts, seq_len=seq_len, stride=seq_len)

    native_densities: List[float] = []
    native_enrich: List[float] = []
    residual_corr_native: List[float] = []
    fmc_stats: Dict[str, Dict[str, List[float]]] = {}

    generator = torch.Generator().manual_seed(1234)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    batches = 0

    activation_written = False

    for start in range(0, len(indices), batch_size):
        if batches >= max_batches:
            break
        batch_idx = indices[start : start + batch_size]
        src = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        valid = (src != PAD_ID) & (src != MASK_ID)
        if valid.sum().item() == 0:
            continue

        logits, metrics = model(src, src_key_padding_mask=(src == PAD_ID), skip_hard=True)
        bp = metrics["boundary_probs"].float()
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            src.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view_as(src).float()

        # A cheap local-change proxy. This is not hidden-state curvature, but it
        # gives the fixed controller one non-loss local signal.
        raw = src.float()
        delta = torch.zeros_like(raw)
        delta[:, 1:] = (raw[:, 1:] - raw[:, :-1]).abs()

        fmc_scores = _build_fmc_scores(bp=bp, ce=ce, delta=delta, src=src, valid=valid)

        if (activation_json or activation_plot) and not activation_written:
            fmc_score = fmc_scores["oracle"]
            activation = capture_activations(model, src, fmc_score, ce, max_positions=min(seq_len, 192))
            if activation_json:
                out = Path(activation_json)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(activation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if activation_plot:
                write_activation_plot(activation, activation_plot)
            activation_written = True

        native_density = bp[valid].mean().item()

        native_densities.append(native_density)
        native_enrich.append(_topk_enrichment(bp, ce, valid, native_density))

        s = bp[valid].float()
        r = ce[valid].float()
        if s.numel() > 4:
            s = (s - s.mean()) / s.std(unbiased=False).clamp(min=1e-6)
            r_norm = (r - r.mean()) / r.std(unbiased=False).clamp(min=1e-6)
            residual_corr_native.append((s * r_norm).mean().item())

        for mode, fmc_score in fmc_scores.items():
            bucket = fmc_stats.setdefault(mode, {"density": [], "enrichment": [], "corr": []})
            bucket["density"].append(fmc_score[valid].mean().item())
            bucket["enrichment"].append(_topk_enrichment(fmc_score, ce, valid, native_density))
            s_mode = fmc_score[valid].float()
            if s_mode.numel() > 4:
                s_mode = (s_mode - s_mode.mean()) / s_mode.std(unbiased=False).clamp(min=1e-6)
                r_norm = (r - r.mean()) / r.std(unbiased=False).clamp(min=1e-6)
                bucket["corr"].append((s_mode * r_norm).mean().item())

        batches += 1

    def mean(xs: Iterable[float]) -> float:
        vals = [x for x in xs if not math.isnan(float(x))]
        return float(sum(vals) / max(len(vals), 1))

    fmc_summary = {
        mode: {
            "density": mean(values["density"]),
            "residual_enrichment": mean(values["enrichment"]),
            "residual_corr": mean(values["corr"]),
        }
        for mode, values in sorted(fmc_stats.items())
    }

    oracle = fmc_summary.get("oracle", {})
    return {
        "checkpoint": ckpt_path,
        "seq_len": seq_len,
        "batches": batches,
        "native_density": mean(native_densities),
        "native_residual_enrichment": mean(native_enrich),
        "native_residual_corr": mean(residual_corr_native),
        "fmc_modes": fmc_summary,
        # Backward-compatible aliases for older plotting scripts. These now mean
        # oracle FMC and should not be interpreted as deployable runtime scores.
        "fmc_score_density": oracle.get("density", float("nan")),
        "fmc_residual_enrichment": oracle.get("residual_enrichment", float("nan")),
        "fmc_residual_corr": oracle.get("residual_corr", float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe FLUED FMC boundary-control hypothesis")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--activation-json", default=None)
    parser.add_argument("--activation-plot", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    result = probe_checkpoint(
        ckpt_path=args.ckpt,
        data_path=args.data_path,
        seq_len=args.seq_len,
        max_lines=args.max_lines,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        device=device,
        activation_json=args.activation_json,
        activation_plot=args.activation_plot,
    )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
