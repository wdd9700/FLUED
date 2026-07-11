"""Train a tiny causal surprise probe for FLUED-v3 diagnostics.

This script does not change FLUED. It answers one narrow question:

Can a small causal model, seeing only previous bytes, predict the positions
where a trained FLUED checkpoint has high reconstruction residual?

If yes, the residual signal can be approximated at runtime and may be useful
for boundary control. If no, the earlier FMC oracle result is mostly a
non-deployable teacher signal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import ByteReconstructionDataset
from flued.model import MASK_ID, PAD_ID
from tools.analysis.v3_0.fmc_boundary_probe import (
    _bigram_surprise,
    _infer_model_from_ckpt,
    _standardize,
    _topk_enrichment,
)


class CausalSurpriseProbe(nn.Module):
    def __init__(self, vocab_size: int = 258, d_model: int = 96, hidden: int = 160, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.rnn = nn.GRU(d_model, hidden, num_layers=layers, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden)
        self.reg_head = nn.Linear(hidden, 1)
        self.byte_head = nn.Linear(hidden, vocab_size)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        causal_src = torch.full_like(src, PAD_ID)
        causal_src[:, 1:] = src[:, :-1]
        x = self.embedding(causal_src.clamp(min=0, max=MASK_ID))
        h, _ = self.rnn(x)
        h = self.norm(h)
        residual_pred = self.reg_head(h).squeeze(-1)
        byte_logits = self.byte_head(h)
        return residual_pred, byte_logits


def _load_texts(data_path: str, max_lines: int) -> List[str]:
    texts: List[str] = []
    with open(data_path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            if line:
                texts.append(line)
    if not texts:
        raise RuntimeError(f"no non-empty text loaded from {data_path}")
    return texts


@torch.no_grad()
def _residual_labels(flued: nn.Module, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (src != PAD_ID) & (src != MASK_ID)
    logits, metrics = flued(src, src_key_padding_mask=(src == PAD_ID), skip_hard=True)
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        src.view(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(src).float()
    return ce, metrics["boundary_probs"].float(), valid


def _corr(score: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    s = score[valid].float()
    t = target[valid].float()
    if s.numel() <= 4:
        return float("nan")
    s = (s - s.mean()) / s.std(unbiased=False).clamp(min=1e-6)
    t = (t - t.mean()) / t.std(unbiased=False).clamp(min=1e-6)
    return (s * t).mean().item()


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if not math.isnan(float(x))]
    return float(sum(vals) / max(len(vals), 1))


@torch.no_grad()
def evaluate(
    flued: nn.Module,
    probe: CausalSurpriseProbe,
    dataset: ByteReconstructionDataset,
    indices: List[int],
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    probe.eval()
    native_corr: List[float] = []
    bigram_corr: List[float] = []
    learned_corr: List[float] = []
    native_enrich: List[float] = []
    bigram_enrich: List[float] = []
    learned_enrich: List[float] = []

    batches = 0
    for start in range(0, len(indices), batch_size):
        if batches >= max_batches:
            break
        src = torch.stack([dataset[i][0] for i in indices[start : start + batch_size]]).to(device)
        ce, bp, valid = _residual_labels(flued, src)
        pred, byte_logits = probe(src)
        learned = _standardize(pred, valid)
        bigram = _standardize(_bigram_surprise(src, valid), valid)
        density = bp[valid].mean().item()

        native_corr.append(_corr(bp, ce, valid))
        bigram_corr.append(_corr(bigram, ce, valid))
        learned_corr.append(_corr(learned, ce, valid))
        native_enrich.append(_topk_enrichment(bp, ce, valid, density))
        bigram_enrich.append(_topk_enrichment(bigram, ce, valid, density))
        learned_enrich.append(_topk_enrichment(learned, ce, valid, density))
        batches += 1

    return {
        "batches": batches,
        "native_corr": _mean(native_corr),
        "bigram_corr": _mean(bigram_corr),
        "learned_corr": _mean(learned_corr),
        "native_enrichment": _mean(native_enrich),
        "bigram_enrichment": _mean(bigram_enrich),
        "learned_enrichment": _mean(learned_enrich),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train causal surprise probe")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-lines", type=int, default=20000)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--byte-loss-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(1234)

    flued = _infer_model_from_ckpt(args.ckpt, device)
    for p in flued.parameters():
        p.requires_grad_(False)

    texts = _load_texts(args.data_path, args.max_lines)
    dataset = ByteReconstructionDataset(texts=texts, seq_len=args.seq_len, stride=args.seq_len)
    if len(dataset) < 16:
        raise RuntimeError("dataset too small for probe training")

    gen = torch.Generator().manual_seed(20260626)
    perm = torch.randperm(len(dataset), generator=gen).tolist()
    split = max(1, int(len(perm) * 0.9))
    train_indices = perm[:split]
    eval_indices = perm[split:]
    if not eval_indices:
        eval_indices = train_indices[-min(len(train_indices), args.batch_size * args.eval_batches):]

    probe = CausalSurpriseProbe().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for step in range(1, args.train_steps + 1):
        probe.train()
        offset = ((step - 1) * args.batch_size) % len(train_indices)
        batch_idx = train_indices[offset : offset + args.batch_size]
        if len(batch_idx) < args.batch_size:
            batch_idx += train_indices[: args.batch_size - len(batch_idx)]
        src = torch.stack([dataset[i][0] for i in batch_idx]).to(device)

        with torch.no_grad():
            ce, _, valid = _residual_labels(flued, src)
            target = _standardize(ce, valid).clamp(-5.0, 5.0)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred, byte_logits = probe(src)
            reg_loss = F.mse_loss(pred[valid].float(), target[valid].float())
            byte_loss = F.cross_entropy(
                byte_logits.view(-1, byte_logits.size(-1)),
                src.view(-1),
                ignore_index=PAD_ID,
            )
            loss = reg_loss + args.byte_loss_weight * byte_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            metrics = evaluate(
                flued=flued,
                probe=probe,
                dataset=dataset,
                indices=eval_indices,
                batch_size=args.batch_size,
                max_batches=args.eval_batches,
                device=device,
            )
            print(
                "step={step} loss={loss:.4f} reg={reg:.4f} byte={byte:.4f} "
                "learned_corr={lc:.4f} bigram_corr={bc:.4f} native_corr={nc:.4f} "
                "learned_enrich={le:.4f} bigram_enrich={be:.4f} native_enrich={ne:.4f}".format(
                    step=step,
                    loss=loss.item(),
                    reg=reg_loss.item(),
                    byte=byte_loss.item(),
                    lc=metrics["learned_corr"],
                    bc=metrics["bigram_corr"],
                    nc=metrics["native_corr"],
                    le=metrics["learned_enrichment"],
                    be=metrics["bigram_enrichment"],
                    ne=metrics["native_enrichment"],
                ),
                flush=True,
            )

    result = evaluate(
        flued=flued,
        probe=probe,
        dataset=dataset,
        indices=eval_indices,
        batch_size=args.batch_size,
        max_batches=args.eval_batches,
        device=device,
    )
    result.update({
        "checkpoint": args.ckpt,
        "seq_len": args.seq_len,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "probe_params": sum(p.numel() for p in probe.parameters()),
    })
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
