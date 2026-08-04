"""BLT entropy-patching boundary audit on the 128k BPE subset ruler.

Loads the freshly trained ByteLanguageModel (v3+v4, 25M, 20K steps), computes
per-position next-byte entropy on the same v4 samples used by
boundary_bpe_subset_audit, calibrates the global entropy threshold to a target
median patch size (~21B, user granularity), then reports the BPE-subset ratios
(exact and +-3B tolerance) for comparison with the FLUED checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blt_baseline.model import ByteLanguageModel  # noqa: E402
from tools.analysis.v3_6.boundary_bpe_subset_audit import bpe_boundary_offsets, sample_texts  # noqa: E402


def entropy_cuts(ent: list[float], theta: float) -> set[int]:
    # boundary before byte i if next-byte entropy at i exceeds theta
    return {i for i, h in enumerate(ent) if h > theta and i > 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-median-bytes", type=float, default=21.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    cli = parser.parse_args()

    device = torch.device(cli.device if cli.device == "cuda" and torch.cuda.is_available() else "cpu")
    payload = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = ByteLanguageModel(
        vocab_size=cfg.get("vocab_size", 257),
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"], num_layers=cfg["num_layers"],
        max_len=cfg.get("max_seq_len", 512),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    tokenizer = Tokenizer.from_file(cli.tokenizer)
    texts = sample_texts(Path(cli.corpus), cli.num_samples, cli.seed)

    entropies = []
    with torch.no_grad():
        for text in texts:
            ids = torch.tensor([[b + 1 for b in text.encode("utf-8")]], dtype=torch.long, device=device)
            logits = model(ids)[1][0].float()  # (T, V) — forward returns (hidden, logits)
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            h = -(p * logp).sum(dim=-1)  # nats, per position
            entropies.append(h.tolist())

    # calibrate theta: choose the entropy quantile that yields median patch ~= target
    all_h = sorted(h for ent in entropies for h in ent)
    best = None
    for q in [x / 100 for x in range(50, 99)]:
        theta = all_h[int(len(all_h) * q)]
        lens = []
        for text, ent in zip(texts, entropies):
            cuts = sorted(entropy_cuts(ent, theta)) + [len(text.encode("utf-8"))]
            prev = 0
            for c in cuts:
                lens.append(c - prev)
                prev = c
        lens.sort()
        med = lens[len(lens) // 2]
        if best is None or abs(med - cli.target_median_bytes) < abs(best[1] - cli.target_median_bytes):
            best = (theta, med, q)
    theta, median_at, q = best

    bpe_sets = [bpe_boundary_offsets(tokenizer, t) for t in texts]
    total = hits = near = 0
    lens_all = []
    for text, ent, bpe_set in zip(texts, entropies, bpe_sets):
        cuts = entropy_cuts(ent, theta)
        hits += len(cuts & bpe_set)
        near += sum(1 for c in cuts if any((c + d) in bpe_set for d in range(-3, 4)))
        total += len(cuts)
        prev = 0
        for c in sorted(cuts) + [len(text.encode("utf-8"))]:
            lens_all.append(c - prev)
            prev = c
    lens_all.sort()
    n = len(lens_all)
    result = {
        "theta": theta, "theta_quantile": q, "cuts": total,
        "subset_ratio": hits / max(total, 1),
        "subset_ratio_near_pm3B": near / max(total, 1),
        "chunk_bytes_median": lens_all[n // 2], "chunk_bytes_p10": lens_all[n // 10],
        "chunk_bytes_p90": lens_all[int(n * 0.9)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blt_entropy_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
