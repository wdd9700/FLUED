"""Boundary linguistic-drift audit: FLUED cuts must be a subset of BPE token cuts.

Metric (user-specified, 2026-08-02): tokenize the same text with a 128k-class
byte-level BPE (trained on corpus v4); every FLUED boundary (byte offset) that
does NOT coincide with a BPE token boundary is a word-internal cut (drift).
Report: subset ratio (1.0 = every cut lands on a BPE boundary), chunk-length
stats, and per-checkpoint comparison.

Byte-offset alignment: with the ByteLevel pre-tokenizer each byte maps to
exactly one unicode char, so a token's byte length == len(token string).
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID  # noqa: E402
from tools.train.v3_6.train_v36 import build_model  # noqa: E402


def bpe_boundary_offsets(tokenizer: Tokenizer, text: str) -> set[int]:
    enc = tokenizer.encode(text, add_special_tokens=False)
    boundaries = set()
    offset = 0
    for tok in enc.tokens:
        offset += len(tok)  # ByteLevel: 1 char == 1 byte
        boundaries.add(offset)
    boundaries.discard(len(text.encode("utf-8")))
    return boundaries


def flued_cut_offsets(model, text: str, device, tau_cut: float) -> set[int]:
    raw = text.encode("utf-8")
    ids = torch.tensor([[b + 1 for b in raw]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, confidence, valid, _ = model._encode(ids)
        cuts, _utf8_cont, _overflow = model._cuts(ids, confidence, valid)
    positions = cuts[0].nonzero(as_tuple=True)[0].tolist()
    # token index i == boundary before byte i; index 0 is the forced first-chunk
    # start (not a model decision), so it is excluded from the subset metric.
    return {int(p) for p in positions if p > 0}


def sample_texts(corpus: Path, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    files = sorted(corpus.glob("*.txt")) if corpus.is_dir() else [corpus]
    sizes = [(f, f.stat().st_size) for f in files]
    rows = []
    while len(rows) < count:
        f, size = rng.choice(sizes)
        with f.open("rb") as fh:
            fh.seek(int(size * rng.uniform(0.0, 0.94)))
            block = fh.read(65536).decode("utf-8", errors="ignore")
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        for line in lines:
            n = len(line.encode("utf-8"))
            if 200 <= n <= 512 and "�" not in line:
                rows.append(line)
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="repeatable; label=path or path")
    parser.add_argument("--config", required=True, help="resolved_config.json of the run (model shape source)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-cut", type=float, default=0.0, help="override tau_cut (tanh space); 0 = use config value")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    cli = parser.parse_args()

    tokenizer = Tokenizer.from_file(cli.tokenizer)
    texts = sample_texts(Path(cli.corpus), cli.num_samples, cli.seed)
    bpe_sets = [bpe_boundary_offsets(tokenizer, t) for t in texts]
    config = json.loads(Path(cli.config).read_text(encoding="utf-8"))
    device = torch.device(cli.device if cli.device == "cuda" and torch.cuda.is_available() else "cpu")

    results = {}
    for spec in cli.checkpoint:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        payload = torch.load(path, map_location=device, weights_only=False)
        run_args = dict(config)
        saved = payload.get("args") or {}
        for key in ("per_chunk_readout", "summarizer_type", "summarizer_dit_layers"):
            if key in saved:
                run_args[key] = saved[key]
        run_args.setdefault("per_chunk_readout", False)
        run_args.setdefault("summarizer_type", "slot")
        run_args.setdefault("summarizer_dit_layers", 2)
        model = build_model(Namespace(**run_args)).to(device)
        model.load_state_dict(payload["model"], strict=False)
        model.eval()
        tau = cli.tau_cut if cli.tau_cut > 0 else float(run_args.get("tau_cut", 0.94))
        model.config.tau_cut = tau

        total_cuts = 0
        in_subset = 0
        near_subset = 0
        chunk_lens = []
        per_row = []
        for text, bpe_set in zip(texts, bpe_sets):
            cuts = flued_cut_offsets(model, text, device, tau)
            hits = len(cuts & bpe_set)
            near = sum(1 for c in cuts if any((c + d) in bpe_set for d in range(-3, 4)))
            total_cuts += len(cuts)
            in_subset += hits
            near_subset += near
            prev = 0
            for c in sorted(cuts) + [len(text.encode("utf-8"))]:
                chunk_lens.append(c - prev)
                prev = c
            per_row.append({"text": text[:60], "cuts": len(cuts), "hits": hits})
        chunk_lens.sort()
        n = len(chunk_lens)
        results[label] = {
            "cuts": total_cuts,
            "subset_ratio": in_subset / max(total_cuts, 1),
            "subset_ratio_near_pm3B": near_subset / max(total_cuts, 1),
            "chunk_bytes_median": chunk_lens[n // 2] if n else 0,
            "chunk_bytes_p10": chunk_lens[int(n * 0.1)] if n else 0,
            "chunk_bytes_p90": chunk_lens[int(n * 0.9)] if n else 0,
        }
        print(f"[audit] {label}: {results[label]}", flush=True)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bpe_subset_audit.json").write_text(
        json.dumps({"tokenizer": cli.tokenizer, "corpus": cli.corpus, "num_samples": cli.num_samples,
                    "seed": cli.seed, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
