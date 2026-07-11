"""
train_bpe.py — Fast BPE tokenizer training via HuggingFace tokenizers (Rust backend).

Usage:
  python train_bpe.py --vocab-size 8192 --max-lines 500000
  python train_bpe.py --vocab-size 16384 --min-bytes 4294967296  # ≥4 GB random sample
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

CORPUS = r"data/corpus.txt"
OUTPUT_DIR = Path("checkpoints/bpe_tokenizer")


def text_iterator(data_path: str, max_lines: int) -> Iterator[str]:
    """Yield stripped non-empty lines from corpus. max_lines=0 means all lines."""
    count = 0
    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
                count += 1
                if max_lines > 0 and count >= max_lines:
                    break


def random_sample_iterator(
    data_path: str, min_bytes: int, seed: int = 42
) -> Iterator[str]:
    """Yield a uniform random sample of lines totalling at least min_bytes.

    Two-pass: (1) count total bytes, (2) accept each line with
    probability = min_bytes / total_bytes, then continue until
    collected bytes ≥ min_bytes.  This gives every line an equal
    and independent chance of being selected (Bernoulli sampling).
    """
    rng = random.Random(seed)

    # ── Pass 1: measure total bytes ──────────────────────────────
    print(f"  Pass 1: measuring corpus size...")
    t0 = time.perf_counter()
    total_bytes = 0
    total_lines = 0
    with open(data_path, "rb") as f:
        for line in f:
            total_bytes += len(line)
            total_lines += 1
    print(f"    lines={total_lines:,}  bytes={total_bytes:,}  "
          f"[{time.perf_counter() - t0:.1f}s]")

    # ── Pass 2: Bernoulli sample ─────────────────────────────────
    p = min(min_bytes / max(1, total_bytes), 1.0)
    target_str = f"{min_bytes/1e9:.1f} GB" if min_bytes >= 1e9 else f"{min_bytes/1e6:.0f} MB"
    print(f"  Pass 2: random sampling (p={p:.4f}, target ≥ {target_str})...")
    t0 = time.perf_counter()
    collected_bytes = 0
    collected_lines = 0
    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if rng.random() < p:
                yield line
                collected_bytes += len(line.encode("utf-8"))
                collected_lines += 1
            # Continue sampling even after hitting min_bytes to keep
            # the Bernoulli property — but we *do* stop eventually to
            # avoid loading the full file when p is small.
            if collected_bytes >= min_bytes and collected_lines > 100000:
                break

    elapsed = time.perf_counter() - t0
    print(f"    sampled={collected_lines:,} lines  "
          f"bytes={collected_bytes:,} ({collected_bytes/1e6:.0f} MB)  "
          f"[{elapsed:.1f}s]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--max-lines", type=int, default=500000,
        help="Max lines to use (0 = all). Overridden by --min-bytes.")
    parser.add_argument(
        "--min-bytes", type=int, default=0,
        help="Minimum bytes of random sample (e.g. 4294967296 for 4 GB).")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--corpus", type=str, default=CORPUS,
                        help="Path to corpus text file.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build iterator ────────────────────────────────────────────
    if args.min_bytes > 0:
        iterator = random_sample_iterator(args.corpus, args.min_bytes, seed=args.seed)
        desc = f"random sample ≥ {args.min_bytes/1e9:.1f} GB"
    elif args.max_lines == 0:
        iterator = text_iterator(args.corpus, 0)
        desc = "ALL lines (streaming)"
    else:
        iterator = text_iterator(args.corpus, args.max_lines)
        desc = f"{args.max_lines:,} lines"

    # ── Train BPE (Rust backend) ──────────────────────────────────
    print(f"Training BPE (vocab={args.vocab_size}, {desc})...")
    t0 = time.perf_counter()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iterator, trainer)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  vocab={tokenizer.get_vocab_size()}")

    # ── Save ─────────────────────────────────────────────────────────
    tokenizer.save(str(out / "tokenizer.json"))
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": ["<pad>", "<bos>", "<eos>", "<unk>"],
        "pad_id": 0, "bos_id": 1, "eos_id": 2, "unk_id": 3,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    size_kb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1024
    print(f"Saved → {out}/  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
