# FLUED: FLexible Unified Encoder-Decoder

Tokenization-free learned boundary compression for byte-level language modeling.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.11 / CUDA 12.8](https://img.shields.io/badge/pytorch-2.11%20cu128-red.svg)](https://pytorch.org/)

FLUED is an experimental Transformer autoencoder that learns soft byte boundaries instead of relying on a fixed tokenizer. The current active line is **FLUED v2**: a 328M-parameter denoising reconstruction model with tied encoder/decoder blocks, type-aware boundary priors, and a frozen-segmenter downstream language-model evaluation.

This repository is the source of truth for the FLUED work:

```text
E:\projects\FLUED\FLUED
```

The CTM-OCR migration copy is not the active FLUED repository.

## Current Status

FLUED v2 is functional and stable for E1 reconstruction, but it is **not yet a finished replacement for BPE**. The latest fair downstream comparison at 2048 original bytes and 100K steps shows:

| Method | Context Budget | Steps | BPB | KV / 1KB | Notes |
|--------|----------------|-------|-----|---------|-------|
| BPE-8K | 2048 original bytes | 100K | **0.8066** | 164.0 | Current strongest baseline |
| BPE-16K | 2048 original bytes | 100K | 0.8165 | 149.1 | Fair byte-denominator fix applied |
| BPE-32K | 2048 original bytes | 100K | 0.8205 | 135.6 | Fair byte-denominator fix applied |
| FLUED v2 | 2048 original bytes | 100K | 0.8732 | 546.1 | Stable, but still behind BPE |
| BLT theta=0.3 | 2048 original bytes | 100K | 2.3996 | 554.9 | Current reproduction is weak |

BPB means bits per byte. Lower is better.

The old fixed-token BPE runs are not valid for comparison because `--max-seq-len 2048` meant 2048 BPE tokens, not 2048 original bytes. Use only the 100K `2048byte` D1 logs for fair reporting.

## Architecture

```text
raw byte ids [B, T]
    |
    v
embedding + dynamic sinusoidal position encoding
    |
    v
TiedTransformerBlock x 24
  - LayerNorm
  - scaled dot-product self-attention
  - SwiGLU feed-forward
    |
    v
boundary_head(delta hidden) -> sigmoid boundary probabilities
    |
    v
soft assignment matrix A
  - pool:   A^T H -> Z
  - expand: A Z   -> H_expanded
    |
    v
reverse tied blocks
    |
    v
tied output projection -> byte logits [B, T, 258]
```

| Component | Value |
|-----------|-------|
| Parameters | 327,789,569 |
| Layers | 24 tied blocks |
| Hidden size | 1024 |
| Attention heads | 16 |
| Feed-forward | SwiGLU, hidden 3072 |
| Vocabulary | 258: PAD=0, bytes=1..256, MASK=257 |
| Dropout | 0.0 |
| Assignment window | 128 semantic window, still O(T^2) memory internally |

Important implementation note: `assignment_window` masks impossible assignments, but the current `_soft_assignment()` still builds a full `[B, T, T]` matrix. It is fine for 512-byte E1 and usable in the current experiments, but true 2048/4096-byte FLUED pretraining needs a real banded or streaming assignment implementation.

## E1 Training Objective

The current stable E1 setup is mixed clean/denoising reconstruction:

```text
clean bytes
  -> with probability denoise_prob, replace spans with MASK_ID
  -> FLUED
  -> predict the clean bytes
```

Default denoising settings:

| Parameter | Value |
|-----------|-------|
| `denoise_prob` | 0.7 |
| `corrupt_rate` | 0.15 |
| `span_mask_prob` | 0.7 |
| `span_min`, `span_max` | 1, 8 |
| `latent_consistency_weight` | **0.0** |

Latent consistency was tested with mean squared error (MSE) between clean and corrupted expanded latents. It caused large loss spikes and boundary collapse, so it is disabled in the current stable v2 runs.

## E1 Results

### A-Class: Three-Seed Stability

All runs use `latent_consistency_weight=0`, fixed 512-byte chunks, and v2 denoising reconstruction.

| Seed | Eval Acc | m/n | bp_std | CJK bp |
|------|----------|-----|--------|--------|
| 42 | 0.9991 | 0.470 | 0.407 | 0.122 |
| 123 | 0.9993 | 0.498 | 0.391 | 0.182 |
| 999 | **0.9996** | 0.491 | 0.394 | 0.092 |
| Mean | 0.9993 +/- 0.0005 | 0.486 +/- 0.028 | 0.397 +/- 0.016 | 0.132 +/- 0.090 |

### Denoising Ratio Ablation

| `denoise_prob` | Eval Acc | m/n | Status |
|----------------|----------|-----|--------|
| 0.3 | 0.9987 | 0.496 | complete |
| 0.5 | 0.9992 | 0.495 | complete |
| 0.7 | 0.9991 | 0.470 | baseline |
| 0.9 | 0.9997 | 0.527 | complete |

Higher denoising does not hurt reconstruction accuracy here, but it tends to increase m/n, meaning less compression.

### Compression Target Ablation

| Weight | Target | Eval Acc | m/n | Status |
|--------|--------|----------|-----|--------|
| 0.30 | 0.20 | 0.9992 | 0.486 | complete |
| 0.30 | 0.30 | 0.0000 | NaN | failed around step 27.7K |
| 0.30 | 0.45 | 0.9995 | 0.501 | complete |
| 0.30 | 0.60 | NaN | NaN | failed around step 27.1K |
| 0.25 | 0.30 | 0.9992 | 0.483 | complete |
| 0.25 | 0.60 | running | running | local run in progress |

The current compression controller is not strong enough. It often converges to roughly 0.48-0.53 m/n even when target compression changes. Aggressive compression loss can also destabilize the boundary head.

## Evidence Locations

Key local result file:

```text
results_summary.json
```

Fair 100K downstream logs and checkpoints are archived under:

```text
K:\FLUED_archive\cloud_5090_D1_20260610\westc_100k_20260613
K:\FLUED_archive\cloud_5090_D1_20260610\westd_blt_100k_20260614
```

Representative logs:

```text
K:\FLUED_archive\cloud_5090_D1_20260610\westc_100k_20260613\d1_flued_v2_2048byte_100k\d1_flued_v2_2048byte_100k.log
K:\FLUED_archive\cloud_5090_D1_20260610\westc_100k_20260613\d1_bpe_8k_2048byte_100k\d1_bpe_8k_2048byte_100k.log
K:\FLUED_archive\cloud_5090_D1_20260610\westd_blt_100k_20260614\d1_blt_bpb_2048_theta03_100k\d1_blt_bpb_2048_theta03_100k.log
```

## Quick Start

Install the same core stack used by the current experiments:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install tokenizers numpy tqdm
```

CPU smoke test:

```bash
python -m flued.e1_stage_a --preset smoke_cpu
```

Representative v2 E1 run:

```bash
python -m flued.e1_stage_a \
  --preset class300m_16gb \
  --data-path /path/to/corpus_v3.txt \
  --ckpt-dir checkpoints/e1_v2_seed42 \
  --max-steps 50000 \
  --seq-len 512 \
  --stride 256 \
  --grad-accum-steps 12 \
  --denoise-prob 0.7 \
  --corrupt-rate 0.15 \
  --latent-consistency-weight 0 \
  --amp --amp-dtype bf16
```

Representative downstream run:

```bash
python -m flued.e3_train \
  --preset 500m \
  --model flued \
  --flued-ckpt checkpoints/e1_v2_seed42/e1_step50000.pt \
  --data-path /path/to/corpus_v3.txt \
  --max-seq-len 2048 \
  --max-steps 100000 \
  --amp --amp-dtype bf16
```

## Comparison Scope

The current D1 comparison controls the original-byte context budget and the downstream training steps. It does not prove that FLUED is better than BPE. The strongest current claim is narrower:

```text
FLUED v2 learns stable differentiable byte boundaries and can train a frozen-segmenter downstream LM,
but current compression control and downstream efficiency still need work.
```

Known engineering gaps:

- True banded/streaming soft assignment for long contexts.
- Better compression schedule instead of a single fixed penalty.
- Mixed-length E1 training instead of only fixed 512-byte chunks.
- A cleaner public launcher for the valid 2048-byte D1 matrix.
- Stronger BLT reproduction before making claims against BLT.

## Repository Map

```text
flued/model.py          FLUED v2 model
flued/e1_stage_a.py     E1 denoising reconstruction trainer
flued/e3_train.py       Downstream language-model training
flued/e3_downstream.py  FLUED/BPE/BLT downstream wrappers
flued/data.py           Byte encoding helpers
tools/eval/             Evaluation and summary scripts
tools/launcher/         Local/cloud launch scripts, some historical
```

Treat `tools/launcher/run_d1_bpb.ps1` as historical until it is replaced with the final 2048-byte fair matrix launcher.

## License

MIT. See [LICENSE](LICENSE).
