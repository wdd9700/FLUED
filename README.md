# FLUED: FLexible Unified Encoder-Decoder

**Tokenization-free learned boundary compression for language modeling.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/pytorch-2.11-red.svg)](https://pytorch.org/)

FLUED is a single Transformer autoencoder that learns to segment raw byte sequences into semantic units via **differentiable boundary detection** — no vocabulary, no fixed patches, no external tokenizer.

---

## Why FLUED?

Tokenizers are a bottleneck. They introduce language biases, fail on typos, and need separate training. FLUED replaces tokenization with one learnable `boundary_head` — a linear layer optimized for **reconstruction fidelity**.

Unlike entropy-based patching (BLT) or similarity merging (H-Net), FLUED boundaries are end-to-end differentiable.

---

## Architecture

```
raw bytes [B, T]
    │
    ▼
┌─────────────────────────────┐
│  Embedding + Pos Encoding    │
│  TiedTransformerBlock × 24   │  ← shared enc/dec weights, SwiGLU
└──────────────┬──────────────┘
               │
    ┌──────────▼──────────┐
    │  boundary_head       │  ← Linear(1024 → 1)  →  sigmoid → p∈[0,1]
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Soft Assignment     │
    │  AᵀH → Z (pool)      │
    │  A Z → Ĥ (expand)    │
    └──────────┬──────────┘
               │
               ▼
       tied output projection  →  byte logits [B, T, 258]
```

| Component | Detail |
|-----------|--------|
| Parameters | **328M** |
| Layers | 24 tied (shared enc/dec) |
| d_model | 1024, nhead=16 |
| FFN | SwiGLU (4096→3072) |
| Vocab | 258 (PAD + 256 bytes + MASK) |

---

## Results (v0.4)

### A-Class: Reconstruction Stability (3 seeds)

| Seed | Eval Acc | m/n | bp_std | CJK bp |
|------|----------|-----|--------|--------|
| 42 | 0.9991 | 0.470 | 0.407 | 0.122 |
| 123 | 0.9993 | 0.498 | 0.391 | 0.182 |
| 999 | **0.9996** | 0.491 | 0.394 | 0.092 |
| **Mean** | **0.9993±0.0005** | **0.486±0.028** | **0.397±0.016** | **0.132±0.090** |

### Denoise Ratio Ablation

| Denoise | Acc | m/n |
|---------|-----|-----|
| 30% | 0.9987 | 0.496 |
| 50% | 0.9992 | 0.495 |
| 70% | 0.9991 | 0.470 |
| 90% | 0.9997 | 0.527 |

### Compression Target Ablation

| Weight | Target | Acc | m/n |
|--------|--------|-----|-----|
| 0.25 | 0.30 | 0.9992 | 0.483 |
| 0.25 | 0.60 | — | running |
| 0.30 | 0.20 | 0.9992 | 0.486 |
| 0.30 | 0.45 | 0.9995 | 0.501 |

Full results: [`results_summary.json`](results_summary.json)

---

## Quick Start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Smoke test (CPU, 2 min)
python -m flued.e1_stage_a --preset smoke_cpu

# Full training (GPU, 16 GB VRAM)
python -m flued.e1_stage_a --preset class300m_16gb \
    --data-path /path/to/corpus.txt --max-lines 50000 --seed 42
```

---

## Comparison

| Method | Boundary Signal | Differentiable | Extra Modules |
|--------|----------------|----------------|---------------|
| **FLUED** | reconstruction | ✅ | **+1 Linear** |
| BLT (Meta) | entropy | ❌ | dual-model |
| H-Net++ | cosine sim | ❌ | hierarchical |
| ByteFlow Net | coding-rate | ✅ | rate-distortion |

---

## Citation

```bibtex
@software{flued2025,
  title = {FLUED: FLexible Unified Encoder-Decoder},
  year = {2025--2026},
  url = {https://github.com/wdd9700/FLUED},
}
```

## License

MIT — see [LICENSE](LICENSE)
