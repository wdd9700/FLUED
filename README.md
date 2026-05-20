# FLUED — Stage A Autoencoder Experiments

**Fluid Language Unified Embedding-matrix Discrete-continuous Converter**

This repository implements Stage A (autoencoder pretraining) experiments
comparing three ~300 M-parameter Transformer backbones on raw-text
reconstruction. The goal is to validate FLUED's dynamic semantic compiler
front-end against two well-understood baselines before adding the full
generation stack.

---

## Baselines

| Model | Location | Tokenisation | Key novelty |
|-------|----------|--------------|-------------|
| **FLUED** | `flued/model.py` | Byte-level (UTF-8) | Dynamic segmenter: SGL gates + AttenRes drive soft span pooling |
| **BPE-Transformer** | `bpe_baseline/model.py` | 64 k BPE subwords | Vanilla encoder–decoder; clean statistical baseline |
| **BLT** | `blt_baseline/model.py` | Byte-level, fixed-size patches | Local↔global two-level Transformer; reduces global sequence length |

All three models are trained as autoencoders: encode the input sequence to a
latent representation and reconstruct it with teacher forcing.
**Loss = cross-entropy reconstruction + optional model-specific auxiliary loss.**

---

## Directory Structure

```
FLUED/
├── flued/
│   ├── config.py      # ModelConfig / TrainConfig dataclasses + CLI argparse
│   ├── data.py        # UTF-8 utils, SimpleBPE, ByteTextDataset, BPETextDataset,
│   │                  #   get_dataloader, dynamic span utilities
│   ├── model.py       # FLUED Stage A autoencoder (DSC / AttenRes / SGL)
│   └── train.py       # Unified Trainer, build_model, eval_step, set_seed
├── bpe_baseline/
│   └── model.py       # BPETransformerAutoencoder
├── blt_baseline/
│   └── model.py       # BLTAutoencoder (LocalEncoder → Patcher → Global → LocalDecoder)
├── tests/
│   ├── conftest.py    # sys.path setup for pytest
│   ├── test_reconstruction.py  # Model forward / backward / trainer tests
│   └── test_utf8.py            # Data pipeline UTF-8 edge-case tests
└── README.md
```

---

## Requirements

```
Python  >= 3.10
PyTorch >= 2.2
pytest  >= 7.0   (for tests)
```

Install:

```bash
pip install torch pytest
```

No other dependencies are required.

---

## Quick Start

### Train FLUED (small, CPU smoke test)

```bash
python -m flued.train \
    --model-type flued \
    --size small \
    --max-steps 500 \
    --batch-size 8 \
    --log-interval 100 \
    --device cpu
```

### Train BPE baseline (small)

```bash
python -m flued.train \
    --model-type bpe \
    --size small \
    --max-steps 500 \
    --batch-size 8 \
    --device cpu
```

### Train BLT baseline (small)

```bash
python -m flued.train \
    --model-type blt \
    --size small \
    --max-steps 500 \
    --batch-size 8 \
    --device cpu
```

### Full 300 M experiment (Ryzen 9950X3D / RTX 5080 16 GB)

```bash
# Prepare a plain-text corpus (one document per line, UTF-8)
python -m flued.train \
    --model-type flued \
    --size 300M \
    --data-path /path/to/corpus.txt \
    --max-steps 50000 \
    --batch-size 16 \
    --max-seq-len 512 \
    --lr 1e-4 \
    --warmup-steps 2000 \
    --log-interval 200 \
    --eval-interval 2000 \
    --save-interval 5000 \
    --output-dir checkpoints/flued_300M \
    --device cuda
```

Repeat with `--model-type bpe` and `--model-type blt` to run all three
baselines under the same training conditions.

---

## Architecture Details

### FLUED — Dynamic Semantic Compiler (Stage A)

```
Byte ids [B, T]
  │
  ├─ Embedding + PositionalEncoding
  │
  ├─ ShallowEncoder (2–4 Transformer layers)
  │     └─ records cross-layer AttenRes = Σ_l w_l · (H^{l+1} − H^l)
  │
  ├─ SGLGatingModule
  │     input: [hidden ‖ attenres]  →  3 sigmoid gates per position
  │       γ_compress  high → merge with previous span
  │       γ_expand    high → force semantic boundary here
  │       γ_bridge    high → write bridge potential for long-range link
  │
  ├─ DynamicLatentEncoder  (differentiable soft span pooling)
  │     acc_t = γ_compress_t · acc_{t−1} + (1−γ_compress_t) · h_t
  │     latent = MLP([acc ‖ attenres])
  │
  ├─ DeepEncoder (remaining Transformer layers)
  │
  └─ TransformerDecoder (causal, teacher-forced)
        → Linear → logits [B, T, vocab_size]
```

**Auxiliary loss**: SGL gate entropy regularisation prevents gate collapse
by maximising the binary entropy of each gate distribution.

### BPE-Transformer

```
BPE token ids [B, T]
  └─ Embedding + PE → TransformerEncoder → TransformerDecoder → Linear → logits
```

No inductive biases; pure statistical learning from subword sequences.
64 k vocabulary trained with `SimpleBPE` directly on the corpus.

### BLT — Byte Latent Transformer

```
Byte ids [B, T]
  ├─ LocalEncoder  (shallow Transformer, byte level)
  ├─ Patcher       (fixed-size patches of P bytes → T/P patches)
  ├─ GlobalTransformer  (operates on T/P patch vectors — O(n²/P²) attention)
  └─ LocalDecoder  (expands patches → bytes → logits)
```

*Note*: The full BLT paper uses entropy-based dynamic patching.
The current implementation uses fixed-size patches as a practical stub;
the `Patcher` class documents where to swap in entropy-based logic.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output (all tests pass):

```
tests/test_utf8.py::TestUTF8ByteEncoding::test_ascii_round_trip           PASSED
tests/test_utf8.py::TestUTF8ByteEncoding::test_cjk_round_trip             PASSED
...
tests/test_reconstruction.py::TestFLUEDAutoencoder::test_forward_output_shape  PASSED
...
```

---

## Experiment Configuration

All hyperparameters are controlled through `flued/config.py`.

### Size presets

| Preset | d_model | nhead | d_ff | enc layers | dec layers | ≈ params |
|--------|---------|-------|------|-----------|-----------|---------|
| `small` | 256 | 4 | 1024 | 4 | 4 | ~3 M |
| `medium` | 512 | 8 | 2048 | 8 | 8 | ~50 M |
| `300M` | 1024 | 16 | 4096 | 12 | 12 | ~300 M |

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `flued` | `flued` / `bpe` / `blt` |
| `--size` | `small` | Size preset (`small` / `medium` / `300M`) |
| `--data-path` | None | Plain-text corpus (one doc per line); uses stub corpus if omitted |
| `--seed` | `42` | Reproducibility seed |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--max-seq-len` | `512` | Sequence chunk length |
| `--batch-size` | `8` | Training batch size |
| `--max-steps` | `5000` | Total gradient update steps |
| `--lr` | `1e-4` | Peak learning rate (AdamW) |
| `--warmup-steps` | `500` | Linear warmup steps |

---

## Evaluation Metric

**Reconstruction accuracy** = fraction of non-padding positions where the
argmax of the output logits matches the target byte/token id exactly.

Reported at every `--eval-interval` steps to `stdout` via Python logging.
Checkpoints saved to `--output-dir`; best checkpoint tracked by eval loss.

---

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage A** | 🔄 current | Autoencoder pretraining — validates DSC + SGL front-end |
| **Stage B** | planned | Hybrid backbone (Transformer-Mamba) + KV Cache integration |
| **Stage C** | planned | Oracle distillation warmup, phonetic embedding, end-to-end generation |
| **Stage D** | planned | vLLM/TensorRT plugin, INT4 quantisation, 32 k context tests |

---

## Reference

> FLUED Architecture Book v1.0 — *Fluid Language Unified Embedding-matrix
> Discrete-continuous Converter* (internal design document, 2025–2026)