# FLUED v0.4 (Minimal)

FLUED v0.4 implements one core idea:

> Semantic units are dynamically compiled during Prefill, and Decode is the inverse process.

This repository is refactored to a minimal local-first architecture with tied encoder/decoder weights as far as practical in PyTorch.

## What is implemented now

- **FLUED v0.4 minimal model** (`flued/model.py`)
  - byte PAD-offset vocabulary (`PAD=0`, bytes `1..256`)
  - shallow tied Transformer stack
  - boundary scorer and dynamic unit compilation
  - span metadata + compression metrics (`m_over_n`, `num_units`, `spans`)
  - tied-weight inverse decode approximation
- **E1 Stage A runner** (`python -m flued.e1_stage_a`)
  - strict reconstruction `x -> x`
  - pass/fail checks for reconstruction accuracy and compression range
- **E2 comparison runner** (`python -m flued.e2_compare`)
  - model options: `flued`, `sentencepiece`, `tiktoken`
  - local Chinese cloze task scaffolding (idiom/connective/coreference-style)
  - perplexity + cloze accuracy report, JSON/CSV-friendly output

## What is intentionally removed from active FLUED core

- SGL three-gate module
- gamma_expand / gamma_bridge / bridge potential
- independent TransformerDecoder as conceptual inverse
- gate-entropy auxiliary loss
- oracle / phonetic / mamba / bridge mechanisms

## Directory structure

```text
FLUED/
├── flued/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── e1_stage_a.py
│   └── e2_compare.py
├── bpe_baseline/
├── blt_baseline/
└── tests/
```

## Install

Required:

```bash
pip install torch pytest
```

Optional (only for selected E2 baselines):

```bash
pip install sentencepiece tiktoken
```

## E1 commands

CPU smoke:

```bash
python -m flued.e1_stage_a --preset smoke_cpu
```

RTX 5080 16GB style run:

```bash
python -m flued.e1_stage_a \
  --preset class300m_16gb \
  --data-path /path/to/corpus.txt \
  --grad-accum-steps 16 \
  --amp --amp-dtype bf16
```

## E2 commands

FLUED + sentencepiece + tiktoken comparison:

```bash
python -m flued.e2_compare \
  --preset smoke \
  --models flued,sentencepiece,tiktoken
```

Sentencepiece-only (custom vocab):

```bash
python -m flued.e2_compare \
  --models sentencepiece \
  --sentencepiece-vocab-size 8192
```

Tiktoken-only:

```bash
python -m flued.e2_compare --models tiktoken
```

Save output:

```bash
python -m flued.e2_compare --output-json e2_results.json --output-csv e2_results.csv
```

## 16GB VRAM notes

- Use preset `class300m_16gb`
- Prefer `batch_size=1..2`
- Use `grad_accum_steps=8..32`
- Use `seq_len=256..512`
- Use AMP (`--amp --amp-dtype bf16` or `fp16`)
- No DDP required (single GPU local execution)

## Help commands

```bash
python -m flued.e1_stage_a --help
python -m flued.e2_compare --help
```

## Future work

- stronger inverse-flow approximations
- larger curated Chinese evaluation sets
- full-scale training recipes for long runs
