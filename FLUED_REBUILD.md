# FLUED v2 Semantic Rebuild

This document tracks the active FLUED rebuild in this repository:

```text
E:\projects\FLUED\FLUED
```

The CTM-OCR migration copy is not the source of truth for this work.

## Goal

Move FLUED from pure byte reconstruction toward semantic denoising reconstruction while preserving the core FLUED identity:

- byte-level input
- dynamic soft segmentation
- type-aware boundary priors
- tied-weight inverse decoder
- simple architecture without a separate tokenizer or standalone decoder

## Architecture

Implemented in:

```text
flued\model.py
```

FLUED v2 keeps PAD-offset byte encoding and adds one explicit corruption token:

```text
PAD_ID=0
raw byte b -> b+1
MASK_ID=257
VOCAB_SIZE=258
```

Main changes:

- Denoising input support with `MASK_ID=257`.
- Existing five-class boundary system is retained:
  - `utf8_cont`
  - `ascii`
  - `cjk`
  - `op`
  - `digit`
- FFN changes from GELU MLP to SwiGLU.
- Default `swiglu_hidden=1536`.
- Self-attention uses PyTorch SDPA kernels.
- Sinusoidal PE grows dynamically for longer sequences.
- Soft assignment is banded with `assignment_window=128` by default.
- Length-aware compression is retained:
  - `target = max(target_compression, min_boundary_units / valid_length)`
- Latent consistency is retained as a training objective, not a model dependency.
- RMSNorm is intentionally not used.

Core path:

```text
byte / mask ids
  -> embedding + dynamic sinusoidal PE
  -> tied Transformer blocks with SDPA + SwiGLU
  -> boundary_head(delta H)
  -> banded log-space soft assignment
  -> z_soft = A^T H
  -> expanded = A z_soft
  -> reverse tied blocks
  -> tied byte projection
```

## Training Objective

The main E1 path is now clean/denoise mixed reconstruction.

For each batch:

```text
clean x
  -> with probability denoise_prob, replace spans with MASK_ID
  -> model(corrupted x)
  -> predict clean x
```

Default denoising settings:

```text
denoise_prob=0.7
corrupt_rate=0.15
span_mask_prob=0.7
span_min=1
span_max=8
latent_consistency_weight=0.03
```

Loss:

```text
CE(model_input -> clean_target)
+ compression/type boundary loss
+ latent_consistency_weight * MSE(expanded_corrupt, stopgrad(expanded_clean))
```

## Checkpoint Compatibility

This is a breaking architecture change.

Old checkpoints are not compatible with FLUED v2:

```text
checkpoints\e1_*
K:\FLUED_archive\**\e1_*.pt
K:\FLUED_backup\**\*.pt
```

Reasons:

- Vocabulary changed from 257 to 258.
- Attention parameter names changed from `nn.MultiheadAttention` to explicit SDPA projections.
- FFN changed from GELU MLP to SwiGLU.
- Soft assignment can be windowed.
- The objective changed from pure reconstruction to denoising reconstruction.

Train FLUED v2 from scratch. Use older checkpoints only for historical comparison.

## Recommended S1 Run

Example RTX 5080 / `soulvlm` command:

```powershell
cd E:\projects\FLUED\FLUED
$env:KMP_DUPLICATE_LIB_OK='TRUE'
conda run --no-capture-output -n soulvlm python -u -m flued.e1_stage_a `
  --preset class300m_16gb `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --ckpt-dir checkpoints\flued_v2_s1 `
  --max-seq-len 2048 `
  --seq-len 2048 `
  --stride 1024 `
  --swiglu-hidden 1536 `
  --assignment-window 128 `
  --target-compression 0.3 `
  --compression-weight 0.125 `
  --min-boundary-units 1.0 `
  --denoise-prob 0.7 `
  --corrupt-rate 0.15 `
  --span-mask-prob 0.7 `
  --span-min 1 `
  --span-max 8 `
  --latent-consistency-weight 0.03 `
  --grad-accum-steps 16 `
  --amp --amp-dtype bf16 `
  --ckpt-every 2500
```

If memory is tight, reduce `--seq-len` to `1024` first. Keep the same architecture; only change the training bucket.

## Long Context Plan

Short/local reconstruction remains the prerequisite. FLUED v2 extends length in stages:

```text
S1: 512/1024/2048 denoising reconstruction
S2: 2048 stable training with banded assignment
S3: 4096 experiment with assignment_window=128 or 256
S4: hierarchical/chunk FLUED only after 4096 is stable
```

Do not switch to a separate sparse Transformer backbone before proving the banded assignment path. For FLUED, the first long-context bottleneck is the assignment matrix, not sinusoidal PE.

## Asset Map

Active repository:

```text
E:\projects\FLUED\FLUED
```

Important files:

```text
flued\model.py          FLUED v2 model
flued\e1_stage_a.py     denoising Stage A runner
flued\data.py           byte helpers, MASK-aware decode
flued\config.py         shared config defaults
FLUED_REBUILD.md        this document
```

Historical archives:

```text
K:\FLUED_archive
K:\FLUED_backup
```

Useful archive material:

```text
K:\FLUED_archive\E_checkpoints\e3_flued_fair.log
K:\FLUED_archive\E_checkpoints\e3_flued_fair_v2.log
K:\FLUED_archive\E_checkpoints\e3_bpe_local.log
K:\FLUED_archive\E_checkpoints\e3_blt_local.log
K:\FLUED_archive\E_checkpoints\E2_COMPARISON_ARCHIVE.md
K:\FLUED_archive\E_checkpoints\PAPER_NARRATIVE.md
```

These archives are for evidence and comparison only, not v2 resume.

## Current Defaults

```text
VOCAB_SIZE=258
MASK_ID=257
swiglu_hidden=1536
assignment_window=128
target_compression=0.3
compression_weight=0.125
min_boundary_units=1.0
denoise_prob=0.7
corrupt_rate=0.15
span_mask_prob=0.7
latent_consistency_weight=0.03
RMSNorm=disabled
```
