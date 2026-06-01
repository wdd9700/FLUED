# AGENTS.md — FLUED Project Guide for AI Coding Agents

> **FLUED** (FLexible Unified Encoder-Decoder): Tokenization-free learned boundary compression for language modeling.
> Semantic units are dynamically compiled by the model during encoding, not predefined by an external tokenizer.

## Quick Reference

| Task | Command |
|------|---------|
| Run tests | `C:\Python314\python.exe -m pytest tests/` |
| Smoke test E1 | `C:\Python314\python.exe -m flued.e1_stage_a --preset smoke_cpu` |
| Full E1 train | See [gpu_retrain_e1.ps1](gpu_retrain_e1.ps1) or [repo memory](memories/repo/flued_v04.md#L88-94) |
| Check GPU | `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` |
| Check running procs | `Get-Process python*` |
| Read checkpoint step | `python -c "import torch; c=torch.load('checkpoints/e1_latest.pt', map_location='cpu', weights_only=False); print(c.get('global_step',0))"` |

## Environment

- **Python**: `C:\Python314\python.exe` (3.14.0) — **required**. Python 3.13.5 crashes with NumPy BLAS FPE on import.
- **PyTorch**: 2.11.0+cu128 (`https://download.pytorch.org/whl/cu128`)
- **GPU**: RTX 5080, 16 GB VRAM, CUDA 13.2
- **CPU**: Set `$env:OMP_NUM_THREADS=4; $env:MKL_NUM_THREADS=4` before training to avoid thread explosion.

## Architecture (3-stage pipeline)

```
E1 (Stage A)  →  E2 (Comparison)  →  E3 (Downstream LM)
   FLUED DSC       FLUED vs BPE/BLT     Frozen encoder + causal LM
   reconstruction   perplexity/cloze     next-byte prediction
```

- **FLUED**: Single Transformer encoder+decoder with **tied weights**. Boundary detection via `sigmoid(Linear(ΔH))`. Soft assignment matrix A does O(T²) pooling — fine for T≤512.
- **BLT** (baseline): Byte Latent Transformer — uses a pre-trained Byte LM's entropy to determine patches.
- **BPE** (baseline): Standard encoder-decoder Transformer on BPE tokens (vocab 8192).

Full architecture doc: [README.md](README.md) · Detailed specs: [repo memory](memories/repo/flued_v04.md)

## Key Files

| File | Role |
|------|------|
| `flued/model.py` | `FLUEDAutoencoder` v0.4 — core model (~540 lines) |
| `flued/config.py` | `ModelConfig`, `TrainConfig`, `SIZE_CONFIGS`, CLI parsing |
| `flued/data.py` | `ByteReconstructionDataset`, PAD-offset encoding (PAD=0, byte b→b+1, vocab=257) |
| `flued/e1_stage_a.py` | E1 training loop (standalone, not using `train.py` Trainer) |
| `flued/e2_compare.py` | E2: multi-model comparison |
| `flued/e3_train.py` | E3: downstream causal LM training |
| `flued/e3_downstream.py` | `FLUEDDownstream`, `BLTDownstream`, `BPEDownstream` wrappers |

## Conventions

- **Checkpoint naming**: `{model}_step{NNNNN}.pt` and `{model}_latest.pt`
- **PAD-offset encoding**: vocab_size=257, PAD=0, byte b → id b+1
- **Tied inverse**: blocks reversed, MHA/FFN subtracted instead of added. Forward and inverse share parameters. `dropout=0.0` in E1 (tied inverse is dropout-sensitive).
- **E1 presets**: `smoke_cpu`, `small_gpu`, `class300m_16gb` — defined in `flued/e1_stage_a.py`
- **skip_hard**: Always pass `skip_hard=True` during training/eval. Hard segmentation is for logging only.

## ⚠️ Critical Pitfalls

1. **🚫 NEVER use wildcards to delete checkpoints.** Always `Get-ChildItem` first, verify, then delete explicitly by name. Keep last 2-3 checkpoints per model. See [checkpoint-safety](memories/repo/flued_v04.md#L163-171).

2. **FP16 entropy/log underflow**: When computing entropy over 257-class softmax in FP16, cast to `.float()` first and use `epsilon=1e-12` (not 1e-8 — below FP16 epsilon). See [FP16 underflow](memories/repo/flued_v04.md#L173-176).

3. **GradScaler overflow**: On overflow skip, metrics accumulate but step doesn't — track `running_micro_count` and rollback. See [overflow fix](memories/repo/flued_v04.md#L178-180).

4. **CUDA context corruption**: After `Stop-Process -Force` on a GPU process, wait 10-30s before launching new training or risk `cudaErrorIllegalAddress`. See [GPU context](memories/repo/flued_v04.md#L186-189).

5. **Duplicate process check**: Always run `Get-Process python*` before launching training to avoid parallel GPU processes. See [process check](memories/repo/flued_v04.md#L191-193).

6. **replace_string_in_file**: Always include ≥3 lines of context before AND after the target to ensure unique match. See [edit safety](memories/repo/flued_v04.md#L182-184).
