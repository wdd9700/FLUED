# AGENTS.md — FLUED Project Guide for AI Coding Agents

> **单一事实源：[`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md)**——默认配置、证据注册表、闸门与待裁定队列。
> 任何实验默认起点是 canonical 配置（v3.6 线 `configs/canonical_v36.json` 为当前默认，`canonical_v35.json` 旧口径保留有效）；`tests/test_canonical_sync.py` 持续校验
> CLI 默认值 == canonical 配置 == CURRENT_STATE.md（v35/v36 两线均受守卫），三处漂移会红灯。新证据落地必须原地更新 CURRENT_STATE.md 并写 changelog。

> **术语注册表：[`docs/TERMS.md`](docs/TERMS.md)**——一切项目术语的唯一登记处，与 CURRENT_STATE 同级。两条长期法则：
> 1. **原地登记**：新术语（缩写、英文造词、非日常名词）落地必须先登记 TERMS.md（原文/中文全称/一句话定义/首现/提出方/日期/状态），否则视为不存在（黑户）；
> 2. **未注册词禁用**：写报告或回复时，凡不在 TERMS.md 的词，首次出现必须内联给出中文全称+一句话定义；英文造词必须挂中文语义锚。AI 提名的术语默认"候选"，用户点头才转正（候选→已注册，与证据状态机同构）。

> **FLUED** (FLexible Unified Encoder-Decoder): Tokenization-free learned boundary compression for language modeling.
> Semantic units are dynamically compiled by the model during encoding, not predefined by an external tokenizer.

## Quick Reference

| Task | Command |
|------|---------|
| Run tests | `python -m pytest tests/` |
| Smoke test E1 | `python -m flued.e1_stage_a --preset smoke_cpu` |
| v3.4 trainer help | `python tools/train/v3_4/train_v34_pos_ar_probe.py --help` |
| v3.4 matrix help | `python tools/launcher/v3_4/run_v34_pos_ar_matrix.py --help` |
| Check GPU | `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` |
| Check running procs | `Get-Process python*` |
| Read checkpoint step | `python -c "import torch; c=torch.load('checkpoints/e1_latest.pt', map_location='cpu', weights_only=False); print(c.get('global_step',0))"` |

## Environment

- **Python**: 3.11+; local validated environments currently use Python 3.12 for CUDA and Python 3.14 for tests.
- **PyTorch**: CUDA 12.8-compatible build for RTX 50-series GPU work.
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

Full project status: [README.md](README.md) · Versioned documentation: [docs/README.md](docs/README.md)

## Key Files

| File | Role |
|------|------|
| `flued/model.py` | `FLUEDAutoencoder` v0.4 — legacy core model (749 lines; v3.6 mainline is `flued/v36/`) |
| `flued/config.py` | `ModelConfig`, `TrainConfig`, `SIZE_CONFIGS`, CLI parsing |
| `flued/data.py` | `ByteReconstructionDataset`, PAD-offset encoding (PAD=0, byte b→b+1, vocab=257) |
| `flued/e1_stage_a.py` | E1 training loop (standalone, not using `train.py` Trainer) |
| `flued/e2_compare.py` | E2: multi-model comparison |
| `flued/e3_train.py` | E3: downstream causal LM training |
| `flued/e3_downstream.py` | `FLUEDDownstream`, `BLTDownstream`, `BPEDownstream` wrappers |
| `flued/v33/` | v3.3 codec implementation |
| `flued/v34/` | v3.4 parallel-memory and rate/emit extension |
| `flued/v36/` | v3.6 KDA-generation mainline model (`FLUEDV36`) |
| `flued/hnet_repro/` | H-Net reproduction baseline (BPB 0.653 anchor) |
| `tools/train/v3_6/train_v36.py` | v3.6 train/eval entrypoint (`--config` eats `configs/canonical_v36.json`) |
| `tools/train/v3_6/train_s0_segmentor.py` | S0 segmentor pretraining entrypoint |
| `tools/train/v3_4/train_v34_pos_ar_probe.py` | v3.4 train/eval entrypoint |
| `configs/v3_4/` | v3.4 reproducible experiment matrices |
| `results/v3.4/5k_ablation/` | Public raw logs and curve artifacts |

## Conventions

- **Checkpoint naming**: `{model}_step{NNNNN}.pt` and `{model}_latest.pt`
- **PAD-offset encoding**: vocab_size=257, PAD=0, byte b → id b+1
- **Tied inverse**: blocks reversed, MHA/FFN subtracted instead of added. Forward and inverse share parameters. `dropout=0.0` in E1 (tied inverse is dropout-sensitive).
- **E1 presets**: `smoke_cpu`, `small_gpu`, `class300m_16gb` — defined in `flued/e1_stage_a.py`
- **skip_hard**: Always pass `skip_hard=True` during training/eval. Hard segmentation is for logging only.

## ⚠️ Critical Pitfalls

1. **Never use wildcards to delete checkpoints.** Always list and verify exact paths first. Keep the latest checkpoint and analysis milestones.

2. **FP16 entropy/log underflow**: When computing entropy over the byte vocabulary in FP16, cast to `.float()` before logarithms.

3. **GradScaler overflow**: On overflow skip, metrics must not advance as if an optimizer step succeeded.

4. **CUDA context corruption**: After force-stopping a GPU process, wait for the CUDA context to clear before launching another run.

5. **Duplicate process check**: Always inspect active Python/GPU processes before launching training.

6. **Versioned paths**: v3 experiment code lives under `tools/*/v3_*`; do not reintroduce pre-reorganization import paths.
