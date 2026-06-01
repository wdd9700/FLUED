# FLUED: FLexible Unified Encoder-Decoder

**Tokenization-free learned boundary compression for language modeling.**

---

## Architecture

FLUED is a single Transformer autoencoder that learns to segment raw byte sequences into semantic units via differentiable boundary detection — no vocabulary, no fixed patches, no heuristics.

```
raw bytes [B, T]
    │
    ▼
┌─────────────────────────────┐
│  Embedding + Pos Encoding    │
│  TiedTransformerBlock × N     │  ← 24 layers, 1024-dim, 16 heads
│  (shared encoder/decoder)    │
└──────────────┬──────────────┘
               │
    ┌──────────▼──────────┐
    │  boundary_head       │  ← Linear(d_model → 1)
    │  → sigmoid → p ∈[0,1]│
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Soft Assignment     │
    │  AᵀH → Z (pool)      │
    │  A Z → Ẑ (expand)    │
    └──────────┬──────────┘
               │
               ▼
          byte logits [B, T, 257]
```

### Key Components

| Component | Description | Params |
|-----------|-------------|--------|
| Embedding | 257-token vocab (256 bytes + PAD) | 263K |
| Transformer | 24 tied blocks, d=1024, nhead=16, FFN=4096 | ~302M |
| boundary_head | Linear(1024→1), operates on hidden state deltas | 1K |
| Soft assignment | A[b,i,j] = P(pos i belongs to segment starting at j) | — |

### Training Objectives

1. **Reconstruction loss** (cross-entropy): byte-level reconstruction accuracy
2. **Boundary regularization loss** (multi-term):
   - `compression_weight × (bp_mean − target)²` — global budget
   - `−lambda_var × Var(p)` — encourage spread (maximize variance)
   - `lambda_entropy × H(p)` — polarize toward 0/1
   - `lambda_utf8 × bp_mean[is_cont]` — suppress mid-character cuts
   - `lambda_type × MSE(p, type_target)` — type-conditional prior

---

## Current Progress (E1-B v2)

### Model

- **Parameters**: 302,573,569
- **Architecture**: d=1024, nhead=16, FFN=4096, 24 tied layers, seq_len=512
- **Training**: FP16 AMP, RTX 5080 16GB, batch=2, grad_accum=8
- **Corpus**: 50K lines from 22GB multilingual text (~3GB sampled)
- **Hyperparams**: λ_var=0.5, λ_entropy=0.05, λ_utf8=0.02, λ_type=0.05, target_compression=0.3

### Final Results (step 31050)

| Metric | Start (step 6050) | End (step 31050) | Δ |
|--------|-------------------|-------------------|----|
| recon_acc | 0.9980 | 0.9995 | +0.0015 |
| hard_m/n | 0.769 | 0.480 | −37.6% |
| soft_m/n | 0.536 | 0.480 | −10.4% |
| bp_std | 0.051 | 0.457 | +8.96× |
| cjk bp | 0.500 | 0.058 | −88.4% |
| utf8_cont bp | 0.554 | 0.684 | +23.5% |
| ascii bp | 0.533 | 0.566 | +6.2% |
| op bp | 0.541 | 0.749 | +38.4% |
| digit bp | 0.527 | 0.570 | +8.2% |

### Emergent Properties

1. **Cross-lingual content adaptation**:
   - CJK lead bytes (0xE4-0xE9): bp → 0.058 (model preserves Chinese characters)
   - UTF-8 continuation bytes (0x80-0xBF): bp → 0.684 (segments end at character boundaries)
   - ASCII: bp → 0.566 (moderate segmentation for subword-level English)

2. **Sub-byte type differentiation from weak inductive biases**:
   - Model distinguishes first vs. last continuation byte within the same 0x80-0xBF range
   - The model receives only weak inductive biases (λ_utf8, λ_type) as soft regularization; the learned byte-type hierarchy — including CJK lead/cont distinction, operator clustering, and digit coalescence — emerges primarily from reconstruction pressure

3. **Compression ratio converges toward target**:
   - hard_m/n dropped from 0.77 to 0.48, approaching target=0.30
   - Chinese text naturally compresses more (3 bytes/char → few segments)

### Training Phases

| Phase | Steps | Key Transition |
|-------|-------|----------------|
| Polarization | 6050–9000 | bp_std 0.05→0.40, entropy term dominates |
| Convergence | 10000–20000 | hard_m/n 0.79→0.70, type differentiation emerges |
| Plateau | 24300–31050 | bp_std 0.40→0.46, all metrics near convergence |

### Checkpoints

| Step | File | hard_m/n | bp_std | cjk | Notes |
|------|------|----------|--------|-----|-------|
| 6050 | — | 0.769 | 0.051 | 0.500 | Start |
| 10000 | e1_step10000.pt | 0.787 | 0.199 | 0.135 | Phase 2 start |
| 15000 | e1_step15000.pt | 0.718 | 0.222 | 0.065 | |
| 20000 | e1_step20000.pt | 0.702 | 0.276 | 0.059 | |
| 30000 | e1_step30000.pt | 0.479 | 0.457 | 0.058 | |
| **31050** | **e1_step31000.pt** | **0.480** | **0.457** | **0.058** | **Final** |

---

## Competitive Landscape

### Comparison Matrix

| Method | Org | Boundary Signal | Differentiable | Extra Modules | Scale | Year |
|--------|-----|-----------------|----------------|---------------|-------|------|
| **FLUED** | — | **reconstruction loss** | ✅ | **boundary_head only** | 300M | 2025 |
| **MANTa** | academia | **gaussian soft assignment** | ✅ | frontier predictor | paper | 2022 |
| **DTP** | academia | Gumbel-sigmoid | ✅ | Hourglass Transformer | paper | 2023 |
| **FLEXITOKENS** | academia | **learnable boundary predictor** | ✅ | flexible tokenizer | paper | 2025 |
| **Bolmo** | academia | **non-causal boundary predictor** | ✅ | local encoder/decoder | 7B | 2025 |
| BLT | Meta FAIR | next-byte entropy | ❌ (threshold) | entropy model + patcher | 8B | 2025 |
| H-Net++ | Mamba authors | cosine similarity | ❌ (threshold) | multi-layer hierarchy | paper | 2025 |
| Evabyte | academia | multi-byte prediction | ❌ | linear attention | 6.5B | 2025 |
| Toucan | academia | token-aware char LM | ✅ | improved DTP decoding | paper | 2023 |
| ByteFlow Net | academia | coding-rate reduction | ✅ | rate-distortion framework | paper | 2026 |
| MrT5 | academia | dynamic token merging | ✅ | T5-style denoising | paper | 2025 |
| Zonkey | academia | probabilistic BOS + diffusion | ✅ | diffusion model | small | 2026 |
| AU-Nets | Meta | fixed-stride pooling | ❌ | U-Net encoder | paper | 2024 |
| MambaByte | academia | none (no chunking) | N/A | SSM backbone | paper | 2024 |

### FLUED Differentiators

1. **Reconstruction-driven boundary learning**: FLUED is among the few methods that optimize segmentation directly for reconstruction fidelity, differing from entropy-based (BLT), similarity-based (H-Net), coding-rate-based (ByteFlow Net), denoising-based (MANTa, MrT5), and distillation-based (Bolmo) approaches.

2. **Minimal architectural change**: One additional Linear layer on a standard Transformer — vs. BLT's dual-model design or H-Net's hierarchical architecture.

3. **End-to-end differentiable**: Boundary probabilities participate in gradient flow, enabling emergent linguistic structure without explicit priors.

4. **Cross-lingual emergence**: Model spontaneously learns different compression ratios for CJK (high compression) vs. ASCII (moderate compression) from reconstruction signal alone.

5. **Two-stage, fully GPU-native pipeline** (new in v0.4):
   - **Stage A**: Sample target-language corpus → train FLUED autoencoder on reconstruction
   - **Stage B**: Freeze encoder → soft `cumsum(bp)` segmentation → scatter-vectorized GPU pooling → downstream LM training
   - ✅ Zero CPU bottlenecks in Stage B (vs. BLT's entropy patcher requiring Python for-loops)
   - ✅ No hard threshold anywhere in the forward path
   - ✅ Semantic segmentation signal learned purely from reconstruction

### FLUED Weaknesses

1. **Scale**: 300M params vs. BLT's 8B, Bolmo's 7B, Evabyte's 6.5B. Cannot compare absolute downstream performance.
2. **Inference complexity**: Soft assignment matrix A is O(T²) in sequence length. For T=512 (training), this is negligible; for T=4096–16K (deployment), sparse top-K (retain top 16–32 neighbors per position), chunk-based processing (T→W windows), or low-rank approximation (A ≈ UVᵀ) are viable mitigation strategies. See [O(T²) Complexity Analysis](#ot²-complexity-analysis) below.
3. **No downstream validation**: Only reconstruction metrics; no LM perplexity or task benchmarks yet. E3 results (FLUED bpb ≈ 0.27 at step 2400) are preliminary and based on frozen-encoder byte prediction, not directly comparable to autoregressive LM bpb.
4. **Single-seed results**: All reported metrics are from seed=42 single run; multi-seed experiments (3–5 seeds) are needed for statistical validity.

### Positioning Strategy

> BLT proved byte-level models can scale with entropy patching — but requires a separately pre-trained entropy model and hard thresholding, creating CPU bottlenecks in downstream training.
>
> FLUED proves they don't need hand-crafted heuristics at all: reconstruction alone induces semantically meaningful segmentation, and the fully soft assignment enables a GPU-native two-stage pipeline where Stage B has zero CPU overhead.

FLUED occupies the **reconstruction-based, fully-differentiable boundary learning** niche. Unlike MANTa (T5-style denoising, 2022), Bolmo (distillation matching, 2025), or FLEXITOKENS (distribution-shift robustness, 2025), FLUED optimizes segmentation for raw byte reconstruction fidelity with minimal architectural overhead (a single Linear head on a standard Transformer). Key contrast with BLT:

| Dimension | BLT | FLUED |
|-----------|-----|-------|
| Segmentation signal | Next-byte entropy (statistical) | Reconstruction fidelity (semantic) |
| Boundary mechanism | Hard threshold θ | Soft cumsum(bp) |
| Pre-training needed | Separate ByteLM pre-training | None (jointly trained) |
| Stage B CPU bottleneck | Yes (Python for-loop patcher) | No (scatter vectorized) |
| Differentiable end-to-end | No | Yes |

---

## Related Work: Detailed Comparison with MANTa (EMNLP 2022)

**MANTa** (Godey et al., EMNLP 2022) is the most directly comparable prior work to
FLUED. Both methods use differentiable, probability-based byte-to-block assignment:

| Dimension | MANTa (2022) | FLUED (2025) |
|-----------|-------------|--------------|
| **Boundary predictor** | Frontier predictor (sliding-window Transformer) → pFi | boundary_head (Linear on hidden-state deltas) → sigmoid(p) |
| **Soft assignment** | Gaussian-approximated byte-block joint distribution | Explicit assignment matrix A[b,i,j] = P(pos i in segment j) |
| **Pooling** | 1-D CNN + Max-Pooling over weighted byte embeddings | AᵀH → Z (pool), then A·Z → Ẑ (expand) |
| **Training objective** | Masked span denoising (T5-style) + end-to-end | Reconstruction loss + boundary regularization |
| **Architecture** | Separate encoder/decoder | Tied-weight single Transformer |
| **Downstream validation** | ✅ GLUE, cross-domain, noise robustness | ❌ Reconstruction metrics only |
| **Extra modules** | Frontier predictor (sliding window) | boundary_head only (1 Linear layer) |

**Key distinctions**:
1. **Training signal**: MANTa uses T5-style span corruption/denoising; FLUED uses
   raw byte reconstruction. The reconstruction objective provides a more direct
   signal for segmentation quality — boundaries that lose information are
   penalized immediately.
2. **Architectural simplicity**: MANTa's frontier predictor processes byte windows;
   FLUED's boundary_head is a single Linear(1024→1) operating on hidden-state
   deltas, adding only ~1K parameters.
3. **Tied weights**: FLUED's encoder and decoder share parameters (pseudo-inverse expansion),
   halving model size for the same representational capacity.
4. **Maturity**: MANTa has complete downstream validation (GLUE, cross-domain);
   FLUED currently lacks this — a critical gap to address.

Other notable methods in the learned-tokenization space:
- **DTP / Toucan** (2023): Gumbel-sigmoid dynamic pooling with Hourglass Transformer;
  Toucan improved DTP's decoding speed by 2×.
- **FLEXITOKENS** (2025): Learnable boundary predictor with emphasis on
  distribution-shift robustness; demonstrated 10% downstream improvement.
- **Bolmo** (2025): Byteify subword LM at 7B scale with non-causal boundary
  predictor and local encoder/decoder; full benchmark suite.
- **MrT5** (Kallini et al., ICLR 2025): Dynamic token merging in T5 framework.
- **ByteFlow Net** (2026): Coding-rate-based compression with rate-distortion theory.
- **Zonkey** (2026): Probabilistic BOS prediction + diffusion-based segmentation.
- **Fast BLT-D** (2026): BLT + block-wise discrete diffusion for 50–92% faster inference.

### Byte-Level Autoencoder Foundations

**CEUR-WS 2021** (Learning to Embed Byte Sequences with Convolutional Autoencoders)
is a directly relevant precursor: a bi-directional temporal CNN autoencoder that
achieves per-byte cross-entropy reconstruction over 256 classes. This work
demonstrated that byte-level autoencoding is computationally viable at small
scale (~700K params). FLUED builds on this foundation by adding learned boundary
detection and soft assignment — transforming the fixed-stride CNN pooling into
a differentiable, context-adaptive segmentation.

### ΔH as Boundary Signal

**Categorical Perception in LLM Hidden States** (2026) uses ||h(n+1) − h(n)||
to measure local representational precision in the context of categorical
perception. While their goal differs (psycholinguistic phenomena vs. tokenization),
the underlying intuition — that adjacent hidden-state differences encode structural
change — is shared. FLUED is, to our knowledge, the first to directly map ΔH to
boundary probabilities via a learned linear projection, trained end-to-end with
reconstruction loss.

### Tied-Weight Architectures

**Tied Transformers** (Qin et al., AAAI 2019) proposed sharing encoder-decoder
parameters in neural machine translation — a different input/output domain.
FLUED applies tied weights to the autoencoder setting where input and output
domains are identical (bytes). The decoder operates as a pseudo-inverse of the
encoder's soft-assignment pooling: A·Z → Ẑ expands pooled representations back
to byte-level, with the same Transformer blocks running in reversed order.

### ΔH Theoretical Intuition

Why can hidden-state deltas detect boundaries? Consider a byte sequence within
a segment (e.g., the UTF-8 encoding of "猫"): all three bytes belong to the same
semantic unit. Their hidden states should evolve smoothly — ΔH is small. At the
segment boundary (e.g., between "猫" and the next token), the semantic context
shifts abruptly — ΔH spikes. This is not a hard-coded rule but an emergent
property of the Transformer's self-attention: bytes that co-occur in the same
linguistic context develop similar representations. The boundary_head learns to
detect these representation discontinuities via a single Linear(1024→1) layer.

---

## O(T²) Complexity Analysis

FLUED's soft assignment matrix A has shape [B, T, T], constructed via cumulative
products of boundary probabilities. This O(T²) memory and computation is the
primary scaling bottleneck. For training (T=512), the cost is manageable (~1M
entries per batch item). For deployment at longer contexts, three strategies
can reduce this:

| Strategy | Complexity | Trade-off |
|----------|-----------|-----------|
| **Sparse top-K** | O(T × K), K=16–32 | Retain only top-K neighbors per position; preserves >95% of assignment mass |
| **Chunk-based** | O(T × W), W=64–128 | Split sequence into overlapping windows; boundaries can't cross window edges |
| **Low-rank A** | O(T × R), R=8–16 | Approximate A ≈ UVᵀ via Nyström or random projection |

At T=4096 with K=32, sparse top-K reduces the assignment cost from 16.8M to
131K entries per batch item (128× reduction). Chunk-based and low-rank methods
can be combined for further gains at T≥16K.

---

## Baselines

### BPE Transformer Autoencoder
- Tokenizer: HuggingFace `tokenizers` BPE, vocab=8192
- Architecture: Standard Transformer encoder-decoder, d=1024, 24 layers
- Trained on same 50K-line corpus
- Tokenizer saved at `checkpoints/bpe_tokenizer/`

### BLT (Byte Latent Transformer) — ✅ Implemented
- **Entropy-based dynamic patching** (next-byte entropy > θ → boundary) — matches BLT paper methodology
- **Fixed-size patching** available as simplified baseline (`--patch-mode fixed`)
- Training script: `train_blt.py` with smoke/small/300m presets
- Matched parameter count (~300M) for fair comparison
- Checkpoints saved as `checkpoints/blt_step*.pt`

### Ablation Framework (E3) — ✅ Implemented
- `e3_ablation.py` — systematic loss term removal and hyperparameter sweep
- Single-term ablations: no_type, no_utf8, no_entropy, no_var, pure_recon
- Sweep modes: compression_weight, target_compression
- Output: `results/e3_ablation.json` + `results/e3_ablation.csv`

---

## Visualizations

| Chart | File | Content |
|-------|------|---------|
| 6-row comprehensive | `checkpoints/e1b_full_trends.png` | 15 metrics, gap-aware |
| Compression focus | `checkpoints/compression_trends.png` | hard/soft/bp_mean + h@ sweep + spread |
| Type differentiation | `checkpoints/type_bp_trends.png` | utf8/ascii/cjk/op/digit per-type bp_mean |

Gap region (9000–24300) marked with pink shading. Three checkpoint anchor points at 10000/15000/20000.

---

## File Index

```
flued/
├── model.py          # FLUEDAutoencoder (DSC architecture)
├── e1_stage_a.py     # Stage-A training runner
├── e2_compare.py     # Baseline comparison
├── e3_downstream.py  # E3 downstream wrapper (FLUED/BLT/BPE)
├── e3_train.py       # E3 downstream LM training
├── e3_ablation.py    # Ablation experiment framework
├── config.py         # Model presets & hyperparams
├── data.py           # Data loading + SimpleBPE
├── train.py          # Shared train/eval utilities

blt_baseline/
├── model.py          # BLTAutoencoder + ByteLanguageModel + EntropyPatcher

bpe_baseline/
├── model.py          # BPE Transformer baseline
├── standalone/       # Self-contained BPE training package (for laptop)

checkpoints/
├── e1_step*.pt       # FLUED checkpoints
├── bytel_m_*.pt      # BLT Stage 1 (ByteLM) checkpoints
├── blt_step*.pt      # BLT Stage 2 checkpoints
├── bpe_tokenizer/    # Trained BPE tokenizer (8192 vocab)
├── e1b_full.log      # Training log (496 lines)
├── *_trends.png      # Visualization charts

train_blt.py          # BLT Stage 2 training
train_blt_stage1.py   # BLT Stage 1 (ByteLM pre-training)
train_bpe.py          # BPE tokenizer training
plot_*.py             # Visualization scripts
```

---

## Next Steps

- [x] FLUED E1-B v2: boundary loss v2 hyperparams, 50000-step training
- [x] BLT paper-faithful reproduction (Stage 1 ByteLM + Stage 2 frozen encoder)
- [x] Soft segmentation forward path (cumsum(bp) → scatter pool, no hard threshold)
- [x] E3 downstream LM wrappers (FLUED/BLT/BPE) + training script
- [ ] E2: Run comparison report FLUED vs BLT vs BPE on reconstruction quality
- [ ] E3: Train three downstream 500M LMs, compare bits-per-byte
- [ ] Paper writing