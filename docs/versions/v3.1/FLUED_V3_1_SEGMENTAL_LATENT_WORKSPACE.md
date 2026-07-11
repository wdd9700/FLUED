# FLUED v3.1 Segmental Latent Workspace Memo

Date: 2026-06-29

This document preserves the recent design turn after the v3 active-memory
experiments, DSpark / ELF / DiffusionGemma review, and the discussion about why
plain reconstruction / coding-rate objectives are insufficient.

It is intentionally separate from `docs/versions/v3.0/docs\versions\v3.0\FLUED_V3_CONTROL.md` so that future context
compression does not erase the current reasoning state.

Current handoff note:

```text
Read `docs/versions/v3.1/docs\versions\v3.1\FLUED_V3_1_REVIEW_AND_NEXT_STEPS.md` first after any context compression.
It summarizes the 2M / seq128 evidence, four-agent review, corrected failure
interpretation, and the next diagnostics that should precede any larger run.

Then read `docs/versions/v3.1/docs\versions\v3.1\FLUED_V3_1_ARCHITECTURE_CN.md` for the current Chinese architecture
explanation and the corrected FLUED codec / backbone / memory role split.

The corrected role split is:
  readout latent is the external backbone interface;
  summarize / memory is an internal FLUED encoder mechanism;
  backbone should not directly consume FLUED internal memory.
```

## 0. Non-Negotiable Goal

FLUED is not primarily a boundary visualizer or a compression trick.

The original goal remains:

```text
Translate a byte stream into latent-space representations that:
  preserve semantic information,
  preserve positional / order information,
  can be decoded back to bytes,
  and reduce the learning / training burden of the downstream backbone.
```

Therefore v3.1 must be judged by three linked properties:

```text
latent quality:
  semantic / positional / logical information is present in the latent.

decoder compatibility:
  the latent can be read out back to bytes without relying on a hidden
  byte-level shortcut.

backbone usefulness:
  the readout latent sequence makes future / downstream modeling easier for
  an external backbone.

internal memory usefulness:
  FLUED memory improves later segmentation, disambiguation, and readout
  formation inside the encoder. It is not a normal backbone-facing interface.
```

Compression and boundary quality are only useful if they serve this translation
goal.

## 1. Current Empirical Ground

The current runnable prototype is:

```text
tools/analysis/v3_0/train_v3_commit_controller_small.py
```

The validated v3 primitive is:

```text
active segment + committed memory
```

The core empirical conclusion is:

```text
active_only:
  lacks global/history awareness.

memory_only:
  lacks high-resolution current-segment information.

active + memory:
  is the best-balanced route so far across prediction quality, globality,
  and computational structure.
```

Therefore the main v3 direction should not be reopened as:

```text
Do we need memory?
```

The remaining question is:

```text
How should active be committed into memory,
how should memory be read,
and what signal decides a good commit?
```

## 2. Relevant Results

### 2M / seq_len=128

```text
active_only                       eval_loss=1.9461  future_loss=1.9620
memory_only                       eval_loss=1.9310  future_loss=1.9478
raw decoder + raw controller      eval_loss=1.8874  future_loss=1.9086
gated decoder + feature ctrl      eval_loss=1.8542  future_loss=1.8754
gated decoder + raw controller    eval_loss=1.8481  future_loss=1.8685
```

Interpretation:

```text
At 2M, decoder-side gated memory fusion was the best structure.
The main gain came from gated decoder fusion, not from replacing raw controller
memory input with simple memory features.
```

### 6M / seq_len=128

```text
6M active_only          eval_loss=1.8714  future_loss=1.8903
6M raw active_memory    eval_loss=1.8428  future_loss=1.8676
6M gated active_memory  eval_loss=1.8982  future_loss=1.9268
```

Interpretation:

```text
Scaling from 2M to 6M helps.

At 6M/128, raw active_memory is the strongest and most stable run.
The gated version has better training reconstruction but a much larger
train/eval gap, suggesting overfitting or mis-generalization at this length.
```

Important correction:

```text
Gating is useful, but not unconditionally default.
Raw active_memory should be the stable baseline for longer-sequence tests.
Gated active_memory should remain a candidate, especially for longer contexts
where memory selection may become more necessary.
```

### ROI Status

The ROI heatmaps are useful for qualitative diagnosis, but the current shapes
are still not semantically well formed.

Observed:

```text
active + memory is more structure-sensitive than active_only.
raw active_memory is especially sensitive to code, templates, entities, mixed
Chinese/English/API/date text, and math-like strings.
```

But:

```text
commit positions still look strange.
They are influenced by byte shape, punctuation, UTF-8 patterns, local syntax,
and training shortcuts.
They are not yet stable semantic-unit boundaries.
```

Therefore current ROI proves:

```text
memory changes the boundary policy in useful directions.
```

It does not prove:

```text
commit == semantic boundary.
```

## 3. seq_len=512 Speed Investigation

Direct 6M / seq_len=512 training was abnormally slow.

Root cause:

```text
V3CommitControllerSmall.forward has a Python loop over seq_len:

  for t in range(seq_len):
    controller
    active_update
    memory_update
    soft write

At seq_len=512 this becomes sequential small-kernel work.
GPU utilization stays low even when VRAM is available.
```

Benchmarks, d_model=384, hidden=384, controller_hidden=512:

```text
gru stride=1   batch=32  2.764 sec/step
mlp stride=1   batch=32  2.891 sec/step
gru stride=4   batch=32  1.594 sec/step
mlp stride=4   batch=32  1.707 sec/step
gru stride=8   batch=32  1.389 sec/step
gru stride=16  batch=32  1.277 sec/step
gru stride=4   batch=64  1.666 sec/step
gru stride=8   batch=64  1.437 sec/step
```

Implemented switches:

```text
--update-cell gru|mlp
--commit-stride N
--torch-compile
```

Findings:

```text
commit_stride is the only useful low-risk speed lever so far.
MLP update did not improve speed.
Increasing batch size did not improve speed meaningfully.
torch.compile works after installing triton-windows, but did not improve speed.
```

Triton / torch.compile:

```text
Installed:
  triton-windows 3.7.1.post27

Test:
  seq_len=512, batch=32, commit_stride=8

eager:         1.194 sec/step over 100 steps
torch.compile: 1.216 sec/step over 100 steps

Conclusion:
  torch.compile is now usable but not beneficial for this recurrent loop.
```

Current 512 recommendation:

```text
Use:
  --update-cell gru
  --commit-stride 8
  --batch-size 32

Do not use MLP update as default.
Do not use torch.compile as default.
```

## 4. What DSpark Actually Contributes

DSpark should not be treated merely as speculative decoding engineering.

The transferable abstraction is:

```text
parallel proposal
+ lightweight serial correction
+ confidence-scheduled commit / verification
```

Original DSpark structure:

```text
parallel backbone:
  proposes a block cheaply.

lightweight sequential head:
  adds intra-block dependency and reduces suffix decay.

confidence head:
  estimates prefix survival probability.

scheduler:
  verifies only high-value prefixes instead of wasting compute on low-confidence
  suffixes.
```

FLUED interpretation:

```text
Parallel boundary / segment proposer:
  handles high-throughput local analysis.

Lightweight serial boundary head:
  corrects local dependency, previous block effects, and prefix consistency.

Confidence / value gate:
  decides what should be committed to memory.
```

This is likely more directly useful than adding a heavy denoising loop.

## 5. What ELF Contributes

ELF's useful abstraction is:

```text
token is not the thinking state;
token is the final readout interface.
```

For FLUED:

```text
segment representation should live in continuous latent space;
byte/token reconstruction should not be the sole training driver;
the latent must remain decoder-compatible, not arbitrary noise.
```

The important constraint:

```text
latent must lie in a decoder-readable basin.
If noise makes the latent underlearned or semantically empty, it increases
model difficulty instead of reducing it.
```

## 6. What DiffusionGemma Contributes

DiffusionGemma's useful abstraction is:

```text
block/canvas-level refinement:
  current block can be edited bidirectionally before final commit.
```

For FLUED:

```text
The current segment may be lightly refined before commit.
But FLUED should not become a full chain-of-thought reasoning engine.
```

Important boundary:

```text
FLUED as encoder/decoder should perform segment-level latent refinement,
not full long-horizon latent CoT.

The future main backbone may be ELF-like.
FLUED should provide segment compression, memory interface, and latent
translation, not own all reasoning.
```

## 7. User-Identified True Missing Pieces

The recent discussion identified three real missing pieces that must be solved
before a bigger architecture is implemented.

### 7.1 Boundary Decision Signal

Question:

```text
How does diffusion / refinement decide boundary?
What feedback or learning signal drives boundary placement?
How is a boundary scored?
```

Current answer:

```text
Boundary should not be driven mainly by reconstruction or coding rate.
It should be driven by commit value.
```

A good boundary is one where commit makes future processing easier:

```text
commit_value =
  future_gain
  + latent_stability
  + memory_usefulness
  - over_commit_cost
  - under_commit_cost
```

Candidate measurable signals:

```text
future AR loss decreases after commit
memory improves future span prediction
current segment latent becomes stable
commit reduces later active burden
boundary remains stable under small semantic-preserving perturbations
```

Do not ask only:

```text
Does this look like a semantic boundary?
```

Ask:

```text
Does committing here create useful memory for what follows?
```

### 7.2 Latent Noise Level / Latent Quality

Question:

```text
How do we decide the noise level of a continuous latent segment?
```

This cannot be a fixed diffusion schedule borrowed from images.

The latent must contain:

```text
semantic information
position information
logic / relation information
decoder-compatible structure
```

If the latent is too noisy or undertrained:

```text
It does not simplify processing.
It makes the downstream model's job harder.
```

Therefore noise should be conditioned by segment state:

```text
high confidence / clear structure / strong memory support:
  low noise or no denoise

uncertain boundary / high prediction error / weak memory match:
  mild refinement

very confused segment:
  do not commit; keep in active or resegment
```

Potential predictors:

```text
latent entropy / variance
future prediction error
memory match score
boundary uncertainty
segment internal consistency
decoder readout confidence
```

This should be called:

```text
confidence-conditioned latent refinement
```

not ordinary unconditional diffusion.

### 7.3 Is Explicit Confidence Data Necessary?

Question:

```text
Do we need confidence labels / confidence head?
Or is a lightweight serial correction head enough?
```

Current answer:

```text
An explicit confidence head is not necessarily required at first.
But some confidence/value variable is necessary.
```

Reason:

```text
commit / freeze / readout are semi-discrete decisions.
Without a quality estimate, the system collapses back toward fixed-budget or
loss-only segmentation.
```

Recommended staged approach:

```text
Stage 1:
  add value / confidence probes for diagnostics only.
  Do not let them control inference yet.

Stage 2:
  if the probes predict useful commits, connect them to commit scheduling.

Stage 3:
  only then consider confidence-gated freeze/readout.
```

## 8. Revised Architecture Name

Working name:

```text
Confidence-Gated Segmental Latent Workspace
```

This should be understood as two sides.

### Encoder / Prefill Side

Goal:

```text
Understand existing input and build memory.
```

Sketch:

```text
long byte/token input
  -> parallel boundary / segment proposer
  -> DSpark-style lightweight serial boundary correction
  -> block latent processor
  -> memory extractor
  -> confidence/value gate
  -> committed memory update
```

Formula:

```text
H_i = EncodeBlock(B_i, M_<i)
M_i = UpdateMemory(M_<i, Extract(H_i))
```

Key principle:

```text
block-internal processing can be parallel/bidirectional;
block-to-block memory remains causal and ordered.
```

### Decoder / Reasoning Side

Goal:

```text
Use latent workspace only where appropriate, without turning FLUED into a full
CoT engine.
```

Sketch:

```text
task / prompt / memory
  -> initialize segment latent canvas
  -> light latent refinement inside current segment
  -> lightweight serial correction for local order / variable transfer
  -> confidence/value gate
  -> freeze selected latent region
  -> background readout
  -> verifier / future AR check
  -> commit, patch, or keep refining
```

Constraint:

```text
FLUED should do segment-level latent refinement.
It should not own full long-horizon reasoning.
```

## 9. What Not To Do Next

Do not immediately implement:

```text
multi-step byte-level diffusion
full latent CoT inside FLUED
large memory matrix
CUDA kernel
unconditional gated decoder as the large-run default
```

Do not make these the main objective:

```text
pure reconstruction
pure coding rate
fixed compression target
fixed noise schedule
```

These are still useful as auxiliary diagnostics, but not as the main driver.

## 10. Next Minimum Experiments

Before implementing the full architecture, define and measure four scores.

### A. Boundary Value

Measure:

```text
Does a commit reduce future AR loss or future span loss?
```

Expected output:

```text
commit_value per boundary
top valuable commits
negative / harmful commits
ROI visualization
```

### B. Latent Quality

Measure:

```text
Does segment latent preserve semantic, positional, and logical information?
Can decoder read it stably?
```

Potential tests:

```text
readout confidence
span prediction
perturbation stability
memory match
latent variance / entropy
```

### C. Memory Usefulness

Measure:

```text
Does memory actually help future prediction compared with active_only?
```

Use:

```text
mask high-value commits
mask low-value commits
top-k commit writes only
no-memory / stale-memory comparison
```

### D. Refinement Necessity

Measure:

```text
Which segments benefit from denoise / serial correction?
Which segments should be left alone?
```

This should guide:

```text
noise level
refinement step count
whether to freeze/readout
```

## 11. Current Code State Notes

Recent implemented switches:

```text
--update-cell gru|mlp
--commit-stride N
--torch-compile
--hybrid-existing-diffusion
```

Important caveat:

```text
--hybrid-existing-diffusion is only a rough placeholder.
It treats denoise samples as current-byte prediction and future samples as AR.
This is not yet the desired confidence-conditioned latent refinement.
Do not treat it as the final diffusion design.
```

Installed local packages during this phase:

```text
triton-windows 3.7.1.post27
pypdf 6.14.2
```

DSpark reference files:

```text
K:/FLUED_archive/references/DSpark_paper.pdf
K:/FLUED_archive/references/DSpark_paper.txt
```

## 12. Immediate Recommended Next Step

Do not continue by running a large 512 training blindly.

Recommended next action:

```text
Write diagnostic tools for:
  boundary value
  latent quality
  memory usefulness
  refinement necessity

Then use the diagnostics to decide:
  whether to add a serial correction head,
  whether confidence/value should be explicit,
  and whether latent refinement is needed at all.
```

Only after these diagnostics show a clear signal should the code move toward:

```text
DSpark-style serial boundary correction
confidence-conditioned latent refinement
confidence-gated memory commit
```

## 13. 2026-06-29 Implementation Plan: 2M Full Architecture First

The next code step is not another abstract design pass. The full v3.1 idea is
now implemented as a separate 2M-scale training script:

```text
tools/analysis/v3_0/train_v3_segmental_workspace_2m.py
```

It is separate from the older active/memory prototype so the previous baseline
remains available for fair comparison.

### 13.1 Implemented Architecture

```text
byte ids
  -> byte embedding
  -> causal GRU encoder
  -> active segment state
  -> committed memory state
  -> commit controller
  -> multi-step latent refinement
  -> lightweight autoregressive correction
  -> byte readout
```

The default 2M-scale route uses:

```text
d_model=192
hidden=192
controller_hidden=256
refine_steps=4
student_refine_steps=1
ar_correction_passes=2
residual_mixer=attn
seq_len=128
```

The exact parameter count is reported in each `summary.json`, because ablations
change the number of trainable modules.

### 13.2 Training-Time Teacher / Inference-Time Student

The central training idea is:

```text
training:
  allow 4-8 latent refinement steps and 1-3 small autoregressive corrections.

target runtime:
  distill this into 1 latent refinement step + 1 lightweight AR correction.
```

The script therefore computes both:

```text
teacher latent:
  refine_steps + ar_correction_passes

student latent:
  student_refine_steps + at most one AR correction
```

and trains with:

```text
student CE loss
latent distillation loss
```

This avoids treating multi-step refinement as the final deployment cost.

### 13.3 Residual Flow

The current implementation adds an AttenRes-style depth residual mixer over
latent refinement states:

```text
states = [initial_latent, refine_1, refine_2, ...]
mixed = depth_attention(states)
```

This is deliberately much smaller than full cross-layer residual attention.
It only mixes the handful of refinement states, so the ablation is cheap and
does not change sequence-level attention complexity.

Available modes:

```text
--residual-mixer attn
--residual-mixer last
--residual-mixer mean
```

`last` is the clean ablation: no cross-step residual selection.

### 13.4 Loss Placement

Main losses:

```text
reconstruction / next-byte CE:
  trains the final teacher readout.

student CE:
  forces the one-step student path to remain usable.

future loss:
  keeps memory predictive.

distillation loss:
  pulls student latent toward the multi-step teacher latent.
```

Diagnostic / auxiliary losses:

```text
commit value loss:
  predicts normalized future span CE from the commit-value head.

confidence loss:
  predicts whether local CE is below the batch median.

adaptive rate pressure:
  uses the existing dual-style rate_lambda update.

commit spread:
  prevents immediate constant commit collapse.
```

Important constraint:

```text
commit value and confidence are probes first.
They are not yet allowed to control inference scheduling.
```

This preserves the current discipline: prove the value signal before making it
part of runtime control.

### 13.5 Metrics To Read First

Do not rank variants by reconstruction alone.

Primary metrics:

```text
eval_loss
eval_future_loss
eval_student_loss
eval_commit_mn
eval_commit_std
eval_commit_corr
eval_commit_enrich
```

Training-history metrics:

```text
student_loss vs teacher recon
distill_loss
value_corr
rate_lambda
residual_alpha_last
commit_std
```

Interpretation:

```text
If teacher improves but student does not:
  multi-step refinement learned something that failed to distill.

If eval_loss improves but eval_future_loss does not:
  readout improved but memory did not become more useful.

If commit_m/n drops while commit_std collapses:
  the model is only obeying budget pressure, not learning useful boundaries.

If residual_alpha_last is always near 1:
  depth residual mixing is not contributing.

If value_corr stays near 0:
  commit value probe is not learning a useful boundary score.
```

### 13.6 Full / Ablation Matrix

Launcher:

```text
tools/launcher/v3_0/run_v3_segmental_workspace_2m_matrix.ps1
```

ROI / latent-state visualizer:

```text
tools/eval/v3_0/eval_v3_segmental_workspace_roi.py
```

It renders:

```text
commit probability
commit value score
confidence score
refinement residual alpha_last
hard spans
top commit ROI contexts
```

Default output:

```text
K:/FLUED_archive/v3_segmental_workspace_20260629
```

Runs:

```text
full_refine4_ar2_attenres_memory
abl_no_refine_ar2_memory
abl_refine4_no_ar_memory
abl_refine4_ar2_no_memory
abl_refine4_ar2_last_residual
abl_refine4_ar2_no_value_probe
```

Decision rule:

```text
The full architecture is only supported if it beats the ablations on both:
  eval_loss / eval_future_loss
  and useful commit diagnostics.

If it only wins reconstruction:
  it is not yet a v3 success.
```

## 14. 2026-06-30 Correction: True v3.1 Parallel Diffusion Route

The previous `train_v3_segmental_workspace_2m.py` implementation is now
classified as a recurrent active-memory baseline, not the intended v3.1 full
architecture.

Reason:

```text
It still had byte-by-byte Python loops in the main active/memory path.
That contradicts the intended design where diffusion / latent refinement is
the main parallel body and AR is only a small correction head.
```

The corrected implementation is:

```text
tools/analysis/v3_0/train_v3_segmental_diffusion_2m.py
```

Corrected architecture:

```text
byte ids
  -> embedding
  -> parallel Transformer segment feature encoder
  -> boundary latent denoise
  -> memory latent denoise
  -> readout latent denoise
  -> small cuDNN GRU correction heads
  -> commit / memory / readout
```

Main path:

```text
parallel latent denoise over [B,T,H]
```

Small AR heads:

```text
boundary correction
memory correction
readout correction
```

The AR heads are constrained by:

```text
small residual gate initialization
delta norm logging
ar_delta_loss
```

### 14.1 Training Style

The route is one model with step annealing, not teacher-student training.

```text
Stage A:
  more denoise steps for boundary / memory / readout.

Stage B:
  denoise and AR steps gradually reduce.

Stage C:
  target deployment form:
    boundary denoise = 1
    memory denoise   = 1
    readout denoise  = 1
    AR correction    = 1
```

Multi-step evaluation is diagnostic. The deploy metric uses the target one-step
schedule.

### 14.2 Current Coarse Experimental Results

Artifacts:

```text
K:/FLUED_archive/v31_segmental_diffusion_20260629/sweep_500
K:/FLUED_archive/v31_segmental_diffusion_20260629/sweep_loss_500
K:/FLUED_archive/v31_segmental_diffusion_20260629/sweep_boundary_value_500
```

Important correction:

```text
The first sweep used next-byte readout while the segment encoder was
bidirectional. That leaked future information and made deploy_loss look
artificially excellent.
```

The corrected default is:

```text
prediction_target = current
future_target     = current
```

This matches FLUED's encoder/prefill role:

```text
the encoder can see the existing segment,
and the readout latent must reconstruct the byte stream.
```

Backbone usefulness should not be read from reconstruction alone. Read:

```text
future_loss from memory / history latent
commit_m/n
commit_std
commit_corr / enrich
value_corr
AR delta cost
deploy-vs-multi gap
```

Current coarse parameter choice:

```text
recon_loss_weight       = 1.0
future_loss_weight      = 1.0
boundary_value_weight   = 0.0
stage_a_ratio           = 0.50
stage_b_ratio           = 0.85
noise_scale             = 0.03
target schedule         = 1/1/1 denoise + 1 AR
```

Why:

```text
future_weight=0.3:
  future/memory signal remains too weak.

future_weight=1.0:
  improves future_loss without the worst commit explosion.

future_weight=2.0:
  improves future_loss slightly more, but pushes commit_m/n too high.

boundary_value_loss:
  did not improve commit quality. It reduced commit variance and m/n but did
  not fix negative commit_corr.
```

Current unresolved issue:

```text
The model learns byte readout quickly, but boundary placement is still not
semantically convincing and commit_corr can remain negative.
```

Interpretation:

```text
High local reconstruction CE is not the same as high commit value.
Commit value must be measured by future/memory usefulness, not by local byte
difficulty alone.
```

Next ablation:

```text
full parallel diffusion
no memory
no AR correction
fixed one-step only
fixed multi-step only
```

### 14.3 Fair 2M / seq128 Ablation Result

Artifact:

```text
K:/FLUED_archive/v31_segmental_diffusion_20260629/ablation_1000_fair
```

Configuration:

```text
seq_len=128
batch_size=64
max_steps=1000
prediction_target=current
future_target=current
recon_loss_weight=1.0
future_loss_weight=1.0
noise_scale=0.03
AR gate hard-scaled to 0.10
```

Result:

```text
fixed one-step:
  deploy_loss=0.0129
  future_loss=2.5436
  commit_m/n=0.570

full anneal + memory + AR:
  deploy_loss=0.0211
  future_loss=2.9085
  commit_m/n=0.602

no AR:
  deploy_loss=0.0212
  future_loss=2.9055
  commit_m/n=0.600

no memory:
  deploy_loss=0.0026
  future_loss=4.4997
  commit_m/n=0.221

fixed max multi-step:
  deploy_loss=0.1575
  future_loss=5.6630
  commit_m/n=0.488
```

Interpretation:

```text
Memory is validated:
  no_memory has much worse future_loss, so committed memory is reducing future
  / history-latent prediction difficulty.

AR is not yet validated:
  full and no_AR are effectively tied after hard delta limiting.

Step annealing is not yet useful at 2M / seq128:
  fixed one-step training beats annealed multi-to-one training.

Multi-step denoise is actively harmful in this small setting:
  fixed_max has the worst future_loss.
```

Current best coarse training policy:

```text
Train directly in the target deployment form:
  boundary denoise = 1
  memory denoise   = 1
  readout denoise  = 1
  AR correction    = 0 or 1, but it is not yet proven useful.

Keep memory enabled.
Keep future_loss_weight around 1.0.
Do not use boundary_value_loss yet.
```

Known defect:

```text
commit_m/n is high around 0.57-0.60 for the useful memory runs.
The next training problem is adaptive compression / commit budget, not readout
reconstruction.
```
