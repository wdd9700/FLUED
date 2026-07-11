# FLUED v3.1 Review and Next Steps

Date: 2026-06-30

This document is a compact handoff for the current FLUED v3.1 state after the
parallel segmental-diffusion implementation, 2M / seq128 sweeps, and the
four-agent design review.

It should be treated as the current working context if the conversation is
compressed.

Chinese architecture note:

```text
Read `docs/versions/v3.1/docs\versions\v3.1\FLUED_V3_1_ARCHITECTURE_CN.md` for the current Chinese explanation of
the v3.1 architecture, the corrected codec/backbone role split, and the open
design details that are not yet finalized.

The Chinese architecture note supersedes this older review wherever this file
appears to expose FLUED internal memory directly to the external backbone.
The corrected interface is:
  FLUED internal memory -> used only inside the encoder;
  readout latent sequence -> the only normal backbone-facing interface.

Current first code target:
  `tools/analysis/v3_1/train_v31_language_codec_2m.py`

This is the small codec-only prototype aligned with the corrected role split.
The older `tools/analysis/v3_0/train_v3_segmental_diffusion_2m.py` remains useful as
an experimental predecessor, but it should not be treated as the final v3.1
interface.
```

## 1. Non-Negotiable Goal

FLUED is trying to solve this problem:

```text
Translate a byte stream into latent-space representations that:
  preserve semantic information,
  preserve positional / order information,
  can be decoded back to bytes,
  and reduce the learning / training burden of the downstream backbone.
```

Therefore reconstruction alone is not success.

The model must eventually prove:

```text
1. byte -> latent:
   latent carries semantic, positional, and structural information.

2. latent -> byte:
   latent can be decoded back to byte stream without relying on a hidden
   shortcut.

3. readout latent -> backbone:
   FLUED emits a readout latent sequence that should reduce the downstream
   backbone's language-learning burden.

4. internal memory -> later encoding:
   FLUED memory is an encoder-side cache for past semantic summaries. It helps
   later segmentation, disambiguation, and readout formation, but is not a
   normal external backbone input.
```

## 2. Current Code State

Current corrected implementation:

```text
tools/analysis/v3_1/train_v31_language_codec_2m.py
```

Predecessor experiments:

```text
tools/analysis/v3_0/train_v3_segmental_diffusion_2m.py
tools/analysis/v3_0/train_v3_segmental_workspace_2m.py
```

The predecessor scripts are not the v3.1 interface. They are useful as evidence
about active memory, denoise steps, and AR correction, but they still mix
boundary / memory / readout behavior in ways that can be mistaken for a
byte-level language model.

Current codec prototype:

```text
byte ids
  -> UTF-8 edge mask
  -> weak boundary starts
  -> shared segment representation
  -> summary latent
  -> internal causal summary memory
  -> readout latent
  -> explicit length head
  -> slot decoder
  -> byte span reconstruction
```

This script cleanly enforces the corrected interface boundary for the codec
stage: only readout latents are the external interface; summary and memory are
internal encoder mechanisms; there is no future / next-byte head.

Key code modules:

```text
CodecCollator:
  builds UTF-8-clean weak segments on CPU / DataLoader workers.

complete_utf8_edge_valid:
  masks chunk-edge partial UTF-8 codepoints before segmentation.

boundary_head:
  predicts boundary logits for ROI and later learned segmentation.

summary_head:
  produces internal summary latents.

causal_summary_memory:
  shifted cumulative summary memory; current segment reads only previous
  summaries.

readout_head:
  maps segment representation + internal memory to readout latent.

length_head / slot_decoder:
  decodes each readout latent back into a variable byte span.
```

Evaluation / diagnosis scripts:

```text
tools/analysis/v3_1/summarize_v31_language_codec.py
tools/analysis/v3_1/train_v31_min_backbone.py
tools/analysis/v3_1/summarize_v31_min_backbone.py
tools/eval/v3_1/eval_v31_language_codec_roi.py
tools/eval/v3_1/eval_v31_language_codec_decoder.py
tools/eval/v3_1/eval_v31_language_codec_memory_ablation.py
tools/eval/v3_1/eval_v31_language_codec_memory_cases.py
```

Fair training convention for architecture comparison:

```text
batch_size: 128
num_workers: 12 on local RTX 5080
stream_samples_per_worker: at least max_steps * batch_size / num_workers

Large-batch runs such as batch=192 or batch=256 are throughput probes only.
They should not be mixed into architecture-quality conclusions because they
change the optimization trajectory and per-step sample budget.
```

## 3. Current Experimental Facts

Current codec result directory:

```text
K:/FLUED_archive/v31_language_codec_2m_20260702
```

Best current 2M / seq128 codec run:

```text
K:/FLUED_archive/v31_language_codec_2m_20260702/codec_10k_utf8clean
```

Summary:

```text
params:              2,010,899
steps:               10,000
eval_mode:           streaming
eval_recon_acc:      0.5063
eval_length_acc:     0.9725
eval_boundary_acc:   0.9354
eval_units_per_byte: 0.1170
train_steps_per_sec: 19.03 on local RTX 5080
```

Decoder diagnostics:

```text
streaming eval:
  recon_acc:                 0.5102
  length_acc:                0.9739
  exact_span_acc:            0.4670
  long_span_recon_acc:       0.3257
  invalid_target_utf8_ratio: 0.0000
  invalid_pred_utf8_ratio:   0.0307

fixed-text eval:
  recon_acc:                 0.3280
  length_acc:                0.9589
  exact_span_acc:            0.0476
  long_span_recon_acc:       0.3089
  invalid_target_utf8_ratio: 0.0000
  invalid_pred_utf8_ratio:   0.0798
```

Memory ablation on streaming eval:

```text
full recon_loss:      1.2476
zero memory:          +1.96% loss, -0.0088 recon_acc
shuffled memory:      +1.76% loss, -0.0073 recon_acc
stale memory:         +1.64% loss, -0.0074 recon_acc
summary_detached:     same as full in no-grad eval
memory_effect:        visible but small
```

ROI / segmentation:

```text
The ROI script now separates raw boundary probability from constrained
executable segmentation. Constrained model segments obey:
  no UTF-8 continuation start;
  max segment length <= max_span;
  invalid chunk-edge UTF-8 fragments are masked out.
```

Current verified conclusions:

```text
1. The codec path is trainable at 2M scale:
   readout latent -> byte span crosses the first-pass recon_acc > 0.5 gate.

2. Length prediction is not the bottleneck:
   length_acc is already about 0.97 on streaming eval.

3. UTF-8 target legality is now fixed:
   invalid_target_utf8_ratio is 0.0 after edge masking and capacity-aware
   weak boundary generation.

4. Decoder quality is still span-length sensitive:
   long spans are much weaker than short spans.

5. Streaming eval and fixed-text eval diverge:
   the model can learn the sampled codec task, but fixed natural text
   generalization is still weak at 10K steps.

6. Internal memory has a measurable but small effect:
   it is not just dead code, but it is not yet a strong semantic memory.
```

Case-based memory diagnostics were added after the random streaming ablation:

```text
tools/eval/v3_1/eval_v31_language_codec_memory_cases.py
```

Representative result:

```text
codec_10k_utf8clean / mean:
  repeated English entity, later subset:
    zero memory loss +27.7%, recon_acc -0.0426

codec_40k_utf8clean / mean:
  repeated English entity, later subset:
    zero memory loss +53.7%, recon_acc -0.0426
  version-number case, later subset:
    max memory-probe loss delta 33.6%

codec_10k_pool_mfl / mean_first_last:
  reconstruction is better, especially on long spans,
  but entity/later memory sensitivity is much weaker.
```

Working interpretation:

```text
mean pooling keeps readout more dependent on internal memory, but reconstructs
long spans poorly.

mean_first_last pooling improves local byte-span payload and long-span
reconstruction, but partially bypasses memory and does not improve masked
latent infill.

Therefore mean_first_last is not a final answer. The next structure should keep
its long-span benefit while preventing readout from becoming only a local
payload shortcut.
```

Minimal backbone result directory:

```text
K:/FLUED_archive/v31_backbone_20260702
```

Strict interface:

```text
latent backbone input: readout latent only
not visible to backbone: FLUED summary / memory / boundary logits
FLUED codec: frozen
decoder: used only to turn filled readout latent back into byte-span metrics
```

Fair 3K comparison:

```text
byte_3k with random byte mask:
  mask_acc = 0.3035
  This is not a fair latent comparison because it masks isolated bytes.

byte_3k_segmentmask:
  contiguous segment-span mask
  mask_acc = 0.1498

latent_3k:
  latent MSE only
  mask_acc = 0.1563

latent_3k_byteaux01:
  latent MSE + 0.1 * masked decoder CE through frozen decoder
  mask_acc = 0.1659

latent_3k_byteaux1:
  latent MSE + 1.0 * masked decoder CE through frozen decoder
  mask_acc = 0.1784
```

Current interpretation:

```text
1. FLUED latent is not proven by random-byte-mask comparisons.
   The byte task must mask comparable contiguous spans.

2. Under segment-mask comparison, readout latent gives a first positive signal.

3. Latent MSE alone is too weakly aligned with the decoder's byte manifold.
   Decoder-aligned masked CE improves the latent backbone, while still keeping
   the FLUED encoder frozen.

4. The current gain is real but small:
   0.1784 vs 0.1498 mask byte accuracy at 3K steps.

5. Next validation must test longer training, fixed-text generalization,
   stronger codec checkpoints, and mask difficulty sweeps before claiming that
   FLUED lowers backbone learning burden.
```

40K codec follow-up:

```text
codec_40k_utf8clean:
  streaming recon_acc:        0.5451
  streaming length_acc:       0.9869
  streaming long_span_acc:    0.3527
  fixed-text recon_acc:       0.3509
  fixed-text long_span_acc:   0.3289
  invalid target UTF-8 ratio: 0.0000

memory ablation:
  zero memory:     +2.76% loss, -0.0109 recon_acc
  shuffled memory: +2.67% loss, -0.0096 recon_acc
  stale memory:    +2.45% loss, -0.0092 recon_acc
```

Backbone with 40K codec:

```text
latent_3k_codec40k:
  latent MSE only
  mask_acc = 0.1641

latent_3k_byteaux1_codec40k:
  latent MSE + 1.0 * masked decoder CE
  mask_acc = 0.1779
```

Updated interpretation:

```text
1. Longer codec training helps reconstruction and memory usefulness, but does
   not remove the long-span weakness.

2. Better codec reconstruction does not automatically improve masked latent
   infill.  The 40K codec keeps non-masked spans better, but masked accuracy is
   essentially tied with the 10K codec.

3. The remaining bottleneck is likely readout latent predictability /
   interpolability and the decoder's long-span byte reconstruction, not only
   raw training steps.
```

Segment pooling follow-up:

```text
pool_mode=mean_first_last:
  segment input = mean state + first state + last state.
  external readout interface is unchanged.
```

Codec result:

```text
codec_10k_pool_mfl:
  params:                    2.085M
  streaming recon_acc:        0.6469
  streaming long_span_acc:    0.5104
  fixed-text recon_acc:       0.5184
  fixed-text long_span_acc:   0.4984
```

This is a large codec improvement over both 10K mean and 40K mean.

However:

```text
memory ablation:
  memory_effect = weak
  zero/shuffled/stale memory only changes loss by about 0.27%-0.42%.

backbone:
  latent_3k_pool_mfl MSE only:      mask_acc = 0.1425
  latent_3k_byteaux1_pool_mfl:      mask_acc = 0.1678
  best old mean latent byteaux1:    mask_acc = 0.1784
  byte segment baseline:            mask_acc = 0.1498
```

Interpretation:

```text
mean_first_last improves reconstruction and long spans, but it weakens the
pressure to use internal memory and does not improve masked latent infill.

This exposes a real objective conflict:
  reconstruction-friendly readout can become a local payload;
  backbone-friendly readout needs predictable, smooth latent structure;
  memory usefulness needs the current segment representation to remain
  context-dependent, not fully solved by local first/last payload.

Next architecture work should preserve the long-span benefit while restoring
memory dependence and latent predictability.  A likely route is separating
payload/detail information from backbone-facing semantic readout, or applying a
smoothness / predictability loss to readout without exposing memory to the
backbone.
```

The old segmental-diffusion evidence below is kept only as predecessor context,
not as the current v3.1 implementation:

Main result directory:

```text
K:/FLUED_archive/v31_segmental_diffusion_20260629
```

Important artifacts:

```text
sweep_500
sweep_loss_500
sweep_boundary_value_500
ablation_1000_fair
fair_smoke_full_300
fair_smoke_no_memory_300
fair_smoke_no_ar_300
```

Fair 2M / seq128 ablation:

```text
K:/FLUED_archive/v31_segmental_diffusion_20260629/ablation_1000_fair
```

Summary:

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

Verified conclusions:

```text
Memory is useful:
  no_memory has much worse future_loss.

AR is not validated:
  full and no_AR are effectively tied.

Step annealing is not useful yet:
  fixed one-step beats annealed multi-to-one training at 2M / seq128.

Multi-step denoise is harmful in this small setup:
  fixed_max has the worst future_loss.
```

Current coarse policy:

```text
Train directly in target deployment form:
  boundary denoise = 1
  memory denoise   = 1
  readout denoise  = 1
  AR correction    = 0 or 1

Keep memory enabled.
Use future_loss_weight around 1.0.
Do not use boundary_value_loss with the current target.
```

Current unresolved problem:

```text
useful memory runs have commit_m/n around 0.57-0.60.
This is too dense and closer to high-frequency soft writing than meaningful
semantic compression.
```

## 4. Important Corrections Already Made

### 4.1 Readout Target

The first diffusion sweep used:

```text
prediction_target = next_byte
```

This was invalid for readout because the segment encoder is bidirectional.
It leaked future information.

Corrected default:

```text
prediction_target = current
future_target     = current
```

This matches FLUED's encoder/prefill role: the encoder sees the existing byte
segment and must produce a decoder-readable latent.

### 4.2 No-Memory Fairness

The no-memory branch originally let `future_head` read `h`, which gave it an
unfair direct path.

Corrected behavior:

```text
future_logits = future_head(hist_memory)
```

When memory is disabled, `hist_memory` is zero. This made the memory ablation
fair and confirmed that memory genuinely helps future_loss.

### 4.3 AR Delta Limiting

Before hard limiting, boundary AR delta could become large enough to violate
the "light correction" premise.

Current hard limit:

```text
delta = tanh(proj(...))
gate  = ar_gate_scale * sigmoid(...)
```

After this, AR delta is tiny. This proves AR is not stealing the main path, but
also means it is not yet useful.

## 5. Subagent Review Synthesis

Four review angles were delegated:

```text
DSpark / DeepSpec:
  AR head and acceptance scheduling.

Mapping Network:
  low-dimensional control space vs high-dimensional payload.

Training signals:
  commit value, future/memory usefulness, intervention metrics.

Latent quality / residual flow:
  position, semantics, readout, memory contribution, stability.
```

The consensus is:

```text
Current v3.1 is closer to the desired architecture than the recurrent baseline,
but its supervision and diagnostics are not yet sufficient.
```

## 6. DSpark Review

Current state:

```text
parallel proposal:
  yes, via parallel latent denoise.

lightweight serial correction:
  partially, via SmallARCorrection.

confidence-scheduled acceptance:
  missing.
```

Problem:

```text
SmallARCorrection is always-on for all valid positions.
It does not select low-confidence tokens or segments.
It does not accept/reject proposed fixes.
```

Missing metrics:

```text
ar_params
active_ar_params
ar_macs_per_token_per_pass
ar_latency_pct
ar_accept_rate
ce_before_ar
ce_after_ar
ar_gain = ce_before_ar - ce_after_ar
ar_gain_by_confidence_bin
accept_precision = P(ar_gain > 0 | accepted)
reject_regret
low_conf_coverage
confidence_auc_for_ar_gain
```

Minimum DSpark-style next step:

```text
Do not enlarge AR.
First add evaluation that compares ar_passes=0 vs ar_passes=1 on the same
checkpoint and records whether AR improves low-confidence regions.
```

## 7. Mapping Network Review

Current state:

```text
boundary_z:
  high-dimensional hidden-size latent. It is control-like but not low-dimensional
  control.

commit:
  scalar control, but too dense and weakly semantic.

memory_z / hist_memory:
  high-dimensional payload, but currently a dense weighted mean.

readout_z:
  high-dimensional readout payload.
```

Problem:

```text
control and payload are not cleanly separated.
boundary_z may still carry byte/payload information.
readout_z has a strong direct h -> readout -> byte path.
memory is useful, but not yet proven to be controlled semantic payload rather
than dense byte-level history pooling.
```

Required distinction:

```text
low-dimensional control_z:
  decides commit, budget, memory/readout gates, freeze/readout scheduling.

high-dimensional payload:
  preserves byte, semantic, entity, position, and structural information.
```

Minimum architecture direction:

```text
1. Add explicit low-dimensional control_z after boundary_z.
   Try dimensions 8 / 16 / 32 / 64 / hidden.

2. Do not feed control_z into byte_head.

3. Let control_z gate memory/readout behavior.

4. Keep memory_write as high-dimensional payload.

5. Add probes to prove control_z cannot decode bytes well, while payload can.
```

## 8. Training Signal Review

Current problem:

```text
value_target is derived from local/span CE.
That answers "where reconstruction is hard", not "where commit helps future".
```

The current `boundary_value_loss` experiment failed for this reason:

```text
It lowered commit variance and m/n a bit, but did not fix commit quality.
```

Correct target:

```text
commit_value_i =
  future_loss_without_write_i
  - future_loss_with_write_i
```

Useful variants:

```text
future_gain:
  writing at i lowers future/span CE.

memory_usefulness:
  memory-only readout improves when i is written.

over_commit_cost:
  write occurred but future gain is near zero or negative.

under_commit_cost:
  suppressing write makes future loss worse.

stability:
  value remains stable under light corruption or perturbation.
```

Important metric replacement:

```text
value_corr_to_delta_future
```

should replace current:

```text
value_corr_to_local_ce
```

Required intervention metrics:

```text
delta_future_loss_by_commit
top_value_commit_gain
low_value_commit_harm
memory_ablation_gap
shortcut_gap
commit_budget_curve
semantic_position_probes
```

## 9. Latent Quality / Residual Review

Current residual flow is structurally understandable:

```text
h -> boundary_z -> commit
h + commit -> memory_z -> memory_write -> hist_memory
h + hist_memory -> readout_z -> byte_head
```

But monitoring is insufficient.

Major risk:

```text
readout reconstruction can be excellent because h carries local/bidirectional
byte information, not because readout_z or memory is a good semantic latent.
```

Position risk:

```text
No explicit positional embedding is visible in the current Transformer encoder.
Position may be carried implicitly by convolution, correction heads, and memory
ordering, but this must be probed.
```

Required probes:

```text
position:
  absolute position bucket, normalized t/T, distance to previous commit,
  pairwise order / distance bucket.

decode:
  byte CE from h, boundary_z, memory_z, hist_memory, readout_z separately.

semantic/entity:
  names, dates, numbers, API identifiers, Chinese/English mixed spans,
  templates, key-value slots.

smoothness:
  intra-segment cosine/L2, cross-boundary jump, boundary jump ratio.

stability:
  clean vs corrupt latent cosine, commit top-k Jaccard, decode CE delta.

memory contribution:
  CE with true memory, zero memory, stale memory, shuffled memory,
  readout without hist_memory residual.

decode basin:
  add noise or quantization to readout_z and measure CE increase.
```

## 10. Immediate Decision

Do not scale the current model yet.

Reason:

```text
Memory is validated,
but commit value, control/payload separation, AR usefulness, and latent quality
are not yet validated.
```

Do not continue optimizing local reconstruction.

Reason:

```text
readout CE is already too easy and can create false positives.
```

Next work should be diagnostic and targeted:

```text
P0. Add counterfactual memory / commit evaluation.
P0. Add AR gain-cost evaluation.
P0. Add control_z vs payload probes.
P0. Add readout shortcut and memory contribution probes.
P1. Add explicit low-dimensional control_z and rerun small sweeps.
P1. Add true prefix-to-suffix future mask evaluation.
```

## 11. Concrete Next Experiments

### 11.1 Counterfactual Commit Value

For a trained checkpoint:

```text
full memory
no memory
stale memory
shuffled memory
top-k commit only
low-k commit only
drop top-k commit
drop low-k commit
```

Report:

```text
future_loss
future_loss_delta
commit_m/n
actual_commit_value
top-k actual-value enrichment
negative/harmful commit ratio
```

### 11.2 AR Gain / Cost

Evaluate the same checkpoint with:

```text
ar_passes = 0
ar_passes = 1
```

Report:

```text
ce_before_ar
ce_after_ar
future_before_ar
future_after_ar
ar_delta
ar_gate
ar_gain / ar_delta
gain by confidence bin
```

### 11.3 Control / Payload Probe

Add or extract:

```text
control_z
boundary_z
memory_z
hist_memory
readout_z
h
```

Probe:

```text
byte reconstruction
position
UTF-8 class
entity / number / API span identity
commit/value prediction
```

Desired result:

```text
control_z predicts commit/value but does not decode bytes well.
payload latents decode bytes and preserve semantic/position information.
```

### 11.4 Prefix-to-Suffix Future Evaluation

The current `future_mask` inherited from older scripts masks suffix but computes
loss on prefix. That is not a true future prediction metric.

Need:

```text
input prefix only or prefix + masked suffix
loss on suffix
compare full/no/stale/shuffled memory
```

## 12. Current Best Short-Term Architecture Hypothesis

Keep:

```text
parallel segment feature encoder
one-step boundary denoise
one-step memory denoise
one-step readout denoise
committed memory
```

Treat as optional / unproven:

```text
AR correction
multi-step denoise
step annealing
boundary_value_loss from local CE
```

Add next:

```text
low-dimensional control_z
counterfactual commit value
memory intervention probes
latent position/semantic probes
```

This is the most faithful current path to the original FLUED goal.
