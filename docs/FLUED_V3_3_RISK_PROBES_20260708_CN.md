# FLUED v3.3 风险 Probe 记录

日期：2026-07-08

本记录用于回应空上下文审计提出的结构怀疑，并把讨论前的最小实验证据固定下来。

## 1. 已执行的 probe

命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python tools/analysis/probe_v33_architecture_risks.py `
  --config configs/v33_full_300m_100m_corpus_v4.json `
  --trials 32 `
  --out-json K:\FLUED_archive\v33_efficiency_bench_20260708\v33_arch_risk_probe_after_user_response.json
```

输出摘要：

```json
{
  "order_sensitivity": {
    "z_content_max_abs_diff_ab_vs_ba": 0.0,
    "readout_z_max_abs_diff_ab_vs_ba": 0.0
  },
  "masked_chunk_ratio": {
    "mask_prob": 0.05,
    "mean_masked_byte_fraction": 0.04765,
    "mean_masked_chunk_fraction": 0.76953,
    "mean_active_chunks": 32.0
  },
  "gate_vs_compute": {
    "soft_readout_units_per_byte": 0.06746,
    "actual_backbone_units_per_byte": 0.125,
    "actual_over_soft": 1.85285
  }
}
```

## 2. 当前已经坐实的问题

### 2.1 chunk 内顺序没有进入 interpreter 表示

`ab` 和 `ba` 这类同字节多重集、不同顺序的输入，在当前 mean-pool interpreter 下：

```text
z_content diff = 0
readout_z diff = 0
```

这说明当前 chunk representation 是近似 bag-of-bytes。decoder 有 slot embedding，但 readout latent 本身没有携带 chunk 内顺序信息。该问题会影响拼写、代码、数字串、变量名、公式等顺序敏感场景。

候选修正：

1. 最小修复：给 chunk 内 byte feature 加 slot/relative position encoding，再 pool。
2. 更符合当前设计方向：借鉴 KDA / Kimi Delta Attention，在 chunk 内加入轻量 delta-state mixer，使 chunk 内局部顺序以低成本进入 pooled/readout。
3. 暂不建议直接堆大 Transformer，因为这会破坏 v3.3 的高效语言编码器定位。

### 2.2 5% byte mask 仍会导致 77% chunk 被遮盖

full 配置已从：

```text
mask_prob=0.15
```

改为：

```text
mask_prob=0.05
```

但由于当前 backbone 遮盖单位是 chunk：

```text
masked_chunks = chunk contains any masked byte
```

在 128-byte chunk 下，即使 byte mask 只有约 5%，仍有约 77% chunk 被判定为 masked chunk。

这说明单纯降低 byte mask 不足以解决 backbone 训练压力。后续更合理的方向是：

1. byte/slot/readout 局部遮盖，而不是整 chunk 遮盖；
2. 或者让 chunk 内被 mask 的部分由 decoder 局部处理，backbone 只预测对应 readout 子槽；
3. 初期训练继续保持较低 mask 率，避免 codec 和 backbone 同时承受过强噪声。

### 2.3 当前 gate 是表达门控，不是计算门控

当前指标：

```text
soft_readout_units_per_byte    = 0.06746
actual_backbone_units_per_byte = 0.125
actual_over_soft               = 1.85
```

原因是 active-only backbone 只按 `chunk_mask` compact：

```text
只要 chunk 存在，该 chunk 的 16 个 readout 都进入 backbone
```

即使 extra readout gate 接近 0，也仍然占用 backbone token。因此必须同时记录：

```text
soft_readout_units_per_byte
actual_backbone_units_per_byte
```

短期不建议直接硬裁剪原始 gate，以免训练不稳定。更合理的方向是独立 silent/emit controller：

```text
每个 chunk:
  fallback readout 永远 emit
  extra readout 默认 silent
  高信息密度 chunk 才 emit extra readout
```

训练早期仍可让 backbone 吃全部 readout 或软权重，稳定后再逐步让 emit 阈值参与真实计算裁剪。

该机制已在当前代码中落地，但要区分两件事：

1. 计算路径已经可以裁剪 extra readout。
2. emit controller 是否学会“简单片段少发声、高密度片段多发声”还需要训练曲线和样本 probe 验证。

最新 probe：

```text
soft_readout_units_per_byte    = 0.06536
soft_emit_units_per_byte       = 0.02178
dense_backbone_units_per_byte  = 0.12500
actual_backbone_units_per_byte = 0.02124
actual_over_dense              = 0.16992
extra_emit_mean                = 0.11920
```

当前训练默认：

```text
emit_compute_mode = masked_or_emitted
emit_threshold    = 0.5
```

这表示 fallback readout 永远发声；被 strict byte mask 覆盖到的 extra readout 必须发声；其余 extra readout 只有 emit score 过阈值才进入 backbone。这样既减少计算，又不把 masked-source 监督路径裁断。

## 3. 已落地的低风险改动

1. full 配置 `mask_prob` 从 0.15 降到 0.05。
2. `train_v33.py` 命令行默认 `--mask-prob` 从 0.15 降到 0.05。
3. 2M 消融配置 `v33_ablation_2m.json` 的 base mask 降到 0.05，并保留 0.10 作为 stress。
4. backbone masked 单位从 chunk-any-mask 改为 readout/span-level mask：

```text
旧逻辑:
  只要 chunk 内任意 byte 被 mask，则该 chunk 的所有 readout 都交给 backbone 预测

新逻辑:
  fallback readout 保留
  masked byte 按 offset 映射到对应 extra readout 槽
  backbone 只预测局部 readout 槽
```

新的 probe 结果：

```text
masked_byte_fraction    ~= 0.04765
masked_chunk_fraction   ~= 0.76953
masked_readout_fraction ~= 0.11487
```

这说明 chunk 级统计仍然很高，但真正交给 backbone 的 masked readout 已经降到局部范围。

5. boundary prior target 改为 threshold-relative：

```text
continue_target = -min(0.95, tau_keep + 0.10)
punct_target    = min(0.95, (tau_trans + tau_cut) / 2)
```

在 full 配置 `tau_cut=0.90, tau_trans=0.75, tau_keep=0.65` 下：

```text
continue_target = -0.75
punct_target    = 0.825
```

避免继续用固定 `0.5` 去监督一个实际需要超过 `0.75` 才进入 transition 的信号。

6. 训练日志新增：

```text
soft_readout_units_per_byte
actual_backbone_units_per_byte
backbone_active_units
masked_chunk_fraction
masked_readout_fraction
masked_backbone_units_per_byte
boundary_continue_target
boundary_punct_target
memory_read_entropy_mean
memory_read_norm_mean
soft_emit_units_per_byte
extra_emit_mean
actual_backbone_units_per_byte
```

1-step 日志验证：

```text
masked_byte_fraction        = 0.04858
masked_chunk_fraction       = 0.79688
masked_readout_fraction     = 0.10742
masked_backbone_units/byte  = 0.01343
soft_readout_units/byte     = 0.06740
soft_emit_units/byte        = 0.02178
actual_backbone_units/byte  = 0.01953
memory_read_entropy_mean    = 2.057
memory_read_norm_mean       = 3.620
```

## 4. 下一步优先问题

1. parallel-local memory：当前默认改为 `parallel_local + bidirectional_no_self`，需要继续验证 self_allowed 始终为 0，且 memory 可见性没有 clean-byte 泄漏。
2. chunk 内顺序建模：优先比较 slot position encoding 与 KDA-like delta mixer。
3. plastic confidence：确认主 loss credit 是否足以塑形 segmentor；必要时加入轻量 attention-residual 风格的跨层 credit residual。
4. memory readout probe：在已有 gate/norm/entropy/self_allowed 之外，增加 top-k source attribution，判断 readout 是否真实参考 other-chunk memory。
5. emit controller：基础路径已落地；下一步观察训练后 extra emit 是否和信息密度、代码/中文/实体密度相关。
6. detach=true/false 配对：只用结果决定“latent 通用性”claim。

补充口径：

```text
decoder 不读 memory。
memory 只服务 encoder interpreter。
prompt encoding 默认可读 other-chunk memory，但屏蔽当前 chunk 自身 memory。
```

最新 no-self visibility probe：

```text
memory_build_mode    = parallel_local
memory_visibility    = bidirectional_no_self
active_chunks_mean   = 32.0
has_context_frac     = 1.0
self_allowed_mean    = 0.0
visible_slots_mean   = 62.0
```

一步 full 配置训练日志中，eval 侧也确认：

```text
eval_memory_self_allowed_mean = 0.0
eval_memory_visible_slots_mean = 124.0
```
