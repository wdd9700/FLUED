# FLUED v3.3 效率审计与当前训练配置

日期：2026-07-08

目标是在不大改 v3.3 框架的前提下，让本地 RTX 5080 训练速度不低于 1 optimizer step/s，同时避免显存溢出和系统分页。

## 1. 当前结论

当前可用的本地全量配置是：

```text
seq_len=4096
max_span=128
max_chunks=128
max_readout_vectors=16
batch_size=1
grad_accum_steps=1
amp=bf16
optimizer=fused_adamw
active_only_backbone=true
save_optimizer=false
chunk_mixer=delta_lite
emit_compute_mode=masked_or_emitted
emit_threshold=0.5
```

该配置保留 v3.3 的核心语义：

1. signed confidence + 双阈值切分。
2. 每个 chunk 先并行形成 local memory summary，再形成 readout matrix。
3. readout matrix 交给外部 latent backbone 做 strict masked-source 补全任务。
4. memory 仅作为 encoder interpreter 的上下文参考，不进入 decoder。
5. 默认 memory 可见性是 `bidirectional_no_self`：可读其他 chunk memory，但屏蔽当前 chunk 自身 memory。

## 2. 为什么不能继续使用 grad_accum=8

`grad_accum_steps=8` 会把 8 次 micro batch 前后向合成 1 次 optimizer step。

在日志上看，这是 1 个 step；在 GPU 上实际是 8 次完整训练计算。因此它会让 optimizer step/s 降到约 `0.11`，并且显存/内存余量变差。

当前研究目标更需要快速迭代和稳定显存。引入 `delta_lite` chunk mixer 后，默认回到 `grad_accum_steps=1`，因为 `grad_accum_steps=2` 会低于 1 step/s。

## 3. 实测 benchmark

benchmark 脚本：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python tools/analysis/v3_3/benchmark_v33_efficiency.py `
  --config configs/v3_3/v33_full_300m_100m_corpus_v4.json `
  --warmup 5 `
  --steps 20
```

| 配置 | step/s | 平均 step 秒数 | 峰值显存 | 结论 |
| --- | ---: | ---: | ---: | --- |
| R16, batch=1, accum=1, active-only backbone | 3.03 | 0.330s | 11.44GB | 最快迭代 |
| R16, batch=1, accum=2, active-only backbone | 1.53 | 0.655s | 13.12GB | 当前本地默认 |
| R16, batch=1, accum=4, active-only backbone | 0.80 | 1.252s | 13.12GB | 低于 1 step/s，不作为默认 |
| R16, delta_lite, batch=1, accum=1 | 3.10 | 0.323s | 13.32GB | emit controller 前的当前上限 |
| R16, delta_lite, emit masked_or_emitted, batch=1, accum=1 | 3.02 | 0.331s | 13.21GB | 当前本地默认 |
| R16, delta_lite, parallel_local no-self memory, batch=1, accum=1 | 2.65 | 0.378s | 13.21GB | 新 memory 默认，仍高于 1 step/s |
| R16, delta_lite, batch=1, accum=2 | 0.90 | 1.106s | 15.00GB | 低于 1 step/s，不作为默认 |
| R16, batch=1, accum=1, full padded backbone | 2.78 | 0.360s | 11.79GB | 可跑，但白算 padding |
| R16, batch=1, accum=8 | 0.11 | 9.17s | 14.04GB | 不适合作为本地默认 |
| R16, batch=2, accum=1 | 0.17 | 5.93s | 16.93GB | 超过 16GB，触发 shared/分页风险 |

关键观察：

- dataloader 不是瓶颈，真实流式数据的 data wait 接近 0。
- optimizer 不是主瓶颈，fused AdamW 每步约 0.015s。
- 主要计算在 forward/loss 和 backward。
- batch=2 会超过独显显存，不能作为 5080 默认方案。

## 4. 已落地的低风险优化

### 4.1 fused AdamW

训练脚本新增：

```text
--optimizer adamw|fused_adamw|foreach_adamw
```

full 配置默认使用 `fused_adamw`。如果 fused 不可用，代码会回退到 foreach AdamW。

### 4.2 active-only backbone

原先 backbone 固定看到 `max_chunks * max_readout_vectors` 个位置。

当前 full 配置为：

```text
max_chunks=128
max_readout_vectors=16
固定长度 = 2048 latent units
```

但 4096 byte 输入在 128 byte 被动 chunk 下通常只有约 32 个 active chunks，即约 512 个 active readout slots。active-only backbone 会把真实 active readout compact 后送入 backbone，同时保留原始 position id，最后 scatter 回原位置。

这不改变 readout 预算、不改变 memory/readout 语义，只减少 padding 计算。

### 4.3 interpreter 重复计算缓存

memory 开启时，`mean_pool` 和 `write_memory` 原先会在 memory read 前后重复计算。

现在 `FLUEDV33.forward()` 会把 pooled chunk representation 和 current write 传给 interpreter 复用，避免同一前向里的重复计算。

### 4.4 checkpoint 不默认保存 optimizer

450M 量级模型保存 AdamW 状态会显著增加检查点体积和保存时内存峰值。

full 配置默认：

```text
save_optimizer=false
```

这适合长训归档和速度测试。如果需要严格断点续训优化器状态，可以命令行打开：

```powershell
--save-optimizer
```

### 4.5 silent/emit readout controller

原先 readout gate 只影响表达强弱，不影响 backbone 实际计算量。也就是说：

```text
soft_readout_units_per_byte    下降
actual_backbone_units_per_byte 不一定下降
```

当前实现把 readout 分成两类：

```text
fallback readout: 每个 active chunk 永远进入 backbone
extra readout: 默认 silent，只有 emit score 超过阈值才进入 backbone
```

训练配置使用：

```text
emit_compute_mode=masked_or_emitted
emit_threshold=0.5
```

`masked_or_emitted` 的含义是：

1. fallback readout 永远进入 backbone。
2. 被 strict byte mask 覆盖到的 extra readout 必须进入 backbone，避免监督被裁掉。
3. 其他 extra readout 只有 emit score 超过阈值才进入 backbone。

初始化时 extra emit bias 为 `-2.0`，因此 extra readout 默认接近 silent：

```text
extra_emit_mean ~= sigmoid(-2) = 0.119
```

最新 probe：

```text
soft_readout_units_per_byte    = 0.06536
soft_emit_units_per_byte       = 0.02178
dense_backbone_units_per_byte  = 0.12500
actual_backbone_units_per_byte = 0.02124
actual_over_dense              = 0.16992
```

也就是说，在不切断 masked readout 监督的前提下，当前实际送入 backbone 的 latent unit 约为 dense readout 方案的 `17%`。这只是计算路径修正，不等价于 emit 已经学会了语义分配；后续训练仍要观察高信息密度文本是否自动打开更多 extra readout。

## 5. 关于 KV cache

### 5.1 memory 的 KV cache

memory 分两类：

```text
prompt-local memory:
  同一次 prompt 内由所有 chunk 并行生成。
  默认 visibility 为 bidirectional_no_self。
  它服务 prefill / encoder，不适合提前缓存为跨请求状态。

committed history memory:
  来自过去对话或过去调用。
  可以 append-only 存储，并缓存 key/value。
```

因此 KV cache 主要服务 committed history memory，避免每次解释当前 prompt 时重复投影全部历史 memory。

当前实现已经支持 committed memory 的 K/V cache：

```text
MemoryState:
  committed
  committed_mask
  committed_key
  committed_value
```

cache 只在 `torch.no_grad()` 的 commit 路径中默认生成；训练态 commit 不缓存，读取时也不会在反向传播路径复用 cache，以免绕开 key/value projection 的梯度。

局部微基准：

```text
history memory slots = 1024
query chunks = 32
d_mem = 1536
top_k = 8
raw read = 0.0144s / 50 reads
cached read = 0.0117s / 50 reads
local speedup = 1.23x
```

但当前训练脚本每个样本通常从空 `memory_state` 开始，主要使用同一前向内的 prompt-local memory bank。这个场景下 KV cache 对训练速度帮助有限，收益不如 active-only backbone 和 emit controller 明显。它主要服务多轮对话、长上下文 prefill、外部系统复用历史 memory 的推理路径。

### 5.2 backbone 的 KV cache

当前训练用的小 backbone 是 `TransformerEncoder`，任务是 strict masked-source latent infill。它不是自回归 decoder，因此训练时不能直接复用 LLM 式 KV cache。

如果未来外部主干换成 AR/decoder-only LLM，则已成型 readout matrix 可以在 prefill 阶段先计算 backbone KV cache。这个优化属于推理路径，不应混入当前 masked infill 训练速度结论。

## 6. 后续 benchmark 矩阵

下一步只需要保留少量高价值矩阵：

| 目的 | 变量 |
| --- | --- |
| 确认 R 上限代价 | R=8 / 16 |
| 确认 memory 成本 | memory on / off |
| 确认 backbone 成本 | backbone on / off / active-only off |
| 确认 objective 成本 | coding-rate 每步 / 每 4 步 / 关闭 |
| 确认数据路径 | N: streaming / local NVMe staging |

已经证明会失败或不建议默认的配置：

- `batch_size=2`：显存超过 16GB。
- `grad_accum_steps=2/4/8`：在 `delta_lite` full 配置下 optimizer step/s 低于目标。
- 每个 checkpoint 保存 optimizer：归档和内存峰值压力过大。
