# FLUED v3.4 全局路径纠偏与消融重跑范围

> 本文记录第一轮纠偏及其当时结果。后续 P0-P4 严格串行实验加入提示级 ALiBi、memory 位置对照和无仿射 LayerNorm，并完成 20K 三组确认；当前默认结论见 [`FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md`](FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md)。

## 1. 根因

旧 v3.4 在 memory 注入后执行：

```text
[B, chunks, readouts, dim]
-> flatten 为 [B, chunks * readouts, dim]
-> 全局 interpreter self-attention
```

这使 `use_memory=false` 的模型仍可让所有 chunk readout 直接通信。它偏离既定职责：interpreter 应精确翻译当前 chunk，跨 chunk 信息只能由排除当前 chunk 的 memory 提供。

旧 no-memory 还继续计算 memory pool、memory attention、memory gate 和 AR memory correction。虽然主任务 residual 最终乘零，但计算没有关闭，AR 辅助损失仍可影响共享 GRU。

因此旧 memory 消融比较的是：

```text
全局 readout attention
vs
全局 readout attention + 无位置、全尺度 memory residual
```

而不是设计目标中的：

```text
chunk-local interpreter
vs
chunk-local interpreter + other-chunk memory
```

## 2. 代码纠偏

1. interpreter 改为每个 chunk 内独立并行运行；
2. no-memory 真正跳过 memory summarizer、memory attention、memory gate 和 AR memory correction；
3. 位置编码支持分层 RoPE、提示级绝对位置和二者混合三种策略；后续确定默认不在 memory 上叠加 chunk 索引 RoPE；
4. memory residual 默认缩放为 0.1；
5. 保持 no-self，当前 chunk memory 仍禁止回读，避免形成 readout 的低维复制捷径；
6. 删除没有激活过的 logic-transition 向量注入；
7. 修复首 byte 在固定 Top-K 中被重复计数的问题；
8. 小模型与正式 `FLUEDV34` 共用同一实现。

## 3. 历史结论影响

### 3.1 必须重跑

| 历史实验 | 为什么受影响 |
|---|---|
| memory on/off、memory 使用率 | 直接研究对象与实际执行路径不符 |
| RoPE / small AR | 旧全局 interpreter 自带跨 chunk RoPE，贡献归因被混合 |
| exact / L2 / uniform boundary | 下游任务梯度经过错误的全局 readout 路径，且 Top-K 有首边界预算错误 |
| uniform -> L2 课程 | 切换后的表示重组发生在错误 interpreter 上 |
| hard / soft emit | emit controller 输入来自错误的全局混合 readout |
| emit value / compute cost | removal delta 和实际 latent 价值被错误 readout 路径改变 |
| boundary bridge | 主任务 credit 经过错误的全局 interpreter |
| boundary weak prior | 先验损失本身有效，但对任务与压缩的贡献量必须重测 |
| structured / plain byte lookup | 方向可能保留，数值贡献和交互必须重测 |
| diffusion noise | interpreter 从全局改为 chunk-local，去噪作用位置改变 |
| codec-only / joint backbone | “必须有补全任务”的原则保留，但数值和 Pareto 必须重测 |
| 吞吐、显存、扩展趋势 | attention 从全局 latent 序列改为 chunk-local，复杂度已经改变 |

### 3.2 可以保留为结构事实

- strict masked-source：先 mask 原始 byte，再编码；
- UTF-8 continuation hard guard；
- decoder 不读取 memory；
- backbone 只接收 readout；
- hard emit 才能减少实际 backbone token，soft gate 本身不减少计算；
- reconstruction-only 不足以证明 backbone-friendly latent，这一点 v1 已经证伪；
- 历史日志与 ROI 可保留为旧实现的反例和问题定位证据。

### 3.3 直接退役

- logic-transition prior：真实 checkpoint 的 66-case ROI 中激活次数为 0；
- 固定 memory 使用率区间：它不监督 memory 是否有任务价值；
- “旧 20K memory 会降低困惑度/提高精度”等任何正向或负向架构 claim。

## 4. 重跑顺序

### P0：Memory 根因矩阵

同种子 5K：

1. memory + chunk RoPE + residual 0.1；
2. no-memory、严格 chunk-local；
3. memory 无 chunk 位置；
4. memory residual 1.0。

先判断 memory 是否有效，以及旧问题主要来自重复全局通道、缺位置还是注入过强。

### P1：纠偏后的完整 5K 消融

重新覆盖位置/AR、边界策略、emit、先验、边界桥、字节表、噪声和联合任务。logic-transition 与 memory 使用率约束不再占实验位。

### P2：延长与多种子

只把 P1 的 Pareto 前三名延长到 20K；memory 最佳和 no-memory 至少一个配对必须达到 15K。随后对最终两组做三种子。

### P3：动态 chunk

在修复后的固定 K 基线上再引入分桶数量和批次级计算对偶约束，不与本轮架构纠偏同时训练。

### P4：完整版

正式配置为约 333M FLUED + 107M 临时 backbone、4096 bytes。只有 P2 稳定后才启动完整训练。

## 5. 确定性重跑结果

### 5.1 位置编码放置

5K、同种子、确定性训练表明：

| 方案 | memory | 重建准确率 | 补全准确率 | 困惑度 | 实际潜向量/字节 |
|---|---:|---:|---:|---:|---:|
| 提示级位置 + 局部 RoPE，scale=0.03 | 开 | 76% | 14% | 35.92 | 0.68 |
| 提示级位置 + 局部 RoPE，scale=0.10 | 开 | 73% | 15% | 37.51 | 0.67 |
| 提示级位置 + 局部 RoPE，scale=0.30 | 开 | 68% | 14% | 39.59 | 0.72 |
| 旧分层 RoPE | 开 | 56% | 14% | 39.32 | 0.78 |

结论不是删除局部顺序信息，而是把两类位置职责拆开：

- 提示级绝对位置在结构化 byte lookup 之后注入，随 byte 内容进入 readout 与 memory；
- segmentor、chunk 内查询池和 chunk-local interpreter 保留局部 RoPE；
- other-chunk memory 不再叠加动态 chunk 索引 RoPE，避免和提示级位置冲突；
- 当前默认 `prompt_plus_local_rope`，`prompt_position_scale=0.03`。

### 5.2 Memory 5K 与 20K

两组都从零训练、同种子、同规模、同位置策略，唯一结构差异为 `use_memory`。5K 使用 20K 轨迹中保存的里程碑检查点，避免另一次训练的随机漂移。

| 步数 | memory | 发声阈值 | 重建准确率 | 补全准确率 | 困惑度 | 实际潜向量/字节 |
|---:|---:|---:|---:|---:|---:|---:|
| 5K | 开 | 0.50 | 88.80% | 13.75% | 35.72 | 0.719 |
| 5K | 关 | 0.50 | 85.41% | 12.61% | 39.31 | 0.718 |
| 20K | 开 | 0.50 | 81.05% | 13.53% | 35.96 | 0.346 |
| 20K | 关 | 0.50 | 97.04% | 13.24% | 37.91 | 0.588 |

默认阈值并不具有相同计算预算，因此还必须比较阈值扫描形成的 Pareto 曲线：

| 约束预算 | memory 阈值/实际量 | memory 困惑度 | no-memory 阈值/实际量 | no-memory 困惑度 | 判断 |
|---:|---:|---:|---:|---:|---|
| 0.59 | 0.34 / 0.607 | 40.01 | 0.50 / 0.588 | 37.91 | 高预算时 memory 较差 |
| 0.46 | 0.42 / 0.457 | 35.82 | 0.54 / 0.455 | 37.85 | memory 更优 |
| 0.39 | 0.46 / 0.397 | 35.87 | 0.56 / 0.391 | 37.93 | memory 更优 |
| 0.34 | 0.50 / 0.346 | 35.96 | 0.58 / 0.335 | 37.89 | memory 更优 |

这组已被后续 P4 覆盖的历史结果，当时纠正了“memory 起倒忙”的单点结论：

1. 当时 5K 的 memory 同时改善重建和补全，说明分支本身可以学习；
2. 20K 时 memory 成为低码率语义辅助，在紧计算预算下改善补全困惑度；
3. memory 对精确重建的长期损害仍然真实，不能被困惑度收益掩盖；
4. `memory_context_norm≈106`、`memory_residual_ratio≈0.21`，说明 memory 与局部 readout 的尺度对齐仍需修正；
5. 下一轮应先做 memory 归一化/残差尺度消融，并在相同实际潜向量预算下比较，不能再只看默认发声阈值。

完整数据见 [`../../../results/v3.4/position_memory_rerun_20260712/`](../../../results/v3.4/position_memory_rerun_20260712/)。

## 6. Claim 边界

本文当时可以声称提示级位置与局部 RoPE 的混合方案优于旧分层位置方案，并且旧 memory 路径在低实际潜向量预算下形成了补全 Pareto 收益。该结论已由 2026-07-13 P4 取代；本文数据只用于说明为什么需要严格 masked-source、真实计算账本、边界 ROI 和逐路径审计。
