# FLUED v3.4 渐进课程、Memory 20K 与边界 ROI 审计

> 历史反例：本文发生在全局 interpreter 路径和 memory 尺度纠偏之前。其负结果用于解释为什么必须重做路径审计，不代表当前 no-self normalized memory。当前结论见 [`FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md`](FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md)。

## 1. 实验范围

本轮只回答三个问题：

1. `uniform -> L2` 的 2000 步连续迁移能否优于第 3000 步硬切换；
2. 同种子、同参数量、从零训练 20K 时，memory 是否降低外部 backbone 的补全难度；
3. 真实 checkpoint 的 hard boundary 是否已经具备可声称的语义自然度。

两组 memory 对照均为 37.26M FLUED + 4.78M 临时 backbone，seed=42，512 bytes，batch=8，5% byte mask。唯一实验变量是 `use_memory`。关闭 memory 时相关参数仍被实例化，保证参数量和初始化顺序一致。

## 2. 渐进课程

课程为：

```text
0-3000:    uniform_budget
3000-5000: 固定 chunk 数，uniform/L2 排序分数按 cosine 连续混合
5000-20000: L2 marginal Top-K
```

硬边界和软边界桥使用同一个退火系数。阶段二开始时与均匀边界完全相同，结束时与纯 L2 相同。

### 2.1 最终固定评估

| 方案 | 重建准确率 | 补全准确率 | 补全 CE | 补全困惑度 | 实际 latent/byte |
|---|---:|---:|---:|---:|---:|
| 渐进课程 + memory | 78.15% | 12.41% | 3.7102 | 41.81 | **0.4326** |
| 渐进课程 - memory | **87.43%** | **13.96%** | **3.5040** | **33.77** | 0.5396 |
| 旧硬切换 + memory | 88.85% | 14.74% | - | - | 0.5225 |

旧硬切换来自前一轮 20K 归档。它与本轮都使用同一版首边界预算实现，但其 `memory_usage_loss_weight=0.02`，本轮为 0；因此只能作为高可信趋势对照，不是唯一变量的严格复跑。

### 2.2 训练阶段均值

| 阶段 | 渐进 + memory 重建 | 渐进 - memory 重建 | 渐进 + memory latent/byte | 渐进 - memory latent/byte |
|---|---:|---:|---:|---:|
| 3000-4000 | 98.66% | 96.42% | 0.9603 | 0.8253 |
| 4000-5000 | 78.40% | 82.29% | 0.9855 | 0.9139 |
| 5000-10000 | 83.48% | 85.28% | 0.8882 | 0.8220 |
| 10000-15000 | 75.56% | 84.40% | 0.5668 | 0.5709 |
| 15000-20000 | 77.48% | 84.27% | 0.4436 | 0.5158 |

本轮结论：渐进迁移消除了硬切换第一瞬间的巨大跌落，但没有保留均匀预热的最终任务优势。L2 权重超过约一半后仍会重排表示，最终结果接近纯 L2 的压缩优先解。**在这条旧路径中，渐进课程不能成为新默认；旧硬切换是当时任务质量更好的参考路线。**

## 3. 本轮旧路径的 Memory 结论

这轮纠偏前的 20K 配对不支持“memory 提高 readout 精度并降低 backbone 困惑度”：

- memory 使实际 latent/byte 降低约 19.8%；
- 但重建下降 9.28 个百分点；
- 补全下降 1.55 个百分点；
- 补全困惑度上升约 23.8%。

在本轮 progressive/旧路径中，memory 更像一种隐式压缩压力：interpreter 使用 memory 后，emit controller 倾向于关闭更多额外 readout，但剩余表示不足以补偿任务损失。该判断已被后续全局路径、位置和尺度纠偏后的 P4 实验取代，不能用于关闭当前 normalized other-only memory。

## 4. 边界 ROI

真实 `progressive_memory_on` checkpoint 在 66 个 case、33 组成对扰动、9626 bytes 上完成 CPU 审计。

### 4.1 可以支持

- 2106 个 UTF-8 continuation byte 上没有 hard cut；
- 多数轻微扰动下边界有一定稳定性，总体 pair hard F1 约 0.896；
- 英文、公式、重复文本和部分代码中，L2 Top-K 常选择空格、标点或运算符附近。

### 4.2 不能支持

- 不能声称已学会语义分块；中文表面自然率仅 59.7%，实体密集文本为 66.1%；
- 不能声称 confidence 驱动最终边界；当前 hard boundary 由 L2 Top-K 决定；
- logic transition 在 66 个 case 中总数为 0，confidence 最大值 0.641，未达到 0.75 阈值；
- 635 个非首边界中 435 个落在空格，存在明显 byte 类别捷径；
- L2 分数差异很小，轻微改名可触发全局 Top-K 重新排序；代码变量改名样本的 pair F1 最低降到 0.429。

具体错误包括中文词组内部切分、英文类名/路径内部切分，以及近似同分候选导致远处边界漂移。当前证据只支持“UTF-8 安全的 L2 固定预算切分”，不支持“语义边界学习”。

## 5. 已修复的预算错误

旧 selector 先按目标 K 做 Top-K，再无条件加入首 byte，导致 63/66 个 case 实际多一个 chunk。现已改为：

```text
首 byte 占用一个预算
额外候选只选择 K-1 个
```

本轮历史 checkpoint 保留旧行为，以保证实验可追溯；后续训练使用修复后的实现。

## 6. 动态 chunk 的修订方案

方向成立：不预测每个样本的固定 K，而是约束批次平均真实计算成本。原提案需要修正两个符号：

\[
C_b = c_{chunk}N_{chunk,b} + c_{latent}N_{emit,b} + c_{attn}N_{emit,b}^{2}
\]

\[
\text{切分净价值}=\text{信息增益}+\text{任务收益}-\text{新增计算成本}
\]

二次项用于近似 Transformer backbone 的注意力成本；系数应由真实 profiler 拟合。按有效 byte 归一化后，约束批次均值：

\[
\bar C=\frac{1}{B}\sum_b\frac{C_b}{L_b},\qquad
L_{primal}=L_{recon}+\alpha L_{completion}+\lambda(\bar C-C_{target})
\]

\[
\lambda\leftarrow\operatorname{clip}\left(\lambda+\eta(\bar C-C_{target}),0,\lambda_{max}\right)
\]

同一个对偶价格同时进入 cut 与 emit 决策，避免两个控制器分别透支预算。UTF-8 continuation、`max_span` 和 `max_chunks` 继续作为安全约束。

### 6.1 调整后的迁移顺序

渐进课程已经表明 2000 步平滑混合并不保留任务上限，因此动态数量实验不应继续把它当默认：

1. 0-3000：均匀边界预热；
2. 第 3000 步：使用当前硬切换参考，固定 K 训练至边界重新稳定；
3. 先开放分桶数量，并只约束批次平均成本；
4. 分桶稳定后再改成动态阈值；
5. 最后加入低频 boundary merge probe，估计移除边界的任务损失增量。

动态阈值前向使用 hard decision，反向继续使用连续概率和 soft boundary bridge。需要注意：soft bridge 只能提供近似 credit，不能让离散 chunk 归属本身完全可微，因此 merge probe 是必要校准，而不是装饰性指标。

## 7. 下一步验收

动态 chunk 进入主线前至少满足：

1. chunk 数严格符合预算账本，不再出现首边界重复计数；
2. `actual_backbone_units_per_byte` 随预算单调变化；
3. 同预算下不低于修复后的固定 K 基线；
4. 总体扰动 pair F1 >= 0.95，各类别 >= 0.90；
5. 中文、实体和代码内部错误显著下降；
6. 真实 profiler FLOPs/时延与成本代理保持稳定相关；
7. 至少三个种子后才决定是否扩大到 300M / 4096 bytes。
