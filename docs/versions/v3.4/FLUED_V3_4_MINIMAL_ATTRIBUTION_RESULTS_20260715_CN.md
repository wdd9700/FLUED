# FLUED v3.4 第一轮最小归因结果

日期：2026-07-15

## 1. 实验范围

本文执行扩模前归因计划中的三个低成本实验：冻结 readout 的 decoder 拟合、同检查点 memory
内容干预、边界课程切换前后的主任务梯度探针。实验没有改变 v3.4 正式结构，也没有重新训练
完整模型。

原始产物：

- `L:\FLUED_archive\v34_attribution_20260715\decoder_isolation_1k`
- `L:\FLUED_archive\v34_attribution_20260715\memory_interventions`
- `L:\FLUED_archive\v34_attribution_20260715\gradient_paths`

## 2. Decoder 函数形态归因

### 2.1 方法

使用 20K `no-memory + shared inverse + diag` 检查点作为冻结 encoder。每个批次只生成一次
readout、chunk 和 emit，然后同步训练：

1. 共享逆函数形态：使用 interpreter blocks、readout pool 和普通 byte lookup 的可训练副本；
2. 独立跨度 decoder：保持其独立参数，从随机初始化开始。

因此两条分支看到完全相同的 readout。共享逆继承 20K 权重，独立 decoder 从随机状态开始，
本实验不能比较谁的最终上限，只用于判断共享逆函数形态是否完全无法拟合固定 readout。

### 2.2 曲线

| 步数 | 共享逆损失 | 共享逆准确率 | 独立损失 | 独立准确率 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.5298 | 21.38% | 5.6731 | 0.78% |
| 100 | 1.8826 | 34.12% | 2.5326 | 26.29% |
| 500 | 1.5656 | 45.56% | 1.7022 | 44.74% |
| 600 | 1.5261 | 47.27% | 1.6100 | 47.69% |
| 1000 | 1.3093 | 56.28% | 1.4395 | 53.42% |

共享逆可训练参数 12,221,446，独立 decoder 为 2,561,025。共享逆在固定 readout 上能够持续
下降并达到 56.28%，因此“共享逆函数形态根本不能解码”被初步否定。原 20K 联合训练停在
约 21% 更可能来自 encoder、边界、emit 和共享逆之间的非平稳联合优化。

该结论不证明共享逆优于独立 decoder：它有 20K 初始化优势且可训练参数更多。下一轮若要比较
上限，应延长拟合并增加参数匹配对照。

## 3. Memory 内容干预

### 3.1 方法

固定 20K other-memory 检查点、输入、掩码和评估种子，只修改送入 interpreter 的 memory：

| 模式 | 重建准确率 | 补全准确率 | 补全困惑度 | 实际 latent/byte | 困惑度变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 29.60% | 12.31% | 42.162 | 0.3624 | +0.000 |
| zero | 30.40% | 12.12% | 42.685 | 0.3788 | +0.523 |
| shuffle-chunk | 29.61% | 12.27% | 42.178 | 0.3627 | +0.016 |
| stale-batch | 29.44% | 12.21% | 42.540 | 0.3653 | +0.378 |

### 3.2 解释

- 置零和跨样本错位会让补全困惑度略差，说明 memory 不是完全无效；
- 样本内 chunk 打乱只使困惑度变化 `+0.016`，几乎等于无影响；
- normal 并未在重建准确率上胜过 zero；
- 干预会改变 emit，从而使实际 latent/byte 略有变化，尚不是严格同预算因果估计。

当前最合理判断是：模型利用了 memory 的某种全局幅度或分布信息，但几乎没有利用“某个历史
chunk 对应某段内容”的关系。它尚未形成设计目标中的语义记忆序列。下一步应先检查
memory attention 的位置敏感性和 summarizer 的跨 chunk 可辨识度，不应直接扩大 memory 参数。

## 4. 边界课程梯度归因

只保留重建、补全和可见字节保持三个主任务，关闭所有边界先验、编码率、emit、memory 和
小 AR 辅助损失。比较 3K 课程切换起点与 6K 动态边界完全接管后的单批次梯度。

| 路径 | 步数 | 边界模式 | Segmentor共享层 L2 | 置信度头 L2 | readout pool L2 | backbone L2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 共享逆 | 3K | 均匀混合，alpha=0 | 0.000 | 0.000 | 2.673 | 0.512 |
| 共享逆 | 6K | 置信度阈值 | 23.045 | 7.437 | 2.993 | 0.386 |
| 独立 decoder | 3K | 均匀混合，alpha=0 | 0.000 | 0.000 | 2.917 | 0.502 |
| 独立 decoder | 6K | 置信度阈值 | 55.741 | 17.542 | 2.369 | 0.673 |

3K 的精确零梯度不是反向传播故障，而是课程在 `alpha=0` 时完全由均匀边界执行，主任务没有
路径塑形 Segmentor。动态边界接管后，Segmentor 梯度瞬间比 readout pool 和 backbone 大一个
数量级以上。这与训练曲线在切换后骤降一致。

因此当前问题不是“Segmentor 没有梯度”，而是课程制造了从零信用分配到强信用分配的梯度
冲击。共享逆同时把 decoder 梯度写回 interpreter/readout pool，使联合非平稳性进一步增加。

## 5. 第一轮结论

| 原问题 | 新定位 | 架构含义 |
| --- | --- | --- |
| 共享逆是否能力不足 | 固定 readout 上可快速拟合 | 首先修训练解耦和梯度路由，不立即删除共享逆 |
| memory 是否承载chunk语义 | 对chunk打乱近乎不敏感 | 当前memory不能作为已验证核心组件 |
| 动态边界为何切换后坠落 | Segmentor主任务梯度从0突增到7--56量级 | 课程必须渐进开放梯度，而不仅渐进混合边界分数 |

## 6. 下一步最小调整候选

1. 为共享逆增加 decoder 梯度缩放或交替更新：先固定 encoder 拟合逆路径，再逐渐联合解冻。
2. 将边界主任务梯度系数从 0 连续升到目标值，独立于 hard boundary 的前向课程。
3. 对 memory 增加位置敏感性探针；若打乱长期无影响，默认关闭 memory，再单独重做 summarizer。
4. 完成上述修正后只跑 5K/10K 配对，不直接启动 300M。

## 7. 用户复核后的修正与新增假设

以下内容保留为待验证问题，不覆盖前述实测结果，架构版本继续保持 **v3.4**。

### 7.1 训练时间基准整体翻倍

需要区分“原训练确实欠时长”和“课程切换机制本身错误”。增加两条 40K 严格对照：

| 组 | 总步数 | 均匀预热结束 | 过渡长度 | 目的 |
| --- | ---: | ---: | ---: | --- |
| S0 | 40K | 6K | 1K | 将原 20K/3K/500 全部按时间放大两倍 |
| S1 | 40K | 6K | 500 | 保持更短的相对过渡比例，测试旧过渡是否占比过高 |

除时间轴外保持语料、batch、优化器、学习率、种子、decoder、memory开关和评估集一致。配置见
`configs/v3_4/v34_boundary_schedule_40k_attribution.json`。

这里需要保留一个反向风险：当前梯度探针显示动态接管后梯度过强，缩短过渡也可能让冲击更
集中。因此 S1 是需要实测的用户假设，不预设一定优于 S0。

### 7.2 Memory 使用率监督确实被关闭

20K memory 修正版配置中的实际值为：

```text
memory_usage_loss_weight = 0.0
memory_usage_min = 0.2
memory_usage_max = 0.5
```

因此原规划中的 20%--50% memory 使用倾向并没有进入损失。当前代码虽记录
`memory_attention_other_share`，但 other-only attention 的合法键全部都是 other-memory，
这个比例接近 1 是结构掩码的必然结果，不代表memory相对于局部readout获得了多少计算权重。
实际局部/memory混合强度由 `memory_gate_mean`、`memory_residual_ratio` 和固定残差尺度决定。

新增三组同尺寸20K实验：关闭监督、权重0.02、权重0.05，目标区间均为0.2--0.5。配置见
`configs/v3_4/v34_memory_usage_supervision_20k.json`。这能回答“负收益是否因为memory梯度和
使用强度不足”，但不能直接证明memory内容已经语义化；训练后仍必须重复normal/zero/shuffle/
stale干预。

### 7.3 当前位置编码事实需要重新表述

本轮 memory 检查点不是简单的“提示级 ALiBi + chunk 内 RoPE”：

1. `position_strategy=layered_rope` 使 Segmentor、chunk pool 和 interpreter 的注意力使用 RoPE；
2. `use_prompt_alibi=true` 又在 Segmentor 的全byte注意力上叠加双向相对距离 ALiBi；
3. `prompt_position_scale=0`，没有向byte payload直接加绝对正弦位置；
4. `memory_use_position=false`，memory cross-attention 不使用 chunk RoPE 或 byte-anchor ALiBi。

ALiBi在这里是相对距离偏置，不是绝对位置向量。memory chunk打乱不敏感是在memory读取层无
显式位置编码的条件下得到的，不能解释成“模型学到了ALiBi的强位置作用”。

用户提出的四组位置消融仍然必要，但当前接口无法完整表达，必须先把位置开关解耦：

| 提示/Segmentor位置 | chunk pool位置 | 状态 |
| --- | --- | --- |
| RoPE | RoPE | 可由当前接口近似表达 |
| ALiBi | RoPE | Segmentor RoPE目前无法独立关闭 |
| RoPE | ALiBi | chunk pool尚无ALiBi实现 |
| ALiBi | ALiBi | 同时缺少两项独立接口 |

该矩阵用于判断位置机制的分工，不应与memory位置编码混为一谈。memory cross-attention是否使用
无位置、chunk RoPE或byte-anchor ALiBi，应作为另一组独立消融。

### 7.4 Memory结论降级

此前“20K足以排除训练不足”的表述过强。修正为：

- 20K足以说明**当前无使用率监督配置**不会自然形成明显的chunk内容依赖；
- 尚不能排除恢复20%--50%使用倾向后，memory得到更强主任务梯度并形成有效表示；
- 也不能排除更长上下文和更大模型才使memory收益出现；
- 但在内容打乱敏感性通过之前，memory仍不能恢复为默认核心组件。

## 8. 保留的未决问题

> 2026-07-16 状态更新：第 1--4 项已由
> [`FLUED_V3_4_ATTRIBUTION_MATRICES_RESULTS_20260716_CN.md`](FLUED_V3_4_ATTRIBUTION_MATRICES_RESULTS_20260716_CN.md)
> 继续验证。40K 和短过渡均未修复坍缩；memory 权重 0.05 形成当前最佳单种子候选，但暴露了
> gate 与实际内容贡献不一致的问题。以下列表保留为历史问题入口。

1. 40K时间轴放大能否消除动态边界接管后的长期损失；
2. 500步短过渡和1000步等比例过渡哪一个更稳定；
3. memory使用率监督能否提高内容敏感性，而不只是放大残差；
4. memory gate监督与实际attention分配是否需要进一步统一；
5. 提示级与chunk级RoPE/ALiBi四组消融的真实最优组合；
6. memory cross-attention自身的位置编码应选择none、chunk RoPE还是byte-anchor ALiBi；
7. 共享逆联合训练的梯度缩放、预热和交替更新哪一种最有效；
8. 编码率、任务收益和真实计算成本如何统一；
9. 2048/4096长度、多种子和300M扩模趋势。
