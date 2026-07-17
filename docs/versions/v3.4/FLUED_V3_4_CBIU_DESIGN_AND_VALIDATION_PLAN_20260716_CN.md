# FLUED v3.4 反事实字节接口效用设计与验证计划

> 日期：2026-07-16  
> 状态：设计冻结；emit 在线训练已完成三轮验证，boundary/memory/small-AR 尚未接管  
> 英文名：Counterfactual Byte-Interface Utility（CBIU）  
> 目标：用同一种可执行的反事实信用，监督 boundary、readout emit、memory 和小自回归修正。

## 1. 为什么需要重新定义监督

v3.4 当前同时存在重建、严格掩码补全、可见字节保持、边界先验、L2 编码率代理、emit
价值、memory 使用率和计算成本等信号。已有归因结果表明，问题不是简单的训练步数不足：

1. hard emit 在动态边界接管前已经造成有效表示容量坍缩；
2. 动态边界接管后 Segmentor 梯度突然增大，形成第二次优化冲击；
3. 固定损失权重不能稳定表达“任务收益是否值得付出真实计算”；
4. memory gate 强度不等于 memory 内容的因果贡献；
5. 重建、补全和压缩可以互相补偿，单一加权和可能掩盖其中一项失败。

因此，CBIU 不再增加一个普通 loss，而是把所有动态结构决策改写为同一个问题：

> 在输入、参数、mask 和随机状态不变时，删除该结构决策会让字节接口的约束目标变好还是变坏？

## 2. 历史位置与原创边界

CBIU 使用的基础思想均有历史来源：

- Shannon 率失真理论定义码率与失真的约束关系；
- Rissanen 最小描述长度同时考虑结构和数据解释成本；
- 信息瓶颈研究压缩输入并保留任务相关信息；
- Optimal Brain Damage / Surgeon 用删除组件后的损失增量衡量重要性；
- ACT、动态路由和 token pruning 将任务收益与额外计算联系起来；
- Pearl 干预和 Shapley 边际贡献提供反事实与交互归因语言；
- BLT、Charformer 和 H-Net 已覆盖动态 byte/character 分块的部分问题。

不能宣称原创的内容包括：率失真、动态分块、删除敏感度、硬门控、动态计算或普通
`task loss + rate penalty`。

可能形成方法贡献的是以下窄而完整的组合：

> 在可逆 byte 接口上，以冻结锚点归一化的三风险最坏项作为统一质量尺度；使用同一干预契约
> 评价 boundary、readout、memory 和小自回归修正；再将真实执行预算的对偶影子价格直接校准
> 为 `[-1, 1]` 的连续决策置信度。

在完成更系统的论文与专利检索前，文档只使用“潜在原创方法”，不使用“首次”。

## 3. 基础字节接口风险

令干净字节流为 `X`，在进入 FLUED 前采样的严格 mask 为 `M`，掩码输入为 `X_masked`。
所有交叉熵统一换算为 `bits / target byte`。

### 3.1 干净重建风险

```text
X -> FLUED encoder -> readout -> shared approximate-inverse decoder -> X_hat
```

`R_rec` 在全部有效 byte 上计算，测试真正的 clean 可逆性。当前训练中“重建带 MASK 的输入”
不能替代该指标。

### 3.2 严格补全风险

```text
X_masked -> FLUED encoder -> backbone -> completed readout -> decoder
```

`R_fill` 只在原始 mask 位置计算。FLUED 从未看到未掩码原文，禁止 clean readout 侧漏。

### 3.3 可见字节保持风险

`R_keep` 在受补全影响的 chunk 中，对未掩码 byte 计算。它约束 backbone 只创造缺失信息，
不能为了补一个位置而破坏已知内容。

### 3.4 无量纲化与最坏项聚合

对每项固定高容量参考系统 `A_rich` 和空接口参考系统 `A_null`：

```math
r_j(A) = (R_j(A) - R_j(A_rich)) / (R_j(A_null) - R_j(A_rich))
```

其中 `j in {rec, fill, keep}`。锚点在一次实验内冻结，禁止和被评模型共同漂移。

统一风险定义为：

```math
rho(A) = max(r_rec, r_fill, r_keep)
```

训练实现可使用温度固定的 `logsumexp` 平滑近似。选择最坏项是公开的鲁棒设计假设：任何一项
失败都不允许被另外两项抵消；它不是自然定律。

## 4. 真实计算约束

训练账本必须统计硬执行后的真实结构，而不是 soft gate 或 latent 范数：

```text
C = chunk construction
  + emitted readout projection
  + compacted backbone FLOPs / latency
  + memory generation and cross-attention
  + small-AR correction
  + boundary / emit side-information cost
```

训练时使用可微 surrogate，定期用硬件 profiler 校准。对预算 `B` 定义 `g=C/B-1`，由对偶变量
或增广拉格朗日更新计算影子价格。若延迟、峰值显存和能耗均有硬要求，应分别设约束，不能先
拍脑袋折成一个固定权重。

## 5. CBIU 定义

设 `A` 为完整执行图，`T_-a(A)` 为删除结构决策 `a` 后、从受影响构造点开始严格重算的图。
定义约束目标：

```math
F(A) = rho(A) + AL(C(A) / B - 1, lambda)
```

保留决策 `a` 的反事实字节接口效用为：

```math
U_a = F(T_-a(A)) - F(A)
```

- `U_a > 0`：删除后更差，应保留；
- `U_a < 0`：删除后更好，应关闭；
- `abs(U_a)`：当前共适应背景下的重要程度。

单点 CBIU 不能直接相加。相邻边界、boundary×emit、memory×AR 等强交互需要分组干预；模块级
四组件只有 16 种组合，可以周期性计算精确 group-Shapley 作为审计，不进入每步训练。

## 6. 四类统一干预契约

| 决策 | `T_-a` 的严格定义 | 不能接受的伪干预 |
|---|---|---|
| Boundary | 强制取消边界、合并相邻 chunk，并重算 readout/memory/compact backbone | 只把 confidence 设零但沿用旧 chunk |
| Extra readout | 关闭可选槽并真正从 backbone 输入中 compact；fallback 永远保留 | 只把向量乘零但仍计算该 token |
| Memory | 跳过对应 memory 的生成和读取；另做 stale/null/order 探针 | 置零但仍执行，却声称节省计算 |
| Small AR | 跳过该次修正及其增量计算，直接使用 one-shot 表示 | 只惩罚修正残差幅度 |

memory 至少区分三种问题：

1. `skip execution`：测信息与计算的联合净价值；
2. `stale batch replacement`：测样本特异内容；
3. `order intervention`：在 byte-anchor 位置机制启用后测顺序价值。

每个 memory 和 small-AR 干预还必须成对报告：

- `total effect`：允许下游 emit 随干预变化，测完整系统总效应；
- `fixed-emit direct effect`：冻结正常路径的 hard emit 图，测组件在相同 backbone latent 容量下的直接效应。

两者之差是由 emit 中介的交互效应。只报告其中一个会把“memory 改变了容量分配”误写成
“memory 内容本身有益/有害”。

## 7. 与 `[-1, 1]` 置信度对齐

令决策头预测：

```math
p_a = P(U_a > 0 | h_a)
s_a = 2 p_a - 1
```

于是现有阈值获得概率解释：

| score | 概率解释 | 行为 |
|---:|---:|---|
| `-1` | 0% | UTF-8 continuation 等结构性禁切 |
| `0` | 50% | 当前证据不足 |
| `0.5` | 75% | 标点等弱先验目标，不足以硬切 |
| `0.75` | 87.5% | chunk 内逻辑转折参考 |
| `0.9` | 95% | 硬 chunk 边界或高把握额外执行 |

训练目标使用 `sigmoid(stopgrad(U_a) / tau)` 或带置信区间的软标签。`tau` 使用 CBIU 的运行中
稳健尺度（EMA/MAD）校准，不为每个组件手调一套温度。

## 8. 现有信号的新位置

| 当前信号 | CBIU 方案中的位置 |
|---|---|
| L2 边际编码率 | 候选边界特征和反事实采样 proposal，不再直接决定最终边界 |
| hard emit + ST | 保留；CBIU 只替换 emit 的价值目标和预算方式 |
| 固定 chunk/readout 数 | 删除为主控制器；只保留 `max_span/max_chunks/fallback` 安全上限 |
| memory 使用率 20%-50% | 降为诊断/短期 bootstrap，不作为永久目标 |
| boundary weak prior | 保留为 UTF-8 硬约束和标点弱先验 |
| 固定十步交替 | 不作为默认；只允许未共享 adapter/head 基于残差触发 catch-up |
| 四段独立课程 | 不采用；短暂全容量 codec 对齐后由同一 CBIU/对偶控制器接管 |

共享 core 上固定十步切换 loss 不是严格的分块坐标优化，因为两边都在修改同一组参数。

## 9. 低成本估计

精确重跑每个候选不可行，采用三级估计：

1. 所有候选用一阶 Taylor/梯度近似粗排；
2. 每个 batch 按 proposal 抽少量动作做共享随机性的精确配对干预；
3. 用精确结果训练轻量 utility head，周期性做模块级组合干预。

采样 proposal 使用：

```text
q = (1 - epsilon) * q_saliency + epsilon * Uniform
```

L2 编码率、当前 confidence 和梯度敏感度可进入 `q_saliency`。保留均匀探索以覆盖低分候选，
必要时用 Horvitz-Thompson 权重得到目标动作分布下的无偏总体估计。训练日志必须报告有效样本量、
最大重要性权重和精确/近似 CBIU 的秩相关。

## 10. 防止模型钻空子

1. CBIU 标签必须 `detach`，不能让 encoder 直接改变自己的评分标准；
2. 使用慢速 EMA 快照或冻结审计 checkpoint 产生标签；
3. full/counterfactual 共享输入、mask、dropout 和扩散噪声；
4. backbone 只能看到 hard emit 后的 compact latent；
5. boundary 图、emit 图和长度模式属于 side channel，必须进入成本账本；
6. 未量化实数 latent 的数量不是 Shannon 码率，论文需区分执行压缩与真实信息码率；
7. zero memory 可能离分布，必须同时做 stale/null/skip-execution；
8. 联合训练只证明共适应可利用度，语义友好度还需 fresh-backbone 和无教师干预测试。

## 11. CBIU 不等于语义自然度

CBIU 首先测量固定模型和预算下的接口可利用度。它还可能混合：

- 序列变短带来的收益；
- 高频 n-gram 和 UTF-8 结构收益；
- 实体、数字、代码标识符的复制收益；
- 真正的实体关系、事件结构和程序行为信息。

因此，语义 claim 必须通过独立、无教师、可证伪的审计：

1. Unicode NFC/NFD、空格和格式变化：语义保持、表面变化；
2. 代码一致变量重命名和等价 AST 重写：行为保持；
3. 改变实体绑定、事件角色、运算符、常量和控制流：语义改变；
4. nonce 实体复制与实体身份追踪分开统计；
5. 无序事实 memory 应对置换稳定，有序事件/代码 memory 应对反转敏感；
6. 在 fresh 小 backbone 和不同规模 backbone 上检查 CBIU 排序是否迁移。

## 12. 最小验证决策链

### V0：离线可辨识性，不改训练

对现有 v3.4 checkpoint 采样 boundary/readout/memory/AR，精确计算 paired CBIU。验收：

- 同一动作在不同 mask seed 上多数同号；
- 当前 confidence 与 `U_a > 0` 的 AUC/Brier/ECE 优于随机；
- 近似 CBIU 与精确 CBIU 的 Spearman 相关为正且稳定；
- zero/stale/skip memory 的解释不互相混写；
- 实际计算差值来自 compact 后执行图。

### V1：emit-only 在线接管

固定边界和 memory，只用 CBIU 替换 extra readout 价值监督。emit 是最干净的反事实。若 5K 内
不能同时改善真实 latent 数与三风险最坏项，则停止扩大。

### V2：boundary + emit

保留 L2 作为 proposal，CBIU 接管边界净收益和 emit 净收益；使用同一计算对偶价格。比较当前
`uniform -> L2` 课程，检查是否消除 3K 后容量坍缩与 6K 后梯度冲击。

### V3：memory 与 small AR

只在 V2 稳定后接入。memory 使用 stale/null/skip 三种目标；small AR 依据额外修正的净收益决定
执行，不采用固定十步交替。

### V4：语义与迁移审计

使用中英文、UTF-8、实体、代码和 ordered/set memory 合成任务；更换 fresh backbone，验证
CBIU 排序是否保留。通过后才能将“backbone-friendly”提升为“语义友好”。

## 13. 当前实现与验证边界

2026-07-17 已完成：

- 固定 rich/null 锚点的三风险归一化与最坏项聚合；
- 单个 `(sample, chunk, extra slot)` 的严格开/关配对重算；
- 必需 masked writable slot 排除、hard compact 与标签停止梯度；
- CBIU quality 和预算对偶两种 emit 在线目标；
- 训练状态随检查点保存和恢复，锚点来源不匹配时拒绝运行；
- emit-only 隔离训练、线性/MLP/slot 三种控制器容量；
- per-action Spearman、AUC、Brier、ECE、同号率和 Top-25% 重合率；
- 分离 encoder、compact、backbone、decoder 的 CUDA 运行时测量。

三轮结果见
[`FLUED_V3_4_CBIU_THREE_ROUND_RESULTS_20260717_CN.md`](FLUED_V3_4_CBIU_THREE_ROUND_RESULTS_20260717_CN.md)。
当前证据支持 CBIU 作为 emit 价值监督候选，但校准尚未通过 boundary 接管门槛。

尚未完成：

- boundary 取消后从 chunk 构造点开始的 strict merge-rebuild；
- memory skip/stale/order 的在线 CBIU 与计算账本；
- small-AR skip 的在线 CBIU；
- 多种子、长序列、fresh backbone、语义迁移和大模型验证；
- profiler 校准后的统一硬件成本对偶约束。

因此，CBIU 已从“纯设计协议”推进到“emit 上得到初步支持的实验方法”，仍不是 v3.4 全组件已验证能力。

## 14. 参考源头

- Shannon, 1959, Coding Theorems for a Discrete Source with a Fidelity Criterion
- Rissanen, 1978, Modeling by Shortest Data Description
- LeCun et al., 1989/1990, Optimal Brain Damage
- Hassibi and Stork, 1992/1993, Optimal Brain Surgeon
- Tishby et al., 1999, The Information Bottleneck Method
- Graves, 2016, Adaptive Computation Time
- Louizos et al., 2018, Learning Sparse Neural Networks through L0 Regularization
- Molchanov et al., 2019, Importance Estimation for Neural Network Pruning
- BLT, 2024, Byte Latent Transformer
- H-Net, 2025, Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
