# FLUED v3.4 统一 5K 消融与曲线分析

## 1. 实验口径

- 17 组全部使用纠正后的 v3.4 两级决策代码。
- FLUED 约 37.26M，临时 backbone 约 4.78M，总计约 42.05M；不是 300M 完整模型。
- 每组 5000 step、batch 8、512 byte、seed 42、同一语料和数据顺序。
- 每 20 step 写一条 JSONL；每组 250 个唯一曲线点。
- 每 500 step 保存可恢复 latest；保留 2500/5000 里程碑。
- 所有组 `eval_truncated_tokens=0`。
- 本轮只能给出单种子结构筛选结论，不能替代完整规模和多种子实验。

## 2. 为什么 500 步结论不够

关键组 identity 曲线：

| 组别 | 500 | 1000 | 2500 | 5000 | 曲线均值 | 末 500 步斜率/1K |
|---|---:|---:|---:|---:|---:|---:|
| 精确 log-det 完整组 | 0.204 | 0.240 | 0.515 | 0.596 | 0.424 | -0.032 |
| L2 coding rate | 0.165 | 0.160 | 0.667 | 0.699 | 0.536 | +0.038 |
| 均匀边界 | 0.202 | 0.565 | 0.961 | 0.984 | 0.775 | -0.005 |
| soft emit 全计算 | 0.213 | 0.257 | 0.576 | 0.701 | 0.462 | +0.005 |

L2 在 1000 步前最差，2500 步后反超，并在 5K 末段仍保持正斜率。500 步会得出完全相反的结论。均匀边界约 1K 后快速收敛，2.5K 已接近平台。

## 3. 最终评估与计算量

| 组别 | eval identity | eval completion | actual latent/byte | 主要结论 |
|---|---:|---:|---:|---|
| full exact | 0.597 | 0.134 | 0.785 | 当前精确编码率基线 |
| L2 coding rate | **0.704** | **0.148** | **0.680** | 在动态压缩方案中支配 exact |
| uniform boundaries | 0.987 | 0.149 | 0.969 | 精度上限高，但接近 byte-level 计算量 |
| soft emit | 0.679 | 0.133 | 1.075 | soft 分数不能带来真实计算压缩 |
| no emit value | 0.448 | 0.134 | 0.626 | 价值监督提高容量和 identity，但没有约束 emit 数量 |
| no compute cost | 0.436 | 0.130 | 0.732 | 当前成本项未形成预期的单调抑制 |
| no memory | 0.499 | 0.134 | 0.461 | memory 提高可逆性，但未提高本轮补全且增加计算量 |
| codec only | 0.647 | 0.014 | 0.444 | 重建更强，但完全没有补全能力 |
| no position / no AR | 0.237 | 0.125 | 0.628 | chunk 内顺序能力严重不足 |
| position only | 0.420 | 0.131 | 0.708 | RoPE 是主要顺序来源 |
| AR only | 0.244 | 0.120 | 0.604 | 小 AR 单独基本无效 |
| plain byte lookup | 0.436 | 0.136 | 0.600 | 结构化 16x16 lookup 显著提高 identity |

## 4. 组件结论

### 4.1 位置与小 AR

RoPE 必须保留。小 AR 只有和 RoPE 同时存在时才显著提高 identity：完整组 0.597，position-only 0.420，AR-only 0.244，全关 0.237。补全差距较小，说明当前小 backbone 已成为补全瓶颈。

### 4.2 Memory

去掉 memory 后 identity 从 0.597 降到 0.499，但 completion 几乎不变，actual latent 从 0.785 降到 0.461。memory 对 codec/readout 可逆性有效，但在本轮 masked completion 上不是免费收益。后续必须画 accuracy-FLOPs Pareto，而不能只说 memory 提升或无效。

### 4.3 边界桥与弱先验

- 去掉软边界桥：identity 0.602，completion 0.127，actual 0.660。
- 去掉边界弱先验：identity 0.566，completion 0.134，actual 0.826。

在固定 Top-K 后，边界桥不再决定“切多少”，只负责让边界位置接受主任务塑形；它对补全有小幅帮助。弱先验仍改善 identity 和一定计算效率，但不再是能否启动训练的必要条件。

### 4.4 去噪与 lookup

无扩散噪声的 identity 更高（0.643），但 completion 更低（0.124）；噪声仍体现“可逆性 vs 下游鲁棒性”权衡。普通 lookup 的 identity 只有 0.436，结构化 byte lookup 的收益在长训练后扩大，应该保留。

### 4.5 Emit 价值与成本

500 步时价值/成本项看起来能阻止全开，但 5K 曲线证明该结论不成立：完整组 actual latent 曾从约 0.29 升到 0.93，再回落到约 0.74；去掉价值或成本后的最终计算量反而不一定更高。

当前 emit 目标主要在帮助模型分配表达容量，没有形成可靠的计算约束。主任务对“打开更多 readout”的梯度长期压过了弱成本项。下一版需要显式的自适应预算/对偶变量或硬总 latent 上限，不能继续仅调一个固定 cost weight。

## 5. Exact、L2 与 Uniform 的真正区别

2.5K 到 5K 的边界稳定性 F1@1：

| 方案 | English | Chinese | Code | Mixed |
|---|---:|---:|---:|---:|
| exact log-det | 0.857 | 0.857 | 0.833 | 1.000 |
| L2 approximation | 0.250 | 0.133 | 0.143 | 0.143 |
| uniform | 1.000 | 1.000 | 1.000 | 1.000 |

因此 exact 的问题不是边界抖动，而是过早固定在次优位置。L2 的边界持续重组，却得到更高 identity、completion 和更低 actual latent，说明其可塑性目前是优势。uniform 提供绝对稳定的对齐，decoder/backbone 最容易学习，但通过接近不压缩的计算量获得上限。

## 6. 当前决策

1. **下一轮动态边界默认候选改为 L2 coding rate**，exact log-det 保留为对照。
2. **RoPE + 小 AR 成组保留**；不再单独启用小 AR。
3. **结构化 byte lookup 保留**。
4. **memory 保留为可选 Pareto 旋钮**，需要计算匹配实验后才能决定默认开启。
5. **emit 必须继续硬前向并真实压紧**；soft-only 明确失败于计算效率。
6. **重构 emit 预算控制**：把实际 latent 总量作为约束，用自适应对偶优化，而非固定 cost 权重。
7. **尝试 uniform warmup -> L2 adaptive 的边界课程**：先让 decoder/backbone 获得稳定对齐，再逐步释放动态边界，兼顾 uniform 的可训练性和 L2 的压缩效率。
8. 在完成上述两项小规模验证前，不扩大到 300M/4096 byte。

## 7. 产物

- 原始实验：`K:\FLUED_archive\v34_rate_emit_all_ablation_5k_20260711`
- 数值汇总：`analysis/curve_summary.{json,csv,md}`
- 交互曲线：`analysis/curves_interactive.html`
- 边界稳定性：`analysis/boundary_stability.{json,md}`
- 每组原始曲线：`<run>/train_log.jsonl`
- 每组检查点：`latest.pt`、`step_002500.pt`、`step_005000.pt`
