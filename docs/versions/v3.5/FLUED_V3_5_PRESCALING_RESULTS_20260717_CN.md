# FLUED v3.5 Pre-Scaling 实验结果与归因分析

> 日期：2026-07-17（深夜）
> 状态：L0-L3 全部工程层实验完成；本文是 scaling 决策前的归因入口
> 归档：
> - `L:\FLUED_archive\v35_l0_codec_20k_20260717`
> - `L:\FLUED_archive\v35_emit_anneal_20k_20260717`
> - `L:\FLUED_archive\v35_l1_backbone_3k_20260717`
> - `L:\FLUED_archive\v35_l2_offline_utility_20260717`
> - 曲线：`各归档内 v35_prescaling_20260717_curves.png`
> 结论级别：38M/512-byte 工程闸门（2 seeds 或单 seed 按层注明）；不是论文级证据。

## 0. 执行摘要

v3.5 分级冻结协议的前四层全部跑完。核心发现：

1. **decoder 结构是 v3.4 全部病态的第一因**。仅把共享逆换成独立 decoder（其余结构
   不变），codec 即饱和（重建 0.993）、顺序通道恢复（order 探针 0.70，共享逆路径曾
   退化到 0.039）、对 emit 冲击天然稳健（阶跃切换不再坍缩）。
2. **分位数预算退火第一次给出平滑压缩路径**：1.0→0.2 预算在 5K 步内渐减，
   经历 -34pp 的率失真代价谷后完全恢复（0.972 @ 0.257 latent/byte），无悬崖、
   无不稳定。阈值控制器在健康 codec 上不坍缩但也学不会压缩（终点仍 0.81）。
3. **动作效用是类级信号**：30,304 条多 mask 离线 CBIU 记录上，类均值查找
   Spearman 0.439，实例级 MLP 仅 0.095（连训练集方差都解释不了）。L3 的正确
   形态是类级查找/聚类头，不是实例级控制器。
4. 预注册闸门有两处未达字面标准（order 探针 0.70<1.0；退火窗口回撤 -34pp<-20pp），
   归因上均有非失败性解释，但按规则如实标记，scaling 决策需显式裁定。

## 1. L0：codec 本体（2 seeds × 20K）

配置：`configs/v3_5/v35_l0_codec_20k.json`。plain lookup + RoPE/prompt ALiBi +
小 AR + **独立 decoder** + uniform 16B 边界（不学动态）+ 无 emit + 无 memory。

| 指标（20K 终点固定评估） | s042 | s123 |
|---|---:|---:|
| 重建 | 0.9930 | 0.9924 |
| 补全 | 0.2499 | 0.2565 |
| PPL | 18.18 | 17.63 |
| 保持 | 0.9909 | 0.9934 |
| actual latent/byte | 1.000 | 1.000 |
| order swap/subst | 0.695 | 0.702 |
| UTF-8 continuation 均值 | -0.998 | -0.998 |

闸门核对（预注册）：

- 重建 ≥ 0.48：**过**（0.993，远超）；
- order 探针 ≥ 1.0：**未过**（0.695-0.702）。如实说明：该阈值取自 d1 独立
  decoder 在 emit+动态边界配置下的 1.17；L0 是 uniform 边界+全容量配置，
  swap 扰动天然小于 substitution（0.70 仍远高于共享逆病态路径的 0.039），
  顺序已注册，阈值标定不当而非通道缺失。**维持标记，待 scaling 决策裁定**；
- 无 NaN/溢出/continuation 硬切：**过**。

latent 质量三视图：

| 视图 | 证据 | 读数 |
|---|---|---|
| 系统重建上限（独立 decoder） | eval 重建 | 0.993 |
| 顺序注册 | order swap/subst | 0.70（病态路径 0.039） |
| 低容量线性探针（冻结 readout，4 类 chunk 内容标签） | probe acc vs majority/shuffled | **98.5%** vs 89.9% / 94.7% |

## 2. emit 引入：阶跃 vs 分位数退火（20K，seed 42）

配置：`configs/v3_5/v35_emit_anneal_20k.json`（基座=L0 两臂 + emit 控制器）。

| 臂 | 3K | 5K | 8K | 20K 终点评估 | actual latent |
|---|---:|---:|---:|---|---:|
| e1 阶跃（warmup 3K 后阈值接管） | 0.890 | 0.953 | 0.909 | 0.9882 / PPL 18.21 | 0.812 |
| e2 分位数退火（3K-8K，1.0→0.2） | 0.983 | 0.871 | **0.645** | 0.9584 / PPL 20.35 | **0.257** |

e2 轨迹：预算渐减 → 代价谷（7K-8K，最大回撤 -34pp）→ **完全恢复**
（9K 0.792 → 10K 0.854 → 15K 0.962 → 20K 0.972 训练值）。

闸门核对（预注册）：

- 退火窗口最大回撤 < 20pp：**未过**（-34pp）。如实说明：回撤是压缩的率失真
  代价而非训练失败——对比 v3.4 同型冲击（0.808→0.146=-66pp 瞬时且永不恢复），
  退火把悬崖摊成了可恢复的谷。**维持标记，待裁定**；
- e2 终点 ≥ e1 终点 - 2pp：**未过但口径失真**——e1 只压到 0.812 latent，
  e2 压到 0.257（3.2 倍预算差），同预算比较不成立。

归因结论：

1. 独立 decoder 使系统对 emit 冲击天然稳健（e1 全程无坍缩）；
2. 阈值控制器**学不会压缩**（emit 概率漂移不定，终点 0.73-0.81）；
3. 分位数退火是目前唯一既平滑又真正到达目标预算的机制，应成为 emit 引入的
   标准方式；其代价谷形态（位置、深度、恢复时长）是 scaling 前值得记录的
   率失真曲线族。

## 3. L1：冻结 L0 训练 fresh backbone（3K）

配置：L0 s042 checkpoint + `--reset-backbone` + `training_scope=backbone_only`。

| | L0 联合 backbone（参照） | L1 fresh backbone 3K |
|---|---:|---:|
| 补全 | 0.2499 | 0.2494 |
| PPL | 18.18 | 18.22 |
| 保持 | 0.9909 | 0.9909 |

结论：接口质量**完全转移**——fresh backbone 3K 即达到联合训练 backbone 的
同等水平。latent 接口不依赖特定 backbone 的共适应记忆，L1 判定通过。
锚点已生成（`anchors_16batch/cbiu_v0.json`，rich/null 全部分离）。

## 4. L2/L3：离线效用数据集与预算分配器

数据集：30,304 条动作记录（8 mask draws × 32 batches × 15 slots），冻结
L0+L1 主体，每条含 chunk 内容特征、on/off 三风险 BPB、锚点归一化效用。
工具：`tools/analysis/v3_5/build_l2_offline_utility_dataset.py`。

分配器评估（样本不相交切分，train 22,735 / test 7,569）：

| 模型 | Spearman | AUC | ECE | 同号率 | top-quartile |
|---|---:|---:|---:|---:|---:|
| 类均值查找（slot×内容桶×位置桶） | **0.439** | 0.625 | 0.232 | 0.602 | 0.499 |
| 实例级 MLP-64（9 特征） | 0.095 | 0.530 | 0.233 | 0.538 | 0.330 |

MLP 训练集 MSE≈1.0（解释方差≈0）：**实例级效用映射在当前特征下不可学**；
类级均值携带真实信号。部署仿真（组内贪心预算 vs oracle）：平均 regret 0.073。

闸门核对（v3.5 §5.4，工程口径）：类级 Spearman≥0.40（类均值查找 0.439，
接近但评估口径为实例级）、AUC≥0.70（0.625 未过）、ECE≤0.10（0.232 未过）。
**L3 未过闸**。

归因结论：动作效用真实存在但**粗粒度**——它是动作类别（slot×内容类型）的
属性，不是逐实例属性。这与 CBIU 多种子（实例级校准 0.55 封顶）互为印证。
L3 的正确形态：类级查找/聚类头 + 实例残差可选；下一轮改进方向是更多
mask draws 降噪、更强 body（独立 decoder 后的干净标签）、更丰富的类级特征。

## 5. 总归因链（v3.4 病态 → v3.5 四层验证）

```text
共享逆联合训练
  → 顺序通道被位置不变捷径抹掉（order 0.039）
  → 重建天花板 21%
  → emit/边界任何冲击都触发坍缩
  → CBIU 标签噪声地板过高（实例级校准 0.55 封顶）
换独立 decoder（其余不变）
  → codec 饱和（0.993）、顺序恢复（0.70）
  → emit 阶跃不再坍缩（但不压缩）
  → 分位数退火成为首个平滑压缩机制
  → 离线标签显现类级结构（类均值 0.439 vs 实例 MLP 0.095）
```

## 6. Scaling 前剩余问题（按优先级）

1. **两处预注册闸门未达字面的裁定**（order 0.70、退火回撤 -34pp）：
   建议接受归因解释并按新标定继续，但必须显式记录裁定。
2. **L3 类级分配器迭代**：更多 draws、类级特征扩充、聚类头替换 MLP；
   过闸后才允许动态边界/预算进入 scaling 配置。
3. **长上下文与真实速度**：2048/4096 上的 FLOPs、外推、encoder 占比
   （v3.4 为 79%）——scaling 前必测。
4. **memory 复测（可选）**：截断 backbone 可见 readout 的协议，给有序
   memory 最后一次举证机会；不进 scaling 默认配置。
5. **论文级证据补跑**：L0/L1 关键结论需要 3-5 seeds + CI 才能写入投稿 claim。

## 7. 复现入口

```powershell
# L0
conda run -n soulvlm python tools/launcher/v3_4/run_v34_pos_ar_matrix.py `
  --matrix configs/v3_5/v35_l0_codec_20k.json `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --out-root L:\FLUED_archive\v35_l0_codec_20k_20260717
# emit 退火对照
conda run -n soulvlm python tools/launcher/v3_4/run_v34_pos_ar_matrix.py `
  --matrix configs/v3_5/v35_emit_anneal_20k.json `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --out-root L:\FLUED_archive\v35_emit_anneal_20k_20260717
# L1
conda run -n soulvlm python tools/train/v3_4/train_v34_pos_ar_probe.py `
  --config <l0 run>\resolved_input.json --init-checkpoint <l0>\latest.pt `
  --reset-backbone --training-scope backbone_only --max-steps 3000 `
  --out-dir <out>
# L2 数据集与 L3 评估
conda run -n soulvlm python tools/analysis/v3_5/build_l2_offline_utility_dataset.py `
  --checkpoint <l1>\latest.pt --anchor-file <anchors>\cbiu_v0.json `
  --out-dir <out> --batches 32 --mask-draws 8
conda run -n soulvlm python tools/analysis/v3_5/train_l3_budget_allocator.py `
  --dataset <out>\l2_offline_utility_dataset.jsonl --out <report>.json
```
