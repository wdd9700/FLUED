# FLUED v3.4 交接复核后续实验报告（decoder 解耦 / CBIU 多种子 / emit 容量课程）

> 日期：2026-07-17（晚，交接复核同日）
> 状态：三组实验全部完成
> 归档：
> - `L:\FLUED_archive\v34_decoder_decoupling_20k_20260717`
> - `L:\FLUED_archive\v34_cbiu_multiseed_20260717`
> - `L:\FLUED_archive\v34_emit_capacity_curriculum_10k_20260717`
> - 曲线：`各归档内 followup_experiments_20260717_curves.png`
> 结论级别：单种子（实验 A/C）与 3×3 多种子（实验 B）的 38M/512-byte 探针结果；用于决定下一步架构路线，不能外推为规模结论。

## 0. 实验动机

2026-07-17 交接全量复核确立了三处 handoff 口径修订（lookup 默认值、decoder blocker 强度、
两阶段故障机制），并据此排定后续顺序：

1. decoder 解耦：共享近似逆是当前最大 blocker（07-15：同预算 21.00%/43.37 vs 独立 40.92%/30.83），
   07-16 归因矩阵建议预热 / 交替更新 / 梯度缩放。若廉价解耦有效，CBIU 多种子可以继续建立在
   共享逆路径上；
2. CBIU MLP-64 多种子校准（准入线：Spearman≥0.30、AUC≥0.65、ECE≤0.10、同号率≥0.65）；
3. 两阶段故障修复：07-16 建议把课程拆为 codec 对齐 / emit 容量 / 边界前向 / 边界梯度四条，
   先验证 emit 容量 warmup 能否阻止 2.5K-4K 坍缩。

## 1. 实验 A：decoder 解耦矩阵（5 臂 × 20K）

### 1.1 协议

- 基座：`configs/v3_4/v34_default_38m_20k.json`，覆盖 `use_memory=false`、
  `use_structured_lookup=false`（采用 07-15 修正版 plain 默认）、diag 编码率、hard-ST emit、
  边界课程 0-3K uniform → 3K-5K 过渡 → confidence_threshold；
- 配置：`configs/v3_4/v34_decoder_decoupling_20k.json`；
- 新增训练旋钮（`tools/train/v3_4/train_v34_pos_ar_probe.py`）：
  `--decoder-warmup-steps`（前置纯重建步数）、`--decoder-alternating-period`
  （重建/补全按周期交替）、`--decoder-loss-scale`（重建损失缩放，近似共享权重上的梯度缩放）；
- 单种子 42，20K 步，512 byte，batch 8，RTX 5080，约 41 分钟/臂。

### 1.2 结果（20K 终点固定评估）

| 臂 | 重建 | 补全 | PPL | 保持 | actual latent/byte |
|---|---:|---:|---:|---:|---:|
| d0 共享逆 baseline | 0.2151 | 0.1231 | 41.76 | 0.1526 | 0.169 |
| **d1 独立 decoder** | **0.5490** | **0.1473** | **29.98** | **0.4514** | 0.506 |
| d2 预热 2K | 0.2277 | 0.1201 | 42.70 | 0.1784 | 0.095 |
| d3 交替 500 | 0.2546 | 0.1122 | 43.77 | 0.1925 | 0.186 |
| d4 梯度缩放 0.3 | 0.2176 | 0.1159 | 41.76 | 0.1690 | 0.298 |

d0 复现 07-15 参考点（21.0%/43.4 → 21.5%/41.8），协议可信。

### 1.3 结论

1. **三种廉价解耦均不能闭合共享逆与独立 decoder 的差距**：重建最高仅 +3.9pp（交替），
   且以补全/PPL 恶化为代价；预热把 latent 压到 0.095 反而加剧容量饥饿；梯度缩放只增加
   latent 用量不改善质量。
2. 联合训练非平稳不是调度层面能修复的问题。共享逆路线若保留，需要结构性方案
   （例如独立参数 + 共享初始化/正则、或交替阶段式全量训练），而不是 loss 层面的微调。
3. 对 CBIU 的含义：在线 CBIU 当前绑定 `decoder_mode=shared_inverse`。CBIU 动作标签的
   绝对水平受共享逆质量限制，但 CBIU vs legacy 的相对比较在同一 decoder 路径下成立
   （实验 B 即在此前提下解读）。

## 2. 实验 B：CBIU MLP-64 多种子校准（3 train seeds × 3 mask seeds）

### 2.1 协议

- 冻结主体：m2 20K 检查点（`v34_attribution_matrices_20260716`），冻结 CBIU 锚点
  （`v34_cbiu_v0_20260716/m2_16batch_fixed_emit/cbiu_v0.json`），锚点与检查点绑定校验开启；
- emit-only 3K，MLP-64 控制器（无 slot embedding，R3 候选），CBIU dual（预算 0.18）或
  legacy target；train seeds {42, 123, 999}；
- 配置：`configs/v3_4/v34_cbiu_multiseed_emit_only_3k.json`；
- 动作校准：`probe_v34_cbiu_action_calibration.py`（新增 `--eval-mask-seed` 参数），
  每个控制器 × mask seeds {1042, 2043, 3044}，16 eval 批次，约 1900 个严格动作/次。

### 2.2 接口指标（3K 终点，6 臂汇总）

| 模式 | 重建 | 补全 | PPL | actual latent/byte |
|---|---:|---:|---:|---:|
| CBIU dual s42 | 0.2321 | 0.1198 | 43.90 | 0.1438 |
| CBIU dual s123 | 0.2340 | 0.1267 | 43.34 | 0.1648 |
| CBIU dual s999 | 0.2211 | 0.1038 | 44.68 | 0.1453 |
| legacy s42 | 0.2134 | 0.1178 | 43.92 | 0.1869 |
| legacy s123 | 0.2159 | 0.1290 | 43.25 | 0.1981 |
| legacy s999 | 0.2066 | 0.1043 | 44.72 | 0.1993 |

CBIU dual 在全部三个种子上以约 23% 更少的 actual latent 取得更高重建，与 R1/R3 单种子
方向一致，多种子下稳定。

### 2.3 动作校准（18 次，每次约 1900 动作）

| 模式 | Spearman | AUC | Brier | ECE | 同号率 |
|---|---:|---:|---:|---:|---:|
| CBIU dual（9 次范围） | 0.15-0.23 | 0.53-0.57 | 0.24 | 0.15-0.18 | 0.57-0.61 |
| CBIU dual（均值） | **0.184** | **0.546** | 0.241 | 0.166 | 0.594 |
| legacy（9 次范围） | 0.10-0.17 | 0.46-0.52 | 0.25 | 0.15-0.19 | 0.49-0.53 |
| legacy（均值） | 0.137 | 0.497 | 0.249 | 0.176 | 0.512 |

准入线：Spearman≥0.30、AUC≥0.65、ECE≤0.10、同号率≥0.65。

### 2.4 结论

1. **未通过准入线**：18/18 个 cell 无一达到任一阈值，最好 cell 为 0.23/0.57/0.15/0.61。
   CBIU 不得接管 dynamic boundary；memory/AR 统一监督更不成立。
2. **CBIU 优于 legacy 是多种子稳健结论**：9/9 个配对 cell 上 Spearman、AUC、同号率全部
   更高；legacy+MLP-64 的 AUC 均值 0.497 接近随机。动作效用信号弱但真实，且不是控制器
   容量能继续挤出的（R3 已试线性→MLP-64→slot）。
3. 按 R3 既定决策：下一步应改**动作特征/标签方差估计**（例如 chunk 内容特征、跨步
   效用平滑、动作聚类后监督），而不是继续扩大控制器。
4. 局限：本实验的 CBIU 标签经由共享逆 decoder 路径计算（实验 A 已确认该路径显著弱于
   独立 decoder），标签绝对噪声水平可能因此被抬高；在 decoder 路线决策后应复测校准。

## 3. 实验 C：emit 容量课程（3 臂 × 10K）

### 3.1 协议

- 与实验 A 同基座（no-memory、plain、diag），max_steps=10K；
- 新增 `--emit-warmup-steps`：前 N 步绕过 hard emit（全部 readout 真实进入主干，
  actual latent/byte≈1.0），第 N 步起控制器接管；
- f0 baseline（emit 从 0 步起启用）；f1 warmup 3K（与边界切换同步接管）；f2 warmup 5K
  （边界过渡完成后接管）；
- 配置：`configs/v3_4/v34_emit_capacity_curriculum_10k.json`。

### 3.2 轨迹关键点（训练日志平滑前原值）

| 臂 | 2.5K | 3K | 4K | 5K | 6K | 10K 终点评估（latent） |
|---|---:|---:|---:|---:|---:|---|
| f0 baseline | 0.744 | 0.846 | 0.732 | **0.218** | 0.317 | 0.2877 / PPL 44.25（0.361） |
| f1 warmup 3K | 0.727 | 0.518 | 0.543 | **0.147** | 0.207 | 0.2332 / PPL 44.16（0.137） |
| f2 warmup 5K | 0.727 | 0.761 | **0.808** | **0.146** | 0.227 | 0.2558 / PPL 44.00（0.202） |

### 3.3 结论

1. **坍缩的直接原因是 hard emit 的容量阶跃，而不是边界切换本身**：f2 在 uniform+全容量
   下一路升至 0.808（补全 0.212，为所有臂历史峰值），emit 接管瞬间跌至 0.146。
   warmup 没有消除坍缩，只是把坍缩推迟到接管时刻。
2. **边界切换与 emit 接管叠加是最差安排**（f1，3K 双重冲击，5K 时 0.147 为全场最低）。
3. 阶跃式容量课程被证伪。07-16 的"emit 容量课程"必须实现为**连续预算退火**
   （例如对偶变量/预算从 1.0 在数千步内渐减至目标，或 hard 阈值随课程移动），
   配合边界梯度独立调度；该连续退火旋钮尚未实现。
4. 10K 终点三者都在坍缩后恢复期，f0 终点最高但 latent 用量是 f1 的 2.6 倍；
   本矩阵用于机制判定，不作为终点优劣排名。

## 4. 综合决策含义（更新到 handoff 决策链）

1. **decoder 路线成为最高优先级结构决策**。廉价解耦已证伪；候选方向：
   (a) 接受独立 decoder（07-15 同预算赢家）并解决其参数量/逆一致性；
   (b) 共享初始化 + 独立参数 + 周期对齐蒸馏；
   (c) 分阶段全量交替训练。300M scaling 维持冻结直至该决策落地。
2. **CBIU 多种子未过准入线**，boundary 接管无限期搁置；下一轮改动作特征与标签
   方差估计。若 decoder 路线改变，校准需在 fresh 检查点上复测。
3. **emit 容量必须连续退火**。实现 `emit_budget_curriculum`（对偶/预算连续衰减）后，
   在 f2 型 uniform 全容量预热之上做 3K-8K 渐减，再评估是否消除坍缩。
4. P0 决策链更新为：decoder 路线决策 → emit 连续退火 → CBIU 新特征/标签多种子复测 →
   memory 因果拆分 → 长上下文与真实速度 → scaling。

## 5. 复现入口

```powershell
# 实验 A
conda run -n soulvlm python tools/launcher/v3_4/run_v34_pos_ar_matrix.py `
  --matrix configs/v3_4/v34_decoder_decoupling_20k.json `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --out-root L:\FLUED_archive\v34_decoder_decoupling_20k_20260717

# 实验 B（训练）与校准
conda run -n soulvlm python tools/launcher/v3_4/run_v34_pos_ar_matrix.py `
  --matrix configs/v3_4/v34_cbiu_multiseed_emit_only_3k.json `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --out-root L:\FLUED_archive\v34_cbiu_multiseed_20260717
conda run -n soulvlm python tools/analysis/v3_4/probe_v34_cbiu_action_calibration.py `
  --checkpoint <run>\latest.pt `
  --anchor-file L:\FLUED_archive\v34_cbiu_v0_20260716\m2_16batch_fixed_emit\cbiu_v0.json `
  --out-dir <out> --eval-mask-seed 1042 --max-eval-batches 16

# 实验 C
conda run -n soulvlm python tools/launcher/v3_4/run_v34_pos_ar_matrix.py `
  --matrix configs/v3_4/v34_emit_capacity_curriculum_10k.json `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --out-root L:\FLUED_archive\v34_emit_capacity_curriculum_10k_20260717
```
