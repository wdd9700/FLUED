# FLUED 当前状态（单一事实源）

> **本文是项目当前默认值、证据状态与待裁定事项的唯一权威入口。**
> 创建：2026-07-17。维护规则见第 1 节。历史版本文档（含 handoff）只记录各自
> 日期的状态；凡冲突，以本文为准。

## 1. 维护规则（先读）

1. 任何新证据落地时，**必须原地更新本文**（默认值、证据状态、闸门状态），
   并在第 6 节 changelog 追加一行带日期的记录；
2. 任何实验的默认起点是 canonical 配置与本文证据状态表——v3.6 线为
   `configs/canonical_v36.json`（当前默认起点），v3.4 收尾系列旧口径
   `configs/canonical_v35.json` 保留有效；
   **不再以版本号或文档创建日期推断当前状态**；
3. `tests/test_canonical_sync.py` 持续校验：训练脚本 CLI 默认值 ==
   canonical 配置 == 本文第 2 节关键值（v35 与 v36 两条线均受守卫）。
   三者漂移时测试红灯；
4. 证据状态只允许四种：`已验证` / `候选` / `已证伪` / `待举证`，
   每条必须挂证据链接与日期；升级降级都要留痕。

## 2. 当前 canonical 默认配置

机器可读源：`configs/canonical_v35.json`（canonical_version: `v35.1-20260717`，
v3.4 收尾系列旧口径，保留有效）与 `configs/canonical_v36.json`
（canonical_version: `v36.1-20260731`，**v3.6 线当前默认起点**：S0 预训
encoder+segmentor 冻结接管、动态边界、4× KDA 状态、k=1 readout 包、
mask 原生两任务）。v3.4 证据基座：v3.5 L0 两 seed 验证（E1）；
v3.6 证据基座：S0 + A/B 对照（E17/E18）。

| 项 | 当前默认 | 证据 |
|---|---|---|
| byte lookup | **plain**（structured 已证伪降级） | E5（07-15 同预算反超） |
| decoder | **独立 decoder**（共享逆已判死） | E1/E4 |
| 边界 | uniform 16B 固定（动态边界待 L3 过闸） | E1 |
| emit 控制器 | **关闭**（实验分支） | E1/E6 |
| emit 引入方式 | 分位数预算退火（若启用） | E6 |
| memory | **关闭**（512B 举证失败；复测条件见 Q5） | E7 |
| 位置/AR | RoPE/prompt ALiBi + 小 AR（成组） | E1/E8 |
| 协议 | strict masked 5%，span 1-8，readout 粒度 | E1 |
| 编码率信号 | diag（训练信号；uniform 下不驱动执行） | E5 |
| 临时 backbone | 384h/3L/4.9M，仅评估用 | E1 |
| 规模 | 38M / 512 byte / batch 8 / lr 2e-4 | E1 |

## 3. 证据状态注册表

| ID | claim | 状态 | 证据（最近） |
|---|---|---|---|
| E1 | 独立 decoder codec 可饱和（重建 0.993，两 seed） | 已验证 | v3.5 L0（2026-07-17） |
| E2 | 共享逆 decoder 同预算大幅落后且不可廉价解耦 | 已验证 | 07-15 报告 + followup 实验 A |
| E3 | 共享逆联合训练抹除 chunk 内顺序通道（0.039 vs 1.17） | 已验证 | followup 实验 A order 探针 |
| E4 | 独立 decoder 恢复顺序通道（0.70-1.17） | 已验证 | 同上 + L0 |
| E5 | plain lookup 20K 同预算优于 structured | 已验证 | 07-15 迁移后报告 |
| E6 | 分位数预算退火是首个平滑压缩机制（0.257 latent 恢复 0.958+） | 已验证（单 seed） | v3.5 e2 |
| E7 | memory（含位置变体、v3.3 串行/并行）在 512-4096B 全可见下无可测增益；v3.3 串行 gate 自学到关闭（gate_mean=0.0000）；memory 还带来 11-26% 吞吐税 | 已验证（单 seed，该条件范围） | memory 长上下文消融 14 臂（2026-07-24） |
| E8 | RoPE 必要；小 AR 单独不能替代位置 | 已验证 | 四组消融 + CBIU V0 |
| E9 | CBIU 动作效用弱而真实、多种子稳定优于 legacy | 已验证 | CBIU 多种子 3×3 |
| E10 | 效用是类级信号，27 参数可加因子模型即最优 L3 形态 | 已验证（单 seed 工程口径） | v3.5 L3 勘误 |
| E11 | CBIU/类级效用达到 boundary 接管准入线 | 已证伪（未过线） | CBIU 多种子 + L3 |
| E12 | emit 阶跃 warmup 可防坍缩 | 已证伪 | followup 实验 C |
| E13 | memory 在截断 backbone 可见性下有增益 | 建议弃用（4096 全可见仍零增益+串行自关门，举证基础消失；待用户裁定） | memory 长上下文消融（2026-07-24） |
| E14 | FLUED latent 接口降低小 backbone 补全难度 | 已验证（小模型单任务） | v3.2.1（2026-07-03） |
| E15 | v3.5 L0 接口质量可转移到 fresh backbone | 已验证（单 seed） | v3.5 L1 |
| E16 | 跨 chunk 语境特征对动作效用零增量（语境顾问假说窗内判死；效用是纯局部类级属性） | 已验证（单 seed 工程口径） | 语境特征增补测试（2026-07-24） |
| E17 | 教师标注管线可行（4B 教师+严格校验：5K→3426 合格→过滤后 2532 条，中位 27B；segmentor S0 预训 F1 0.886 模仿教师、高置信刀与用户重合 86%） | 已验证（单 seed 工程口径） | S0 管线（2026-07-27/30） |
| E18 | v3.6 组件预训路线优于端到端（A: 0.189/34.2 vs B: 0.131/35.9，+5.8pp 且 B 过 12K 退化；端到端桥装死全程）；4× 容量+S0 边界把 k=1 平台从 0.11 抬到 0.19-0.20 但未击穿 | 已验证（单 seed） | s0_vs_e2e 对照（2026-07-31） |
| E19 | 增益归因：S0 动态边界 +4.4pp（全部活性成分），4× 容量单独 +0，k∈{1,4,16} 无差异（~0.19）；K4 首发 NaN 发散但同 seed 重跑干净（bf16 瞬时不稳定，列为稳定性工作项） | 已验证（单 seed） | 归因矩阵（2026-08-01） |
| E20 | 公平对比（masked infilling 同口径）：v3.6 masked acc 0.149 ≈ 瓶颈 HNet-DiT 0.142，但信息传输 1,536 vs ~97,000 标量（~60× 效率）；HNet-DiT 两臂边界均退化（标准臂层级溶解 1 chunk、瓶颈臂切到 2.5B/段）——无显式压缩激励时动态切分无中间态 | 已验证（单 seed） | hnet_dit_fair（2026-08-02） |

## 4. 闸门注册表

| 闸门 | 预注册阈值 | 结果 | 裁定 |
|---|---|---|---|
| L0 重建 | ≥0.48（2 seeds） | 0.993/0.992 | 过 |
| L0 order 探针 | ≥1.0 | 0.695/0.702 | **未过-归因待裁定**（阈值标定不当；0.70 已证顺序注册，Q1） |
| L0 线性探针 | 记录项 | 98.5% vs 89.9%/94.7% | 参考 |
| 退火窗口回撤 | <20pp | -34pp 后完全恢复 | **未过-归因待裁定**（RD 代价谷 vs 训练失败，Q1） |
| CBIU 准入线 | Spearman≥0.30/AUC≥0.65/ECE≤0.10/同号≥0.65 | 0.184/0.546/0.166/0.594 | 未过，boundary 接管搁置 |
| L3 类级校准 | Spearman≥0.40/AUC≥0.70/ECE≤0.10 | 0.433/0.624/- | 未过（AUC） |
| L3.5 闭环 rollout | 长程不恶化+最坏序列有界 | 未执行 | 待 L3 过闸后执行 |

## 5. 待裁定队列（当前优先级排序）

| # | 事项 | 状态 |
|---|---|---|
| Q1 | 两处闸门字面未过的归因裁定（order 0.70；退火 -34pp） | **等待用户裁定** |
| Q2 | e2 退火排名分数改用类级因子效用（消除 legacy 分数混杂） | 待跑（~40 min） |
| Q3 | L3 因子模型（27 参数）正式过闸 + 多 draws 降噪 | 待跑 |
| Q4 | L1 在 0.257 预算下重测 fresh backbone（当前 1.0 偏松） | 待跑 |
| Q5 | memory 长上下文消融（v3.3 串行/并行 + v3.4 并行×2 变体 × 512/2048/4096，14 臂 20K） | **已完成（2026-07-24）：全线零增益，建议关闭 memory 线，待用户裁定** |
| Q6 | 2048/4096 长上下文 + encoder 速度（scaling 前必测）。可行性已实测（2026-07-23）：2048=max_chunks 256/stride 1024/batch 4（峰值 14.3GB，~3.6 step/s，20K≈93min）；4096=max_chunks 1024/stride 2048/batch 1（峰值 12.2GB，~2.3 step/s，20K≈2.4h）；硬闸门 cut_capacity_overflow=0 且 truncated_tokens=0（实测全 0）。注：`expandable_segments` 在本 torch 构建上不支持（启动警告），无效；有效手段是 batch 压到峰值 <15.9GB 分配器抖动红线以下 | 配置标定完成；memory 消融执行中 |
| Q7 | 300M L0 codec（无条件安全的纯 codec 层）启动时机 | 待 Q1 裁定 |

## 6. Changelog

- 2026-07-17：本文创建。canonical 定为 v35.1（独立 decoder/uniform/plain/无 emit/无 memory）；
  旧 v3.4 canonical（`v34_default_38m_20k.json`）降级为历史实验基座；证据表 E1-E15、
  闸门表与待裁定队列 Q1-Q7 建立。
- 2026-07-18：术语注册表 `docs/TERMS.md` 建立（handoff §19 移植 13 条 + v3.4/v3.5 全层
  补登记，AI 提名默认候选待用户转正）；AGENTS.md 新增两条长期法则（术语原地登记、
  未注册词禁用）。术语口径此后以 TERMS.md 为准。
- 2026-07-23：长上下文可行性实测（Q6 batch 标定，数字见 Q6 行）；代码核实两个审计暗礁：
  max_chunks 默认实为 40（非 64），uniform 边界容量不足时被 clamp 成更少更长 chunk
  （静默拉伸，`flued/v34/model.py:790-795`，不报错不截断）；stride 采用半窗相对重叠率
  （2048/1024、4096/2048）。TERMS.md 新增 3 个候选术语（静默拉伸/容量溢出守卫/分配器抖动）。
  更正：`expandable_segments` 在本 torch 构建不支持，batch 标定的有效变量是 batch 大小本身。
- 2026-07-23：用户裁定启动 memory 长上下文消融（Q5 并入 Q6）：v3.3（串行 causal_current/
  past_only、并行 parallel_local/bidirectional_no_self、无 memory，d512/h1536≈12.7M——
  v3.3 架构上限 ~19M，无法到 38M，已声明）× 2048/4096 + v3.4（无/并行 nopos/并行
  chunk_rope，canonical 38M）× 512/2048/4096，共 14 臂 20K 步，归档
  `L:\FLUED_archive\v35_memory_longctx_20260723`。v3.4 512 无 memory 基线复用 L0 s042。
- 2026-07-24：memory 长上下文消融 14 臂全部完成（18h，守卫信号全 0，226 文件 manifest）。
  结果：所有变体 × 所有长度的补全差值均在 ±0.3pp/±0.05 PPL 内（512 的 +0.9pp 在 L0
  两 seed 噪声带内）；干预探针 mem_swap≈0.002-0.006（顺序零注册）而 mem_subst 0.2-0.7
  （内容被读但无下游收益）；v3.3 串行 gate 自学关闭（0.0000）；memory 吞吐税 11-26%。
  E7 范围扩至 4096B，E13 建议弃用，Q5 建议关闭 memory 线（均待用户裁定）。
  曲线：`v35_memory_longctx_effect_curves.png`。
- 2026-07-24：语境特征增补测试完成（用户判一/判二归因的窗内判决）：L2 数据集补跨 chunk
  语境特征后因子模型 Spearman 0.4289 vs 基线 0.4324（Δ-0.0035）、regret 不变——语境顾问
  假说窗内判死，效用为纯局部类级属性（E16 新增）；8.1 叙事与 L3 局部因子形态获反向加固。
  归档 `L:\FLUED_archive\v35_l2_context_features_20260724`；新脚本
  `tools/analysis/v3_5/dump_l2_context_features.py`、`eval_l3_context_augmented.py`。
- 2026-07-25：**v3.6（KDA 世代）规格落定**：`docs/versions/v3.6/FLUED_V3_6_SPEC_20260725_CN.md`。
  关键裁决：整条 prompt 恰好 1 个 readout 包（端点 A，k 扫档 {1,4,16,32}）；无 stride/无跨窗/
  无 TBTT（样本=全上下文）；segmentor 细长型、summarizer 为 projector、decoder 为 encoder 的
  1/3~1/5；Occam 基线=KDA-LM 混合 3:1；率失真前沿为比较口径；AR vs DiT 主干列入 R0.5；
  网页版 K3 的"双通道"提案被用户否决（存档）。语料线交 Codex；正式训练待语料确认。
  五步预备程序启动：落文档（✔）→ KDA 硬件冒烟 → 架构实现 → Triton 优化 → pipeline+评测。
  TERMS.md 新增 v3.6 一节（KDA/状态机/readout 包/相对基线/文档级任务等，双通道记为弃用）。
- 2026-07-25：v3.6 v0 代码落地（`flued/v36/` + `tools/train/v3_6/train_v36.py` + 7 项测试，
  128 全量绿，总参 39M 符合尺寸链）。FLA/triton-windows 冒烟通过（稳态 ~2ms/call）。
  实现修复两则：padding NaN 传染（0×NaN）、bf16 写入状态爆炸（key L2 归一化）。
  512B/20K 可学习性探针（单 readout 包）运行中，结果回填本节。
- 2026-07-27：可学习性探针完成（归档 `L:\FLUED_archive\v36_learnability_probe_20k_20260725`）：
  20K 步 recon 0.117 / 补全 0.114 / PPL 44.4 / 保持 0.117，轨迹在 0.09-0.14 平台波动
  （8K/12K 有 0.14-0.18 尖点但不持续），三任务近同值——边缘分布学习为主，内容传输微弱；
  状态范数 6-11 健康但训练中段有上行趋势。**单 seed 负向早期信号，非终审**：k=1 点的
  容量/优化归因需 k∈{4,32} 对照臂区分（R1 前沿扫档的前置）。07-25 晚学院断电时无任务
  在跑（探针已于 10:30 完成），机器 07-27 11:45 重启，无损失。
- 2026-07-27：v0.1 落地（动态边界端到端+软边界桥、mask 原生两任务、CBIU 对象定为 β、
  readout×4 变体可配置）；冒烟发现边界装死（切分率 0.2% 且下降）与主干零贡献。
  用户裁决新路线：**S0 段落标签独立训练 segmentor → S0.5 GRPO-CBIU 边界微调**
  （NLA 迁移，规格附录 A）；NLA/GRPO/S0 术语已登记。
- 2026-07-31：S0 全线完成：用户手标 16 条（中位 21B 粒度）→ 4B 教师批量标注
  （5K→3426 合格，68.5%）→ subagent 质检（合规 62.5%/明显错误 20%）→ 过滤 2532 条
  （中位 27B）→ S0 SFT（F1 0.886 vs 教师、高置信刀与用户重合 86%、阈值扫描证明
  粒度差距是排序问题非操作点）。A/B 对照（E18）：组件预训 >> 端到端。
  **新默认 `configs/canonical_v36.json`（v36.1）建立，v35 旧口径保留**；
  9B 思考模型教师路线证伪（预算全烧思考）；教师粒度天花板 27B 交 S0.5 RL 裁决。
  进行中：干净归因矩阵（B0/B1/K4/K16，统一 S0 encoder 冻结，隔离边界/容量/k）。
- 2026-08-01/02：归因矩阵完成（E19）——增益全在 S0 边界；K4 发散重跑判为 bf16 瞬时
  不稳定。Kimi K3 开源（2.8T）+ 官方 FlashKDA 检索：推理内核（无反向/SM90+/CUDA12.9/
  K=V=128），训练留在 FLA Triton。专用环境 `kda-kernels` 建成（torch 2.13+cu130：
  fla KDA 训练、mamba_ssm Mamba-2 训练、FlashKDA 推理全可用；subagent 修复了 FlashKDA
  在 MSVC 下 CUtensorMap 64B 对齐丢失的真实内核 bug，建议反馈上游）。H-Net 复现
  （44.5M，BPB 0.653 天花板锚）+ HNet-DiT 公平对比（E20）：masked acc 打平、
  信息效率 ~60×、动态切分无激励必退化。**当前 RD 前沿点：k1/k4/k16 ≈ 0.19 + HNet
  两个参照（天花板 0.324@无压缩 / 瓶颈 0.142@97K 标量）**。
- 2026-08-02：初始化审计修复落地（用户下令 1-4 必修）：① canonical_v36 五值
  （d_pack 1536/kda_head_k 128/kda_head_v 256/max_span 64/tau_cut 0.94）与
  `V36Config`、`train_v36.py` CLI 默认值三方对齐——此前仅显式 `--config` 才生效，
  叠加 shape 兼容加载只报 skipped 数，存在静默半加载风险；② `test_canonical_sync.py`
  扩至 v36 线（CLI/V36Config/本文三源守卫，v35 守卫保留）；③ v36 死代码清理
  （从未调用的 `self.policy`、重复定义的 `WriteHead.to_beta`、重复 `MASK_ID`），
  `flued/__init__.py` 补 v36/hnet_repro 导出；④ 文档索引层同步 v3.6 口径
  （docs/README、根 README 时间线、configs/tools/results README、TERMS §3
  canonical 条目、本文 §1）；⑤ readout 包均值瓶颈登记（v3.6 规格 §13 备注 +
  TERMS 候选术语），S0.5 GRPO 奖励设计须知。
