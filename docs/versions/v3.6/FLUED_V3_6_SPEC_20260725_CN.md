# FLUED v3.6 架构规格（KDA 世代）

> 日期：2026-07-25。状态：全部关键决策已由用户裁决锁定，进入实现阶段。
> 命名约定：**v3.6 = 本文的 KDA 架构世代**；仓库中既有 "v3.5" 文档/配置属于
> v3.4 收尾系列（分级冻结 L0-L5、离线效用数据集、因子模型），继续有效，两者并存。
> 底稿：用户《FLUED v3.5 架构交接文档》（KDA 版）+ 2026-07-23 至 07-25 的逐条裁决。
> 被否决提案存档：网页版 K3 的"双通道"修正（内容 readout 与全局 readout 分流）——
> 用户明确否决，决策权归用户，本文不采用。

## 1. 一句话

**整条 prompt 压缩成恰好 1 个 readout 包**：切 chunk → summarizer 逐段产 memory
（每 chunk 一个）→ KDA 状态机只读 memory 串行更新 → 从终态读出 1 个 readout 包
交给主干。要证明的不是"记忆有用"，而是：**语义预处理值得成为一个独立的、
并行的、可拆分的层**；新颖性锚点是 KDA 的 channel-wise（逐通道）更新粒度 ×
极端压缩率的组合（单纯极端压缩或单纯状态空间模型均有先例，组合无已知先例）。

## 2. 架构

```text
字节流（512-4096B 单样本全上下文，无窗口、无 stride、无跨窗状态）
  → Byte encoder（全并行，普通 byte lookup + RoPE + prompt 级双向 ALiBi）
  → Segmentor（细长型 DiT，one-shot：参数不变、加深、O(L²) 但常数小）
  → chunk 序列（边界 = 写入/读出时刻表，率控制 100% 归边界，emit 已删除）
  → Summarizer（projector 定位：层少宽度足，one-shot，FFN 收尾，每 chunk 产 1 个 memory 向量）
  → 写入头（有真实深度：memory → k, v, α, β；α 逐通道 chrono 式多时间尺度显式重参数化）
  → KDA 状态机（新 interpreter，串行，S_i = (I−β_i k_i k_iᵀ)Diag(α_i) S_{i−1} + β_i k_i v_iᵀ，
    状态随样本生灭——没有跨窗、没有 TBTT）
  → Readout 包（每 prompt 恰好 1 个逻辑包：默认 1 个查询读 1 个 token，FFN 再对齐；宽度 k 为预算旋钮）
  → Backbone（临时 4.9M 冻结规格 → 后续翻倍至 100M 档）
  → 独立 span decoder（重建 + strict masked-source 补全；规模为 encoder 的 1/3~1/5，锥形网络依据）
```

结构性质：
- 字节级 O(L²) 只存在于 segmentor/byte encoder（参数小、常数小，4096 单卡实测可跑）；
  KDA 只跑 chunk 序列（32-256 步，O(n)），与字节长度脱钩——这是"没有 stride"的依据；
- 内容到达主干的唯一路径：chunk bytes → summarizer → 写入 → 状态 → readout。
  自写自读是设计内通道（v3.4 的 no-self 掩码工程由因果递推免费强制执行）；
- 下游无跨 readout 注意力可借（默认 k=1 时主干只看 1 个 token）——v3.4 时代
  "下游语境推断截胡"的冗余路径被结构拆除，需求结构焊死。

## 3. 尺寸链（默认值，可调需记录）

| 环节 | 值 | 依据 |
|---|---|---|
| byte encoder / segmentor | 384d×9L（≈13M，与原 512d×5L 等参数，深 1.8 倍）；备选 320d×13L | 用户裁决：细长型，给深度决策容量，保持 one-shot |
| summarizer | 2L×1024 宽，每 chunk 并行独立 | 用户裁决：projector/流形假设，层少宽度足 |
| 前端合计 | ~20-25M（encoder 略大） | 用户裁决 |
| KDA 状态 | 4 头×64×128 = 32K 标量（≥readout 包 2.6 倍） | 状态容量必须装得下读出 |
| readout 包 | k × 384 标量，k=1 默认（旗舰点），扫档 {1,4,16,32} | 率=比较参数非约束（用户裁决） |
| 训练用 backbone | 4.9M（384h/3L，冻结沿用，保可比性） | 通用性验证时翻倍至 100M 档（J-Space 论据） |
| decoder | encoder 的 1/3~1/5（≈4-5M） | 锥形神经网络实验依据（用户裁决） |

## 4. 基线与比较口径

- **Occam 基线（最先跑）**：KDA-LM，byte 级端到端 decoder-only，**混合版
  KDA 层 : transformer 层 = 3:1**（月之暗面论文最佳比例，用户裁决），
  参数量 = FLUED 前端 + backbone + decoder 总和；前/主干参数配比至少扫两点
  （防"主干大一点就是好"的平庸结论）。
- **相对基线**：同切分、同 KDA 版 interpreter，但每 chunk 直产 1 个 readout、
  **不读 memory**（无状态通道）。与主架构唯一差别 = 有无串行记忆通道，兼作 R1 消融臂。
- **比较口径**：率 = 传输的 FP16/BF16 标量总数（信息学压缩视角），质量 = BPB
  （字节分母纪律）；比**率失真前沿**不比单点（用户裁决：预算是比较参数不是架构约束）。

## 5. 主干未决项：AR vs DiT（预注册判决项）

端点 A 下主干处理的序列不再是 prompt 内 token，而是上传的状态——类似正常模型的
中间层，且天然面向**会话轮次级序列**（多轮 prompt 各产 1 个 readout 包，主干在其上
维持更大的核心空间，即用户引 Anthropic 拆解的 J-Space 角色）。必须回答：AR 主干
是否仍能工作，还是必须上 DiT 主干做多轮潜空间迭代（ELF/何恺明路线），以及这是否
破坏 decoder 任务。判决实验：同 readout 包、同 codec 任务，AR 主干 vs DiT 主干
（多轮迭代），比较补全/重建与收敛稳定性——列入 R1 前的前置探针。

## 6. 评测协议

- **局部任务**（沿用）：strict masked-source 补全（5%、span 1-8、先 mask 后编码）、
  独立 decoder 重建、保持；
- **文档级任务**（新增，全局通道的需求创造——防第二次 gate_mean=0）：
  跨段实体一致性、长程指代、文档主题保持，定义在单条上下文（≤4096B）内部；
- **语料现实**（subagent 抽查结论）：corpus_v3（23.8GB）与 corpus_v4（224.9GB，
  清洗去重+扩源）均无文档边界标记，需"文档重建"工序；v4 的 wiki/STEM 分片是
  评测集首选。语料线由 Codex 负责，就绪前正式训练不启动。

## 7. 实验矩阵与预注册证伪

| 轮次 | 内容 | 判决 |
|---|---|---|
| R0 | KDA-LM 混合 3:1（含配比扫描）vs v3.6 全栈 | 打平 → v3.6 关闭 |
| R0.5 | AR 主干 vs DiT 主干前置探针（§5） | 定主干形态 |
| R1 | 主架构 vs 相对基线，率失真前沿对前沿（含 k 扫档 {1,4,16,32}），512/2048 | 前沿不超 → 记忆通道终局关闭（已给予需求结构，无申诉权） |
| R2 | 4096、≥3 seeds、KDA-LM/BPE/H-Net 全指标对齐、分段计时 | 成立条件最终裁决 |

证伪条件（开跑前冻结）：
1. KDA-LM 同参打平 v3.6 全栈 → 全线关闭；
2. R1 前沿不超相对基线 → 状态通道关闭；
3. 胜出需质量更高**且**分段计时（front/backbone 分开）更快，只赢质量不赢时间判不成立；
4. 任何结论进文档须 ≥3 seeds + 长度外推（2048→4096）验证。

## 8. 工程纪律（继承 v3.4 血泪清单并追加）

1. KDA kernel 优先用 FLA 库，不自造；**已知风险：当前环境无 triton/FLA，
   Windows + torch 2.12 nightly 兼容性是头号工程风险，正式开工前先冒烟**；
2. bf16 下逐通道衰减注意数值稳定（log 域/分块全精度）；
3. 梯度监控记录 clip 前范数；守卫信号（容量溢出/截断/NaN）为每个 run 强制输出；
4. 检查点与 manifest 纪律沿用；每臂 resolved_input.json 可完整复现；
5. 一次只动一个变量；任何新监督必须是按决策的边际信号（全局标量监督是坟场）；
6. ≥3 seeds 才进结论；文档与 README 声明同步更新。

## 9. 当前待办（用户排定顺序）

1. 本文档（✔）与术语登记（✔）；
2. KDA/FLA 硬件可行性冒烟（✔ 2026-07-25：triton-windows 3.7.1 + flash-linear-attention
   0.5.1 + fla-core 0.5.1 装入 soulvlm，RTX 5080 bf16 前向+反向通过，稳态 ~1.9-2.5ms/call
   @[2,256,512]；注意：首次调用 Triton JIT 编译 ~142s，训练脚本需预热或持久化 kernel cache；
   121 测试与 v3.4 训练冒烟均正常；副作用：transformers 5.14.1 带入 rich 15.0.0 与 openxlab
   版本约束冲突告警，本项目不用 openxlab，记录在案）；
3. 架构实现（✔ 2026-07-25 v0：`flued/v36/model.py`（39M：encoder 6.3M/segmentor 18.6M/
   summarizer 2.6M/写入头 1.3M/状态机 0.35M/backbone 4.2M/decoder 5.8M≈1/5 encoder）、
   `tools/train/v3_6/train_v36.py`、`tests/test_v36_smoke.py`（7 项，128 全量绿）。
   纯 PyTorch KDA 递推起步；两个实现级发现：padding chunk 的 NaN 经 0×NaN 传染
   （掩码 softmax 有限化+门控置零修复）、bf16 写入无界导致状态爆炸（观测 7.5e15 尖峰，
   按 delta-net 惯例对 key 做 L2 归一化后状态范数稳定 1-5）。v0 边界为 uniform，
   segmentor 仅记录无梯度；512B/b8 ≈5.6 step/s（双前向：干净+遮蔽）。
   2K 步仅边缘分布学习（acc≈0.11）；20K 可学习性探针完成，见附录 B）；
4. Triton KDA 效率优化（训练+推理）；
5. pipeline 跑通 + 文档级评测工具 + 实验设计冻结；
6. 语料（Codex 线）确认就绪 → 正式训练。

## 10. 2026-07-27 设计修订（用户裁决汇总）

v0.1 已实现并冒烟（2K）：**边界端到端动态化**（阈值切分+UTF-8 续字节守卫+容量安全切+
软边界桥，无课程）、**任务 mask 原生化两任务**（任务一 readout→decoder 精准还原；
任务二 readout→backbone 改写→decoder 全部 512 字节；共享 decoder 等权；masked-source
内生于任务）、**CBIU 对象定为 β 写入门**（锚点 3K 快照离线生成后上线）、
**readout 容量 ×4 变体**（d_k 128/d_v 256/d_pack 1536，矩阵边长×2）。

冒烟发现：边界装死（切分率 0.2% 且下降，靠 max_span 硬约束兜底——规格预言的"摆烂不切"）；
direct≈backbone（主干零贡献，仪器正常）；4× 容量 2K 处同平台。

后续路线（用户 2026-07-27 裁决）：
- **S0：先独立训练 segmentor**（encoder+segmentor，段落标签稠密 BCE，整行打包数据管线；
  语料行=自然段已抽查证实；summarizer/KDA/decoder 搁置并行调试）；
- **S0.5：GRPO-CBIU 边界微调**（NLA 迁移，见附录 A；接管后软边界桥退役）；
- k=1 的去留由 S0/S0.5 后的 4× 容量 20K 曲线裁决。

## 11. 2026-07-31：S0 完成与 A/B 对照（E17/E18）

**S0 生产链**（全部归档）：用户手标 16 条（中位 21B/p90 39B/p99 57B，中英同粒度）→
本地 4B 教师（qwen3-vl-4b，few-shot 12 范本，严格逐字符校验+引号豁免）批量标注
5K→3426 合格（68.5%；9B 思考模型路线证伪：预算全烧思考零产出）→ subagent 质检
（完全合规 62.5%/轻微 17.5%/明显错误 20%，系统性偏粗 +33%）→ 过滤 2532 条
（粒度/括号/引号/特殊 token/英文长尾五规则）→ S0 SFT（encoder+segmentor，
4093 窗口/1534 步/85 秒，**F1 0.886 vs 教师标签**；对用户标注 P=0.866/R=0.533，
阈值扫描证明粒度差距是排序问题非操作点，交 S0.5 裁决）。

**A/B 对照**（512B/20K/4× readout）：A（S0 冻结接管）0.189/34.2PPL，B（全新端到端）
0.131/35.9PPL，**组件预训 +5.8pp 且 B 过 12K 退化、边界装死全程**——端到端桥路线判死，
组件预训成为 v3.6 默认路线。k=1 平台从 0.11 抬到 0.19-0.20 但未击穿。

**当前默认**：`configs/canonical_v36.json`（v36.1：S0 接管+动态边界 tau 0.94+max_span 64+
4× 状态+k=1+mask 原生两任务）；v35 旧口径保留。

**进行中（干净归因矩阵，全部统一 S0 encoder 冻结以隔离变量）**：
- B0：uniform 16B + 1× 状态 + k=1（新任务口径下的旧基线对应物）；
- B1：uniform 16B + 4× 状态 + k=1（容量单独归因）；
- K4/K16：A 配置 + readout_queries 4/16（k 扫档）；
- A（已完成）= S0 动态 + 4× + k=1。边界归因 = A−B1，容量归因 = B1−B0，k 效应 = K4/K16−A。

**decoder 训练纪律**（用户要求慎重）：两任务共享 GlobalSpanDecoder（差值=主干净贡献）；
decoder 每臂从头共训、容量固定 1/5 encoder；masked/unmasked 准确率分裂监控两个子技能；
边界分布在 RL 阶段移动时 decoder 需要重适应期，max_span 跨臂固定 64 保可比。
无课程学习（边界/任务均从第 0 步最终形态）；lr warmup/cosine 属优化器层不视为课程；
CBIU 锚点生成（3K 快照）属仪器校准不视为课程。

## 12. 2026-08-01/02：归因矩阵与公平对比

**归因（E19，统一 S0 encoder 冻结）**：旧基线 0.117 →B0（+S0 encoder/两任务）0.146
→B1（+4× 状态）0.145（容量单独零效应）→A（+S0 动态边界）0.189（边界 +4.4pp，
全部活性成分）→K4/K16 ≈0.192（k 无差异）。K4 首发 NaN 发散@6.6K，同 seed 重跑干净
（bf16 瞬时不稳定，状态范数随 k 上行，列为 S0.5/Triton 阶段稳定性工作项）。

**内核与环境**：Kimi K3（2.8T）开源；官方 FlashKDA（CUTLASS 推理内核）不适用训练
（无反向/SM90+/CUDA12.9/K=V=128）。专用环境 `kda-kernels`（torch 2.13+cu130）：
fla KDA 训练、mamba_ssm Mamba-2 训练、FlashKDA 推理三件套全可用；subagent 修复
FlashKDA 在 MSVC 下 CUtensorMap 64B 对齐丢失的真实内核 bug（补丁在
`FlashKDA/csrc/smxx/`，建议反馈上游）。训练主体留 soulvlm。

**H-Net 复现与公平对比（E20）**：AR 版 H-Net 复现（44.5M，transformer 主干，
next-byte BPB 0.653=全信息天花板锚）。DiT 化后统一 masked infilling 口径
（38.2M=39M 目标）：

| 模型 | masked acc | 全位置 acc | PPL | 信息路径 |
|---|---:|---:|---:|---|
| v3.6 A | **0.149** | 0.189 | 34.2 | 1 readout（1,536 标量） |
| HNet-DiT 标准臂 | 0.324 | 0.968 | 1.14 | byte 全直连（无瓶颈天花板） |
| HNet-DiT 瓶颈臂 | 0.142 | 0.492 | 7.97 | 190 细 chunk（~97,000 标量，2.5B/段） |

结论：最硬指标打平（0.149 vs 0.142），信息传输效率 ~60×；HNet-DiT 两臂边界
全部退化（标准臂层级溶解 1 chunk；瓶颈臂切到 2.5B/段规避压缩）——**无显式压缩
激励时动态切分无中间态**，率控制+CBIU 的必要性获对照组证据。当前 RD 前沿：
k∈{1,4,16}≈0.19@1.5K 标量 + HNet 两参照点。口径：单 seed；HNet-DiT 为 transformer
主干+本方 dechunk 变体（Mamba-2 主干在 kda-kernels 可用，留待 R2 忠实版）。

## 附录 A：NLA 迁移可行性分析（2026-07-27）

**NLA（Natural Language Autoencoders，Anthropic/Transformer Circuits，2026-05-07，
transformer-circuits.pub/2026/nla/）**：一对微调 LM 互为自编码器——AV 把残差流激活
作为单 token 注入固定 prompt 自回归生成自然语言解读，AR 把解读还原为向量，往返余弦
误差=解读忠诚度。训练：API 生成解读做 SFT bootstrap → RL 阶段 AV 用 **GRPO**
（组相对策略优化：同状态采 G 个候选，组内归一化优势，免 value network），
奖励=AR 重建误差负值；**AR 同时持续监督跟上 AV 分布漂移**（移动靶 critic 共进化，
70B 尺度验证稳定）。注入为原样单 token embedding+固定 scale。

**可迁移三要素**：
1. **GRPO 组相对优势**——segmentor 切分与 β 写入是离散动作，软边界桥只是近似梯度
   权宜；同前缀采 G 个切分方案、反事实风险差当奖励、组内排名，是免 value net 的
   真策略优化（手写约百行，不需 Miles/SGLang 基建）；
2. **移动靶 critic 先例**——与我们"锚点定期从最新快照重算"同构，70B 尺度先例；
3. **注入接口经验**——单 token 原样注入+固定 scale，与 1-readout 进主干同构。

**映射表**：AV→segmentor/β（离散结构决策）；AR→CBIU 反事实配对；GRPO→同前缀
多切分组内比较；API 标签 bootstrap→段落标签 S0 预训；FVE→readout 内容覆盖率指标。

**两个必须警惕的差异**：
- 他们的奖励有盲区（Dingeto arXiv:2607.20379 证明重建评分被共谋私有编码攻破，
  因中间载体是可含糊的自然语言）；我们的中间载体是**物理删除操作**，串通无渠道——
  结构性免疫，但评审会问，须在论文中显式对比；
- 尺度：他们 RL 70B，我们 39M，只搬算法不搬基建。

**落地**：S0 段落预训（=SFT bootstrap）→ S0.5 GRPO-CBIU 微调（=RL stage）→
软边界桥退役。已被用户采纳为当前路线（§10）。

## 附录 B：v0 可学习性探针（2026-07-25，uniform 边界三任务旧版）

512B/20K/k=1/d_pack=384：recon 0.117 / 补全 0.114 / PPL 44.4 / 保持 0.117，
轨迹 0.09-0.14 平台（8K/12K 尖点 0.14-0.18 不持续），三任务近同值——边缘分布学习
为主、内容传输微弱（单 seed 负向早期信号，非终审；k=1 去留由 4× 容量+段落边界
版本重测裁决）。归档 `L:\FLUED_archive\v36_learnability_probe_20k_20260725`。
07-25 晚学院断电时无任务在跑，无损失。

## 13. 2026-08-02 备注：readout 包均值瓶颈（package-mean bottleneck）

初始化审计中确认的实现层事实：`FLUEDV36.forward` 里整条 prompt 的 readout 包先经
`package.mean(dim=1)`（k=1 时即包本身）压成**单条件向量**，再加 `chunk_pos` 位置嵌入
广播给所有 chunk——GlobalSpanDecoder 的跨段区分**仅靠 chunk_pos**，内容条件对全部
chunk 完全共享。这是当前实现的显著信息瓶颈：decoder 拿到的逐段差异信号全部来自
位置嵌入而非内容。

对 S0.5 的含义：GRPO 奖励（反事实风险差）经由这条单向量条件通道传导，边界/β 动作的
奖励分辨率受限于该瓶颈；评估 GRPO 收益上界、或设计逐段条件化（如 per-chunk 状态读出）
改造时须先意识到这一点。是否改造、何时改造留待 S0.5 证据裁决；本节仅为备注，不改默认。

## 14. 2026-08-02 预注册：S0.5 GRPO 首轮探针（双臂）

**出发点**：`s05_baseline_3k_20260802`（① 3K 纯两任务快照，eval 0.1895/33.7PPL）。
**锚点**：`s05_cbiu_anchors_20260802`（② 离线生成，β 写入门全 rich/全 null，
三风险全维度可分；仪器校准非课程）。

**GRPO 臂**（`tools/train/v3_6/train_v36_grpo.py`，预注册口径见脚本 docstring）：
- 动作：自由位置边界 Bernoulli（p=sigmoid((conf−tau_cut)/0.15)，UTF-8 续字节/首字节
  非动作）+ β logit 空间高斯扰动（σ=0.5）；采样 detach 注入、logp 按动作位置取均值
  保留梯度（软边界桥退役，真实策略梯度替代近似路径）；
- 奖励 = −robust 归一化风险（三风险 BPB 经 ② 锚点归一化后取 max），组内标准化优势，
  免 value net；G=8、batch 2（wide 16）、同 prompt 共享 mask（同字节同 mask 配对）；
- 冻结 byte_lookup+encoder_blocks，segmentor/write_head/decoder 等可训；
  500 步、lr 5e-5（warmup 50 cosine）、grad_clip 1.0。
- **decoder 重适应控制臂**：`train_v36.py` 同快照续训 500 步（边界冻结、canonical
  默认 lr 2e-4）——已先行完成：0.1891/0.1513/33.7，与 ① 基本持平。

**判定口径**（见结果后不改）：主指标 eval backbone_masked_acc 与 backbone_acc，
GRPO 臂 vs 控制臂同 snapshot 对同 snapshot；副指标 reward_mean 趋势、边界漂移
（chunks_per_sample/hard_cut_fraction 自 17.6/0.031 的移动方向——任务奖励把边界
推向 21B 还是 27B 侧，即 §11 教师粒度天花板的裁决信号）；守卫：truncated_tokens=0、
cut_capacity_overflow=0、nan_skips 记录。单 seed 工程口径，不进结论只定下一步。

**结果（2026-08-02，判定口径不变如实登记）**：GRPO 臂 0.1778/0.1064/42.8PPL，
**全面劣于**控制臂 0.1891/0.1513/33.7 与出发点 0.1895/0.1476/33.7；边界被推到容量
上限（chunks 17.6→58.9/64，~9B/chunk），**cut_capacity_overflow=46.5 守卫破位**。
训练中 reward_mean 从 −0.35 改善到 −0.10 但 eval 质量反降——纯质量奖励（无率项）
下模型发现"切得更碎降低单段预测难度"的捷径：传输率被 k=1 焊死，多切不付代价，
decoder 在 readout 包均值瓶颈下无法区分 59 段（§13 预言的瓶颈在此兑现为惩罚项）。
**首轮教训（下轮修正项）**：GRPO 奖励必须含边界率项（chunks 预算/对偶，cbiu.py 的
增广拉格朗日协议现成）——规格"率控制 100% 归边界"在首轮实现中被遗漏，代价已实测。
归档 `L:\FLUED_archive\s05_grpo_first_probe_20260802`。

**二轮预注册（2026-08-02，同出发点同控制臂）**：奖励 = −(ρ + λ·violation +
0.5·w·violation²)，violation=relu(chunks/32−1)，λ 从 0 做投影对偶上升
（dual_lr 0.05/max 20/aug_w 1.0，沿用 v35 CBIU 超参）；预算 32 chunks/512B
（=16B/chunk，uniform 旧基线粒度，给任务奖励在 19-32 区间内的自由选择空间——
教师 27B≈19 chunks 与用户 21B≈24 chunks 的裁决即在此区间发生）。其余口径同一轮。
判定：主指标对控制臂；若 chunks 贴 32 上限且 overflow 再破位，说明预算仍松，
下轮换更紧预算或线性价目。

**二轮结果（2026-08-02）**：0.1841/0.1293/35.2，好于一轮仍劣于控制臂。训练内采样
计数被率项摁在 27-31（violation≈0、dual 缓升），但 **eval 硬阈值计数 43→49**——
温度 0.15 的涂抹使 Σp（采样期望）与 N_hard（部署规则计数）脱钩 ~1.8 倍：率项摁住
了采样均值，部署规则照样冲向容量。归档 `s05_grpo_second_probe_20260802`。

**三轮预注册与结果（2026-08-02）**：率项改锚硬阈值计数（N_hard，detach 进奖励）。
**失败且机制明确**：N_hard 在同一 prompt 的 G 个采样间是确定性的——组内方差为零，
GRPO 的组内标准化优势把它整个消掉，率项对策略不可见；hard 计数冲到 63-81（dual
打满 20，奖励被罚项主导 −32），质量三轮最差 0.1673/0.1017/43.7。**核心教训：
GRPO 组相对优势只能优化组内有方差的量**——一轮（无率项）边界冲顶、二轮（采样计数，
有方差）率可控但涂抹脱钩、三轮（确定性计数，无方差）率项消失。归档
`s05_grpo_third_probe_20260802`。

**四轮方向（待跑）**：回到二轮的采样计数机制（组内方差来源），用 cut_temperature
0.15→0.05 收拢采样-硬规则涂抹，预算 32 不变；若 dual 长解振荡则改线性价目。另观察
到三轮共性：即使边界受控，eval masked 仍落后控制臂 ~2pp——decoder 重适应滞后与
state_norm 上行（5.8→8.9）是 RL 期固有成本，500 步可能不足以让质量项回正，
四轮起改 2K 步并加 1K 中点 eval。

**四轮修订（2026-08-02，NLA 复查后取代上段方案）**：重读 NLA 原文 Method/Reward
shaping 后确认三条借鉴——① **约束不进组相对奖励**：NLA 的唯一正则（向初始化策略的
KL）是直接损失项；我们的等价物是 E[count]=Σcut_prob（自由位置求和，对 confidence
完全可微，是部署硬计数的光滑松弛——二轮采样计数是其蒙特卡洛估计、三轮硬计数是其
阶梯版），`rate_weight·relu(Σp/32−1)` 直接加 loss，撤销对偶上升与奖励内率项；
② KL 锚定 S0 策略（他们防解释退化=我们防边界漂移）与 ① 功能重叠，留五轮后备；
③ FVE∝log(步数) 证实 RL 收益按数量级算，500 步 probes 本在噪声区——四轮 2K 步
（1K 中点 eval），控制臂同步补 2K 版。已同构无需改的：decoder 每步任务回归跟上
actor（=AR 共进化）、task_loss 不给 segmentor 梯度（=AR 不回传 AV）、风险评分只用
detach 标量（=步内固定评分器）；记录在案的差异：NLA 的 AV/AR 为独立两模型，我们
decoder 与 actor 共享 summarizer/write_head k,v 通路。奖励 −log 变换因 rho 可负
不适用。已知松弛：Σp 是硬计数的下界（冒烟实测 Σp 11.6 vs hard 16），预算等效略松，
看均衡点再校。代码已落地（train_v36_grpo.py R4 版 + 3 smoke 测试通过）。

**四轮定稿（2026-08-02，用户裁决两条）**：
- **判定改双口径**（针对捷径表 #1 移动靶归因混淆）：① 绝对口径——eval
  backbone_acc/masked_acc 对出发点（0.1895/0.1476）；② 相对口径——对 2K 控制臂。
  两口径独立记录，结论分级（双过/单过/双不过）；reward_mean 只作过程指标不进判定；
- **预算瞄准用户标注粒度**（捷径表 #2）：Σp 预算 32→18，按实测松弛
  Σp≈0.73×hard 等效部署 ≈24 段（用户手标中位 21B/chunk）；段更少同时是架构级
  计算成本优化——KDA 串行递推步数与 decoder span 计算量均随段数线性下降。
  若均衡点 hard 明显偏离 24（松弛比漂移），下轮按实测比重校。
- 监控项全表（捷径表 #1-#10）随 run 记录：hard/Σp 双计数、β_mean（新增，
  监视写入门饱和）、state_norm、reward_std、grad、overflow=0 守卫。

**四轮结果（2026-08-02，双口径判定如实登记）**：GRPO 臂 2K = 0.1782/0.1477/35.7，
控制臂 2K = 0.1906/0.1492/33.1（出发点 0.1895/0.1476/33.7）。判定：绝对口径
masked 持平（+0.0001）、backbone −1.1pp；相对口径 masked −0.15pp（实质打平）、
backbone −1.2pp——**backbone 双不过、masked 双平，分级=单过（masked）**。
**边界裁决大成功**：hard 计数停在 24.3（≈用户手标 21B 粒度），Σp 全程 9.9-16.1
低于预算 18，**率项全程未激活**——2K 尺度下质量奖励自选停点就是用户粒度附近；
β_mean 0.086-0.160 未饱和（写入门存活）；state_norm 5.6 健康；overflow/truncated
全程 0。masked 为四轮 GRPO 最佳且趋势未封顶（eval@1K→2K：0.1461→0.1477）。
诚实归因警示：率项未激活，故 R4 不是率机制的对照实验；停点 24 来自质量奖励
自身动力学（2K 尺度）+ 种子/历史，机制隔离留后续。注意段数 17.6→24.3 相对
S0 基线是增多（KDA 串行步数 +38%），"段更少"是相对 32 预算/R1 灾难而言。
归档 `L:\FLUED_archive\s05_grpo_r4_2k_20260802`。
**五轮方向（预注册待跑）**：从 R4 臂 checkpoint 续训至 10K（NLA：FVE∝log 步数，
backbone 缺口随 decoder 重适应收敛应继续收窄），同口径双判定；若 10K 仍
backbone 不过而 masked 过，进入"masked 优先"口径讨论（masked 是 RD 前沿口径）。

## 15. 2026-08-02 预注册：summarizer 容量全因子消融（S0.6，优先于 GRPO 续训）

**用户假设（待证）**：质量天花板不在 KDA 状态机与 readout 数量（E19 已证两者扩容
零效应），而在 summarizer。**用户的 KDA 附注**：KDA 门控机制复杂、需更多训练步数，
属"下限更低上限更高"的潜在天生缺陷——若全组合臂增益集中在训练后段出现即印证。

**协议**（用户裁决）：每臂 20K 步；只从 `s05_baseline_3k` 快照加载 S0 四前缀
（byte_lookup/encoder_blocks/segmentor_blocks/segmentor_head，新 `--init-prefixes`
机制）并冻结，**其余模块全部从头初始化**（变量干净）；数据/种子/eval 口径同 canonical。

**因子与假设映射**（2³ 全因子 + A0 基线，共 8 臂）：
- S = summarizer_slots 4→16（池化质量：装得细不细）；
- H = summarizer_hidden 1024→2048（处理能力：算得够不够）；
- M = d_mem 512→1024（表达量：每段 memory 向量装得下多少）。
- 臂：A0 基线 / A1=S / A2=H / A3=M / A4=SH / A5=SM / A6=HM / A7=SHM。

**判定**（见结果后不改）：主指标 eval backbone_masked_acc；**相对 A0 ≥2pp 才算
容量有效**（单 seed 噪声纪律）；A0 兼作协议验证（应复现 ≈0.149，复现失败则全矩阵
作废重查"从头"协议）。因子主效应=含该因子 4 臂均值 − 不含 4 臂均值。
**解读纪律**：若全部臂不动，结论是"瓶颈在更下游（readout 包均值通道）"而非
"summarizer 无罪"——装得下但取不出的情形留给 decoder 条件化改造实验区分。
不上"每段 2 memory 向量"臂（用户裁决，KDA 串行步数翻倍与省算力方向相反）。
归档 `L:\FLUED_archive\s05_summarizer_matrix_20260802`（跑完补 manifest）。

**结果（2026-08-02，2pp 阈值判定）**：8/8 臂零报错完成。masked acc：A0 0.1540、
S 0.1525、H 0.1524、M 0.1512、SH 0.1482、SM 0.1507、HM 0.1526、SHM 0.1490——
**因子主效应 S −0.24pp / H −0.16pp / M −0.09pp，全部远低 2pp 阈值，最佳臂是 A0
本身。用户假设（天花板在 summarizer）在 20K/512B 尺度下被证伪**：池化槽数、
隐藏宽度、memory 尺寸任一方向扩 2-4 倍均无效应，A0 协议验证通过（0.154≈0.149
预期）。按解读纪律，结论指向**瓶颈在 summarizer 下游——readout 包均值条件通道**
（§13）：summarizer 装得下，decoder 取不出。KDA"增益后段集中"附注未获印证
（A0 与 SHM 训练曲线全段重合，无后段发散；但 20K 对"KDA 需要更多步数"的检验
仍可能不足）。下一步方向（待用户裁定）：decoder 逐段条件化改造（per-chunk 状态
读出替代 mean+pos 广播），或 GRPO 10K 续训（R4 臂 masked 趋势未封顶）。
归档已带 manifest（41 项 SHA256）。

## 16. 2026-08-02：S0.5 决策路线全景（截至今日的完整决策树与结果索引）

```
S0 完成（F1 0.886，教师 27B vs 用户 21B 粒度之争待裁决）
  → ① 3K 纯两任务基线（0.1895/33.7，快照=s05_baseline_3k）
  → ② 离线 CBIU 锚点（β 门 rich/null 三风险全可分，s05_cbiu_anchors）
  → ③ GRPO 边界+β 微调（规格附录 A 的 NLA 迁移）：
      R1 纯质量奖励 → 失败（碎切捷径冲 59/64 段；教训：k=1 焊死传输率时多切无代价）
      R2 +采样计数字率项 → 半失败（训练内受控，温度涂抹使部署硬计数脱钩 ~1.8×）
      R3 率项锚硬计数 → 失败（确定性罚项组内零方差，被 GRPO 组相对优势消掉）
      ——核心教训：组相对优势只能优化组内有方差的量
      NLA 复查 → 三借鉴：约束走可微直接损失（E[count]=Σcut_prob）、
                 KL 锚定初始化（后备）、RL 收益按 log(步数)计
      R4 定稿（用户裁决：判定=绝对+相对双口径；预算 18≈部署 24 段锚用户粒度）
        → masked 双口径打平控制臂（0.1477 vs 0.1492，四轮最佳未封顶），
          backbone −1.2pp 未过；**边界裁决成功：hard 停点 24.3≈用户 21B，
          教师粒度之争了结**；β 未饱和、率项未激活、守卫全 0
  → S0.6 summarizer 容量全因子（用户假设：天花板在 summarizer）
      8 臂×20K 从零（仅 S0 前缀接管）：S/H/M 主效应 ≤0.24pp 全无效
      → 假设证伪，瓶颈锁定下游 readout 包均值条件通道（§13 疑点升级为头号嫌疑）
```

**沉淀的通用方法论**（后续轮次直接复用）：
1. 奖励捷径必查表（捷径表 #1-#10，§14）：移动靶归因、部署/训练规则脱钩、
   无成本动作饱和、组内方差消失；
2. 约束与奖励分通道（NLA 结构）：可微直接损失管约束，组相对优势管质量；
3. 双臂纪律：decoder 重适应控制臂先行，判定看臂间差+绝对值双口径；
4. 从零消融协议：`--init-prefixes` 只接管 S0 四前缀，其余从头，变量干净；
5. 2pp 单 seed 显著性阈值 + 协议验证臂（A0 复现失败则全矩阵作废）。

**当前待裁定岔口（Q8）**：decoder 逐段条件化改造 vs GRPO 10K 续训，意义对比
见 CURRENT_STATE §5。

## 17. 2026-08-02 预注册：S0.7 逐段条件化诊断臂（Q8 已裁定，用户选改造）

**诊断逻辑**（用户裁决采纳）：summarizer 扩容零效应（E22）把瓶颈锁在 readout 包
均值条件通道（§13）；本臂是对该嫌疑的判决性实验，同时是路线生死的诊断——
**若 masked 跳升，证明信息一直在 KDA 状态里、只是读出接口太弱（路线活）；
若仍不动，说明信息在压进 1 个包的过程中真丢失（对"整条 prompt 1 包"路线本身是重击）**。

**改造**（新 `per_chunk_readout` 开关，canonical 默认 False 不受影响）：KDA 串行
递推中每消费一段即从当前状态读出该段条件向量（(B,C,q,d_pack)），替代
`package.mean(dim=1)+chunk_pos` 广播；backbone 输入从 1 token 变 C tokens。
`train_v36_grpo.py` 同步改造（后续 GRPO 在同一条件化口径下继续）。

**率口径修正（诚实记账）**：传输标量从 k=1 的 1,536 升至 C×d_pack
（24 段×1536≈36.9K）——"k=1 旗舰点"叙事修正为"前沿点"；但仍仅为 HNet-DiT
瓶颈臂（~97K）的 ~38%，RD 前沿上仍便宜 ~2.6×。判决仍走前沿对前沿。

**协议**：canonical + `--per-chunk-readout`，从零消融协议（只接管 S0 四前缀），
20K 步，同数据/种子/eval。对照 = S0.6 A0 基线（0.1903/0.1540/33.9）。
**判定**（见结果后不改）：eval backbone_masked_acc 相对 A0 **≥+2pp（≥0.174）=
通道是瓶颈，路线活，转入逐段条件化主线**；**<+2pp = 信息在压缩中真丢失，
回到"整条 prompt 1 包"路线的根本讨论**。守卫照旧（truncated/overflow=0）。
归档 `L:\FLUED_archive\s07_perchunk_20k_20260802`（跑完补 manifest）。

**结果（2026-08-02，结构性大阳性 + 预注册口径部分失当，如实登记）**：
eval backbone_acc **0.3507**（A0 0.1903，+16.0pp）、eval direct_acc 0.3497、
eval backbone_ppl **12.12**（A0 33.86）——**信息确实在 KDA 状态里，逐段读出
让 decoder 取到了，"整条 prompt 1 包"路线活**。但预注册判据 masked acc
0.1413（A0 0.1540，−1.3pp）**字面未过线**：masked 补全测的是"编码前就被遮蔽、
从未进过状态的字节"，靠的是上下文推断而非检索——均值通道从来不是这项子技能
的瓶颈（甚至解锁检索后模型对上下文推断的投入略降）。**结论：预注册把 masked
当检索指标用属口径部分失当（同 Q1 类归因待裁定）；诊断问题本身已由
unmasked/PPL 证据一锤定音——通道曾是检索瓶颈，逐段条件化是正确方向。**
附带观测：state_norm 8.07 偏高但稳定；本臂传输标量 17.6 段×1536≈27K，
全位置 acc 0.351 对比 HNet-DiT 瓶颈臂 0.492@97K、masked 0.141 对 0.142@97K，
RD 前沿上仍 ~3.6× 便宜。归因裁定与 canonical 切换（v36.2?）待用户。
归档已带 manifest。

**用户后续裁定（2026-08-02）**：① masked 补全为必备项，不降级（口径归因
讨论关闭）；② S0.7 不设默认，`per_chunk_readout` 保持默认关（canonical 维持
v36.1）；③ 新条件化下 GRPO 是否重跑，待另行裁定。

## 18. 2026-08-02 预注册：S0.8 DiT summarizer 前后对比（用户裁决恢复字面形态）

**背景**：口径考古确认 summarizer 在代码史中从未是 DiT（用户心智源自 v3.1 doctrine
与 v3.4 的双 DiT 组件记忆，详见 §16 与当日调查报告）；用户裁决：**不按类比处理，
直接把 summarizer 改成字面 one-shot DiT**（新 `summarizer_type="dit"`：
DiTStyleBlock×2 逐段并行 + 掩码均值池 + FFN，5.06M vs slot 版 2.55M；
canonical 默认仍为 `"slot"`），并**用原实验条件重测对比**。

**两臂（协议与对应原臂逐项一致：从零消融、同数据/种子/eval、20K 步）**：
- D0 = A0 条件（均值通道）+ DiT summarizer，对照 A0（0.1903/0.1540/33.9）；
- D7 = S0.7 条件（per_chunk_readout）+ DiT summarizer，对照 S0.7
  （0.3507/0.1413/12.1）。

**判定**（见结果后不改，2pp 阈值）：DiT 相对 slot 同条件 Δ≥+2pp（任一主指标）
= projector 形态曾是隐藏瓶颈，summarizer 形态升级进 canonical 讨论；全部 |Δ|<2pp
= slot projector 与 DiT 等效，形态问题关闭（维持 slot 默认，省 2.5M 参数）。
**尺寸分析工作项（本节前先行登记，结论随 D0/D7 一并交付）**：① interpreter 参数
构成——KDA 递推本身零参数（alpha_logit+readout_query 仅 1,024），state_machine
的 3.94M 几乎全是 readout realign MLP；write_head 2.11M 单层 trunk 扩张
512→1,544 门控值/段，疑似比状态机更可能卡准确性；② backbone 4.76M 在 S0.7 上
direct≈backbone（0.3497 vs 0.3507），通道打开前主干零贡献——"主干过小"在通道
打开后是否成为新约束，需要 backbone 容量臂（3L→6L、ffn 2048，d_backbone=384
因共享字节表锁定）验证；③ 边际拐点判读：全部容量旋钮（summarizer 三因子、
状态 4×、k）边际零效应 vs 唯一结构改造（S0.7）+16pp——项目不是全面过了边际
拐点，而是容量边际零、结构边际正；S0.7 后 backbone/写入头容量臂才首次值得测。
归档 `L:\FLUED_archive\s08_dit_summarizer_20260802`（跑完补 manifest）。

**S0.8 结果（2026-08-05，双臂完成）**：
- D0（均值通道 + DiT）：0.1950/0.1527/32.1 vs A0 0.1903/0.1540/33.9——全部 |Δ|<2pp，
  均值通道下形态无关（通道卡死一切，下游什么都摸不到）；
- D7（逐段条件化 + DiT）：**unmasked 0.5491 / PPL 5.81** vs S0.7 slot 0.3507/12.12——
  **+19.8pp、PPL 再腰斩**；masked 0.1428 持平（与口径一致：masked 测推断非检索）。
- **判定：projector 形态确实是隐藏瓶颈——但只在通道打开后才显形**（串联瓶颈的
  遮蔽效应：均值通道是第一瓶颈时，summarizer 形态好坏摸不到差异；逐段条件化
  打通后，slot projector 的欠训/欠表达立刻成为新瓶颈）。用户直觉（summarizer
  是瓶颈）部分平反：它是第二瓶颈，不是第一瓶颈。
- **RD 前沿含义**：D7 全位置 0.549 @ ~27K 标量 **反超 HNet-DiT 瓶颈臂 0.492@97K**，
  且便宜 3.6×（无压缩参照 0.968@262K 仍远）；masked 0.143@27K ≈ 0.142@97K。
- 守卫：overflow/truncated 全 0；D7 有 1 次 NaN skip（bf16 瞬时，K4 前科同类，
  稳定性工作项记录）；state_norm 5.63 健康。
