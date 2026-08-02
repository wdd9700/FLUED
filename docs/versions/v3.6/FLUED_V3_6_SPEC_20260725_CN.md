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
