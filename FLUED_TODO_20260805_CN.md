# FLUED 待办与讨论结论（2026-08-05，压缩上下文接力用）

> 单一事实源 `docs/CURRENT_STATE.md`；术语 `docs/TERMS.md`；主线规格 `docs/versions/v3.6/FLUED_V3_6_SPEC_20260725_CN.md`（§1-§20）。
> 本文是当前待办快照。canonical：**v36.2-20260805**（S0′ 前端 + DiT summarizer + 逐段读出 + S1.0 三任务，训练入口 `tools/train/v3_6/train_v36_s1.py`）。

## 0. 当前最强数字（v36.3-attn 时代，20K/512B/单 seed）

**E30（S0′′ 前端，attn 主干，目前仍是最强基线）**：direct 0.836 / unmasked 0.830 / backbone 0.791 / PPL 2.25 / masked 0.160 / predict_cos 0.915 / 段数 23.5。
归档 `L:\FLUED_archive\s12_s0pp_20k_20260805`。
v36.4/v36.5 三臂（s13 预测 v2.0 / s14 per-readout+锚0.1 / s14b per-readout+锚1.0）全部负结果（E31/E32），attn 主干 + E30 配置仍是当前基线。

## 0.1 基线裁定——**前提已被 E34 改写（2026-08-06）**

旧三选项建立在 E32（per-readout 判死）之上，但 E34 查明 E32 的两臂都跑在 v2.1 中毒体制里，判决暂停。新事实：latent 预测配方下 attn 主干完全健康（s16 复现 E30）；复制捷径（双向注意力+字节 CE）才是真凶。待定问题重排为：
- **s17（在跑）**：mlp+latent——若回 ~0.8，per-readout 平反，E32 机制叙事作废；若仍低，监督密度机制在干净体制下确认。
- **s18（在跑）**：attn+latent+无状态通道 vs s16——R1 健康体制正式判决。
- causal-attn 的动机改写：不再是"保监督密度"，而是**拆掉复制捷径后让预测字节 CE 复活**（诚实版 v2）——若 s17 显示 mlp 健康，causal-attn 与 mlp 是同族替代（都无未来可抄），未必需要；若 mlp 仍死，causal-attn 是 attn 唯一诚实化路径。
队列中容量消融/GRPO/翻页曲线三项**暂停**，待基线裁定后按新基线重排。

## 0.2 当前队列（2026-08-06 用户裁定：性能优化 → R1 → R0；三选一顺延）

1. **性能优化（已完成，E33）**：指标免同步、fla chunk_kda 默认（canonical v36.6 起）、b16 换页判死、推理流式原语 `stream_step` 落地。速度修正：20K 全程口径新旧码持平，fla 价值在字节级长度与 R0 使能。
2. **E34 体制判别（已闭环）**：v2.1 复制捷径定罪（s15 0.377 vs s16 0.832 单变量），canonical 切 v36.7（预测默认 latent），fla 核经 s16 复现赦免。
3. **R1 相对基线（重跑中）**：中毒对 s15/s15b 仅作参照；健康体制判决对 = s16（已完成 0.832）vs s18（attn+latent+无状态，在跑）。判决（规格 §7）：前沿不超相对基线 → 记忆通道终局关闭。
4. **E32 重审（在跑）**：s17（mlp+latent，其余全同）——per-readout 平反或确认。
5. **R0 Occam 基线（代码就绪待排）**：`flued/v36/kda_lm.py` + `tools/train/v3_6/train_kda_lm.py`，KDA:transformer=3:1 byte 级直吃，臂 A d512/L12/ffn1792（48.21M）/ 臂 B d448/L16/ffn1536（≈47.3M）对齐全栈 47.2M，同 H-Net 复现协议，BPB 口径。判决：同参打平 → v3.6 全线关闭；胜出需质量+分段计时双赢。
6. 归因链闭环后回到 §0.1 新三选一，再重排容量消融/GRPO/翻页曲线。

## 1. 讨论已定论项（2026-08-05 与用户讨论）

1. **masked 指标口径**：任务是"从未进状态的字节靠语境推断"，0.14-0.15 是任务难度上限（三条独立证据：direct≈backbone 打平 / HNet 瓶颈臂同水平 / S1.0 独占角色也不升）。对内迭代不作判决线，对外对比保留。eval 128 样本标准误 ±0.6pp，跨 run ±1pp 内差异不判读。**任务范围澄清（2026-08-05 与用户确认）：mask 散布整段 512B，主干对整张 readout 矩阵全局补全、decoder 还原全部字节；masked 只是"遮蔽位计分"的评分口径，不是"只补当前段"；唯一严格绑定下一段的是预测任务。**
2. **GRPO R4 的"语言学漂移"被高估**：BPE 子集尺实测 97.1% 精确命中（E25），±1 字偏移是标点归属风格不是切词。
3. **S0′ 教师轮成功**：K2.5 + 规则固化 + 非思考；旧 4B 标签退役；原 5K 纠偏判定不需要（E27 大胜）。
4. **Q1/Q5 已裁定关闭**（阈值失当+代价谷；memory 线关闭不影响当前路径——写入头/KDA 只消费 summarizer 产物）。

## 2. 待办（优先级序）

### T1 S0′ 扩量（用户判断：还有潜力）——**已闭环（E30 大胜）**
- 结果：7,363 条双源合并（k25 1,285 锚 + pro 6,078 增量，硬上限预筛）→ S0′′（F1 0.756）→ 下游 direct 0.836 / unmasked 0.830 / PPL 2.25 / masked 0.160 / predict_cos 0.915，较 E28 +4.6~4.8pp。教师切换（K2.5→v4-pro）与扩量判定完全成功。

### T2 GRPO 在 v36.2 上重跑（奖励比例微调）
- ~~必须先在 v36.2 形态下重算 CBIU 锚点~~ **锚点已重算（2026-08-05，S1.0 语义 + 混合口径，`s11_mixedmask_20k_20260805/cbiu_anchors/`，rich/null 三维全可分）**；探针已适配三任务风险定义（reconstruction=as-encoded 保真）。
- 奖励比例：R4 是 rl_weight=1.0、Σp 预算 18；新起点已是用户粒度（23 段），率约束角色从"拉回来"变"守住"，预算改锚 S0′ 起点（Σp≈23，或对偶自校准）；rl_weight 可降 0.3-0.5（R4 里 pg 梯度占比偏大）。
- 控制臂：train_v36_s1.py 同快照续训。

### T3 masked 任务变体（2026-08-05 用户已定口径：40/60 混合）
- **40% 整 UTF-8 字 mask + 60% 整 BPE 词 mask（用 128k 参照尺）**：整字 mask 锻炼单字理解；整词 mask 测语义推断、但占比不过半以**避免把架构养成高级 BPE**。现行"1-8B 随机 span 会切碎单字"的口径废弃。
- **代码已落地（2026-08-05，v36.3）**：`train_v36_s1.py` `make_mixed_mask`（`--mask-mode mixed` 默认，byte_span 保留开关；专用 CPU 生成器保 eval 确定性；`mask_rate` 实测入日志）。单测 5 项 + 实语料 smoke 通过（8×512 batch 1.0ms；实测速率 0.060 vs 目标 0.05，span 粒度尾差，各臂同码同种子对比有效）。
- **对比口径统一工作项（用户提醒）**：① 现仓 HNet 复现/HNet-DiT 两臂的 Mamba-2 已被换成 causal transformer（`flued/hnet_repro/model.py` docstring 已披露）；Mamba-2 忠实版是 R2 候选（kda-kernels 环境已备 mamba_ssm 2.3.2）；② **HNet-DiT 瓶颈臂同口径 20K 重跑进行中**（`checkpoints/hnet_dit_bottleneck_mixed_20k_20260805`，`train_hnet.py` 已支持 `--mask-mode mixed`），标准臂待排；③ R2 时的最终对比需统一：同一 mask 口径、同一 BPB/acc 定义、同一 eval 集。

### T4 与 H-Net AR 的预测能力对齐评测
- 现状不对齐：H-Net 复现是 next-byte BPB 0.653（全字节上下文无压缩，天花板锚）；我们的是潜空间 predict_cos 0.898，不可直接比。
- **首轮已测（2026-08-05，`eval_v36_predict_decode.py` on s11 checkpoint）**：零样本 decoder 复用 byte acc 0.099 / BPB 10.1——比均匀随机还差（自信地错），证明潜空间预测信息不能经现有 decoder 白拿；这是下限口径。
- 下一步：训一个轻量探针读头（冻结主干，backbone_out[i] → 小读头 → 第 i+1 段字节），得 byte acc/BPB 上界估计，与 0.653 比时注明任务差异（next-chunk vs next-byte、压缩 vs 无压缩）。

### T5 残余工作项
- state_norm 上行（S1.0 16.3 / S0′ 15.4，此前 2-8）：预测任务推高状态范数，bf16 稳定性观察项（K4 NaN 前科）。
- 1024/2048/4096 scaling 试探（Q6 标定已完成，见 CURRENT_STATE）。
- FlashKDA 补丁上游 issue（`L:\FLUED_archive\flashkda_msvc_patch_20260802\UPSTREAM_ISSUE_DRAFT_EN.md` 待提交）。
- push 状态：main 已被 HEAD（b799c39，v36.2）强制覆盖（2026-08-05，第 4 次重试成功）；分支与 main 同点。

## 4. 生成线与容量阶梯（2026-08-05 用户裁定）

> **范围冻结（2026-08-05 用户裁定）：当前只跑已规划的四项——s14 v36.5 基线 →
> 容量消融矩阵 → GRPO 重跑 → 翻页曲线测量。不加新项。全部完成后彻底更新/筛查
> 一次默认配置（canonical + CURRENT_STATE + 规格），之后再考虑是否重测 E31
> 核心内容。**

**生成式矩阵模型（翻页）**：实际运用形态是主干在全新 latent 矩阵上生成回复，接近/达到矩阵承载上限（当前 512B，后期 1024B）就切入全新矩阵；新 prompt 输入即新矩阵。预测路径的字节可读性是硬前置（见下）。

**已落地**：预测任务 v2（v36.4）——`--predict-mode decode`：backbone_out[i] 经**冻结 decoder**（functional_call 全参数 detach，梯度只回主干）解第 i+1 段字节计 CE 为主损失；潜空间 MSE 降为 0.1 弱风格锚（保主干输出风格与 encoder 一致）；predict_byte_acc 成为日常指标。

**翻页信号（先实测后决定）**：延长前缀衰减曲线到 64/128 段，分域（代码/文本/数学）测最优翻页点方差——方差小则字节预算翻页（保守执行），方差大才上门控 MLP（3 层，决定 encoding/生成何时切新矩阵，适配不同信息密度）。

**容量消融矩阵（各轴独立、全部从零，segmentor 从 S0′′ 快照接管；1x 已有=当前值，看拐点定组件比例，不拍脑袋）**：
| 轴 | 当前 1x | 待测点 | 备注 |
|---|---|---|---|
| backbone 主干 | 4.76M | 30M（后期 100M 须配任务升级/冻结 codec 验证通用性） | 最优先，生成线核心 |
| decoder | 6.5M | 1.5x（~9.7M）/ 2x（~13M） | 所有任务的最终裁判，不能是瓶颈/残差截流者 |
| write_head 写入头 | 2.1M | 加宽（interpreter 合计目标 ~10M 量级内定） | memory→KDA 唯一翻译器 |
| summarizer | 5.06M | 重测容量探针后再定（E22 旧形态零效应，勿按旧直觉直接加） | 最后动 |
- 预算与入口：均为 train_v36_s1.py + canonical 改单轴 + 20K/512B，对照 E28。

## 3. 环境速查

- 训练：`C:\Users\74090\Miniconda3\envs\soulvlm\python.exe`（直接调，勿 conda run）；测试/绘图：`py -3.14`；内核：`kda-kernels` 环境（需 PYTHONUTF8=1）。
- 语料：v3 `E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt`；v4 `P:\FLUED_corpus\FLUED_corpus_v4\shards\`（57 shards, 230G）。
- Moonshot token：`C:\Users\74090\.moonshot_api_key`（勿进仓库）。
- 训练纪律：OMP/MKL=4；分配器红线 15.9/16.3GB；启动前查重进程。
