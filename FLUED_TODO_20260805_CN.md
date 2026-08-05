# FLUED 待办与讨论结论（2026-08-05，压缩上下文接力用）

> 单一事实源 `docs/CURRENT_STATE.md`；术语 `docs/TERMS.md`；主线规格 `docs/versions/v3.6/FLUED_V3_6_SPEC_20260725_CN.md`（§1-§20）。
> 本文是当前待办快照。canonical：**v36.2-20260805**（S0′ 前端 + DiT summarizer + 逐段读出 + S1.0 三任务，训练入口 `tools/train/v3_6/train_v36_s1.py`）。

## 0. 当前最强数字（v36.2，20K/512B/单 seed）

direct 保真 0.772 / unmasked 0.765 / PPL 2.74 / masked 0.148 / predict_cos 0.898 / 段数 23（≈用户 21B 粒度）。
RD 前沿：0.765@35K 标量 vs HNet-DiT 瓶颈臂 0.492@97K。归档 `L:\FLUED_archive\s10p_s0p_20k_20260805`。

## 1. 讨论已定论项（2026-08-05 与用户讨论）

1. **masked 指标口径**：任务是"从未进状态的字节靠语境推断"，0.14-0.15 是任务难度上限（三条独立证据：direct≈backbone 打平 / HNet 瓶颈臂同水平 / S1.0 独占角色也不升）。对内迭代不作判决线，对外对比保留。eval 128 样本标准误 ±0.6pp，跨 run ±1pp 内差异不判读。
2. **GRPO R4 的"语言学漂移"被高估**：BPE 子集尺实测 97.1% 精确命中（E25），±1 字偏移是标点归属风格不是切词。
3. **S0′ 教师轮成功**：K2.5 + 规则固化 + 非思考；旧 4B 标签退役；原 5K 纠偏判定不需要（E27 大胜）。
4. **Q1/Q5 已裁定关闭**（阈值失当+代价谷；memory 线关闭不影响当前路径——写入头/KDA 只消费 summarizer 产物）。

## 2. 待办（优先级序）

### T1 S0′ 扩量（用户判断：还有潜力）
- 现状：1,686 条 v4 K2.5 标签 → S0′。证据：direct/unmasked 曲线 20K 步仍在上升未饱和；±1 字刀口偏移（时|候类）是已知长尾。
- **教师切换（用户 2026-08-05 裁定）：停用 K2.5（价格过高），改用 DeepSeek v4 flash**；K2.5 扩量轮已叫停（仅消耗 ~50 次调用）。风险：教师更换可能带来标签分布漂移——开工前先同样本对比 DeepSeek 与 K2.5 的合格率/粒度分布（200 条试点），分布不一致则以 K2.5 存量 1,686 条为锚、DeepSeek 只作增量并记录来源字段。
- 方案：标 6-8K 条（`--rules-file S05_TEACHER_RULES_CN.md` + 非思考）；DeepSeek 为 OpenAI 兼容协议，脚本换 endpoint/model 即可（token 自备后给我）。
- 之后：S0′′ SFT（train_s0_segmentor.py，从零）→ S1.0 条件 20K 评测臂对照 E27。

### T2 GRPO 在 v36.2 上重跑（奖励比例微调）
- 必须先在 v36.2 形态下重算 CBIU 锚点（`tools/analysis/v3_6/probe_v36_cbiu_anchors.py`——注意它当前只兼容旧两任务语义，需要适配 S1.0 三任务的风险定义）。
- 奖励比例：R4 是 rl_weight=1.0、Σp 预算 18；新起点已是用户粒度（23 段），率约束角色从"拉回来"变"守住"，预算改锚 S0′ 起点（Σp≈23，或对偶自校准）；rl_weight 可降 0.3-0.5（R4 里 pg 梯度占比偏大）。
- 控制臂：train_v36_s1.py 同快照续训。

### T3 masked 任务变体（2026-08-05 用户已定口径：40/60 混合）
- **40% 整 UTF-8 字 mask + 60% 整 BPE 词 mask（用 128k 参照尺）**：整字 mask 锻炼单字理解；整词 mask 测语义推断、但占比不过半以**避免把架构养成高级 BPE**。现行"1-8B 随机 span 会切碎单字"的口径废弃。
- **代码已落地（2026-08-05，v36.3）**：`train_v36_s1.py` `make_mixed_mask`（`--mask-mode mixed` 默认，byte_span 保留开关；专用 CPU 生成器保 eval 确定性；`mask_rate` 实测入日志）。单测 5 项 + 实语料 smoke 通过（8×512 batch 1.0ms；实测速率 0.060 vs 目标 0.05，span 粒度尾差，各臂同码同种子对比有效）。
- **对比口径统一工作项（用户提醒）**：① 现仓 HNet 复现/HNet-DiT 两臂的 Mamba-2 已被换成 causal transformer（`flued/hnet_repro/model.py` docstring 已披露）；Mamba-2 忠实版是 R2 候选（kda-kernels 环境已备 mamba_ssm 2.3.2）；② mask 口径变更后 HNet-DiT 瓶颈臂需同口径重跑（一个 20K run），对外对比表才有效；③ R2 时的最终对比需统一：同一 mask 口径、同一 BPB/acc 定义、同一 eval 集。

### T4 与 H-Net AR 的预测能力对齐评测
- 现状不对齐：H-Net 复现是 next-byte BPB 0.653（全字节上下文无压缩，天花板锚）；我们的是潜空间 predict_cos 0.898，不可直接比。
- 方案：给 S1.0 checkpoint 补"预测路径字节级解码"评测（backbone_out[i] → decoder → 第 i+1 段字节），得 byte acc/BPB，与 0.653 比时注明任务差异（next-chunk vs next-byte、压缩 vs 无压缩）。

### T5 残余工作项
- state_norm 上行（S1.0 16.3 / S0′ 15.4，此前 2-8）：预测任务推高状态范数，bf16 稳定性观察项（K4 NaN 前科）。
- 1024/2048/4096 scaling 试探（Q6 标定已完成，见 CURRENT_STATE）。
- FlashKDA 补丁上游 issue（`L:\FLUED_archive\flashkda_msvc_patch_20260802\UPSTREAM_ISSUE_DRAFT_EN.md` 待提交）。
- push 状态：main 已被 HEAD（b799c39，v36.2）强制覆盖（2026-08-05，第 4 次重试成功）；分支与 main 同点。

## 3. 环境速查

- 训练：`C:\Users\74090\Miniconda3\envs\soulvlm\python.exe`（直接调，勿 conda run）；测试/绘图：`py -3.14`；内核：`kda-kernels` 环境（需 PYTHONUTF8=1）。
- 语料：v3 `E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt`；v4 `P:\FLUED_corpus\FLUED_corpus_v4\shards\`（57 shards, 230G）。
- Moonshot token：`C:\Users\74090\.moonshot_api_key`（勿进仓库）。
- 训练纪律：OMP/MKL=4；分配器红线 15.9/16.3GB；启动前查重进程。
