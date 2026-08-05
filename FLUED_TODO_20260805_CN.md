# FLUED 待办与讨论结论（2026-08-05，压缩上下文接力用）

> 单一事实源 `docs/CURRENT_STATE.md`；术语 `docs/TERMS.md`；主线规格 `docs/versions/v3.6/FLUED_V3_6_SPEC_20260725_CN.md`（§1-§20）。
> 本文是当前待办快照。canonical：**v36.2-20260805**（S0′ 前端 + DiT summarizer + 逐段读出 + S1.0 三任务，训练入口 `tools/train/v3_6/train_v36_s1.py`）。

## 0. 当前最强数字（v36.3，20K/512B/单 seed）

E28 混合 mask 口径主臂：direct 0.789 / unmasked 0.782 / PPL 2.637 / masked 0.141（噪声带内持平）/ predict_cos 0.862 / 段数 23.5；较 E27 byte_span 口径三主项 +1.8pp/+1.7pp/−0.10，口径切换零回退。
归档 `L:\FLUED_archive\s11_mixedmask_20k_20260805`（含 S1.0 语义 CBIU 锚点 + 预测解码评测）。

## 1. 讨论已定论项（2026-08-05 与用户讨论）

1. **masked 指标口径**：任务是"从未进状态的字节靠语境推断"，0.14-0.15 是任务难度上限（三条独立证据：direct≈backbone 打平 / HNet 瓶颈臂同水平 / S1.0 独占角色也不升）。对内迭代不作判决线，对外对比保留。eval 128 样本标准误 ±0.6pp，跨 run ±1pp 内差异不判读。
2. **GRPO R4 的"语言学漂移"被高估**：BPE 子集尺实测 97.1% 精确命中（E25），±1 字偏移是标点归属风格不是切词。
3. **S0′ 教师轮成功**：K2.5 + 规则固化 + 非思考；旧 4B 标签退役；原 5K 纠偏判定不需要（E27 大胜）。
4. **Q1/Q5 已裁定关闭**（阈值失当+代价谷；memory 线关闭不影响当前路径——写入头/KDA 只消费 summarizer 产物）。

## 2. 待办（优先级序）

### T1 S0′ 扩量（用户判断：还有潜力）
- 现状：1,686 条 v4 K2.5 标签 → S0′。证据：direct/unmasked 曲线 20K 步仍在上升未饱和；±1 字刀口偏移（时|候类）是已知长尾。
- **教师切换已定案（2026-08-05）：v4-pro + 纯净规则 + 关思考 + temp 0.2**。五轮 200 条配对试点：flash 各配置全线失败（中文 2× 粗或修中丢英），pro 一次贴锚（ZH 23.8B/EN 26.5B vs 锚 23.6/29.8，合格率 93%）。10 条 subagent 抽检：带条件可用（1/10 硬违规剔除；介词悬空 4/10 建议 R7 补反例——全量合并后加机器预筛：中文段 >21 字/英文 >9 词零容忍退回）。
- **全量 8K 标注完成（6,572 条，82.2% 合格率与 K2.5 轮同水平）；合并经硬上限预筛（>21字/>9词剔除 1,032 条，含 K2.5 锚自身 401 条）得 7,226 条**（k25 1,285 + deepseek_v4pro 5,941，来源字段已落，`outputs/s05_teacher_merged_v4_20260805`）；合并后粒度 k25 24.8B vs pro 25.3B——对齐。**S0′′ 从零 SFT 已启动**（`checkpoints/s0pp_pro_v4_sft_20260805`，超参同 S0′ 18 epochs）；failures 返工轮（1,428 条）并行，回收后重合并。
- 之后：S0′′ SFT（train_s0_segmentor.py，从零）→ S1.0 条件 20K 评测臂对照 E27/E28。

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

## 3. 环境速查

- 训练：`C:\Users\74090\Miniconda3\envs\soulvlm\python.exe`（直接调，勿 conda run）；测试/绘图：`py -3.14`；内核：`kda-kernels` 环境（需 PYTHONUTF8=1）。
- 语料：v3 `E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt`；v4 `P:\FLUED_corpus\FLUED_corpus_v4\shards\`（57 shards, 230G）。
- Moonshot token：`C:\Users\74090\.moonshot_api_key`（勿进仓库）。
- 训练纪律：OMP/MKL=4；分配器红线 15.9/16.3GB；启动前查重进程。
