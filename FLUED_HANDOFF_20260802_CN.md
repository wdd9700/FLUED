# FLUED 交接文档（2026-08-02，opencode → Kimi Code CLI）

> 写给下一位协作者（Kimi Code CLI 中的 agent）。本文是当前状态的完整快照。
> 单一事实源仍是 `docs/CURRENT_STATE.md`；术语以 `docs/TERMS.md` 为准（未登记词视为不存在，新词先登记再用，AI 提名默认"候选"待用户转正）。

## 0. 授权问题专项（用户明确要求注意）

上一个环境的最大效率杀手：**agent 访问工作区外位置需要用户手动授权，用户不在电脑前时任务空转**。下一位请执行：
1. **开工第一天就和用户确认预授权清单**，把下列高频位置一次性批准（或配置 Kimi Code CLI 的自动允许规则）：
   - `L:\FLUED_archive\`（全部实验归档，读写）
   - `N:\FLUED_corpus_v4\`、`E:\projects\SoulMamba\`（语料，只读）
   - `C:\Users\74090\AppData\Local\Temp\opencode\`（临时产物专用目录，读写）
   - 网络检索（webfetch arXiv/GitHub/HuggingFace）
   - pip 安装（仅限指定 conda 环境内）
2. **所有临时产物一律进上述 Temp 目录**，正式产物直接写 L: 归档，避免触发新位置授权；
3. **长任务用主控脚本模式**（见 §6 的 master.ps1 范式）：分离进程、自写日志、自持 manifest，全程零交互——启动后即可离开，不需要任何中途授权；
4. 需要用户决策的事项**攒批提问**，不要逐条问。

## 1. 状态快照

- 分支 `copilot/implement-stage-a-experiments`，本地领先 origin 多个 commit，**从未 push**（用户知情，勿主动 push/commit 除非被要求）；
- 测试：`py -3.14 -m pytest -q` = **146 全绿**（含 Codex 的 corpus_v5 18 项；其导入路径已由 tests/conftest.py 补 tools/data）；
- GPU 空闲，无后台任务在跑；
- 未提交改动一堆（见 `git status`，全部是有意保留的工作树，勿清理）。

## 2. 三个 Python 环境（用途严格分开）

| 环境 | 路径 | 用途 | 关键坑 |
|---|---|---|---|
| `soulvlm` | `C:\Users\74090\Miniconda3\envs\soulvlm\python.exe` | **训练主环境**（torch 2.12+cu128，triton-windows 3.7.1，fla 0.5.1，transformers 5.14.1） | `conda run` 有 GBK 编码 bug——**永远直接调 python.exe 全路径**；Triton 首次 JIT ~142s 属正常 |
| `kda-kernels` | `C:\Users\74090\Miniconda3\envs\kda-kernels\python.exe` | 内核专用（torch 2.13+cu130，fla 0.5.2，mamba_ssm 2.3.2+causal-conv1d 1.6.2，FlashKDA 本地补丁版） | 直接调 python.exe 需 `$env:PYTHONUTF8=1`（GBK 区域问题，激活脚本已固化但裸调不经过它） |
| `py -3.14` | 系统 | 测试+绘图（有 matplotlib） | 无 GPU torch |

训练前惯例：`$env:OMP_NUM_THREADS=4; $env:MKL_NUM_THREADS=4; $env:KMP_DUPLICATE_LIB_OK="TRUE"`。长上下文（2048+）注意：峰值 ≥15.9/16.3GB 会触发分配器抖动（吞吐骤降 10-20×），batch 要压到峰值留 2GB 余量；`expandable_segments` 在该 torch 构建上不支持，无效。

## 3. v3.6 架构与当前默认（`configs/canonical_v36.json`）

一句话：整条 prompt（512B）→ encoder+segmentor（S0 预训，冻结）动态切分 → summarizer 逐段产 memory → KDA 状态机串行消费 → **整条 prompt 恰好 1 个 readout 包**（d_pack=1536，4× 状态 d_k128/d_v256）→ backbone → 共享 span decoder。两个任务：readout→decoder 精准还原、readout→backbone 改写→decoder 全部还原（mask 原生，5% span 1-8 先于编码）。

关键超参：tau_cut=0.94（S0 校准的 tanh 空间值）、max_span=64、lr 2e-4、无课程、无 stride（窗口=样本）。
S0 权重：`L:\FLUED_archive\s0_segmentor_sft_20260727\latest.pt`，加载用 `--init-checkpoint ... --freeze-prefixes byte_lookup,encoder_blocks,segmentor_blocks,segmentor_head`。
代码：`flued/v36/model.py`、`tools/train/v3_6/train_v36.py`（`--config` 可吃 canonical_v36.json）。

## 4. 证据速览（全部单 seed，细节见 CURRENT_STATE E17-E20）

- **组件预训 >> 端到端**：A（S0 接管）0.189/34.2PPL vs B（e2e）0.131/35.9，e2e 桥装死+退化，判死；
- **归因**：增益全在 S0 动态边界（+4.4pp）；4× 容量单独零效应；k∈{1,4,16} 无差异（~0.19）；K4 首发 NaN 发散但同 seed 重跑干净（bf16 瞬时不稳定，状态范数随 k 上行——S0.5/Triton 阶段工作项）；
- **公平对比（masked infilling 同口径）**：v3.6 masked acc **0.149** ≈ HNet-DiT 瓶颈臂 0.142，但信息传输 1,536 vs ~97,000 标量（**~60× 效率**）；HNet-DiT 两臂边界全退化（无激励动态切分无中间态）——CBIU 必要性的对照组证据；
- **天花板锚**：AR H-Net 复现（44.5M）next-byte BPB 0.653；
- **S0 教师粒度天花板 27B vs 用户口味 21B**：阈值扫描证明是排序问题，留 S0.5 用任务奖励裁决哪个离任务最优更近。

## 5. 工件与归档（L:\FLUED_archive\，均带 SHA256SUMS）

- `s0_teacher_labels_20260727`（5K 标注 3426 合格+过滤 2532 条+质检）；
- `s0_segmentor_sft_20260727`（S0 权重+训练日志）；
- `v36_learnability_probe_20k_20260725`（旧 uniform 三任务基线 0.117）；
- `v36_s0_vs_e2e_20260727`（A/B 对照）；
- `v36_attribution_matrix_20260731`（B0/B1/K4/K16+K4 重跑）；
- `hnet_repro_512_20k_20260801`（AR H-Net）、`hnet_dit_fair_20260802`（DiT 两臂）；
- 更早：`v35_*` 系列（v3.4 收尾证据链）。
仓库内新代码：`flued/v36/`、`flued/hnet_repro/`、`tools/train/v3_6/`（train_v36.py、train_s0_segmentor.py）、`tools/baselines/hnet/train_hnet.py`、`tools/analysis/v3_6/build_s0_teacher_dataset.py`、`FlashKDA/`（本地补丁版，见 §7）。

## 6. 下一步（优先级序）

1. **S0.5（主线）**：从 A 臂 checkpoint（`v36_s0_vs_e2e_20260727\arm_a_s0\latest.pt`）出发——① 3K 步纯两任务训练存快照；② 离线算 CBIU 锚点（全 rich/全 null 参照，复用 `tools/train/v3_4/cbiu.py` 的锚点协议与 v3.5 L1 的生成脚本模式）；③ GRPO 微调边界+β 写入门（同前缀采 G 个切分方案、反事实风险差组内排名、免 value net；NLA 迁移设计见规格附录 A）。**预警**：边界分布移动后 decoder 需重适应期，对比实验必须带"decoder 重适应"控制臂（规格 §11 纪律）；
2. **RD 前沿补点**：把 k1/k4/k16（0.19@1.5K 标量）与 HNet 两参照点画成正式前沿图（py -3.14 matplotlib 已有脚本模式可仿）；
3. **FlashKDA 补丁反馈上游**（`FlashKDA/csrc/smxx/` 三个文件的 CUtensorMap 64B 对齐修复，`git diff` 可导出； MoonshotAI/FlashKDA issue）；
4. **语料线**：Codex 负责 corpus_v5（`tools/data/build_corpus_v5.py` 等，测试已绿），就绪后正式训练切 v5；N 盘 v4 备用；
5. **R2 候选**：kda-kernels 环境跑 Mamba-2 主干忠实版 H-Net、FlashKDA 推理计时。

## 7. 操作纪律（违反这些等于制造事故）

- **单一事实源 `docs/CURRENT_STATE.md`**：新证据落地必须原地更新+changelog；证据四态（已验证/候选/已证伪/待举证）；
- **术语**：`docs/TERMS.md` 先登记再使用；报告里未登记词首次出现必须内联给中文全称+一句定义；
- **预注册**：阈值/口径实验前写死，见结果后不改；
- **归档**：每臂 resolved_input/stdout/train_log/summary + SHA256 manifest，禁通配符删 checkpoint；
- **文档口径**：历史日期文档不回改，只追加；README/AGENTS.md 指针同步；
- **说话风格**（用户明确要求）：精准人话，术语最少化且必须当场解释，不造词（造词也要登记成候选）。
