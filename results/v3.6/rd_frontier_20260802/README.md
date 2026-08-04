# RD Frontier 2026-08-02

一句话口径：**率 = 每条 512-byte prompt 传输的标量总数**（v3.6 为 `readout_queries × d_pack`，HNet-DiT 为字节/chunk 数 × d_model=512）；**失真 = strict masked-source 补全准确率**（`eval_backbone_masked_acc` / `eval_masked_acc`，越高失真越低）；**单 seed=42**，均 20k steps、corpus_v3。天花板锚 AR H-Net BPB 0.653 是不同指标（next-byte），仅作图内文本注释，不进入 acc 坐标系。

来源归档目录（均在 `L:/FLUED_archive/` 下，只读）：

- `v36_s0_vs_e2e_20260727/arm_a_s0`（v3.6 k=1）
- `v36_attribution_matrix_20260731/k4_s0_4x_rerun`（v3.6 k=4）
- `v36_attribution_matrix_20260731/k16_s0_4x`（v3.6 k=16）
- `v36_attribution_matrix_20260731/b0_uniform_1x_k1`、`b1_uniform_4x_k1`（uniform 上下文点）
- `v36_learnability_probe_20k_20260725`（旧 uniform 基线；该 run 未记录 d_pack，按同属 1× KDA 状态族绘于 384，仅上下文参考）
- `hnet_dit_fair_20260802/hnet_dit_std`、`hnet_dit_bottleneck`（HNet-DiT 参照；归档无显式标量数记录，率按 d_model=512 推导）
- `hnet_repro_512_20k_20260801`（BPB 天花板锚，注释用）

复现：`py -3.14 tools/plotting/plot_v36_rd_frontier.py`（纯 matplotlib，无 torch）。
