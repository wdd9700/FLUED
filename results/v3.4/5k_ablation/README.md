# FLUED v3.4 5K Ablation Archive

这是 2026-07-11 完成的 17 组小规模结构消融公开归档。

## 实验口径

| 项目 | 值 |
| --- | --- |
| FLUED 参数量 | 37,264,917 |
| 临时补全主干 | 4,783,232 |
| 总参数量 | 42,048,149 |
| 训练步数 | 每组 5,000 |
| 日志频率 | 每 20 步 |
| 种子 | 42 |
| 序列长度 | 512 bytes |
| 状态 | 17/17 完成；无截断 token |

## 文件

- [`logs/`](logs/)：每组 250 个训练记录点的原始 JSONL。
- [`analysis/curve_summary.csv`](analysis/curve_summary.csv)：用于表格的扁平汇总。
- [`analysis/curve_summary.json`](analysis/curve_summary.json)：完整快照、面积、尾段斜率。
- [`analysis/boundary_stability.md`](analysis/boundary_stability.md)：2.5K 与 5K 边界稳定性。
- [`analysis/curves_interactive.html`](analysis/curves_interactive.html)：可交互训练曲线。
- [完整中文分析](../../../docs/versions/v3.4/FLUED_V3_4_5K_ABLATION_ANALYSIS_20260711_CN.md)。

## 历史 5K 结构筛选结论

L2 边际编码率是本轮效果、补全和实际潜向量数量之间最均衡的候选；均匀边界是近乎不压缩的上限对照；精确行列式编码率边界稳定，但更早固化在次优位置。该结论只来自小规模单种子筛选，已由后续 20K 位置/memory 实验继续修正。
