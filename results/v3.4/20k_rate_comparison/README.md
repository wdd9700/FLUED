# FLUED v3.4 20K Rate and Curriculum Comparison

四组实验均从头训练 20K，控制模型、数据、seed、batch 和学习率曲线一致。

| 方案 | eval reconstruction | eval completion | actual latent/byte |
| --- | ---: | ---: | ---: |
| exact marginal rate | 0.7784 | 0.1337 | 0.4928 |
| L2 marginal rate | 0.7985 | 0.1198 | 0.4431 |
| uniform boundaries | 0.9969 | 0.1457 | 0.5905 |
| **uniform 3K -> L2** | **0.8885** | **0.1474** | **0.5225** |

文件：

- [`logs/`](logs/)：四组完整 JSONL，每 20 步记录一次。
- [`analysis/curve_summary.csv`](analysis/curve_summary.csv)：快照和终点评估。
- [`analysis/curve_summary.json`](analysis/curve_summary.json)：完整曲线统计。
- [`analysis/curves_interactive.html`](analysis/curves_interactive.html)：交互曲线。
- [中文分析报告](../../../docs/versions/v3.4/FLUED_V3_4_20K_RATE_CURRICULUM_ANALYSIS_20260711_CN.md)。

模型检查点和训练语料不纳入 Git。
