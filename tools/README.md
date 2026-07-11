# Tool Index

工具先按职责、再按研究版本组织：

```text
tools/
  analysis/       日志汇总、曲线、机制审计
  eval/           ROI、边界、memory、严格补全评估
  launcher/       PowerShell/Python 实验矩阵入口
  train/          正式训练入口
  data/           语料构建
  debug/          v1/v2 与基线诊断
  plotting/       v1/v2 历史绘图
```

各职责目录中的 `v3_0`、`v3_1`、`v3_2`、`v3_3`、`v3_4` 子目录对应研究版本。没有版本子目录的工具属于 v1/v2 或跨版本共享工具。

当前主线入口：

```text
tools/train/v3_4/train_v34_pos_ar_probe.py
tools/launcher/v3_4/run_v34_pos_ar_matrix.py
tools/analysis/v3_4/analyze_v34_5k_curves.py
tools/analysis/v3_4/probe_v34_boundary_stability_5k.py
```
