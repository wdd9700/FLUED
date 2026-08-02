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

各职责目录中的 `v3_0` 至 `v3_6` 子目录对应研究版本。没有版本子目录的工具属于 v1/v2 或跨版本共享工具。

当前主线入口（v3.6）：

```text
tools/train/v3_6/train_v36.py            # 主训练器，--config 吃 configs/canonical_v36.json
tools/train/v3_6/train_s0_segmentor.py   # S0 segmentor 预训
tools/analysis/v3_6/build_s0_teacher_dataset.py  # S0 教师标注数据生产
```

v3.4 收尾系列入口（CBIU 锚点协议与离线效用数据集是 S0.5 的复用对象）：

```text
tools/train/v3_4/train_v34_pos_ar_probe.py
tools/train/v3_4/cbiu.py
tools/analysis/v3_4/probe_v34_cbiu.py
tools/analysis/v3_5/build_l2_offline_utility_dataset.py
```
