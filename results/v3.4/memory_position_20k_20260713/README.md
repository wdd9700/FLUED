# FLUED v3.4 Memory / Position 20K Archive

本目录公开 2026-07-13 完成的 P3 归一化筛选与 P4 20K memory 确认实验。P4 三组使用同一随机种子、模型尺寸、数据和训练计划，并全部从零训练。训练检查点体积较大，不进入 Git；日志、解析后配置、最终汇总和里程碑阈值扫描均完整保留。

## 目录

- `p3/`：5 组 5K memory 归一化/尺度实验，每组含 `train_log.jsonl`、`summary.json` 和解析配置。
- `p4/`：3 组 20K 从零训练实验，每组含完整训练日志和最终汇总。
- `threshold_trajectory/`：3K/6K/9K/12K/15K/18K/20K 的全部阈值扫描 JSON。
- `threshold_trajectory.csv`：展平后的 105 行曲线数据，便于直接绘图和比较。
- `p3_p4_summary.{csv,json}`：8 组实验的统一最终指标表。
- `checkpoint_sha256.txt`：本机三组 P4 `latest.pt` 的校验哈希。

## P4 最终结果

| 组别 | 重建 | 补全 | 困惑度 | 实际 latent/byte |
| --- | ---: | ---: | ---: | ---: |
| no-memory | 87.43% | 11.64% | 45.05 | 0.44 |
| **other-only historical memory + affine-free LayerNorm + fixed 0.10** | **96.89%** | **13.80%** | **35.76** | 0.58 |
| other + detached current-memory | 85.44% | 12.95% | 38.63 | 0.53 |

完整方法、阈值扫描和结论边界见[中文分析报告](../../../docs/versions/v3.4/FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md)。

三个 `summary.json` 仍保留 `memory_usage_min/max` 兼容字段，但 `memory_usage_loss_weight=0`，实际没有启用 20%-50% 固定使用率约束。
