# FLUED Documentation Index

本目录按研究版本组织。版本目录保存当时的设计、实验和判断，不会为了匹配当前结论而改写历史；当前口径以仓库根目录 `README.md` 和 v3.4 文档为准。

## 阅读顺序

1. [研究回顾](research/FLUED_RESEARCH_RETROSPECTIVE_CN.md)：从 v1 到 v3.3 的问题、失败和设计转向。
2. [v3.4 实现基线](versions/v3.4/FLUED_V3_4_IMPLEMENTATION_BASELINE_CN.md)：当前可运行架构。
3. [v3.4 编码率与发声控制修正](versions/v3.4/FLUED_V3_4_RATE_EMIT_CORRECTION_20260710_CN.md)：边界与计算预算的修正。
4. [v3.4 5K 消融分析](versions/v3.4/FLUED_V3_4_5K_ABLATION_ANALYSIS_20260711_CN.md)：当前最完整的实验结论。
5. [公开训练日志](../results/v3.4/5k_ablation/README.md)：原始曲线、数值摘要和交互图。

## 版本目录

| 目录 | 定位 | 状态 |
| --- | --- | --- |
| [v0.4 / v1](versions/v0.4/) | 最小假设：可微边界、可逆字节压缩 | 历史证据 |
| [v2](versions/v2/) | 去噪重建、类型边界先验、328M tied inverse | 稳定但训练动力学难调 |
| [v3.0](versions/v3.0/) | 主动语义段、memory、surprise 与新评估表 | 设计与机制探索 |
| [v3.1](versions/v3.1/) | segmental latent workspace 和小规模 codec | 历史原型 |
| [v3.2](versions/v3.2/) | 严格 masked-source、paired backbone 评估 | 已验证关键证据 |
| [v3.3](versions/v3.3/) | 完整 byte-to-latent 接口与消融入口 | 架构基线 |
| [v3.4](versions/v3.4/) | 并行 memory、边际编码率、readout emit | 当前主线 |

## 公共研究材料

| 文档 | 用途 |
| --- | --- |
| [研究回顾](research/FLUED_RESEARCH_RETROSPECTIVE_CN.md) | 版本演进和证据边界 |
| [Tokenizer-free LM 研究图谱](research/TOKENIZER_FREE_LM_LANDSCAPE_CN.md) | 领域调研与邻近工作 |
| [v3 系列检查点重评估](research/evidence/v3-family/FLUED_V3_CHECKPOINT_REEVALUATION_CN.md) | 历史检查点的新评估口径 |
| [v3 系列完整指标表](research/evidence/v3-family/FLUED_V3_FULL_METRIC_TABLE_REEVALUATION_CN.md) | 编解码、主干、latent、memory、boundary 指标 |
| [Corpus v4 去重计划](research/CORPUS_V4_DEDUP_PLAN_CN.md) | 数据治理方案 |
| [官网展示文案](research/FLUED_WEBSITE_SHOWCASE_CN.md) | Alethic Insight 官网材料 |

## 证据口径

- **当前可直接引用**：v2 三种子、D1 2048-byte/100K、公平 masked-source v3.2.1、v3.4 5K 单种子消融。
- **研究附录可引用**：v1 历史 E3、v3.1 ROI、早期 memory 机制测试。
- **反例**：latent consistency 爆炸、固定压缩权重 NaN、clean encode 后再 mask 造成的信息泄露。
- **未完成 claim**：v3.4 尚未完成多种子、300M、4096-byte 和同算力外部系统比较。
