# FLUED Documentation Index

本目录按研究版本组织。版本目录保存当时的设计、实验和判断，不会为了匹配当前结论而改写历史。
**当前口径以 [`CURRENT_STATE.md`](CURRENT_STATE.md)（单一事实源）与 [`TERMS.md`](TERMS.md)（术语注册表）为准**；
本索引与版本目录只提供历史脉络。

## 阅读顺序（当前主线）

1. [CURRENT_STATE.md](CURRENT_STATE.md)：当前默认配置、证据注册表（E1-E20）、闸门与待裁定队列（Q1-Q7）。
2. [TERMS.md](TERMS.md)：术语注册表与汇报协议（未注册词禁用）。
3. [v3.6 架构规格](versions/v3.6/FLUED_V3_6_SPEC_20260725_CN.md)：当前主线（KDA 世代）唯一规格书，含 §10-§12 演进修订（S0/S0.5 路线、归因矩阵、HNet 公平对比）与附录 A（NLA/GRPO 迁移设计）。
4. [v3.5 分级冻结设计](versions/v3.5/FLUED_V3_5_STAGED_FREEZE_AND_OFFLINE_UTILITY_CN.md) 与 [v3.5 预 scaling 结果](versions/v3.5/FLUED_V3_5_PRESCALING_RESULTS_20260717_CN.md)：v3.4 收尾系列（分级冻结 L0-L5、离线效用数据集）。注意命名：仓库内 "v3.5" 指该系列，不是 KDA 世代。
5. [研究回顾](research/FLUED_RESEARCH_RETROSPECTIVE_CN.md)：从 v1 到 v3.4 的问题、失败和设计转向。
6. v3.4 及更早版本目录：细节按需经各目录 README 索引定位，结论已被 CURRENT_STATE 蒸馏。

## 版本目录

| 目录 | 定位 | 状态 |
| --- | --- | --- |
| [v0.4 / v1](versions/v0.4/) | 最小假设：可微边界、可逆字节压缩 | 历史证据 |
| [v2](versions/v2/) | 去噪重建、类型边界先验、328M tied inverse | 稳定但训练动力学难调 |
| [v3.0](versions/v3.0/) | 主动语义段、memory、surprise 与新评估表 | 设计与机制探索 |
| [v3.1](versions/v3.1/) | segmental latent workspace 和小规模 codec | 历史原型 |
| [v3.2](versions/v3.2/) | 严格 masked-source、paired backbone 评估 | 已验证关键证据 |
| [v3.3](versions/v3.3/) | 完整 byte-to-latent 接口与消融入口 | 架构基线 |
| [v3.4](versions/v3.4/) | 并行 memory、边际编码率、readout emit | 历史主线（结论已蒸馏进 CURRENT_STATE） |
| [v3.5](versions/v3.5/) | 分级冻结 L0-L5、离线 CBIU 效用数据集（v3.4 收尾系列） | 有效；L3 未过闸 |
| [v3.6](versions/v3.6/) | KDA 世代：整条 prompt 恰好 1 个 readout 包 | **当前主线** |

## 公共研究材料

| 文档 | 用途 |
| --- | --- |
| [研究回顾](research/FLUED_RESEARCH_RETROSPECTIVE_CN.md) | 版本演进和证据边界 |
| [Tokenizer-free LM 研究图谱](research/TOKENIZER_FREE_LM_LANDSCAPE_CN.md) | 领域调研与邻近工作 |
| [v3 系列检查点重评估](research/evidence/v3-family/FLUED_V3_CHECKPOINT_REEVALUATION_CN.md) | 历史检查点的新评估口径 |
| [v3 系列完整指标表](research/evidence/v3-family/FLUED_V3_FULL_METRIC_TABLE_REEVALUATION_CN.md) | 编解码、主干、latent、memory、boundary 指标 |
| [Corpus v4 去重计划](research/CORPUS_V4_DEDUP_PLAN_CN.md) | 数据治理方案 |
| [官网展示文案](research/FLUED_WEBSITE_SHOWCASE_CN.md) | Alethic Insight 官网材料 |

## 数据与审计快照

| 文档 | 用途 |
| --- | --- |
| [corpus v5 管线手册](data/README_CN.md) | v5 增量清洗管线操作约束（10 条硬规则） |
| [语料来源审计](data/SOURCE_AUDIT_20260725_CN.md) | 各来源体积/校验状态/排除清单 |
| [许可分层](data/Y_LICENSE_MATRIX_20260725_CN.md) | Y 盘语料许可矩阵 |
| [2026-07-19 审计导出](audit_export_20260719/) | 当日控制面文档冻结快照（勿当现状） |

## 证据口径

- **当前证据注册表在 [CURRENT_STATE.md](CURRENT_STATE.md) §3**（E1-E20，四态证据状态机，注意单 seed 限定语）；本节以下为 v3.4 时代的历史口径，保留不删。
- **可直接引用（历史口径）**：v2 三种子、D1 2048-byte/100K、公平 masked-source v3.2.1、v3.4 20K 单种子配对实验。
- **研究附录可引用**：v1 历史 E3、v3.1 ROI、早期 memory 机制测试。
- **反例**：latent consistency 爆炸、固定压缩权重 NaN、clean encode 后再 mask 造成的信息泄露。
