# FLUED v3.4

- [`FLUED_V3_4_FOLLOWUP_EXPERIMENTS_20260717_CN.md`](FLUED_V3_4_FOLLOWUP_EXPERIMENTS_20260717_CN.md)：交接复核后的三组后续实验（2026-07-17 晚）。decoder 预热/交替/梯度缩放三种廉价解耦证伪；CBIU MLP-64 多种子（3×3）未过准入线但稳定优于 legacy；emit 容量阶跃 warmup 证伪，坍缩由 hard emit 容量阶跃直接造成，需要连续预算退火。

- [`FLUED_V3_4_CBIU_THREE_ROUND_RESULTS_20260717_CN.md`](FLUED_V3_4_CBIU_THREE_ROUND_RESULTS_20260717_CN.md)：三轮自主实验的正式结论。CBIU 已能改善 extra-readout 动作排序并真实减少主干 latent，但当前校准不足以接管 boundary/memory；包含三轮表格、完整曲线、运行时拆分和 K 盘归档索引。
- [`FLUED_V3_4_ARCHITECTURE_GRADIENT_DECISION_TRACE_CN.md`](FLUED_V3_4_ARCHITECTURE_GRADIENT_DECISION_TRACE_CN.md)：v0.4-v3.4 架构谱系、当前数据流、梯度路由、位置/小 AR 决策和关键代码索引。
- [`FLUED_V3_4_CBIU_DESIGN_AND_VALIDATION_PLAN_20260716_CN.md`](FLUED_V3_4_CBIU_DESIGN_AND_VALIDATION_PLAN_20260716_CN.md)：反事实字节接口效用（CBIU）的统一监督定义、数学约束、历史边界、反作弊要求和 V0-V4 验证决策链。emit 已完成在线验证，其他结构仍按准入门槛推进。
- [`FLUED_V3_4_CBIU_V0_RESULTS_20260716_CN.md`](FLUED_V3_4_CBIU_V0_RESULTS_20260716_CN.md)：冻结 20K 检查点的首轮 CBIU 结果。三风险锚点有效；small AR 和部分 extra readout 显示稳定正效用；memory 的直接效用弱，且过去的结论混入了 memory→emit 容量中介效应。

该追踪文档现额外记录 ELF、BLT、ByteFlow、DiffusionGemma 与 H-Net 的可核验借鉴边界，以及 CTM-OCR 中“高重建不等于下游接口友好”的工程触发证据。

当前实证状态：固定置信度阈值边界、readout 级补全、RoPE/ALiBi + 小 AR、普通 byte lookup
和硬 readout emit 已获得修正版实验支持。2026-07-16 长程归因进一步表明：hard emit 在动态边界
接管前已造成容量坍缩，边界接管随后带来第二次梯度冲击；单纯延长到 40K 或缩短过渡无法恢复。
memory 使用率权重 0.05 在当前单种子小模型上首次改善率失真前沿，但其 gate 强度不等于真实
内容依赖，仍需多种子和长上下文确认。共享近似逆 decoder 的首要问题仍是联合训练非平稳。

> **最新两组归因矩阵：** [`FLUED_V3_4_ATTRIBUTION_MATRICES_RESULTS_20260716_CN.md`](FLUED_V3_4_ATTRIBUTION_MATRICES_RESULTS_20260716_CN.md)。该文包含 2×40K 边界课程、3×20K memory 使用率、严格 no-memory 对照、修复后的 memory 干预与下一步架构决策。

> **硬盘迁移后实验总报告：** [`FLUED_V3_4_POST_MIGRATION_EXPERIMENTS_20260715_CN.md`](FLUED_V3_4_POST_MIGRATION_EXPERIMENTS_20260715_CN.md)。该文汇总 2026-07-15 完成的边界闭环、核心路径、位置/小 AR、编码率、lookup/emit、memory 与 decoder 长程实验，是当前结论的最高优先级入口。

> **先读自查报告：** [`FLUED_V3_4_FULL_SELF_AUDIT_20260714_CN.md`](FLUED_V3_4_FULL_SELF_AUDIT_20260714_CN.md)。它区分当前代码事实、历史实验观察和仍未落地的设计，不再把逐位置 L2 能量代理、单次噪声训练或独立 decoder 骨架表述成完整边际编码率、扩散模型和 tied-inverse decoder。
>
> **自查后的正式修正：** [`FLUED_V3_4_CORE_CORRECTION_AND_RERUN_20260714_CN.md`](FLUED_V3_4_CORE_CORRECTION_AND_RERUN_20260714_CN.md)。one-shot 路线保留；默认改为置信度阈值切分、对角边际编码率训练信号、readout 级补全和共享 interpreter 权重的近似逆 decoder。

建议阅读顺序：

1. [`FLUED_V3_4_FOLLOWUP_EXPERIMENTS_20260717_CN.md`](FLUED_V3_4_FOLLOWUP_EXPERIMENTS_20260717_CN.md)
2. [`FLUED_V3_4_CBIU_THREE_ROUND_RESULTS_20260717_CN.md`](FLUED_V3_4_CBIU_THREE_ROUND_RESULTS_20260717_CN.md)
3. [`FLUED_V3_4_CBIU_DESIGN_AND_VALIDATION_PLAN_20260716_CN.md`](FLUED_V3_4_CBIU_DESIGN_AND_VALIDATION_PLAN_20260716_CN.md)
4. [`FLUED_V3_4_CBIU_V0_RESULTS_20260716_CN.md`](FLUED_V3_4_CBIU_V0_RESULTS_20260716_CN.md)
5. [`FLUED_V3_4_ATTRIBUTION_MATRICES_RESULTS_20260716_CN.md`](FLUED_V3_4_ATTRIBUTION_MATRICES_RESULTS_20260716_CN.md)
6. [`FLUED_V3_4_POST_MIGRATION_EXPERIMENTS_20260715_CN.md`](FLUED_V3_4_POST_MIGRATION_EXPERIMENTS_20260715_CN.md)
7. [`FLUED_V3_4_ATTRIBUTION_PLAN_20260715_CN.md`](FLUED_V3_4_ATTRIBUTION_PLAN_20260715_CN.md)
8. [`FLUED_V3_4_MINIMAL_ATTRIBUTION_RESULTS_20260715_CN.md`](FLUED_V3_4_MINIMAL_ATTRIBUTION_RESULTS_20260715_CN.md)
9. [`FLUED_V3_4_RERUN_MATRIX_20260714_CN.md`](FLUED_V3_4_RERUN_MATRIX_20260714_CN.md)
10. [`FLUED_V3_4_FULL_SELF_AUDIT_20260714_CN.md`](FLUED_V3_4_FULL_SELF_AUDIT_20260714_CN.md)
11. [`FLUED_V3_4_CORE_CORRECTION_AND_RERUN_20260714_CN.md`](FLUED_V3_4_CORE_CORRECTION_AND_RERUN_20260714_CN.md)
12. [`FLUED_V3_4_IMPLEMENTATION_BASELINE_CN.md`](FLUED_V3_4_IMPLEMENTATION_BASELINE_CN.md)
13. [`FLUED_V3_4_RATE_EMIT_CORRECTION_20260710_CN.md`](FLUED_V3_4_RATE_EMIT_CORRECTION_20260710_CN.md)
14. [`FLUED_V3_4_5K_ABLATION_ANALYSIS_20260711_CN.md`](FLUED_V3_4_5K_ABLATION_ANALYSIS_20260711_CN.md)
15. [`FLUED_V3_4_20K_RATE_CURRICULUM_ANALYSIS_20260711_CN.md`](FLUED_V3_4_20K_RATE_CURRICULUM_ANALYSIS_20260711_CN.md)
16. [`../../../results/v3.4/20k_rate_comparison/`](../../../results/v3.4/20k_rate_comparison/)
17. [`FLUED_V3_4_PROGRESSIVE_MEMORY_ROI_20260712_CN.md`](FLUED_V3_4_PROGRESSIVE_MEMORY_ROI_20260712_CN.md)
18. [`FLUED_V3_4_BOUNDARY_ROI_PROTOCOL_CN.md`](FLUED_V3_4_BOUNDARY_ROI_PROTOCOL_CN.md)
19. [`../../../results/v3.4/progressive_memory_20k/`](../../../results/v3.4/progressive_memory_20k/)
20. [`FLUED_V3_4_GLOBAL_PATH_CORRECTION_AND_RERUN_20260712_CN.md`](FLUED_V3_4_GLOBAL_PATH_CORRECTION_AND_RERUN_20260712_CN.md)
21. [历史位置/memory 纠偏结果](../../../results/v3.4/position_memory_rerun_20260712/)
22. [`FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md`](FLUED_V3_4_MEMORY_POSITION_20K_ANALYSIS_20260713_CN.md)
23. [`../../../results/v3.4/memory_position_20k_20260713/`](../../../results/v3.4/memory_position_20k_20260713/)

`FLUED_V3_4_ABLATION_RESULTS_20260710_CN.md` 是早期 1K pilot，7 月 10--13 日文档是历史筛选；涉及位置、lookup、emit、memory、decoder 和编码率默认路径时，以 2026-07-15 硬盘迁移后实验总报告为准。
