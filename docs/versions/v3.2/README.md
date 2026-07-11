# FLUED v3.2 / v3.2.1

- [`FLUED_V3_2_DESIGN_SUPPLEMENT_CN.md`](FLUED_V3_2_DESIGN_SUPPLEMENT_CN.md)：设计规范。
- [`FLUED_V3_2_EXECUTION_GOALS_CN.md`](FLUED_V3_2_EXECUTION_GOALS_CN.md)：迁移、训练和验收记录。

这一阶段最重要的结果是 strict masked-source：先在原始 byte 输入上 mask，再让 FLUED 和 paired backbone 处理，消除了 clean readout 侧漏。
