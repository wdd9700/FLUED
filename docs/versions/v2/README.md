# FLUED v2

v2 引入 328M tied Transformer、SwiGLU、类型边界先验和混合 clean/denoising 重建。

- [`FLUED_REBUILD.md`](FLUED_REBUILD.md)：架构和训练重建记录。
- [`../../../results/v2/`](../../../results/v2/)：三种子、消融和公平 D1 汇总。

结论：重建和边界学习稳定，但去噪、压缩控制和下游 BPB 之间存在难调的微分动力学冲突。
