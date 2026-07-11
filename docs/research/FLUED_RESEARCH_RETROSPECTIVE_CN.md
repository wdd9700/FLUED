# FLUED v1 到 v3.4 研究回顾

本文档用于公开说明 FLUED 的研究过程、失败路径、纠偏过程和当前可辩护 claim。它不是宣传稿，而是研究归档。

> 版本说明：正文第 1-8 节冻结了截至 v3.3 的历史判断；文末 v3.4 补记是当前实现和实验口径。若两者冲突，以 v3.4 5K 消融和仓库根目录 README 为准。

## 1. 起点问题

FLUED 的起点不是“做一个新 tokenizer”，而是：

```text
如何把字节流翻译为潜空间表示，使其符合语义、带有位置信息、可被还原为字节流，并降低外部 backbone 的学习难度？
```

传统 BPE/WordPiece 路线把语言预处理成离散分类问题。它工程上有效，但语义先验被固化在 token embedding 中，局部语境变化时容易产生不可逆的早期决策。FLUED 尝试把这一步改成连续的、可训练的、可还原的语言编码接口。

## 2. v1：软切分自编码器

v1 的核心是 soft boundary autoencoder：

```text
byte stream -> soft segmentation -> latent units -> tied inverse decoder -> bytes
```

更准确地说，v1 是 FLUED 的最小假设验证：

```text
如果 reconstruction loss 能把 boundary head 训练出非均匀、类型相关的边界分布，
并且 latent units 仍可被 tied decoder 还原为 bytes，
那么“可微 byte boundary 可以学习”这件事成立。
```

历史 E1v5 结果支持这个最小假设：

| 指标 | 数值 |
| --- | ---: |
| recon_acc | 0.9999 |
| m/n | 0.379 |
| bp_std | 0.443 |
| 压缩倍数 | 约 2.64x |

历史 E3 20K 下游对比中，v1 还出现过一个强阳性 BPB 信号：

| Method | BPB |
| --- | ---: |
| FLUED v1 | 1.2114 |
| BPE | 1.4786 |
| BLT reproduction | 2.6371 |

因此 v1 相比该历史 BPE run 的 BPB 改善约为 18.1%。但这个结果属于历史探索口径，
不能和后续 2048 original-byte / 100K 的公平 D1 矩阵混用。

收获：

1. 可微 byte boundary 可以稳定训练。
2. reconstruction accuracy 很容易做高。
3. 只靠 reconstruction 会让模型走向恒等映射，不足以证明语义建模。

问题：

1. 切分边界的语义质量缺少强评估。
2. 压缩率控制不稳定。
3. 下游 backbone 是否受益没有被直接证明。
4. 短序列还原能力不稳，暴露了固定窗口训练和还原协议的局限。
5. 大量关键参数仍像 magic number，例如 `target_compression=0.225`、`compression_weight=0.02`、`lambda_type=0.05` 等。
6. type prior 不是无害项，历史记录显示 `lambda_type=0.05` 曾导致 operator/digit boundary collapse。

## 3. v2：去噪重建和 328M tied model

v2 做了较大规模的 tied encoder-decoder：

| 项目 | 值 |
| --- | --- |
| 参数量 | 约 328M |
| d_model | 1024 |
| 层数 | 24 |
| 注意力头 | 16 |
| FFN | SwiGLU |
| 词表 | PAD + 256 bytes + MASK |
| 训练 | clean/reconstruction + denoising |

三种子 E1 reconstruction 结果稳定：

| Seed | Eval Acc | m/n | bp_std | CJK bp |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.9991 | 0.470 | 0.407 | 0.122 |
| 123 | 0.9993 | 0.498 | 0.391 | 0.182 |
| 999 | 0.9996 | 0.491 | 0.394 | 0.092 |

去噪比例消融显示，v2 的问题不是“学不会重建”，而是去噪、压缩和边界分化之间存在训练动力学冲突：

| denoise_prob | Eval Acc | m/n |
| ---: | ---: | ---: |
| 0.3 | 0.9987 | 0.496 |
| 0.5 | 0.9992 | 0.495 |
| 0.7 | 0.9991 | 0.470 |
| 0.9 | 0.9997 | 0.527 |

较高去噪比例仍能得到很高重建准确率，但 m/n 明显上升，说明模型倾向于保留更多 latent units，
压缩行为变弱。换言之，去噪能迫使模型利用上下文，但也会和压缩压力发生冲突。

压缩扫描进一步暴露了可塑性问题：

| compression_weight | target_compression | 结果 |
| ---: | ---: | --- |
| 0.30 | 0.20 | acc 0.9992, m/n 0.486 |
| 0.30 | 0.30 | 约 27.7K step NaN |
| 0.30 | 0.45 | acc 0.9995, m/n 0.501 |
| 0.30 | 0.60 | 约 27.1K step NaN |
| 0.25 | 0.30 | acc 0.9992, m/n 0.483 |

这说明 v2 并不是简单“训练得不够”，而是训练信号的位置和比例会改变系统的微分动力学。
当压缩约束过强时，boundary head gradient 会进入不稳定区；当去噪过强时，模型又倾向于降低压缩。

关键纠偏：

1. `latent_consistency_weight=0.03` 的均方误差一致性损失会造成 latent loss 爆炸。
2. 爆炸后 boundary head 被拉向常数解，bp_std 上不去，切分塌缩。
3. 结论是：latent consistency 不应进入主线训练目标。

## 4. v2 D1 下游对比

后续修正了早期不公平对比：BPE 的 `max_seq_len=2048` 不能解释成 2048 原始 byte。最终使用 2048 original bytes、100K steps 做公平对比。

| Method | Context Budget | Steps | BPB |
| --- | --- | ---: | ---: |
| BPE-8K | 2048 original bytes | 100K | 0.8066 |
| BPE-16K | 2048 original bytes | 100K | 0.8165 |
| BPE-32K | 2048 original bytes | 100K | 0.8205 |
| FLUED v2 | 2048 original bytes | 100K | 0.8732 |
| BLT theta=0.3 | 2048 original bytes | 100K | 2.3996 |

结论：

1. FLUED v2 稳定，但没有击败 BPE。
2. 历史上“FLUED v1 接近或优于 BPE”的口径不能直接拿来当当前公平结论。
3. BLT reproduction 弱，不能代表 BLT 本身弱。

## 5. v3 / v3.1：从重建转向语言编码器

v3 的思想转变：

```text
重建不是最终目的；核心是 latent representation 是否让外部 backbone 更容易理解和补全语言。
```

v3.1 引入：

1. 当前语义段 readout latent。
2. 当前语义段 summary memory。
3. 小 backbone 进行 masked completion probe。

早期现象显示 memory 有帮助，但后来发现评估协议不够严格：如果先用 clean 输入产生 readout，再遮掉某些 readout，后续 readout 或 memory 可能已经携带被遮盖信息。

v3.1 language codec 的代表性结果：

| Run | 参数量 | recon_acc | length_acc | boundary_acc | units/byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| codec_10k_pool_mfl | 约 2.01M | 0.6469 | 0.9721 | 0.9353 | 0.1170 |

这说明 v3.1 已经能形成可解释的 ROI 切分和较稳定的长度还原；但它仍不能证明 memory 是主线。
早期 zero/shuffle/stale memory ablation 只能说明模型确实使用过 memory，
不等于 memory 在严格下游补全中提供通用收益。

v3.1 minimal backbone 的早期正信号是：

| Run | mask_acc |
| --- | ---: |
| byte segment-mask baseline | 0.1498 |
| latent_3k_byteaux1 | 0.1784 |

但这批结果包含 clean encode 后再遮 readout/segment 的历史口径风险，因此只能作为机制探索或上界参考。

## 6. v3.2 / v3.2.1：严格 masked-source 纠偏

v3.2.1 的关键修正：

```text
先在 byte 输入层面 mask，再让 FLUED encode。
```

这避免了 clean encode 侧漏。严格 paired-backbone 结果：

| Run | memory | mask_acc | delta_acc vs byte | byte_CE | delta_CE vs byte |
| --- | --- | ---: | ---: | ---: | ---: |
| v3.2.1 no-memory masked 15k | false | 0.1898 | +0.0458 | 3.1424 | +0.2358 |
| v3.2.1 memory masked 15k | true | 0.1897 | +0.0457 | 3.1473 | +0.2308 |
| byte baseline | false | 0.1440 | 0.0000 | 3.3782 | 0.0000 |

结论：

1. v3.2.1 是当前最强的 v3-family evidence。
2. no-memory 已经显著超过 byte baseline。
3. memory 没有在这个结果中给出额外稳定收益，因此不能作为默认 claim。

重要细节：

1. v3.2 初版 strict backbone 没有过关，byte baseline 约 0.1472，而 v3.2 no-memory / memory 约 0.1451 / 0.1458。
2. v3.2.1 的关键修复不是更换 backbone，而是先训练 masked-source codec：从被 mask 的 byte 输入产生 readout，再还原 clean masked bytes。
3. 15K masked-source codec 的 direct masked accuracy 约 0.190，length accuracy 约 0.925-0.932，boundary accuracy 约 0.944，units/byte 约 0.108。
4. 接入 strict backbone 后，no-memory latent readout 从 byte baseline 0.1440 提升到 0.1898，这是当前 v3-family 最干净的正证据。
5. memory path 在部分 stress case 中不是死代码，但通用 strict backbone 没给出稳定额外收益。

## 7. v3.3：公开架构截止点

v3.3 收束为一个 byte-to-latent decision interface：

```text
byte lookup
-> memory-free signed segmentor
-> dual-threshold chunk policy
-> chunk builder
-> memory-conditioned interpreter
-> readout latent + delayed memory write
-> external backbone
-> tied-inverse decoder
```

主要原则：

1. segmentor 不读 memory。
2. interpreter 可以读过去 memory。
3. current chunk memory 不能被 current chunk 自己读到。
4. memory 是分支，不是默认主线。
5. 严格 masked-source 是所有补全评估的底线。
6. reconstruction 是必要指标，但不能单独证明语义质量。

## 8. 当前 claim

可以公开主张：

1. FLUED 是一种 tokenizer-free byte-to-latent language interface 研究路线。
2. v2 证明了可微边界和 tied decoder 的稳定重建能力。
3. v2 公平 D1 对比显示 FLUED 尚未击败 BPE。
4. v3.2.1 证明严格 masked-source 下，FLUED readout latent 可以帮助小 backbone 超过 byte baseline。
5. v3.3 给出了更清晰、可复现、可消融的架构边界。
6. v1 可以作为历史强阳性信号展示，但必须标明它不是当前公平矩阵结论。

不应公开主张：

1. FLUED 已替代 BPE。
2. FLUED 已击败 BLT / H-Net / ByteFlow 等近期 tokenizer-free 系统。
3. memory branch 已经被证明是主线。
4. reconstruction accuracy 等于语义质量。
5. v3.3 已经完成系统实验或已经优于现有 tokenizer-free 系统。

## 9. 证据索引

证据分三层使用，避免把历史探索、附录分析和官网主结论混写。

### 9.1 官网可直接引用

| 阶段 | 实验名 | 关键指标 | 文件路径 | 可信度/注意事项 |
| --- | --- | --- | --- | --- |
| v2 | A-class 三种子稳定性 | eval_acc: 0.9991 / 0.9993 / 0.9996；m/n: 0.470 / 0.498 / 0.491；均值 0.9993 +/- 0.0005 | `<archive-root>\v2_final_seeds\a_class_v2_summary.json`；`<repo-root>\README.md` | 高。支撑“稳定可微 byte-boundary / reconstruction”，不支撑“优于 BPE”。 |
| v2 | D1 2048-byte / 100K 公平比较 | BPE-8K 0.8066 BPB；BPE-16K 0.8165；BPE-32K 0.8205；FLUED-v2 0.8732；BLT theta=0.3 2.3996 | `<archive-root>\cloud_5090_D1_20260610\westc_100k_20260613\...`；`<archive-root>\cloud_5090_D1_20260610\westd_blt_100k_20260614\...`；`<repo-root>\README.md` | 很高。官网应明确“FLUED 稳定但落后 BPE”。 |
| v3.2.1 | strict masked-source backbone | byte baseline mask_acc 0.1440；最佳 latent no-memory 0.1898，delta +0.0458；CE 3.1424，delta_CE +0.2358 | `<archive-root>\v3_strict_backbone_full_table_20260703\strict_backbone_full_table.md`；`<archive-root>\v32_strict_backbone_20260703_masked_codec_15k\strict_backbone_summary.md` | 很高。当前 v3-family 最强证据：严格先 mask byte，再给 FLUED。 |
| v3.3 | 架构与消融入口 | `train_v33.py`、`run_v33_ablation_matrix.*`、`summarize_v33_ablation.py`、`configs/v3_3/v33_ablation_2m.json` | `<repo-root>\docs\versions\v3.3\FLUED_V3_3_ARCHITECTURE_CN.md`；`<repo-root>\docs\versions\v3.3\FLUED_V3_3_ABLATION_INTERFACE_CN.md` | 中高。是当前公开架构端点和复现实验入口，不是已完成结果。 |

### 9.2 研究附录可引用

| 阶段 | 实验名 | 关键指标 | 文件路径 | 可信度/注意事项 |
| --- | --- | --- | --- | --- |
| v2 | AB2 denoise 扫描 | denoise 0.3-0.9 均 >0.9987 acc；更高 denoise 往往 m/n 更高，压缩更弱 | `<repo-root>\results_summary.json`；`<archive-root>\ab2_denoise_20260612` | 高。用于解释 denoising 稳定性和压缩张力。 |
| v2 | AB1 compression 扫描 | weight=0.3 时 tc=0.30/0.60 约 27K step NaN；weight=0.25 tc=0.30 稳定 | `<repo-root>\results_summary.json`；`<archive-root>\ablation_20260611`；`<archive-root>\ab1_weight_0.3` | 高。用于说明 compression control 仍是 blocker。 |
| v3.1 | codec_10k_pool_mfl ROI | PASS；loss 6.3395 -> 0.9667；recon 0.6469；length 0.9721；boundary 0.9353；units/byte 0.1170 | `<archive-root>\v31_language_codec_2m_20260702\sweep_summary.md`；`<archive-root>\v31_language_codec_2m_20260702\codec_10k_pool_mfl\roi_constrained.md` | 中高。适合讲可解释分段，但样本和任务不是 strict masked-source 主证据。 |
| v3 audit | checkpoint evidence audit | 217 个 summary；strict masked backbone 中 v3.2.1 no-memory 0.1898，memory 0.1895；memory 没有默认胜出 | `<archive-root>\v3_checkpoint_audit_20260703\checkpoint_evidence_audit.md` | 高。适合支撑“memory branch 仍是实验分支”。 |
| v3 audit | full metric table | v321 masked 15k codec_mask_acc: no-memory 0.1643，memory 0.1716；boundary_F1 约 0.86；memory R@5 很低 | `<archive-root>\v3_checkpoint_audit_20260703\full_metric_table_representative\probe_suite.md` | 中。指标丰富，但要和 paired strict backbone 分开解读。 |

### 9.3 历史/反例，不宜放主叙事

| 阶段 | 实验名 | 关键指标 | 文件路径 | 可信度/注意事项 |
| --- | --- | --- | --- | --- |
| v1 | E2 重建历史对比 | FLUED ppl 1.39；BPE token-level ppl 1.21；BLT byte-level ppl 1.13 | `<archive-root>\E_checkpoints\E2_COMPARISON_ARCHIVE.md` | 中。BPE 是 token-level CE，不能和 byte-level 直接比。 |
| v1 | E3 20K 历史对比 | FLUED 1.2114 BPB；BPE 1.4786；BLT 2.6371 | `<archive-root>\E_checkpoints\E2_COMPARISON_ARCHIVE.md` | 中。旧 fixed-token / 20K 口径，不应覆盖 v2 D1 2048-byte / 100K 公平矩阵。 |
| v3/v3.1 | 早期 memory-positive clean tests | clean codec、legacy backbone 中 memory 有时看起来有收益 | `<archive-root>\v3_checkpoint_audit_20260703\checkpoint_evidence_audit.md` | 低到中。审计结论已指出多为 clean reconstruction 或 legacy objective，不是 leakage-safe claim。 |

官网最干净路线：

```text
v2 稳定边界学习
-> v2 公平 D1 仍落后 BPE
-> v3.2.1 strict masked-source latent 确实降低小 backbone 补全难度
-> v3.3 作为下一步架构/消融接口
```

不要把 v1 的 FLUED > BPE 历史 E3 结果和 v2 D1 公平矩阵混写。

## 10. 下一步研究路线

优先级：

1. 使用 v3.3 2M/128 核心矩阵验证接口、指标和 memory/no-memory 分歧。
2. 扩展到 6M/512，观察 scaling 是否改善 ROI 形状和 backbone gain。
3. 对 memory 做 zero/shuffle/stale/patching causal effect 分析。
4. 将 readout rate 从固定 target 改为更自适应的编码成本压力。
5. 探索 diffusion backbone，但必须保证 1-step 或少步数版本有工程意义。

研究底线：

```text
任何“帮助 backbone”的 claim，都必须在 byte 输入层面先 mask，且必须和 byte baseline / no-memory branch 做公平对比。
```

## 11. v3.4 补记：并行 memory 与真实计算门控

v3.4 将 v3.3 的串行 memory 路径改为并行的逐 chunk 总结：每个 memory 只读取本 chunk 的字节，interpreter 同时读取其他 chunk 的 memory，但屏蔽当前 chunk 的 memory。decoder 不读取 memory，只反转 byte-to-readout 翻译。

同时引入两级容量决策：

```text
边际编码率 -> 决定 chunk 边界
emit controller -> 决定每个 chunk 的 1-16 个 readout 中哪些真正进入 backbone
```

37M FLUED、5K 步、单种子结构筛选的主要结果：

| 方案 | 重建准确率 | masked completion | 实际 latent/byte |
| --- | ---: | ---: | ---: |
| 精确边际编码率完整结构 | 0.5970 | 0.1343 | 0.7852 |
| L2 边际编码率 | **0.7041** | **0.1477** | **0.6804** |
| 均匀边界 | 0.9873 | 0.1485 | 0.9694 |
| 软 emit、不实际压缩 | 0.6795 | 0.1334 | 1.0752 |

当前结论不是“v3.4 已经解决切分”，而是：L2 边际编码率、RoPE、小 AR 修正头、结构化 byte lookup 和硬 emit 是下一轮最值得保留的组合；memory 仍是效果/成本分支；固定计算成本权重仍不能可靠控制真实 latent 数量。

完整证据见 [v3.4 5K 消融分析](../versions/v3.4/FLUED_V3_4_5K_ABLATION_ANALYSIS_20260711_CN.md) 和 [公开日志](../../results/v3.4/5k_ablation/README.md)。
