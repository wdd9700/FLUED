# FLUED v3 系列 checkpoint 重评估结论

日期：2026-07-03

本轮目标不是继续调训练，而是先把已有 v3 / v3.1 / v3.2 / v3.2.1 checkpoint 的证据重新整理，明确哪些结论来自严格评估，哪些只是历史机制验证。

## 1. 评估范围

归档根目录：

```text
K:\FLUED_archive
```

新生成的审计目录：

```text
K:\FLUED_archive\v3_checkpoint_audit_20260703
```

输出文件：

```text
K:\FLUED_archive\v3_checkpoint_audit_20260703\runs.csv
K:\FLUED_archive\v3_checkpoint_audit_20260703\runs.json
K:\FLUED_archive\v3_checkpoint_audit_20260703\checkpoint_evidence_audit.md
K:\FLUED_archive\v3_checkpoint_audit_20260703\probe_suite\probe_suite.csv
K:\FLUED_archive\v3_checkpoint_audit_20260703\probe_suite\probe_suite.json
K:\FLUED_archive\v3_checkpoint_audit_20260703\probe_suite\probe_suite.md
```

覆盖：

| family | run_count |
| --- | ---: |
| v3 | 114 |
| v3.1 | 65 |
| v3.2 | 29 |
| v3.2.1 | 9 |

其中 217 个 `summary.json` 被统一归档分析。实际 forward probe 覆盖了 7 个代表性 codec checkpoint：

1. v3.1 `codec_40k_utf8clean`
2. v3.1 `codec_10k_pool_mfl`
3. v3.2 `stage3_v32_mfl_nomemory_10k`
4. v3.2 `stage3_v32_mfl_memory_10k`
5. v3.2 `stage3_v32_mfl_random_10k`
6. v3.2.1 `v321_mfl_nomemory_masked_15k`
7. v3.2.1 `v321_mfl_memory_masked_15k`

早期 v3 commit-controller / segmental diffusion / segmental workspace checkpoint 架构和任务目标不同，当前只纳入日志层面，不做同口径性能排名。

## 2. 统一口径

本轮分四类证据：

### 2.1 Clean codec

输入是完整 clean byte，目标是重建原始 byte。

该口径能证明 codec 训练能否收敛，但不能证明 masked-source 场景下的语言编码器质量。

### 2.2 Strict masked-source codec

先在原始 byte 输入上做 byte/span mask，然后 FLUED 只能看到 masked source。

目标：

```text
masked_source -> FLUED encoder -> readout latent -> frozen decoder -> clean masked bytes
```

该口径避免 clean readout / clean segment 侧漏，是当前主线最可信评估。

### 2.3 Strict masked-source backbone

同样先在 byte 输入上 mask。FLUED 只能看 masked source，外部小 backbone 只消费 readout latent。

目标：

```text
masked_source -> FLUED readout -> small backbone -> frozen decoder -> clean masked bytes
```

主指标是：

```text
Delta_acc = acc(latent backbone) - acc(byte baseline)
Delta_loss = loss(byte baseline) - loss(latent backbone)
```

### 2.4 Probe suite

新加的实际诊断，不参与训练。

指标包括：

```text
direct_mask_acc:
  不接外部 backbone，只用 masked-source readout 直接解 masked byte。

masked_readout_cos:
  masked-source readout 与 clean-source oracle readout 的余弦相似度。

boundary_f1 / boundary_ece:
  boundary hard prediction 和 weak boundary label 的 F1 / 校准误差。

memory_readout_cos:
  retrieved memory context 与当前 readout 的余弦相似度。

ablation_memory_zero_loss_delta:
  zero memory 后 masked byte loss - full memory masked byte loss。
  正值表示 memory 对当前 masked reconstruction 有帮助。
```

## 3. 核心结果

### 3.1 strict masked-source backbone

| run | mask_acc | mask_loss | keep_acc | codec_memory |
| --- | ---: | ---: | ---: | --- |
| old byte baseline | 0.1472 | 3.3684 | - | - |
| old v3.2 no-memory | 0.1451 | 3.4551 | 0.5215 | false |
| old v3.2 top-k memory | 0.1458 | 3.4667 | 0.5460 | true |
| v3.2.1 byte baseline | 0.1440 | 3.3785 | - | - |
| v3.2.1 no-memory | 0.1898 | 3.1423 | 0.5436 | false |
| v3.2.1 top-k memory | 0.1895 | 3.1473 | 0.5448 | true |

结论：

```text
v3.2.1 no-memory latent 相比 byte baseline:
  mask_acc +0.0458

v3.2.1 top-k memory 相比 no-memory:
  mask_acc -0.0003
```

因此，当前严格口径下可以确认：

1. masked-source codec training 让 latent 明显降低了 backbone 补全难度。
2. 当前 top-k memory 没有带来 backbone 主指标收益。

### 3.2 strict masked-source codec

| run | steps | masked_recon_acc | keep_recon_acc | length_acc | boundary_acc | retrieval_entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3.2.1 memory 3k | 3000 | 0.1581 | 0.3612 | 0.8799 | 0.9409 | 1.2630 |
| v3.2.1 no-memory 3k | 3000 | 0.1575 | 0.3653 | 0.8788 | 0.9409 | 0.0000 |
| v3.2.1 random 3k | 3000 | 0.1576 | 0.3601 | 0.8777 | 0.9408 | 1.3317 |
| v3.2.1 memory 15k | 15000 | 0.1910 | 0.5744 | 0.9316 | 0.9439 | 1.1726 |
| v3.2.1 no-memory 15k | 15000 | 0.1900 | 0.5751 | 0.9250 | 0.9446 | 0.0000 |

结论：

```text
15k memory - no-memory masked_recon_acc = +0.0010
15k memory - no-memory keep_recon_acc   = -0.0007
```

memory 有极小 masked reconstruction 提升，但 keep reconstruction 略降。这个量级不足以支持“memory 是默认主线收益”。

### 3.3 probe suite

| checkpoint | direct_mask_acc | direct_mask_loss | masked_readout_cos | keep_readout_cos | boundary_f1 | boundary_ece | memory_readout_cos | zero_memory_loss_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3.1 codec_40k_utf8clean | 0.0334 | 10.9573 | 0.3796 | 0.9966 | 0.8174 | 0.0190 | 0.0152 | - |
| v3.1 codec_10k_pool_mfl | 0.0391 | 10.7671 | 0.3785 | 0.9974 | 0.8134 | 0.0197 | 0.0074 | - |
| v3.2 mfl no-memory 10k | 0.0167 | 13.9006 | 0.4328 | 0.9732 | 0.8047 | 0.0557 | 0.0000 | - |
| v3.2 mfl memory 10k | 0.0201 | 13.1889 | 0.4355 | 0.9870 | 0.8076 | 0.0190 | 0.0204 | 0.0000 |
| v3.2 mfl random 10k | 0.0191 | 12.4481 | 0.4121 | 0.9880 | 0.8262 | 0.0130 | -0.0088 | 0.0000 |
| v3.2.1 no-memory masked 15k | 0.1658 | 3.2960 | 0.3956 | 0.9917 | 0.8608 | 0.0145 | 0.0000 | - |
| v3.2.1 memory masked 15k | 0.1697 | 3.2908 | 0.4051 | 0.9931 | 0.8670 | 0.0128 | -0.0176 | 0.0000 |

关键解释：

1. v3.2.1 masked codec 的 `direct_mask_acc` 从旧 clean codec 的约 0.02-0.04 提升到约 0.166-0.170，说明训练信号位置修正是实质性改进。
2. v3.2.1 memory 的 `masked_readout_cos` 比 no-memory 略高，说明 memory 对 masked-source readout 保真有轻微帮助。
3. 但普通 batch 上 `zero_memory_loss_delta = 0`，说明当前 memory 对普通 masked reconstruction 没有可测因果贡献。
4. `memory_readout_cos` 不稳定，v3.2.1 memory 甚至为负，说明 memory context 与 readout 没有形成清晰同向语义几何。
5. boundary 的 F1 / ECE 在 v3.2.1 上最好，但它仍只是 weak boundary label 质量，不是语义边界质量。

### 3.4 strict memory stress

| file | memory | retrieval | full_acc | full_loss | zero_loss_delta | shuffled_loss_delta | stale_loss_delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| memory_topk | true | topk | 0.0244 | 4.2035 | 0.3181 | 0.1704 | 0.4155 |
| nomemory | false | none | 0.0244 | 4.2092 | -0.0005 | -0.0009 | -0.0001 |

解释：

1. 在专门构造的实体 / 代码标识符 / 版本号 / 中文重复词样本上，top-k memory path 是活的。
2. 但总体 full_acc 与 no-memory 相同，full_loss 只好约 0.0057。
3. 因此 memory stress 只能证明 memory 有局部功能，不能证明它是通用收益。

## 4. 重新得出的结论

### 4.1 当前最强结论

```text
masked-source codec training 是当前唯一通过严格泄露控制后仍明确有效的方向。
```

它带来的不是小改进，而是：

```text
strict backbone mask_acc:
  byte baseline       0.1440
  v3.2.1 no-memory    0.1898
```

这说明 FLUED readout latent 确实能降低外部 backbone 的 masked-byte 补全难度。

### 4.2 对 memory 的新判断

旧结论“memory 明显更有用”需要降级。

更准确的说法是：

```text
memory path 是活的；
memory 在特定 stress 样本上有因果作用；
memory 在当前通用 128-byte masked-source 主任务上没有稳定总体收益；
memory 暂时不应作为默认主线。
```

原因：

1. clean reconstruction 里 memory 可能只是训练依赖，不等于独立模型收益。
2. 同 checkpoint zero/shuffle/stale 退化只能证明模型用过 memory，不证明有 memory 模型强于 no-memory 模型。
3. v3.2.1 的公平 no-memory 对照已经很强，盖住了 memory 的边际贡献。
4. memory-readout 几何没有形成清晰关联。
5. 当前通用短序列任务对 memory 的需求不够强。

### 4.3 对 latent 的新判断

latent 有效，但语义质量还没有被充分证明。

已经证明：

```text
latent > byte baseline on strict backbone
```

尚未证明：

```text
latent 形成了稳定、可解释、语义化的连续表示空间。
```

缺少：

1. probe / MDL probe
2. control task selectivity
3. CKA / RSA 几何分析
4. 扰动稳定性
5. 与外部 backbone hidden state 的结构对齐

### 4.4 对 boundary 的新判断

boundary 比之前更稳，但还不是语义分割证据。

已有：

```text
boundary_f1 约 0.81-0.87
boundary_ece 约 0.013-0.056
```

这只能说明它拟合 weak boundary label 还可以。

还不能说明：

```text
边界符合语义 chunk；
soft confidence 和 hard boundary 的 ROI 有稳定语义对应；
boundary 对下游任务有真实 utility。
```

## 5. 下一步先补评估，不先加 loss

当前不建议马上把新信号塞进训练监督。顺序应该是：

### 5.1 latent 评估

先做：

```text
MDL probe / control probe:
  byte type
  UTF-8 start / continuation
  chunk 内相对位置
  span length
  punctuation / whitespace
  digit / version / entity / code identifier marker

geometry:
  effective rank
  CKA / RSA
  masked-source vs clean-oracle gap
  轻微扰动稳定性
```

只有当 probe 证明 latent 缺少某类信息，才考虑把对应信号加入辅助监督。

### 5.2 memory 评估

先做：

```text
memory slot probe:
  从 memory 预测当前 chunk 的实体 / 变量 / 主题 / 数字串摘要。

memory-readout retrieval:
  正样本 = 后续引用同一实体或变量的 readout。
  负样本 = 无关 chunk readout。
  指标 = Recall@k / AUC / InfoNCE。

causal patching:
  正确 memory patch 到错误样本，masked loss 应下降。
  错误 memory patch 到正确样本，masked loss 应上升。
```

只有这些成立，才考虑 memory contrastive / ranking loss。

### 5.3 boundary 评估

先做：

```text
tolerance F1:
  允许 +/-1 或 +/-2 byte 近邻边界。

WindowDiff / Pk:
  看分割序列质量，而不是逐点 accuracy。

ECE / Brier:
  评估 boundary confidence 是否校准。

boundary utility:
  learned segmentation vs random length-matched segmentation
  看 codec / backbone loss 差值。
```

只有 boundary utility 明确为正，才考虑加更强 boundary loss。

## 6. 暂不建议加入训练的信号

暂不加入：

```text
clean/corrupt latent MSE
```

原因：历史 v2 已证明它会压倒训练，破坏 boundary / memory 分化。

暂不加入：

```text
让 memory 直接预测 next byte
```

原因：这会把 FLUED 推向无 codec 的字节级语言模型，偏离语言编码器定位。

暂不加入：

```text
动态 memory gate / overflow slots
```

原因：当前固定 memory slots 还没有证明通用收益，继续加复杂度会掩盖根因。

## 7. 当前路线决策

默认主线：

```text
no-memory masked-source codec
```

实验分支：

```text
top-k memory / chunk memory sequence
```

memory 分支继续的条件：

1. 长序列 / 代码 / 实体密集任务上超过 no-memory。
2. memory slot probe 证明 memory 存了有用摘要。
3. causal patching 证明正确 memory 对 masked reconstruction/backbone infill 有因果贡献。
4. memory-readout retrieval 证明 memory 和后续引用有结构对应。

如果这些条件不成立，memory 不应进入默认 v3.2 主线。

