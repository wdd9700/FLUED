# FLUED v3 全新评估表重评估

日期：2026-07-03

这份文档回应一个具体问题：不能只按旧路径重算旧结论，而要把新的评估表真正套到历史 checkpoint 上，重新判断 FLUED v3 系列的有效信号、伪信号和缺口。

## 1. 本轮做了什么

新增/更新脚本：

```text
tools/eval/eval_v3_checkpoint_probe_suite.py
```

这个脚本现在按五类指标重新评估 checkpoint：

```text
1. Codec:
   masked_recon_acc / CE
   keep_recon_acc / CE
   length_acc

2. Backbone:
   byte baseline CE / acc
   latent CE / acc
   Delta_CE / Delta_acc

3. Latent:
   online-MDL probe
   control-task selectivity
   CKA
   perturbation stability
   clean-oracle gap

4. Memory:
   full-zero/shuffle/stale Delta_CE
   memory probe online-MDL
   memory-readout Recall@k / InfoNCE
   patching causal effect
   entity/code/CJK stress cases

5. Boundary:
   F1 / tolerance-F1
   Pk / WindowDiff
   ECE / Brier
   boundary utility gap
   perturbation stability
```

其中 Backbone 项已经补齐为独立 sweep：对代表性历史 checkpoint 分别训练同构小主干，并统一采用 strict masked-source 输入。也就是说，byte mask 发生在 FLUED encoder 接触输入之前，clean byte 只作为 target，不允许先 clean encode 再遮 readout。

## 2. 输出位置

代表性 checkpoint 细评：

```text
K:\FLUED_archive\v3_checkpoint_audit_20260703\full_metric_table_representative
```

全量 latest checkpoint 轻量评估：

```text
K:\FLUED_archive\v3_checkpoint_audit_20260703\full_metric_table_latest
```

Backbone strict masked-source paired sweep：

```text
K:\FLUED_archive\v3_strict_backbone_full_table_20260703
```

主要文件：

```text
probe_suite.csv
probe_suite.json
probe_suite.md
strict_backbone_full_table.csv
strict_backbone_full_table.json
strict_backbone_full_table.md
```

全量 latest 覆盖：

```text
34 checkpoints
33 completed
1 error: stage1_seed_1k
```

`stage1_seed_1k` 失败原因是早期 V32 保存参数和当前 V32 构造参数不一致，缺少 memory 模块权重，已记录在 CSV/JSON 中，不参与排序。

## 3. 新评估表的关键结果

### 3.1 Codec：v3.2.1 masked 系列明显领先

按 `codec_mask_acc` 排名：

| rank | checkpoint | codec_mask_acc |
| ---: | --- | ---: |
| 1 | v321_mfl_nomemory_masked_15k | 0.1659 |
| 2 | v321_mfl_memory_masked_15k | 0.1603 |
| 3 | v321_mfl_nomemory_masked_3k | 0.1393 |
| 4 | v321_mfl_memory_masked_3k | 0.1309 |
| 5 | v321_mfl_random_masked_3k | 0.1222 |

按 `codec_mask_CE` 越低越好：

| rank | checkpoint | codec_mask_CE |
| ---: | --- | ---: |
| 1 | v321_mfl_memory_masked_15k | 3.3347 |
| 2 | v321_mfl_nomemory_masked_15k | 3.3593 |
| 3 | v321_mfl_nomemory_masked_3k | 3.4140 |
| 4 | v321_mfl_random_masked_3k | 3.4363 |
| 5 | v321_mfl_memory_masked_3k | 3.4416 |

结论：

```text
masked-source codec training 是实质性改进，不只是旧指标下看起来更好。
```

它在 strict masked-source 直接解码上显著超过 v3.1/v3.2 clean codec 旧路线。

### 3.2 Length：v3.2.1 masked 系列解决了旧版本长度预测弱的问题

按 `length_acc` 排名：

| rank | checkpoint | length_acc |
| ---: | --- | ---: |
| 1 | v321_mfl_memory_masked_15k | 0.8719 |
| 2 | v321_mfl_nomemory_masked_15k | 0.8388 |
| 3 | v321_mfl_memory_masked_3k | 0.7852 |
| 4 | v321_mfl_nomemory_masked_3k | 0.7722 |
| 5 | v321_mfl_random_masked_3k | 0.7625 |

旧 clean codec 大多在 0.1-0.3 区间。

结论：

```text
v3.2.1 masked objective 不只是提高 masked byte acc，也明显提高 segment length 可预测性。
```

这说明它更接近真正 codec，而不是只靠局部分类。

### 3.3 Latent：信息可提取性强，但“语义质量”仍不能直接证明

按 latent type selectivity 排名：

| rank | checkpoint | latent_type_selectivity_bits |
| ---: | --- | ---: |
| 1 | stage3_v32_mfl_memory_10k | 3.2846 |
| 2 | stage3_v32_mean_random_10k | 3.2416 |
| 3 | codec_40k_utf8clean | 3.1625 |
| 4 | v321_mfl_nomemory_masked_15k | 3.1538 |
| 5 | stage2_causal_memory_10k | 3.0820 |

按 latent length selectivity 排名：

| rank | checkpoint | latent_length_selectivity_bits |
| ---: | --- | ---: |
| 1 | v321_mfl_memory_masked_15k | 5.7312 |
| 2 | codec_40k_utf8clean | 5.2678 |
| 3 | v321_mfl_nomemory_masked_15k | 5.2439 |
| 4 | stage2_causal_nomemory_10k | 5.1872 |
| 5 | stage3_v32_mean_random_10k | 4.9539 |

解释：

1. readout latent 中确实包含可被轻量 probe 提取的 byte type / span length 信息。
2. v3.2.1 masked 15k 在 length 信息上很强。
3. type 信息最强的不完全是 v3.2.1，说明 byte type 这类低层信息并不是最终判断语义质量的充分指标。

结论：

```text
latent 不是空表示；它包含结构信息。
但当前 probe 主要证明“形式/类型/长度信息可提取”，还没有证明高层语义表示质量。
```

### 3.4 CKA / 扰动稳定性：高稳定不等于好表示

clean-oracle CKA 排名前几名多为旧 speed/short 实验：

| checkpoint | clean_CKA |
| --- | ---: |
| speed_b192_w8 | 0.8856 |
| speed_b64_w0 | 0.8853 |
| speed_b256_w4 | 0.8755 |
| speed_b128_w4 | 0.8524 |

这些 checkpoint 的 codec masked acc 并不强。

结论：

```text
CKA / perturbation stability 不能单独作为好坏指标。
过高稳定性可能只是表示变化不足，而不是语义更好。
```

这类指标应该作为诊断，不应该直接变成训练目标。

### 3.5 Memory：新指标明确不支持“当前 memory 已形成可用语义记忆”

memory type selectivity 有正值：

| checkpoint | memory_type_selectivity_bits |
| --- | ---: |
| v321_mfl_memory_masked_3k | 2.7785 |
| v321_mfl_random_masked_3k | 2.6776 |
| stage3_v32_mfl_memory_10k | 2.5998 |
| stage3_v32_mean_random_10k | 2.4690 |
| v321_mfl_memory_masked_15k | 2.4207 |

但 memory-readout retrieval 很差：

```text
best mem_R@5 ≈ 0.0039
```

普通 batch 的 memory patch CE：

```text
ablation_memory_zero_loss_delta = 0.0 for all memory-enabled latest runs
```

解释：

1. memory slot 里能被 probe 读出一些低层类型信息。
2. 但 memory slot 和 readout 没形成可检索对应关系。
3. 普通 masked-source batch 上，把 memory 置零不改变 masked CE。
4. 因此 memory 目前更像旁路激活，而不是稳定有用的内部语义记忆。

结合之前 strict memory stress：

```text
stress case 里 memory path 是活的；
普通 batch 里 memory 没有通用因果收益。
```

新结论：

```text
当前 memory 不应进入默认主线。
它只保留为长程 / 代码 / 实体密集任务的实验分支。
```

### 3.6 Boundary：v3.2.1 masked 系列的边界质量也更强

boundary F1 前几名：

| checkpoint | boundary_F1 |
| --- | ---: |
| v321_mfl_memory_masked_15k | 0.8712 |
| v321_mfl_nomemory_masked_15k | 0.8670 |
| stage2_causal_memory_10k | 0.8566 |
| stage3_v32_mean_random_10k | 0.8545 |

Pk 最低，即越好：

| checkpoint | Pk |
| --- | ---: |
| v321_mfl_nomemory_masked_15k | 0.1178 |
| v321_mfl_memory_masked_3k | 0.1188 |
| codec_40k_utf8clean | 0.1199 |
| v321_mfl_random_masked_3k | 0.1205 |

boundary utility gap：

| checkpoint | utility_gap |
| --- | ---: |
| codec_40k_utf8clean | 2.1368 |
| v321_mfl_nomemory_masked_15k | 1.5251 |
| v321_mfl_memory_masked_15k | 1.1927 |
| codec_10k_pool_mfl | 1.1646 |

解释：

1. v3.2.1 masked 系列不只是重建强，boundary F1 / Pk 也在前列。
2. boundary utility gap 为正，说明 learned boundary 比随机等量切分更有用。
3. 但 `codec_40k_utf8clean` 的 utility gap 仍最高，说明 clean codec 的边界可能在局部重建上仍有优势。

新结论：

```text
boundary 不是死组件。
v3.2.1 masked training 没有破坏 boundary，反而整体改善了边界质量。
```

不过，当前 boundary 仍是 against weak label 的评估，不能直接宣称语义边界已经正确。

## 4. 重新得到的总判断

### 4.1 哪条路线真正变强了

```text
v3.2.1 masked-source codec 是当前主线。
```

原因不是单一 reconstruction，而是多项指标共同支持：

1. masked byte acc 最高。
2. masked CE 最低。
3. length acc 显著领先。
4. latent length 信息可提取性强。
5. boundary F1 / Pk / utility gap 均在前列。
6. strict backbone paired sweep 显示，只有 v3.2.1 masked-source latent 明显强于 byte baseline。

### 4.2 哪个旧判断被推翻了

旧判断：

```text
memory 明显更有用。
```

新判断：

```text
memory path 是活的，但当前 memory 没有形成通用有效的语义记忆。
```

原因：

1. memory probe 有信息，但 memory-readout Recall@5 近乎 0。
2. 普通 batch zero-memory patch CE = 0。
3. strict backbone paired sweep 中 memory 不优于 no-memory。
4. stress case 只能证明局部功能，不能证明通用收益。

### 4.3 哪个新问题被发现了

新发现：

```text
高 CKA / 高扰动稳定性不一定对应好 codec。
```

旧 speed 系列 checkpoint 的 CKA 很高，但 masked acc 不强。说明表示稳定性可能来自表示变化不足，而不是语义质量更好。

因此：

```text
CKA / stability 只能作为诊断指标，不适合单独作为训练监督。
```

### 4.4 当前最有价值的监督候选

可以考虑进入训练监督的，不是 memory MSE，而是：

```text
1. masked-source codec objective
2. length / span consistency
3. boundary utility / calibration 辅助
4. readout 的可探针性约束，谨慎低权重
```

暂时不建议加入：

```text
1. clean/corrupt latent MSE
2. memory-readout 强行相似
3. memory 直接预测 next byte
4. CKA/stability 直接作为 loss
```

原因是这些目标容易让模型走捷径，或者偏离语言编码器定位。

## 5. Backbone：paired sweep 已补齐

这一步不是 checkpoint forward，而是重新训练 paired backbone：

```text
同一语料
同一 byte/span mask
同一 strict masked-source 输入
同一 2 层小 Transformer 主干
同一 3000 step 训练预算
```

输出位置：

```text
K:\FLUED_archive\v3_strict_backbone_full_table_20260703
```

结果：

| rank | run | family | memory | mask_acc | delta_acc | byte_CE | delta_CE |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | v321_mfl_nomemory_masked15k | v32 | false | 0.1898 | +0.0458 | 3.1424 | +0.2358 |
| 2 | v321_mfl_memory_masked15k | v32 | true | 0.1897 | +0.0457 | 3.1473 | +0.2308 |
| 3 | v31_codec10k_pool_mfl | v31 | false | 0.1468 | +0.0028 | 3.4551 | -0.0770 |
| 4 | v31_codec40k_utf8clean | v31 | false | 0.1463 | +0.0023 | 3.4753 | -0.0972 |
| 5 | v32_stage3_mfl_memory_10k | v32 | true | 0.1458 | +0.0018 | 3.4666 | -0.0885 |
| 6 | v32_stage3_mfl_nomemory_10k | v32 | false | 0.1449 | +0.0009 | 3.4552 | -0.0771 |
| baseline | byte_3k_strict_mask | byte | false | 0.1440 | 0.0000 | 3.3782 | 0.0000 |

解释：

```text
v3.2.1 masked-source codec 是唯一显著降低小主干补全难度的路线。
```

它相对 byte baseline 的 masked-byte accuracy 提升约 4.58 个百分点，byte CE 降低约 0.236。v3.1/v3.2 clean codec 虽然 acc 略高于 byte baseline，但 CE 更差，说明它们没有稳定降低主干学习难度，只是命中率有轻微随机/分布优势。

Memory 结论：

```text
memory-enabled v3.2.1 与 no-memory 几乎打平，但没有超过 no-memory。
```

因此，当前不能继续说“memory 明显更有用”。更准确的说法是：memory 路径可运行，局部 stress case 有效，但普通 masked-source backbone 任务下尚未形成通用收益。

## 6. 当前决策

默认主线：

```text
v3.2.1 no-memory masked-source codec
```

保留分支：

```text
v3.2.1 top-k memory
```

但 memory 分支必须用以下指标重新证明自己：

```text
memory-readout Recall@k 明显提升；
zero/shuffle/stale patch CE 有正 delta；
entity/code/CJK stress 不只是局部样本有效；
strict backbone Delta_CE / Delta_acc 超过 no-memory。
```

在这些成立之前，memory 不参与默认主线训练监督。
