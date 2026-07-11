# FLUED v3.3 消融接口说明

本文档说明 v3.3 公开代码的训练参数、矩阵消融入口和验收指标。目标不是一次性证明 FLUED 已经优于 BPE，而是让后续研究可以直接复现实验轴，并快速判断某个组件是否值得保留。

## 1. 入口文件

```text
tools/train/train_v33.py                    单次训练 / 评估入口
tools/launcher/run_v33_ablation_matrix.py   JSON 矩阵消融 runner
tools/launcher/run_v33_ablation_matrix.ps1  Windows PowerShell 包装
tools/analysis/summarize_v33_ablation.py    汇总 summary.json 为表格
configs/v33_ablation_2m.json                2M 级核心消融矩阵
```

单次 smoke：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe tools\train\train_v33.py `
  --config configs\v33_no_memory_smoke.json `
  --device cuda `
  --amp
```

矩阵消融：

```powershell
.\tools\launcher\run_v33_ablation_matrix.ps1 `
  -Matrix configs\v33_ablation_2m.json `
  -DataPath E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  -Device cuda `
  -BatchSize 128
```

只跑部分实验：

```powershell
.\tools\launcher\run_v33_ablation_matrix.ps1 `
  -Only no_memory_backbone,memory_rank4_backbone `
  -MaxSteps 1000
```

Dry run 检查命令：

```powershell
.\tools\launcher\run_v33_ablation_matrix.ps1 -DryRun
```

汇总结果：

```powershell
python tools\analysis\summarize_v33_ablation.py --root checkpoints\v33_2m_core
```

## 2. 配置优先级

`train_v33.py` 支持 JSON 配置和命令行覆盖：

```text
默认值 < JSON config < CLI argument
```

例如配置里 `batch_size=128`，命令行传 `--batch-size 64`，最终使用 64。

布尔参数支持正反向覆盖：

```text
--use-memory / --no-use-memory
--use-backbone / --no-use-backbone
--amp / --no-amp
--streaming-train / --no-streaming-train
--strict-masked-source / --no-strict-masked-source
```

配置文件中出现未知字段会直接报错，避免 typo 造成“以为改了参数，实际没改”的问题。

每次运行都会在输出目录保存：

```text
resolved_config.json      train_v33.py 最终实际参数
resolved_input_*.json     matrix runner 生成的输入配置
train_log.jsonl           训练过程
summary.json              最终评估摘要
latest.pt                 最新检查点
```

## 3. 核心参数组

### 3.1 模型规模

| 参数 | 含义 |
| --- | --- |
| `d_model` | byte lookup、segmentor 和 chunk 特征维度 |
| `d_z` | 输出给外部 backbone 的 readout latent 维度 |
| `d_mem` | memory 表示维度 |
| `hidden` | segmentor/interpreter/decoder 内部隐藏维度 |
| `max_chunks` | 每个样本最多产生多少 latent chunk |
| `max_span` | 每个 chunk 最多覆盖多少 byte |

### 3.2 切分策略

| 参数 | 含义 |
| --- | --- |
| `tau_cut` | signed confidence 高于该阈值时形成明确切分 |
| `tau_trans` | 进入软过渡区域的阈值 |
| `tau_keep` | 低于负阈值时强制延续，常用于 UTF-8 continuation byte |
| `boundary_loss_weight` | 弱边界先验权重，不应成为主驱动 |
| `boundary_credit_loss_weight` | backward-only 的边界置信度塑形权重 |
| `boundary_credit_backbone_weight` | plastic credit 中 backbone token loss 的相对权重 |

v3.3 的原则是：segmentor 只看当前输入 byte/context，不读 memory；memory 只给 interpreter 使用。前向切分使用双阈值后的 hard boundary / transition / force-continue；连续 confidence 不进入 interpreter 内容表示，只通过边界先验和 plastic credit assignment 接收反向监督。

### 3.3 严格遮盖协议

| 参数 | 含义 |
| --- | --- |
| `strict_masked_source` | 是否在输入 byte 层面先遮盖，再交给 FLUED |
| `mask_prob` | 遮盖有效 byte 的比例 |
| `mask_span_min/max` | 连续遮盖 span 的长度范围 |

严格协议是必须项。不能先 clean encode 再遮 readout，因为那会让后续 readout 或 memory 携带被遮盖信息。

### 3.4 损失信号

| 参数 | 作用 |
| --- | --- |
| `masked_loss_weight` | 被遮盖 byte 的还原损失 |
| `visible_loss_weight` | 未遮盖 byte 的保持损失 |
| `length_loss_weight` | chunk 长度预测损失 |
| `boundary_loss_weight` | 弱语义边界先验 |
| `boundary_credit_loss_weight` | 用 detached token difficulty 塑形普通边界 confidence |
| `max_readout_vectors` | 每个 chunk 最多可打开的 readout vector 数量；第 1 个为硬兜底，其余由可反传 gate 控制 |
| `rate_loss_weight` | readout 数量预算压力 |
| `coding_rate_loss_weight` | readout 表达分散度 / 编码率压力 |
| `target_rate` | 目标 readout units / byte |
| `rate_loss` | `upper` 只惩罚超预算，`l2` 惩罚偏离目标 |

当前不再使用 v2 中造成崩溃的 latent consistency MSE（均方误差）主损失。

当前全量 v3.3 配置中，chunk 被动上限为 128 byte，`max_readout_vectors=16`，`target_rate=0.0625`。这表示每个 chunk 至少有 1 个 readout 兜底，平均预算约 8 个 readout，最多可打开 16 个；数量压力只惩罚超预算，避免模型为了压缩率牺牲高密度语义片段。plastic boundary credit 不是前向特征注入，而是把重建/主干 token loss 作为 detached credit 分配给 signed confidence。

### 3.5 Memory 分支

| 参数 | 作用 |
| --- | --- |
| `use_memory` | 是否启用 causal low-rank sequence memory |
| `memory_rank` | 每个 chunk 写入的低秩槽数量 |
| `memory_top_k` | memory 读取时选择多少历史槽 |

memory 是分支，不是默认主线。它必须在严格 masked-source 和 paired-backbone 条件下证明收益，否则不作为 claim。

### 3.6 Backbone 探针

| 参数 | 作用 |
| --- | --- |
| `use_backbone` | 启用小 latent backbone |
| `backbone_loss_weight` | backbone masked-byte 任务损失权重 |
| `detach_backbone_input` | 是否阻断 backbone loss 回传到 readout |
| `detach_backbone_keep` | 未遮盖 latent 是否 detach，降低侧漏和 shortcut |

backbone 探针用于回答：FLUED 的 latent 是否真的降低小主干补全难度，而不是只会自重建。

## 4. 默认 2M 核心矩阵

`configs/v33_ablation_2m.json` 默认包含：

| 实验 | 问题 |
| --- | --- |
| `no_memory_codec` | codec 本体是否能学会严格遮盖重建 |
| `memory_rank4_codec` | memory 是否影响 codec 自身 |
| `no_memory_backbone` | 无 memory 的 latent 是否帮助小主干 |
| `memory_rank4_backbone` | memory 是否在严格条件下带来额外收益 |
| `rate_off_backbone` | 不加 rate pressure 是否更好 |
| `rate_035_backbone` | 更强 readout 压缩是否破坏补全 |
| `mask_010_backbone` | 较轻遮盖下是否更像保持任务 |
| `mask_025_backbone` | 较重遮盖下是否仍能稳定 |

## 5. 验收指标

每个实验至少看这些字段：

| 指标 | 解释 |
| --- | --- |
| `eval_decoder_mask_acc` | FLUED decoder 还原被遮盖 byte 的能力 |
| `eval_decoder_visible_acc` | 未遮盖内容是否被破坏 |
| `eval_backbone_mask_acc` | 小 backbone 通过 latent 补全 byte 的能力 |
| `eval_leakage_gap` | backbone 相对 decoder 直接输出的增益，异常大时要查侧漏 |
| `eval_readout_units_per_byte` | 压缩率行为 |
| `eval_length_acc` | chunk 长度是否可预测 |
| `eval_boundary_loss` | 弱边界先验是否失控 |
| `eval_boundary_credit_loss` | 主任务 credit 对普通边界 confidence 的塑形强度 |
| `eval_boundary_credit_target_std` | plastic credit 分配是否有有效动态范围 |
| `steps_per_sec` | 工程可扩展性 |

## 6. 决策链条

1. 如果 `no_memory_backbone` 明显优于 byte baseline，说明 readout latent 有价值。
2. 如果 `memory_rank4_backbone` 只在 clean encode 下有效，在 strict masked-source 下无效，则早期 memory 结论属于侧漏或任务不公平。
3. 如果 memory 在 strict 条件下稳定提升，同时 attention/patching 证明不是未遮盖信息作弊，memory 才能进入主线 claim。
4. 如果 rate pressure 降低 `readout_units_per_byte` 但显著伤害 masked accuracy，说明压缩预算仍需要自适应或分阶段训练。
5. 如果 `visible_acc` 大幅下降，说明 FLUED 不是在翻译输入，而是在改写输入，这违背语言编码器定位。
6. 如果 `decoder_mask_acc` 高但 `backbone_mask_acc` 不高，说明 codec 自重建强，不代表 latent 对 backbone 友好。

## 7. 当前建议

先跑 2M/128 的核心矩阵，确认接口和指标趋势。只有当趋势稳定后，再扩大到 6M/512 或更长上下文。扩大规模前不应再引入新的结构信号，否则无法区分是 scaling 带来的收益，还是结构变更带来的收益。
