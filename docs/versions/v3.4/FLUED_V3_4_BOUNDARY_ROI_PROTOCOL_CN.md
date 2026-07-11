# FLUED v3.4 边界 ROI / 切分行为 CPU 评估协议

## 目的

本协议用于检查 v3.4 在固定文本上的边界与切分行为，不用于训练或改变 checkpoint。评估同时保留模型的有符号 confidence、策略层 hard cut、`ChunkBuilder` 的可执行 chunk 起点、逻辑转折、UTF-8 continuation 和 `max_span` 触发的强制边界，避免把这些信号混成一个“边界率”。

## 运行方式

在仓库根目录执行：

```powershell
python tools/eval/v3_4/eval_v34_boundary_roi.py --device cpu --output-dir outputs/v34_boundary_roi
python tools/analysis/v3_4/render_v34_boundary_roi.py outputs/v34_boundary_roi/v34_boundary_roi.json --output-dir outputs/v34_boundary_roi
```

使用已有 v3.4 checkpoint 时，评估脚本会读取 checkpoint 的 `args` 重建 `FLUEDV34Probe`，然后加载 `model`：

```powershell
python tools/eval/v3_4/eval_v34_boundary_roi.py `
  --device cpu `
  --checkpoint checkpoints/v34_rate_emit_5k/full/latest.pt `
  --output-dir outputs/v34_boundary_roi/full_5k
python tools/analysis/v3_4/render_v34_boundary_roi.py `
  outputs/v34_boundary_roi/full_5k/v34_boundary_roi.json `
  --output-dir outputs/v34_boundary_roi/full_5k
```

无 checkpoint 的默认模式使用配置中的小型随机初始化模型，仅用于接口和渲染 smoke，不作为模型质量结论。`--cases` 可替换固定样本 JSON，`--seq-len` 和 `--max-cases` 用于缩短 smoke 范围。脚本是 inference-only，不调用训练入口、不保存 checkpoint，也不请求 GPU；默认设备为 CPU。

## 固定样本

`configs/v3_4/v34_boundary_roi_cases.json` 中的样本是显式保存的 UTF-8 文本，当前包含 66 个 case、33 个 pair，覆盖：

- 中文和英文自然段、逻辑转折、长文本；
- Python、SQL、JSON、路径、类名、配置项和实体密集文本；
- 数字、版本号、指标、小数、负数、百分号、公式和不等式；
- 重复 token、重复短句、混合中英代码；
- 标点、空格、tab、LF/CRLF、括号和换行；
- 中文、日文、韩文、emoji、组合符号等多字节 UTF-8；
- 短文本起点、接近 `max_span` 的结构化长样本。

每个 case 明确带有 `category`、`pair_id` 和 `variant`（`base`/`perturbed`）。每个 pair 只做一个轻微扰动，覆盖标点变化、相邻字母/实体替换、空格/换行变化、代码变量改名和中文语序变化，便于做边界稳定性对比。每个 case 还带有 `audit_targets`，用于人工审阅，不是监督标签。评估不会声称这些固定文本拥有唯一正确的切分。

## 输出字段与边界定义

输出文件为 `v34_boundary_roi.json`，每个 case 都保留 `category`、`pair_id`、`variant`；每个 byte 都包含 `signed_confidence`、chunk/slot 坐标、UTF-8 状态、模型边界和逻辑/预算标记；每个 case 还包含 chunk 级 readout 预算。

| 字段 | 定义 |
|---|---|
| `signed_confidence` | `FLUEDV34Probe.encode()` 中的 `tanh(boundary_logits)`，范围约为 `[-1, 1]`；正值表示 cut pressure，负值表示 continuation pressure。 |
| `requested_model_boundary` | 策略或 coding-rate selector 请求的 hard cut，包含可能因容量而未执行的请求。 |
| `model_hard_boundary` | `out.policy.hard_cut`，即经过 v3.4 policy 和容量安全处理后交给 `ChunkBuilder` 的模型 hard cut。第一个有效 byte 也会作为 chunk 起点。 |
| `hard_chunk_boundary` | 从 `chunk_id`/`chunk_offset==0` 提取的实际可执行 chunk 起点，包含模型边界和自动边界。 |
| `logic_transition` | v3.4 两阈值策略中的 `tau_trans < confidence <= tau_cut` soft transition；它不是 hard cut。 |
| `utf8_continuation` | raw byte 在 `0x80..0xBF` 的 UTF-8 continuation。v3.4 policy 会禁止这些位置成为 model hard cut。 |
| `forced_max_span_boundary` | 实际 chunk 起点但不是 model hard boundary，通常由 `ChunkBuilder` 在累计 slot 达到 `max_span` 时插入。它是预算/容量边界，不应解释成模型学到的语义边界。 |
| `force_continue` | 模型输出的结构性 continuation 保护；当前 v3.4 主要由 UTF-8 continuation guard 贡献。 |

### 容量边界的注意事项

`FLUEDV34Probe._capacity_safe_cuts()` 先限制可执行请求数量，`ChunkBuilder` 随后还会根据 `max_span` 补充 `span_cut`。因此不能只统计 `policy.hard_cut` 来计算实际 chunk 数；本协议同时输出 `model_hard_boundary_count`、`hard_chunk_boundary_count` 和 `forced_max_span_boundary_count`。

## Readout / chunk 预算

每个 case 的 `budget` 保存配置上限：`max_chunks`、`max_span`、`fixed_chunk_budget`、`bytes_per_chunk_budget`、`configured_readout_vectors` 和最大 readout slots。`summary` 保存实际 active chunks、每 chunk 字节数、`emit_hard` 的硬 readout 数、`emit_soft` 的软预算和每 byte 比例。chunk 明细进一步列出每个 chunk 的 `readout_slot_mask`，便于判断第一个 fallback slot 与额外 slot 的使用情况。

这些是行为/预算诊断，不等于下游任务准确率，也不等于压缩率或 BPB。不同 checkpoint、不同 `max_span`、不同 `boundary_mode` 的结果必须分开比较。

## HTML 审阅

renderer 输出单文件、自包含 HTML，不依赖网络资源：

- 底色按 signed confidence 着色，蓝色偏 continuation pressure，红色偏 cut pressure；
- 红色上边表示 model hard boundary；
- 黄色右边表示 logic transition；
- 蓝色下边表示 UTF-8 continuation；
- 紫色左边表示 forced max-span boundary。

同一个 byte 可以同时具有多种标记，HTML 会叠加显示，避免掩盖 UTF-8 continuation 与预算边界的重合。每个 byte 的 tooltip 还包含原始 byte、hex、chunk/slot 和 confidence。

## Smoke 验证

最小人工审阅命令：

```powershell
python tools/eval/v3_4/eval_v34_boundary_roi.py --device cpu --max-cases 3 --output-dir outputs/v34_boundary_roi_smoke
python tools/analysis/v3_4/render_v34_boundary_roi.py outputs/v34_boundary_roi_smoke/v34_boundary_roi.json --output-dir outputs/v34_boundary_roi_smoke
```

验证 JSON 中 `device` 为 `cpu`、case 数为 3、每个 case 有 `bytes`/`chunks`/`budget`；验证 HTML 中存在四种边界图例和 UTF-8 元数据。完整固定样本运行不改变上述定义。

## 已知限制

1. 无 checkpoint 的随机初始化运行只验证接口、字段和渲染，不支持质量结论。
2. 固定样本没有人工 gold boundary；`audit_targets` 是审阅提示，不是 F1 标签。
3. `model_hard_boundary` 和 `forced_max_span_boundary` 可能在同一段附近相邻，且多个 byte 标记可以重合；应以 byte 级字段和 chunk offset 为准。
4. 如果输入文本超过 `seq_len`，评估会按原始 UTF-8 byte offset 截断并在 JSON 标出 `truncated_to_seq_len=true`；byte-char 标签不会对截断前缀执行 `errors=replace` 再编码。正式比较应保持相同 `seq_len`。
5. CPU 运行速度取决于 checkpoint 规模和 `seq_len`；本协议不运行 GPU 训练，也不把 CPU wall-clock 当成模型指标。
