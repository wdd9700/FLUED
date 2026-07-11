# FLUED v3.2 执行目标、迁移路径与验收标准

日期：2026-07-03

本文是 v3.2 的执行协议。目的不是重新解释架构，而是保证上下文压缩后仍能继续实现、训练和判断。

先读顺序：

```text
1. FLUED_V3_1_ARCHITECTURE_CN.md
2. FLUED_V3_2_DESIGN_SUPPLEMENT_CN.md
3. 本文
```

## 0. 当前决策链条

### 0.0 上下文压缩恢复要求

新执行者在只读取以下三份文档后，必须能恢复 v3.2 的执行状态：

```text
FLUED_V3_1_ARCHITECTURE_CN.md
FLUED_V3_2_DESIGN_SUPPLEMENT_CN.md
FLUED_V3_2_EXECUTION_GOALS_CN.md
```

恢复后必须能独立说清：

```text
1. v3.2 不覆盖 v3.1 基线脚本。
2. v3.2 新增并行入口，例如 train_v32_language_codec_2m.py。
3. Stage 1 只替换 FactorizedByteSeed，保持 v3.1 decoder / weak boundary smoke。
4. Stage 2 一次性接入 fixed r memory slots + top-k retrieval + memory-conditioned interpreter。
5. backbone 只消费 readout latent。
6. boundary 不接收 memory。
7. decoder 不接收 memory。
8. 每阶段必须先跑 py_compile、CPU smoke、GPU 1k，再决定是否更长训练。
9. 每阶段开始前必须判断哪些任务可以交给 subagent 并行处理。
```

如果新执行者无法恢复以上信息，先补文档，不进入实现。

### 0.1 已经被否定的路线

```text
纯 reconstruction:
  可以证明 codec 能重建，但不能证明 latent 有语义、可预测、对 backbone 有帮助。

固定 target compression:
  v2 已经证明 m/n 难控，且压缩率不等于有效语义压缩。

latent consistency MSE:
  v2 崩溃实验证明它会压倒其他 loss，破坏 boundary / memory 分化。

backbone 直接读取 FLUED memory:
  破坏接口边界。memory 是 FLUED encoder 内部机制，不是 backbone 输入。

segmentation 读取过去 memory:
  强但耦合。v3.2 主线先不用，防止 boundary 和 memory 互相污染。

decoder 读取 encoder memory:
  会让解码依赖编码历史状态，部署和缓存语义不干净。
```

### 0.2 当前保留的核心判断

```text
1. FLUED 是语言编码器 / codec，不是 byte LM。
2. readout latent 是唯一对外接口。
3. summary / memory 是内部机制。
4. boundary 只决定当前输入的可执行 chunk。
5. interpreter 负责把 chunk + 过去 memory 翻译为 readout latent。
6. summarizer 负责把当前 chunk 写成后续可用 memory。
7. decoder 只把 latent unit 反编译为 byte span。
```

### 0.3 v3.2 的一句话目标

```text
用 memory-free boundary + memory-conditioned interpreter + causal append-only summary，
把 v3.1 的“能重建的 codec”推进成“对 backbone 有用的语言编码器”。
```

### 0.4 目标模式监督规则

当前 Codex active goal 已设置为 v3.2 迭代监督目标。
系统 goal 工具只能标记完成 / 阻塞，不能直接改写 active objective。
实际执行时以本文作为可恢复目标协议；需要细化口径时，更新本文和计划状态，
不强行重建 goal，避免目标口径反复。

每次进入新阶段时必须先做三件事：

```text
1. 复述当前阶段目标、行动、验收标准和停止条件。
2. 检查是否有可并行 subagent 任务。
3. 更新当前计划状态，不把未验证结果写成已完成。
```

可交给 subagent 的任务优先级：

```text
1. 只读代码锚点盘点。
2. 训练日志 / summary 表核对。
3. ROI / memory-case 样本审计。
4. launcher 参数与公平性审计。
5. 消融结果归因草案。
```

主线程保留当前阻塞路径：

```text
1. 核心代码修改。
2. 训练启动与失败处理。
3. 最终决策和文档口径更新。
```

## 1. 从 v3.1 到 v3.2 的平滑迁移原则

不能一次性把所有想法塞进去。迁移要保证每一步都有可比较基线。

### 1.1 v3.1 中必须保留的东西

```text
保留 batch=128 公平口径。
保留 seq_len=128 / max_span=16 的第一轮小规模验证。
保留 length_head + slot decoder 作为当前 decoder 骨架。
保留 ROI / decoder / memory ablation / memory-case 诊断脚本思路。
保留最小 backbone 的 masked-source 公平比较。
保留 summary.json / train_log.jsonl / sweep_summary.md 的工程闭环。
```

原因：

```text
这些是可比性基础。没有它们，v3.2 的结果无法判断是结构提升还是实验口径变化。
```

### 1.2 v3.1 中必须改的东西

```text
byte embedding:
  改为 16x16 factorized byte coordinate seed。

memory path:
  从 shifted cumulative summary memory 改为 chunk memory sequence。

memory read:
  从简单前缀均值 / 全局历史压缩改为 top-k sparse retrieval。

interpreter:
  显式接收 segmented chunk feature + retrieved past memory。

metrics:
  把 units/byte 拆成 readout_units_per_byte 和 memory_slots_per_byte。
```

原因：

```text
这些是 v3.2 与 v3.1 的本质差别。
如果不改，仍然只是在 v3.1 codec 上调 pooling / head。
```

### 1.3 v3.1 中暂时不能动的东西

```text
decoder 不要马上换成复杂 tied inverse。
segmentation 不要马上读 memory。
memory 不要马上做动态 gate / overflow。
AR 小修正头不要马上变成默认开启。
不要马上拉到 seq_len=512 或更大。
```

原因：

```text
先验证 v3.2 的职责分工是否成立，再扩大上下文和组件复杂度。
```

## 2. 组件分级：必须加、逐步加、全量后删着消融

### 2.1 必须立即加入或修改的组件

这些是 v3.2 的身份组件。不加就不是 v3.2。

#### A. FactorizedByteSeed

行动：

```text
新增 16x16 factorized byte coordinate seed。
byte b -> hi/lo -> row_embed + col_embed + byte_type_feature。
替换当前 nn.Embedding(vocab_size, d_model) 的 byte 主入口。
```

验收：

```text
1. 参数量统计中 byte seed 参数显著少于 258 x d_model lookup。
2. CPU smoke test 通过。
3. 1k 训练 loss 正常下降，无 NaN。
4. 与 v3.1 mean 相比，初期 recon_acc 不应完全崩溃。
```

失败判断：

```text
如果训练完全学不动，先加 small residual byte lookup 做消融，
但主线仍优先 factorized seed。
```

#### B. Memory-free Boundary

行动：

```text
把 boundary / segmentation 明确封装成不读取 memory 的模块。
输入只能是当前 byte seed / local context。
输出 hard executable segments + soft confidence。
```

验收：

```text
1. 代码结构上 boundary forward 不接收 memory 参数。
2. ROI 中 model_start 是可执行 hard segment。
3. UTF-8 continuation 不作为 segment start。
4. max_span 约束有效。
5. soft boundary confidence 被记录，但不作为推理时全 soft segmentation。
```

失败判断：

```text
如果边界质量差，先改 boundary loss / weak prior / confidence，
不要立刻让 boundary 读取 memory。
```

#### C. ChunkMemorySequence

行动：

```text
把 memory 从单个累计历史向量改成 chunk memory 序列。
每个 chunk 固定 r 个 memory slots。
```

验收：

```text
1. summary 输出形状为 [B, U, r, H] 或等价展开。
2. memory_pool 只包含当前 chunk 之前的 memory。
3. 当前 chunk 的 summary 不反哺当前 chunk interpreter。
4. summary.json 记录 memory_slots_per_byte。
```

失败判断：

```text
如果 memory-case 无变化，优先检查 retrieval 和 summarizer，
不要马上加动态 gate。
```

#### D. MemoryConditionedInterpreter

行动：

```text
显式实现：
  segmented chunk feature + retrieved past memory -> readout latent
```

验收：

```text
1. zero / shuffled / stale memory 会影响 readout 或 decoder 指标。
2. memory-case 中实体、版本号、代码标识符场景有可测 delta。
3. backbone 输入仍只包含 readout latent。
```

失败判断：

```text
如果 codec 重建强但 memory ablation 无差异，
说明 readout 又走了局部 payload 捷径。
下一步要限制 local-only path 或增强 memory-conditioned path。
```

### 2.2 可以逐步加入的低风险组件

这些组件能逐个加，因为它们不会彻底改变训练闭环。

#### E. Top-k Sparse Retrieval

行动：

```text
先实现 top_k = 4 / 8 的 memory retrieval。
query 来自当前 chunk feature。
key/value 来自过去 memory slots。
```

验收：

```text
1. retrieval_topk 被记录。
2. retrieval_entropy 被记录。
3. retrieved_distance_distribution 被记录。
4. top-k retrieval 不显著拖慢 2M/128 训练。
```

先后顺序：

```text
先做 all-past attention 或 simple top-k 可运行版本；
再优化为稀疏实现；
最后才考虑 DSA 风格更激进稀疏。
```

#### F. Decoder-aligned Auxiliary Loss

行动：

```text
沿用 v3.1 最小 backbone 经验：
latent MSE 不够时，通过 frozen decoder CE 对齐 latent 与 byte manifold。
```

验收：

```text
1. masked latent infill 中 byte accuracy 提升。
2. CE 不反向破坏 FLUED encoder 主结构。
3. 不重新引入 latent consistency MSE。
```

#### G. Memory-case Diagnostics

行动：

```text
把已有 memory_cases 适配 v3.2。
增加实体密集、代码变量密集、中文指代密集样本。
```

验收：

```text
1. full memory 优于 zero/shuffled/stale。
2. mean_first_last 弱 memory 依赖问题不再出现或被减轻。
3. 失败样本能定位是 retrieval 错还是 memory 内容不足。
```

#### H. Byte Seed Ablation

行动：

```text
比较：
  256 lookup
  16x16 full table
  16x16 row+col factorized
  row+col+byte_type
```

验收：

```text
1. factorized seed 不显著损害 codec 可训练性。
2. factorized seed 不降低 memory-case 和 backbone 指标。
3. 如果 full table 只提高 reconstruction 但不提高 backbone，则不作为主线。
```

### 2.3 必须全量加入后再删着做消融的组件

这些组件强耦合，单独加可能看不出真实效果。

#### I. Memory-conditioned Interpreter + Chunk Memory + Retrieval

原因：

```text
只加 memory slots 但不让 interpreter 有效读取，看不出 memory 价值。
只加 retrieval 但 memory 内容差，也看不出 retrieval 价值。
只加 interpreter 但 memory 是累计均值，会退回 v3.1。
```

正确策略：

```text
先全量实现：
  fixed r memory slots
  top-k retrieval
  memory-conditioned interpreter

然后删着做消融：
  no memory
  zero memory
  shuffled memory
  stale memory
  no retrieval, use mean memory
  random retrieval
  local-only interpreter
```

验收：

```text
如果全量版本有效，而删掉 memory/retrieval 后下降，
才能说明 v3.2 的 memory 路径真实成立。
```

#### J. Small AR Correction

原因：

```text
小 AR 头只有在已有并行 proposal 和 decoder/interpreter 主路径后才有意义。
单独加 AR 可能只是补主路径缺陷，误导结构判断。
```

正确策略：

```text
第一版默认关闭。
全量 v3.2 跑通后，加 AR correction。
然后删着比较：
  ar_off
  ar_on_limited_delta
  ar_on_decoder_only
  ar_on_interpreter_only
```

验收：

```text
1. AR delta norm 很小。
2. AR 参数占比很小。
3. AR latency 可接受。
4. AR 前后 CE / span acc 有明确改善。
5. 如果收益不清楚，默认关掉。
```

#### K. Dynamic Memory Gate / Overflow Slots

原因：

```text
动态 gate 会改变 memory budget、retrieval、训练信号和推理效率。
它不能在固定 memory slots 没跑清楚之前加入。
```

正确策略：

```text
先固定 r=2/4。
如果实体密集样本证明固定 r 不够，再加 overflow。
```

验收：

```text
overflow 只在实体/变量/专名密集样本明显改善，
且 memory_slots_per_byte 与速度成本可控时才保留。
```

## 3. 推荐实现阶段

### Stage 0：文档与接口冻结

行动：

```text
1. 完成 FLUED_V3_2_DESIGN_SUPPLEMENT_CN.md。
2. 完成本文。
3. 在 README 或 v3.1 review 中链接本文。
```

验收：

```text
1. 文档明确 v3.2 的五条约束。
2. 文档明确哪些组件必须现在加，哪些后续加。
3. Mermaid 图可以作为标准实现参考。
```

决策：

```text
如果文档仍把 memory 写成 backbone 输入，停止实现，先修文档。
```

### Stage 1：Byte seed + boundary-freeze smoke

行动：

```text
在 v3.1 codec 基础上最小替换 byte seed。
保持原 weak boundary 和 decoder 不变。
```

目的：

```text
先排除 16x16 factorized seed 导致训练不可用的风险。
```

验收：

```text
CPU smoke test 通过。
GPU 1k step loss 下降。
recon_acc 不低于同条件 v3.1 太多。
```

决策：

```text
如果失败：
  做 byte seed ablation。
如果通过：
  进入 Stage 2。
```

### Stage 2：全量 memory path 装上

行动：

```text
一次性加入：
  chunk memory sequence
  fixed r memory slots
  top-k retrieval
  memory-conditioned interpreter
```

目的：

```text
避免只装半条 memory path 导致错误负结论。
```

验收：

```text
1. py_compile 通过。
2. CPU smoke test 通过。
3. GPU 1k / 3k 小训练无 NaN。
4. summary 记录 memory_slots_per_byte、retrieval_topk、retrieval_entropy。
5. memory ablation 有非零可解释影响。
```

决策：

```text
如果 codec 直接崩：
  先查 tensor shape / retrieval / memory leakage。

如果 codec 能训但 memory 无效：
  做 no-retrieval / random-retrieval / zero-memory 消融。

如果 memory 有效但重建差：
  改 decoder 或 chunk feature，而不是删 memory。
```

### Stage 3：与 v3.1 公平对比

行动：

```text
batch=128
seq_len=128
max_span=16
2M 级别
1k / 3k / 10k
```

对比对象：

```text
v3.1 mean
v3.1 mean_first_last
v3.2 full
v3.2 no-memory
v3.2 random-retrieval
```

验收：

```text
v3.2 full 至少满足：
  codec 不低于 v3.1 mean 太多；
  long_span 明显好于 v3.1 mean 或接近 mean_first_last；
  memory-case 强于 mean_first_last；
  memory ablation 有清晰 delta；
  efficiency 没有不可接受退化。
```

决策：

```text
如果 v3.2 full 只提高 reconstruction：
  说明又走了 payload 捷径。

如果 v3.2 full 只提高 memory-case 但 codec 差：
  decoder/readout path 不够。

如果 v3.2 full 两者都不行：
  回到 Stage 1/2 分别定位 seed 和 memory path。
```

### Stage 4：最小 backbone 验证

行动：

```text
冻结 v3.2 codec。
接最小 backbone。
做 byte/span masked-source latent infill。
```

验收：

```text
至少对比：
  byte/span masked-source byte baseline
  v3.1 best latent
  v3.2 full latent
  v3.2 no-memory latent
```

关键指标：

```text
mask_acc
keep_acc
masked byte acc through frozen decoder
latent loss
decoder CE auxiliary effect
```

决策：

```text
如果 codec 指标好但 backbone 不好：
  readout latent 可预测性不足。

如果 backbone 好但 decoder keep 差：
  decoder path 不稳。

如果 no-memory 与 full 一样：
  memory 仍是装饰品。
```

### Stage 5：小 AR / overflow / 长序列

只有 Stage 3/4 给出正信号后才进入。

行动顺序：

```text
1. small AR correction
2. overflow memory slots
3. seq_len=256 / 512
4. 更大模型
```

验收：

```text
每个新增组件都必须有成本-收益表：
  accuracy gain
  memory-case gain
  backbone gain
  latency / steps/s / samples/s cost
```

## 4. 文件级行动清单

### 4.0 当前 v3.1 代码锚点

v3.1 基线必须保留，不直接覆盖：

```text
tools/analysis/train_v31_language_codec_2m.py
tools/analysis/summarize_v31_language_codec.py
tools/eval/eval_v31_language_codec_roi.py
tools/eval/eval_v31_language_codec_decoder.py
tools/eval/eval_v31_language_codec_memory_ablation.py
tools/eval/eval_v31_language_codec_memory_cases.py
tools/analysis/train_v31_min_backbone.py
tools/launcher/run_v31_language_codec_2m_5080.ps1
tools/launcher/run_v31_min_backbone_5080.ps1
```

v3.1 中可复用的函数 / 类：

```text
CodecCollator:
  UTF-8 edge mask、weak boundary、segment target packing。

complete_utf8_edge_valid:
  chunk 边缘 UTF-8 合法性处理。

weak_boundary_starts:
  第一版 hard executable segment 的弱边界来源。

build_segments:
  [B,T] -> [B,U,S] target packing。

segment_mean_pool / segment_edge_pool:
  第一版 chunk feature baseline。

V31LanguageCodec2M:
  decoder、length_head、slot_decoder、byte_head 可作为 v3.2 初版骨架。

move_codec_batch:
  batch 搬运逻辑可复用。
```

v3.2 第一版需要新增而不是覆盖：

```text
FactorizedByteSeed
V32LanguageCodec2M
SparseMemoryRetriever
ChunkMemorySummarizer
MemoryConditionedInterpreter
```

第一批建议新增：

```text
tools/analysis/train_v32_language_codec_2m.py
tools/analysis/summarize_v32_language_codec.py
tools/eval/eval_v32_language_codec_memory_cases.py
tools/eval/eval_v32_language_codec_decoder.py
tools/eval/eval_v32_language_codec_roi.py
tools/launcher/run_v32_language_codec_2m_5080.ps1
```

可复用 v3.1：

```text
CodecCollator 的 UTF-8 edge mask / weak boundary 原型
decoder length bucket 统计
memory-case 文本样本
minimal backbone 训练口径
summary markdown 结构
```

不要直接覆盖：

```text
tools/analysis/train_v31_language_codec_2m.py
```

原因：

```text
v3.1 是当前可复现基线。v3.2 应该新脚本并行存在，避免改坏基线。
```

## 5. 第一版张量形状协议

建议第一版统一：

```text
B = batch size
T = seq_len
U = chunk / readout units
S = max_span
R = memory slots per chunk
K = retrieval top-k
H = hidden size
V = byte vocab size
```

核心张量：

```text
src_ids:             [B, T]
byte_seed:           [B, T, H]
local_h:             [B, T, H]
boundary_logits:     [B, T]
seg_ids:             [B, T]
seg_mask:            [B, U]
chunk_h:             [B, U, H]
memory_slots:        [B, U, R, H]
past_memory:         [B, <=U*R, H]
retrieved_memory:    [B, U, K, H]
retrieval_weights:   [B, U, K]
readout:             [B, U, H]
length_logits:       [B, U, S]
byte_logits:         [B, U, S, V]
targets:             [B, U, S]
```

不变量：

```text
boundary module 不接收 past_memory。
interpreter 接收 retrieved_memory。
summarizer 输出 memory_slots。
memory_slots_i 只能进入后续 chunk 的 past_memory。
backbone 只接收 readout。
decoder 不接收 memory_slots / past_memory。
```

## 6. 最小失败判据

任何实验出现以下情况，不继续长训：

```text
loss NaN / Inf
grad 爆炸
readout_units_per_byte 塌缩到全切或全合并
memory_slots_per_byte 异常增长
length_acc 不涨
recon_acc 长期不涨
invalid_target_utf8 > 0
invalid_pred_utf8 明显恶化
full memory 与 zero/shuffled/stale 无差异
retrieval entropy 塌缩且 memory-case 无收益
backbone latent 不优于 byte/span masked-source byte baseline
```

## 7. 当前下一步

下一步不是直接大训，而是：

```text
1. 把本文链接进 v3.2 设计补充文档。
2. 新建 train_v32_language_codec_2m.py。
3. 先实现 FactorizedByteSeed + v3.1 旧路径 smoke。
4. 再一次性接入完整 memory path。
5. 跑 1k / 3k 小训练和诊断。
```

判断标准：

```text
如果 FactorizedByteSeed 都过不了，先别讨论 memory。
如果 FactorizedByteSeed 通过但 memory path 无效，重点查 interpreter/retrieval。
如果 memory path 有效但 backbone 无收益，重点查 readout latent predictability。
```

## 8. 具体命令模板

### 8.1 编译检查

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m py_compile `
  tools\analysis\train_v32_language_codec_2m.py `
  tools\analysis\summarize_v32_language_codec.py
```

### 8.2 CPU smoke

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python tools\analysis\train_v32_language_codec_2m.py `
  --out-dir K:\FLUED_archive\v32_language_codec_2m_20260703\smoke_cpu `
  --device cpu `
  --seq-len 64 `
  --stride 32 `
  --batch-size 4 `
  --max-steps 5 `
  --d-model 64 `
  --hidden 64 `
  --encoder-layers 1 `
  --ffn-dim 256 `
  --max-span 8 `
  --log-every 1 `
  --ckpt-every 5
```

### 8.3 GPU 1k smoke

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python tools\analysis\train_v32_language_codec_2m.py `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --out-dir K:\FLUED_archive\v32_language_codec_2m_20260703\stage1_seed_1k `
  --device cuda --amp `
  --streaming-train --streaming-eval `
  --stream-samples-per-worker 200000 `
  --seq-len 128 --stride 64 `
  --batch-size 128 `
  --num-workers 12 `
  --prefetch-factor 4 `
  --max-steps 1000 `
  --warmup-steps 300 `
  --max-eval-batches 32 `
  --d-model 192 --hidden 192 --nhead 4 --encoder-layers 2 --ffn-dim 768 `
  --min-span 2 --max-span 16 --max-units 128 `
  --log-every 250 `
  --ckpt-every 1000
```

### 8.4 诊断与汇总

```powershell
python tools\analysis\summarize_v32_language_codec.py `
  K:\FLUED_archive\v32_language_codec_2m_20260703 `
  --out-path K:\FLUED_archive\v32_language_codec_2m_20260703\sweep_summary.md

python tools\eval\eval_v32_language_codec_decoder.py `
  --ckpt K:\FLUED_archive\v32_language_codec_2m_20260703\stage1_seed_1k `
  --data-path E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt `
  --streaming-eval `
  --batch-size 128 `
  --max-batches 32 `
  --device cpu

python tools\eval\eval_v32_language_codec_memory_cases.py `
  --checkpoint K:\FLUED_archive\v32_language_codec_2m_20260703\stage1_seed_1k `
  --device cpu
```

## 9. 数值验收阈值

这些阈值用于决定是否进入下一阶段，不是论文指标。

### Stage 1：FactorizedByteSeed

```text
py_compile: 必须通过。
CPU smoke: 必须生成 latest.pt / summary.json / train_log.jsonl。
GPU 1k: loss 必须下降，无 NaN / Inf / grad explosion。
recon_acc: 相对同条件 v3.1 mean 1k 下降不超过 20%。
length_acc: 不应明显失效，至少 > 0.70。
throughput: steps/s 下降超过 30% 必须停下分析。
```

2026-07-03 Stage 1 初步结果：

```text
run:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage1_seed_1k

model_version:
  v3.2-stage1-factorized-byte-seed

结果:
  params:              1,970,003
  steps:               1,000
  eval_recon_acc:      0.3839
  eval_length_acc:     0.9059
  eval_boundary_acc:   0.9334
  eval_units_per_byte: 0.1170
  train_steps_per_sec: 17.87
  max_memory_mb:       3307

对照:
  v3.1 speed_b128_w12 1k eval_recon_acc = 0.4014
  v3.2 相对下降约 4.4%，低于 20% 停止线。

判断:
  Stage 1 通过。FactorizedByteSeed 没有破坏训练友好度，
  可以进入 Stage 2 的 full memory path。
```

### Stage 2：full memory path

```text
full vs zero/shuffled/stale memory:
  至少一个模式产生稳定非零 delta。

memory-case:
  至少一个实体/代码/版本号样本方向一致地显示 full memory 更好。

retrieval:
  retrieval_entropy 不应完全塌缩。
  retrieval_topk / memory_slots_per_byte 必须进入 summary。

efficiency:
  steps/s 下降超过 30% 时，必须记录 retrieval_topk、memory_slots_per_byte、max_memory_allocated_mb 后再决定是否继续。
```

2026-07-03 Stage 2 严格 causal memory 初步结果：

```text
run:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_1k

model_version:
  v3.2-stage2-causal-past-memory-interpreter

配置:
  byte_encoder_causal:    True
  memory_read_scope:     past_only
  boundary_reads_memory: False
  decoder_reads_memory:  False
  memory_slots_per_chunk: 2
  memory_topk:            4

训练 / eval:
  params:                         2,192,147
  steps:                          1,000
  eval_recon_acc:                 0.3909
  eval_length_acc:                0.9090
  eval_boundary_acc:              0.9385
  eval_readout_units_per_byte:    0.1170
  eval_memory_slots_per_byte:     0.2339
  eval_retrieval_entropy:         1.2745
  eval_retrieval_past_only_violation_count: 0
  eval_first_unit_memory_norm:    0.0
  train_steps_per_sec:            16.56
  max_memory_allocated_mb:        3449

同条件 no-memory 独立训练对照:
  run:              stage2_causal_nomemory_1k
  eval_recon_acc:   0.3865
  eval_length_acc:  0.9051
  train_steps_per_sec: 17.92

同一 checkpoint memory ablation:
  full:     recon_loss 1.9183, recon_acc 0.3898, length_acc 0.9082
  zero:     recon_loss 1.9481, recon_acc 0.3807, length_acc 0.7399
  shuffled: recon_loss 1.9537, recon_acc 0.3815, length_acc 0.8042
  stale:    recon_loss 1.9495, recon_acc 0.3817, length_acc 0.7774

判断:
  Stage 2 最小 causal memory path 通过。
  同一 checkpoint 下 memory 被置零 / 打乱 / 滞后都会退化，
  说明 interpreter 确实使用过去 memory，而不是只靠 readout payload 捷径。
  当前仍只是 1k 初步证据，下一步应跑 3k/更长训练，并补 memory-case / ROI 样本。
```

2026-07-03 Stage 2 3k 复核结果：

```text
run:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_3k

训练 / eval:
  params:                         2,192,147
  steps:                          3,000
  eval_recon_acc:                 0.4539
  eval_length_acc:                0.9450
  eval_boundary_acc:              0.9402
  eval_readout_units_per_byte:    0.1170
  eval_memory_slots_per_byte:     0.2339
  eval_retrieval_entropy:         1.2276
  eval_retrieval_past_only_violation_count: 0
  train_steps_per_sec:            19.69
  max_memory_allocated_mb:        3449

同一 checkpoint memory ablation:
  full:     recon_loss 1.5021, recon_acc 0.4541, length_acc 0.9453
  zero:     recon_loss 1.5378, recon_acc 0.4408, length_acc 0.8722
  shuffled: recon_loss 1.5489, recon_acc 0.4414, length_acc 0.9073
  stale:    recon_loss 1.5419, recon_acc 0.4418, length_acc 0.8945

判断:
  3k 时 memory 贡献没有消失，且 delta 大于 1k。
  可以进入 memory-case / ROI 样本分析；不要只凭整体 recon_acc 继续判断。
```

2026-07-03 Stage 2 样本级诊断：

```text
memory-case:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_3k\memory_cases.md

ROI heatmap:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_3k\roi_heatmap.html

样本:
  entity_repeat_en
  code_identifier
  cjk_reference
  version_number

结果:
  entity_repeat_en:
    memory 贡献最清楚。
    entity 子集最大 loss delta 约 9.58%，later 子集最大 loss delta 约 14.32%。

  cjk_reference:
    zero/stale memory 会降低 entity/later 的 accuracy 或 length_acc，
    但 shuffled memory 有局部反向收益。

  code_identifier / version_number:
    结果混合，部分子集里 shuffled 或 zero memory 的局部 recon_acc 更高。
    这说明当前 3k 模型的 retrieval 还没有稳定学出语义选择，
    不能把 memory 贡献解释成“已经理解代码标识符 / 版本号一致性”。

判断:
  Stage 2 的 memory path 是有效的，但 retrieval 质量还不稳定。
  下一步若继续训练，应同时保留整体 ablation、memory-case、ROI heatmap，
  不能只看 recon_acc；如果更长训练后 shuffled 仍经常优于 full，
  应优先检查 retrieval query/key 训练信号，而不是扩大模型。
```

2026-07-03 Stage 2 10k 复核结果：

```text
run:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_10k

训练 / eval:
  params:                         2,192,147
  steps:                          10,000
  eval_recon_acc:                 0.5762
  eval_length_acc:                0.9691
  eval_boundary_acc:              0.9428
  eval_readout_units_per_byte:    0.1170
  eval_memory_slots_per_byte:     0.2339
  eval_retrieval_entropy:         1.1568
  eval_retrieval_past_only_violation_count: 0
  train_steps_per_sec:            22.13
  max_memory_allocated_mb:        3449

同一 checkpoint memory ablation:
  full:     recon_loss 1.1405, recon_acc 0.5763, length_acc 0.9691
  zero:     recon_loss 1.8622, recon_acc 0.4779, length_acc 0.9049
  shuffled: recon_loss 1.9787, recon_acc 0.4573, length_acc 0.8636
  stale:    recon_loss 1.9359, recon_acc 0.4680, length_acc 0.8778

样本级诊断:
  memory-case:
    K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_10k\memory_cases.md
  ROI heatmap:
    K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_10k\roi_heatmap.html

  entity_repeat_en:
    full memory 整体最好；entity 子集 loss delta 约 10.54%。
    later 子集里 shuffled loss 更低但 acc 不更好，说明局部 retrieval 仍不稳定。

  cjk_reference:
    10k 后 full memory 明显最好。
    later 子集最大 loss delta 约 85.34%，entity 子集约 45.23%。

  code_identifier:
    full 在 all/later loss 上优于 zero/shuffled，但 stale 在 entity 子集局部更好。
    说明标识符场景仍未稳定学会“正确历史槽位”。

  version_number:
    all/later full 更好，但 entity 子集 shuffled loss 更低。
    说明数字/版本号一致性还存在 retrieval 误选或过拟合偶然性。

判断:
  欠训练是 3k 样本级混乱的重要原因，继续到 10k 后整体 memory 贡献大幅变强。
  但 retrieval 还没有在所有专名/代码/版本号场景形成稳定语义选择。
  下一步先做同训练预算的 no-memory 10k 与 v3.1 对比，
  再决定是改 retrieval 训练信号，还是进入 Stage 3 backbone 验证。
```

2026-07-03 Stage 2 no-memory 10k 对照：

```text
run:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_nomemory_10k

配置:
  memory_slots_per_chunk: 0
  byte_encoder_causal:    True
  memory_enabled:         False

结果:
  params:                         1,970,003
  steps:                          10,000
  eval_recon_acc:                 0.5363
  eval_length_acc:                0.9699
  eval_boundary_acc:              0.9436
  eval_readout_units_per_byte:    0.1170
  train_steps_per_sec:            24.99
  train_samples_per_sec:          3198
  max_memory_allocated_mb:        3333

与 full memory 10k 对比:
  full memory params:             2,192,147
  full memory eval_recon_acc:     0.5762
  full memory train_samples_per_sec: 2833

判断:
  在同训练步数、同 batch、同 seq_len/max_span 下，
  causal past memory 带来约 +0.040 绝对 recon_acc。
  代价是约 +11.3% 参数和约 -11.4% samples/s。
  这证明 memory path 不只是同 checkpoint ablation 的训练依赖，
  但下一轮公平比较仍应补一个参数量匹配的 no-memory 对照或 v3.1 同预算对照。
```

### Stage 3：v3.1 vs v3.2

```text
v3.2 full:
  recon_acc 不低于 v3.1 mean 太多。
  long_span_recon_acc 应接近 v3.1 mean_first_last，而不是回到 v3.1 mean 的短板。
  memory-case 应明显强于 v3.1 mean_first_last。

如果 v3.2 只提高 reconstruction 但 memory-case 弱：
  判为 payload shortcut。

如果 v3.2 memory-case 强但 codec 差：
  判为 decoder/readout path 不稳。
```

2026-07-03 Stage 3 公平对比结果：

```text
统一口径:
  batch_size: 128
  seq_len:    128
  max_span:   16
  max_steps:  10000
  eval:       streaming, max_eval_batches=32
  device:     local RTX 5080

v3.1 对照:
  K:\FLUED_archive\v31_language_codec_2m_20260702\codec_10k_utf8clean
  K:\FLUED_archive\v31_language_codec_2m_20260702\codec_10k_pool_mfl

v3.2 对照:
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_memory_10k
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage2_causal_nomemory_10k
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mean_random_10k
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_nomemory_10k
  K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_random_10k

结果表:
  v3.1 mean:
    params 2.011M, recon_acc 0.5063, length_acc 0.9725, steps/s 19.03

  v3.1 mean_first_last:
    params 2.085M, recon_acc 0.6469, length_acc 0.9721, steps/s 19.64

  v3.2 mean no-memory:
    params 1.970M, recon_acc 0.5363, length_acc 0.9699, steps/s 24.99

  v3.2 mean random-retrieval:
    params 2.192M, recon_acc 0.5443, length_acc 0.9695, steps/s 24.17

  v3.2 mean top-k memory:
    params 2.192M, recon_acc 0.5762, length_acc 0.9691, steps/s 22.13

  v3.2 mean_first_last no-memory:
    params 2.044M, recon_acc 0.7584, length_acc 0.9799, steps/s 25.25

  v3.2 mean_first_last random-retrieval:
    params 2.267M, recon_acc 0.7559, length_acc 0.9768, steps/s 23.70

  v3.2 mean_first_last top-k memory:
    params 2.267M, recon_acc 0.7568, length_acc 0.9780, steps/s 21.87

Stage 3 诊断:
  mean 口径:
    v3.2 top-k memory > random-retrieval > no-memory > v3.1 mean。
    说明在弱 chunk feature 下，memory payload 和 top-k retrieval 都有独立收益。

  mean_first_last 口径:
    v3.2 no-memory ≈ top-k memory ≈ random-retrieval，且三者都明显超过 v3.1 mean_first_last。
    说明 first/last edge feature 显著增强 readout，
    使当前 fixed-r memory retrieval 对 reconstruction 几乎冗余。

  mfl top-k 同 checkpoint ablation:
    K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k\memory_ablation_streaming.md

    full:     recon_loss 0.7116, recon_acc 0.7571, length_acc 0.9777
    zero:     recon_loss 0.7220, recon_acc 0.7531, length_acc 0.9704
    shuffled: recon_loss 0.7299, recon_acc 0.7512, length_acc 0.9681
    stale:    recon_loss 0.7242, recon_acc 0.7532, length_acc 0.9701

  mfl memory-case / ROI:
    K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k\memory_cases.md
    K:\FLUED_archive\v32_language_codec_2m_20260703\stage3_v32_mfl_memory_10k\roi_heatmap.html

判断:
  Stage 3 codec 层面通过。
  v3.2 的 best codec route 当前不是“memory 越多越好”，
  而是 mean_first_last chunk feature 先把 readout 做强；
  memory 在弱 feature 下有帮助，但在强 feature 下重建收益很小。

  这不是否定 memory：
    memory 的价值应转向 Stage 4 backbone / mask latent infill / 长上下文一致性验证，
    而不是继续用局部 reconstruction 单指标逼它证明语义。

  下一步进入 Stage 4：
    冻结 v3.2 mfl no-memory、v3.2 mfl top-k memory、v3.1 best latent，
    接最小 backbone 做 byte/span masked-source latent infill。
```

### Stage 4：minimal backbone

```text
2026-07-03 口径修正:
  旧版 segment-mask latent infill 会先用 clean text 生成 readout / segment，
  再遮掉部分 readout。
  这会让 clean segment 结构和未 mask readout 携带被遮内容的信息，
  不能作为严格主线证据。

严格主线:
  先在 byte 输入上做 byte/span mask。
  FLUED 只能看到 masked source。
  segmentation / summary / memory / readout 全部基于 masked source 重新生成。
  segmentation 只是 FLUED 内部执行结果，不作为 mask 采样依据，也不作为 backbone 额外输入。
  clean source 只作为最终监督标签。
  backbone 仍然只接收 readout latent，不接收 FLUED memory。

byte baseline:
  使用同一 byte/span mask 的 masked-source byte baseline。
  不再使用 clean segment-mask baseline 作为主对照。

v3.2 latent:
  旧 v3.1 best latent mask_acc=0.1784 是 legacy clean-readout 口径，
  只能作为历史上界参考，不能作为严格验收线。

如果 v3.2 full 与 no-memory 一样：
  memory 仍未证明有用。
```

2026-07-03 Stage 4 legacy clean-readout 结果：

```text
归档:
  K:\FLUED_archive\v32_backbone_20260703

结果:
  byte segment baseline:
    mask_acc 0.1493

  v3.2 mfl no-memory MSE:
    mask_acc 0.1370

  v3.2 mfl memory MSE:
    mask_acc 0.1283

  v3.2 mfl no-memory byte_aux=1:
    mask_acc 0.1642

  v3.2 mfl memory byte_aux=1:
    mask_acc 0.1660

判断:
  legacy 口径下 v3.2 latent 略高于 byte segment baseline，
  但没有达到旧 v3.1 best latent 0.1784。
  该口径存在 clean readout / clean segment 侧漏，不能作为主线结论。
```

2026-07-03 Stage 4 strict masked-source 结果：

```text
归档:
  K:\FLUED_archive\v32_strict_backbone_20260703

严格脚本:
  tools/analysis/train_v32_strict_masked_backbone.py

严格定义:
  byte/span mask 先作用于输入 bytes。
  FLUED 在 masked source 上重新做 segmentation 和 encode。
  训练 loss 只在原始 masked byte 位置上计算。
  mask 的采样和评估都不依赖 clean segment。
  不复用 clean seg_ids / clean seg_mask / clean readout。

结果:
  strict byte baseline:
    run:      byte_3k_strict_mask
    mask_acc: 0.1472

  strict v3.2 mfl no-memory:
    run:      latent_v32_mfl_nomemory_3k_strict_fast
    mask_acc: 0.1451
    keep_acc: 0.5215
    mask_length_acc: 0.7050

  strict v3.2 mfl top-k memory:
    run:      latent_v32_mfl_memory_3k_strict_fast
    mask_acc: 0.1458
    keep_acc: 0.5460
    mask_length_acc: 0.7364

判断:
  去掉 clean segment / clean readout 侧漏后，
  当前 v3.2 latent backbone 没有超过 byte baseline。
  memory 对 mask_byte_acc 的增益很小，仅约 +0.0007；
  但 memory 对 keep_acc 和 mask_length_acc 有正向影响。

  这说明：
    1. 之前 legacy latent 优势至少部分来自侧漏或任务定义偏宽。
    2. 当前 memory tensor 没有直接暴露给 backbone，不属于“backbone 读 memory 作弊”。
       但 memory-enabled readout 仍可能在 FLUED 内部携带 memory-conditioned signal。
    3. 但 memory 可能形成了一个内部多 LOD 表示，
       对上下文保真和长度恢复有轻微帮助。
    4. 真正短板是 readout latent 对 masked byte 内容的可预测性不足。

下一步:
  不进入 Stage 5 大模型。
  先做 strict 任务下的 readout predictability 诊断：
    - 比较 masked-source readout 与 clean-source readout 的距离。
    - 看 masked unit 周边 readout 是否携带足够位置 / 类型 / 局部上下文信息。
    - 检查 memory retrieval 在 strict masked source 下是否集中到有效历史片段。
```

2026-07-03 Stage 4 strict readout predictability 诊断：

```text
脚本:
  tools/eval/eval_v32_strict_readout_diagnostics.py

输出:
  K:\FLUED_archive\v32_strict_backbone_20260703\readout_diag_nomemory.json
  K:\FLUED_archive\v32_strict_backbone_20260703\readout_diag_memory.json

诊断方式:
  对同一 masked-source segmentation：
    1. 用 masked source 编码得到 masked readout。
    2. 用 clean source 编码得到 clean oracle readout。
    3. 比较 masked / clean readout 距离。
    4. 用 frozen decoder 直接解 masked readout，看不接 backbone 时有多少信息。
  这里的 clean oracle readout 只作为离线诊断靶标；
  不允许作为 backbone 输入、训练目标输入，或严格主线的可部署流程。

no-memory:
  direct_mask_byte_acc:      0.0174
  direct_keep_byte_acc:      0.5598
  masked_unit_readout_cos:   0.3913
  masked_unit_readout_l2:    0.7011
  keep_unit_readout_cos:     0.9725

top-k memory:
  direct_mask_byte_acc:      0.0271
  direct_keep_byte_acc:      0.6049
  masked_unit_readout_cos:   0.4703
  masked_unit_readout_l2:    0.6793
  keep_unit_readout_cos:     0.9864
  retrieval_entropy:         1.2748
  memory_context_norm:       0.2898

判断:
  memory 在 strict masked-source 下确实改善了 readout 保真：
    masked unit cosine 从 0.391 提高到 0.470；
    direct masked byte acc 从 0.017 提高到 0.027；
    keep byte acc 从 0.560 提高到 0.605。

  但绝对值仍然很低。
  当前 masked-source readout 距离 clean oracle readout 太远，
  backbone 只能在信息不足的 latent 上补全，因此 3k backbone mask_acc
  只能接近 byte baseline，而不能明显超过。

结论:
  Stage 4 未通过“v3.2 latent 明显降低 backbone 难度”的强验收。
  已证明不是 backbone 直接读取 memory tensor 作弊；
  更像是 memory 形成了轻微多 LOD 辅助，但 readout 训练目标还没有让 masked-source latent
  携带足够可预测的语义 / 位置 / 类型信息。

下一步:
  暂不扩大模型。
  优先重构训练任务：
    - 在 FLUED codec 训练期加入 masked-source denoising readout / decoder objective。
    - 不使用 clean segment / clean readout 作为 backbone 输入。
    - 让 memory 对 masked-source readout 的贡献直接进入训练目标。
```

### Stage 4.1：strict masked-source codec repair

```text
2026-07-03 落地入口:
  tools/analysis/train_v32_masked_codec_2m.py
  tools/launcher/run_v32_masked_codec_2m_5080.ps1

任务定义:
  clean bytes -> byte/span mask -> masked bytes。
  FLUED 只能看到 masked bytes。
  segmentation / summary / memory / readout 都从 masked source 生成。
  clean bytes 只作为 decoder loss target。

明确禁止:
  不用 clean segment 采样 mask。
  不用 clean seg_ids / clean seg_mask 作为输入。
  不用 clean readout 作为训练输入。
  不把 FLUED memory 暴露给 backbone。

训练指标:
  masked_recon_loss / masked_recon_acc:
    主指标，判断 masked-source readout 能不能还原被遮 byte。

  keep_recon_loss / keep_recon_acc:
    防止模型为了补 masked byte 把未遮 byte 解码带崩。

  length_acc / masked_length_acc:
    判断 masked-source segmentation 后的长度预测是否还能工作。

  boundary_acc:
    仍只对 masked source 的 weak boundary 负责，不读取 memory。

  retrieval_entropy / memory_context_norm / memory_slots_per_byte:
    判断 memory path 是否实际参与 interpreter。

首轮建议:
  先跑 v3.2.1 mfl top-k memory 3k 步 smoke。
  如果 masked_recon_acc 明显高于旧 strict backbone 直接解码诊断，
  再跑 no-memory / random-retrieval 对照。

验收:
  py_compile 通过。
  CPU smoke 无 shape / NaN。
  GPU 3k loss 下降，masked_recon_acc 上升。
  summary.json 记录 clean_segment_used=false、clean_readout_used=false、mask_granularity=byte_span。

决策:
  如果 masked codec 仍学不动，优先检查 byte seed / masked segment packing / decoder path。
  如果 masked codec 学得动但 strict backbone 仍无收益，说明 readout 可解码但不可预测，
  下一步应改 readout 训练信号或接更贴近真实下游的 backbone 任务。
```

2026-07-03 3k 实验结果：

```text
masked-source codec 归档:
  K:\FLUED_archive\v32_masked_codec_2m_20260703

codec 对照:
  v321_mfl_memory_masked_3k:
    masked_recon_acc 0.1581
    keep_recon_acc   0.3612
    length_acc       0.8799
    steps/s          11.50

  v321_mfl_nomemory_masked_3k:
    masked_recon_acc 0.1575
    keep_recon_acc   0.3653
    length_acc       0.8788
    steps/s          11.44

  v321_mfl_random_masked_3k:
    masked_recon_acc 0.1576
    keep_recon_acc   0.3601
    length_acc       0.8777
    steps/s          10.94

判断:
  masked-source codec objective 是有效改动：
    direct masked byte acc 从旧 strict readout diagnostic 的 0.017/0.027
    提高到约 0.158。

  但 3k 时 top-k memory 没有拉开 no-memory / random：
    top-k 0.1581，no-memory 0.1575，random 0.1576。
    当前收益主要来自训练信号位置修正，不是 memory retrieval。

strict backbone 回测归档:
  K:\FLUED_archive\v32_strict_backbone_20260703_masked_codec

strict backbone 对照:
  byte_3k_strict_mask_recheck:
    mask_acc 0.1441

  latent_v321_mfl_nomemory_maskedcodec_3k:
    mask_acc 0.1617

  latent_v321_mfl_memory_maskedcodec_3k:
    mask_acc 0.1600

判断:
  在严格 byte/span masked-source 口径下，v3.2.1 latent 首次稳定超过 byte baseline。
  这证明“先训练 codec 的 masked-source readout/decoder objective，再接 backbone”
  比直接拿 clean reconstruction codec 接 backbone 更合理。

  memory 仍不是正贡献项：
    no-memory 0.1617 高于 top-k memory 0.1600。
    下一步不要扩大 memory；应先让 masked-codec 训练更充分，
    或做 memory 训练信号 / 专用数据场景消融。
```

2026-07-03 15k 延长实验结果：

```text
masked-source codec 归档:
  K:\FLUED_archive\v32_masked_codec_2m_20260703

codec 对照:
  v321_mfl_memory_masked_15k:
    masked_recon_acc 0.1910
    keep_recon_acc   0.5744
    length_acc       0.9316
    steps/s          12.29

  v321_mfl_nomemory_masked_15k:
    masked_recon_acc 0.1900
    keep_recon_acc   0.5751
    length_acc       0.9250
    steps/s          12.63

判断:
  15k 后 masked-source codec 继续提升：
    3k 约 0.158 -> 15k 约 0.190。
  但 top-k memory 仍没有显著分化：
    top-k 只比 no-memory 高 0.0010，且 no-memory 的 keep_recon_acc 更高。
  这不是可靠 memory 正贡献。

strict backbone 回测归档:
  K:\FLUED_archive\v32_strict_backbone_20260703_masked_codec_15k

strict backbone 对照:
  byte_3k_strict_mask_recheck:
    mask_acc 0.1440

  latent_v321_mfl_nomemory_maskedcodec15k_3k:
    mask_acc 0.1898
    keep_acc 0.5436

  latent_v321_mfl_memory_maskedcodec15k_3k:
    mask_acc 0.1895
    keep_acc 0.5448

判断:
  15k masked-codec 明显降低了 strict backbone 补全难度：
    byte baseline 0.1440 -> latent 0.1898。

  但 memory 仍没有提供 backbone 收益：
    no-memory 0.1898，top-k memory 0.1895。

  当前最稳结论:
    v3.2.1 的正确方向是 masked-source codec training。
    active memory path 在通用语料 / 128 序列 / 当前 loss 下仍近似装饰品。

复现入口:
  tools/launcher/run_v32_masked_codec_2m_5080.ps1
  tools/launcher/run_v32_strict_masked_backbone_v321_5080.ps1
```

2026-07-03 memory stress 诊断：

```text
诊断入口:
  tools/eval/eval_v32_masked_memory_stress.py
  tools/launcher/run_v32_masked_memory_stress_5080.ps1

归档:
  K:\FLUED_archive\v32_masked_codec_2m_20260703\memory_stress_15k

严格口径:
  在原始 byte 流中定位第二次及后续出现的实体 / 变量 / 术语。
  先把这些 byte 位置替换为 MASK_ID。
  FLUED 只能看到 masked source。
  segmentation / readout / memory 全部从 masked source 重新生成。
  loss 只统计被 mask 的 term byte。

top-k memory checkpoint:
  full:
    masked_acc  0.0244
    masked_loss 4.2035

  zero:
    masked_acc  0.0000
    masked_loss 4.5215

  shuffled:
    masked_acc  0.0122
    masked_loss 4.3739

  stale:
    masked_acc  0.0000
    masked_loss 4.6189

no-memory checkpoint:
  full:
    masked_acc  0.0244
    masked_loss 4.2092

  zero / shuffled / stale:
    与 full 几乎无差异，这是预期，因为没有 memory path。

分项观察:
  CJK repeat case 中 top-k memory 的 full 明显优于 zero：
    full masked_acc 0.0833，loss 3.3861；
    zero masked_acc 0.0000，loss 4.3836。

  code_identifier / version_number / English entity case 中 acc 仍为 0，
  但部分 loss 有轻微改善或混合。

判断:
  memory path 不是死代码：
    对 top-k checkpoint，full > zero / shuffled / stale，尤其 CJK repeat。

  但 memory path 仍不是主线收益：
    top-k full 与 no-memory full 的总体 masked_acc 相同，loss 只好 0.0039；
    strict backbone 15k 回测 no-memory 仍略高于 top-k。

当前决策:
  v3.2.1 主线先采用 no-memory masked-source codec，作为最强最简可复现路线。
  memory 保留为实验分支，不进入默认主线。
  后续如果继续研究 memory，应先改训练信号或构造更强的长程/代码/实体密集任务，
  而不是增加 memory slots、top-k、overflow 或动态 gate。
```
