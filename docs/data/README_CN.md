# FLUED corpus_v5 增量清洗管线

该管线只读处理 W/X/Y 新下载语料，在 `N:\FLUED_corpus_v5_increment_20260725` 生成训练可读的 UTF-8 文本增量。

## 不可破坏的约束

1. 不修改 W/X/Y 原始下载，也不修改 `N:\FLUED_corpus_v4`。
2. 从 corpus_v4 只读复制 SQLite 哈希库，按相同的空白折叠、小写化和 Blake2b-128 合同，对 E 盘 `corpus_v3.txt`、N 盘 corpus_v4 和本轮全部来源做精确去重。
3. “零重复”只指规范化文本级精确重复，不虚构语义近重复为零。
4. FineWeb-Edu 先于 FineWeb，清洗后完全相同的网页保留高质量来源。
5. 代码保留原缩进，只接受逐记录许可白名单；未知许可、vendor、生成物、lock 文件、密钥和压缩代码全部拒绝。
6. Qwen3.5-9B 只返回 `keep/drop_paragraphs/split_after`，正文永远由原段落重组；失败、超时、缺 ID、越界、整篇删除或异常响应全部回退到确定性切分。
7. 已知不完整、乱码或许可来源不清的数据不进入正式输入。
8. 正式构建前必须完成匹配当前配置的深度验证；输入文件、配置或基线库发生变化后拒绝续跑。
9. 单实例运行锁和 pending journal 防止并发写坏与中断重复追加；恢复前强制核对状态库、索引和全部已有分片的连续覆盖关系。
10. 全局输出上限 300 GiB，并为 N 盘保留至少 80 GiB 可用空间。

## 当前输入

| 来源 | 处理方式 | 许可口径 |
|---|---|---|
| FineWeb-Edu 2025-08 | 50% 确定性抽样，教育评分至少 3 | ODC-By 1.0 + Common Crawl 条款 |
| FineWeb-Edu 2024-51 | 45% 确定性抽样，教育评分至少 3 | ODC-By 1.0 + Common Crawl 条款 |
| FineWeb 2024-51 | 18% 确定性抽样，语言评分至少 0.82 | ODC-By 1.0 + Common Crawl 条款 |
| codeparrot/github-code | 25% 确定性抽样，逐记录宽松许可过滤 | 原仓库许可白名单 |
| Y: OASST2、QASC | 全量；使用数据集专用结构化格式器 | Apache-2.0 / CC-BY-4.0 |
| Y: HotpotQA、SQuAD、Dolly、BoolQ、ARC-Easy | 全量；保留上下文、选项和答案 | CC-BY-SA-3.0/4.0，公开时保留归因和相同方式共享义务 |

Y 盘其余 18 个候选因非商业限制、许可未知、上游版权不清或具体版本冲突而排除。Y 盘代码指令集不与 GitHub-code 混合：其记录缺少可核验的原仓库许可。C4/Falcon/codeparrot 的部分下载、已进入 corpus_v4 的来源、已确认乱码的数据和临时缓存均显式排除。

## Qwen 审核

LM Studio 当前接口：

```text
model: qwen/qwen3.5-9b
quantization: Q4_K_M
context: 128000
parallel: 8
flash attention: enabled
```

生产配置为 4 路并发、每请求 1 篇文档、60 秒超时、最多 50000 篇。Qwen 只用于 Y 盘混合结构文本的疑难旁路；FineWeb/FineWeb-Edu 使用其质量分数和确定性清洗，GitHub 代码使用许可与代码规则，三者均不在正式写入主路径调用 Qwen。长但干净的多段正文不再仅因“异质”进入模型。

2026-07-25 的真实 FineWeb-Edu 校准中，64 条输入产生 32 条 Qwen 审核；8 次组级响应不完整均被拆成单条自动重试，最终 `qwen_fallback=0`。异步审核与下一批 CPU/NAS 清洗重叠，同规模耗时由约 64.5 秒降至 39.4 秒。完整输入候选记录在 `qwen_inputs.jsonl`，决策和失败尝试记录在 `qwen_audit.jsonl`。

同日的压力测试还确认：`128k context × 8 parallel` 会占满 16 GB 显存，把 16 篇长文同时送入 9B 模型会触发 180 秒超时。运行时已改为 `32k × 4`；32 条结构化探针全部有效，21.9 秒完成。Qwen 是疑难样本旁路，不是全语料分段器；超时记录保留在审计日志，正文自动回退且不会被删除。

接口探针：

```powershell
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe .\probe_lmstudio_qwen.py `
  --workers 8 --batch-size 2 --output .\lmstudio_probe.json
```

## 执行顺序

先深度验证。Parquet 会读取并解压全部文本列，gzip 会验证 CRC、UTF-8 和每一行 JSON。结果按文件缓存，中断后继续：

```powershell
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe .\build_corpus_v5.py `
  --config .\corpus_v5_sources_20260725.json `
  --validate-only
```

只有 `validation_manifest.json` 覆盖全部启用来源且 `invalid=0` 时，正式构建才会启动：

```powershell
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe .\build_corpus_v5.py `
  --config .\corpus_v5_sources_20260725.json
```

构建完成后必须运行独立验证器。它要求分片清单存在，并检查参考库零重叠、状态库新增数等于索引数、索引连续覆盖分片全部字节、每条内容哈希和分片 SHA-256：

```powershell
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe .\verify_corpus_v5.py `
  --output-dir N:\FLUED_corpus_v5_increment_20260725 `
  --reference-db N:\FLUED_corpus_v4\state\dedupe_hashes.sqlite3
```

无需安装 pytest 的快速测试：

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe -m py_compile build_corpus_v5.py verify_corpus_v5.py test_build_corpus_v5.py
```

小样本必须使用独立输出目录和 `--dry-run`，避免生成不完整的正式状态：

```powershell
C:\Users\74090\Miniconda3\envs\soulvlm\python.exe .\build_corpus_v5.py `
  --config .\smoke_qwen_config.json --dry-run --max-records-per-source 100
```

## 产物

```text
N:\FLUED_corpus_v5_increment_20260725\shards\
N:\FLUED_corpus_v5_increment_20260725\shards.txt
N:\FLUED_corpus_v5_increment_20260725\state\dedupe_hashes.sqlite3
N:\FLUED_corpus_v5_increment_20260725\manifests\validation_manifest.json
N:\FLUED_corpus_v5_increment_20260725\manifests\shard_manifest.csv
N:\FLUED_corpus_v5_increment_20260725\reports\source_stats.csv
N:\FLUED_corpus_v5_increment_20260725\reports\FINAL_REPORT_CN.md
N:\FLUED_corpus_v5_increment_20260725\logs\qwen_audit.jsonl
```

长任务由 `run_corpus_v5_managed.ps1` 托管。`reports\managed_status.json` 记录运行、完成或失败状态；构建退出码允许正常完成或达到安全容量上限，随后必须由独立验证器返回 0 才标记 `complete`。
