# FLUED corpus_v4 本地语料整理计划

## 目标

生成一份新的 FLUED 通用训练语料 `corpus_v4`：

- 输出位置：`<corpus-v4-root>`
- 编码：UTF-8
- 形式：多个文本分片，整体视为一份语料
- 去重：按规范化后的文档/行内容哈希去重
- 保留：`<corpus-path>`
- 禁动：`N:\ocr_datasets`

## 完成状态

已完成构建与清理。最终报告：

```text
<corpus-v4-root>\reports\FINAL_REPORT_CN.md
```

最终语料：

```text
<corpus-v4-root>\shards
<corpus-v4-root>\shards.txt
```

实际输出为 57 个 UTF-8 分片，总计 224.929GiB。`N:\ocr_datasets` 和 E 盘 `corpus_v3.txt` 均已保留。

## 当前来源判断

已有训练主要使用：

```text
<corpus-path>
```

该文件约 22.14GB，SHA256：

```text
ED4AB97810776CE7E181D3C123729B3FD4EB6BB00BC03593AF1F669D8CFC0FCE
```

它来自早期 `build_v3_corpus_smart.py` 的配比构建，原始候选约 135.7GB，覆盖中文、英文、数理、代码、多模态描述。主要输入路径对应历史 Linux 挂载，映射到当前本机后大致是：

- `O:\v11`
- `O:\soulvlm_data`
- 部分 E 盘临时语料

因此 `O:\v11` 不应默认再次全量加入，否则会把“构建后的语料”和“构建前的源数据”重复输入，导致比例失真。

## 盘点结论

### N 盘

- `N:\ocr_datasets` 是 OCR 禁动区。
- 空间最宽裕，适合放最终语料。

### O 盘

- `O:\v11` 是主要历史源数据盘。
- `O:\soulvlm_data` 更像早期 raw/cache 目录。
- `O:\v11\_tokenizer_train_v1.txt`、`_tokenizer_train_v3*.txt` 是旧拼接语料候选，后续清理前需要确认。

### W / X 盘

- 主要是 `Corpus\general\commoncrawl-wet`。
- W 盘另有 `enwiki-latest-pages-articles.bz2` 和 `zhwiki-latest-pages-articles.bz2`。
- W/X Common Crawl 更像跨盘分片，不是简单重复。
- 有 `.tmp` 未完成文件，默认不进入构建。

### Y / Z 盘

有不少 v3 未明确覆盖的新源：

- FineWeb / FineWeb-Edu
- Cosmopedia
- C4
- Proof-Pile
- StarCoderData
- multilingual-cc
- FineMath / OpenWebMath / arxiv-summarization

同时存在 HF cache 和 temp 下载目录，删除前要区分是否已经合并进 `v11`。

## 构建策略

脚本：

```text
<repo-root>\tools\data\build_corpus_v4.py
```

默认输出：

```text
<corpus-v4-root>
```

默认输入：

1. `corpus_v3.txt` 作为基线。
2. W 盘 Wikipedia dump。
3. W/X Common Crawl WET，按源限额加入。
4. Y/Z 中较明确的新源。

默认不加入：

- `O:\v11` 原始源全量。
- `Y:\hf_cache` / `Z:\hf_cache`
- `Y:\temp_hf_dl` / `Z:\temp_hf_dl`
- 所有 `.tmp` 文件
- `N:\ocr_datasets`

原因：这些目录要么已经进入 v3，要么是缓存/临时分片，要么是 OCR 禁动区。

## 去重定义

每条文本先做规范化：

- 统一 UTF-8 解码，非法字节忽略。
- 去除控制字符。
- 规范化空白。
- HTML 实体解码。
- 对 wiki XML 做轻量模板、链接、ref 清理。

然后用规范化文本的小写版本计算 `blake2b-128` 哈希，写入 SQLite。

正式构建时 SQLite 放在 C 盘，避免 N 盘大量随机写拖慢速度：

```text
<state-dir>\dedupe_hashes.sqlite3
```

这不是语义去重，但足够避免同一文档/同一行内容重复进入最终训练语料。

## 运行方式

先 dry-run：

```powershell
cd <repo-root>
python tools\data\build_corpus_v4.py --dry-run
```

正式构建：

```powershell
python tools\data\build_corpus_v4.py --target-gb 320 --state-dir <state-dir>
```

可恢复重跑；已经完成的源会在 SQLite 状态中标记。

## 输出文件

```text
<corpus-v4-root>\shards\corpus_v4_00000.txt
<corpus-v4-root>\shards\corpus_v4_00001.txt
...
<corpus-v4-root>\manifests\selected_sources.csv
<corpus-v4-root>\reports\source_stats.csv
<state-dir>\dedupe_hashes.sqlite3
```

## 删除原则

删除只在最终语料构建成功、报告生成、且确认没有遗漏后执行。

明确保留：

- `<corpus-path>`
- `N:\ocr_datasets`
- `<corpus-v4-root>`
- FLUED 工程和训练检查点归档

候选删除，待最终确认：

- `O:\v11\_tokenizer_train_v1.txt`
- `O:\v11\_tokenizer_train_v3.txt`
- `O:\v11\_tokenizer_train_v3_clean.txt`
- W/X/Y/Z 中已成功纳入 `corpus_v4` 且不再需要保留原始格式的语料目录
- `hf_cache` / `temp_hf_dl` 中确认已整理到 `v11` 或 `corpus_v4` 的缓存

当前不自动删除任何源数据。

## W/X/Y/Z 后续补充名单

建议后续补充或完善：

- DCLM filtered web
- FineWeb / FineWeb-Edu 完整分片
- Dolma
- SlimPajama / RedPajama
- C4 / mC4
- OSCAR / CC100
- Wikipedia 多语种 dump
- The Stack v2 / StarCoderData
- OpenWebMath / FineMath / Proof-Pile-2
- arXiv / PubMed / PMC-OA
- 中文百科、新闻、问答、论坛清洗语料
- 代码专用：CodeSearchNet、GitHub code corpora
- OCR/文档方向单独维护，不混入 `corpus_v4` 删除流程
