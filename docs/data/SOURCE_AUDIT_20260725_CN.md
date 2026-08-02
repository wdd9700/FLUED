# corpus_v5 来源与风险审计（2026-07-25）

## 基线

- `E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt`：23,777,636,695 bytes。
- `N:\FLUED_corpus_v4`：57 个文本分片，约 224.93 GiB。
- `N:\FLUED_corpus_v4\state\dedupe_hashes.sqlite3`：83,896,715 个规范化精确哈希，已覆盖 corpus_v3 和 corpus_v4 的合格条目。
- corpus_v5 不重写基线，只复制哈希库并写增量分片。

## 输入状态

| 来源 | 发现文件 | 原始体积 | 当前决策 | 证据/备注 |
|---|---:|---:|---|---|
| FineWeb-Edu 2025-08 | 15 Parquet | 约 21.34 GiB | 条件纳入 | `.incomplete` 排除；深度校验 15/15 通过 |
| FineWeb-Edu 2024-51 | 50 Parquet | 约 35.98 GiB | 条件纳入 | 教育评分至少 3；完整深读校验 50/50 通过 |
| FineWeb 2024-51 | 200 Parquet | 约 362.58 GiB | 条件纳入 | 在 Edu 后处理；语言评分至少 0.82；完整深读校验 200/200 通过 |
| codeparrot/github-code | 1044 Parquet | 约 279.87 GiB | 逐记录纳入 | 完整深读校验 1044/1044；原仓库许可白名单；排除 vendor、生成物、密钥、压缩代码 |
| Y 盘许可通过层 | 7 gzip JSONL | 约 0.25 GiB 压缩数据 | 纳入 | OASST2/QASC/HotpotQA/SQuAD/Dolly/BoolQ/ARC-Easy 均逐文件深读通过 |
| Y 盘其余候选 | 18 个数据集 | 不计入正式输入 | 排除 | 非商业、未知许可、上游权利不清或版本冲突；详见 `Y_LICENSE_MATRIX_20260725_CN.md` |

## 许可

FineWeb 与 FineWeb-Edu 的原始数据集卡声明 ODC-By 1.0，同时受 Common Crawl 使用条款约束。正式发布语料清单必须保留归属和条款链接。

`codeparrot/github-code` 的数据集卡将总体许可标为 `other`，但每条记录包含原仓库许可。管线不把数据集整体 Apache 声明当作文件许可，而只保留明确属于 MIT、Apache-2.0、BSD、ISC、CC0、Unlicense、Zlib、MPL-2.0 等白名单的记录；缺失许可直接拒绝。

Y 盘混合数据集不能共享一个许可结论，因此配置拆成七个独立来源，状态库逐条保留 `source`，清单逐来源记录许可和官方链接。`transformers-code`、`codealpaca`、`code-contests` 因缺少逐记录原仓库许可，不进入本轮代码来源。

## 明确排除

- `Y:\v11\C4-en`：进度记录未完成，约 243.8 GiB 部分 gzip。
- `Y:\v11\Falcon-RefinedWeb`：下载日志记录失败。
- `Y:\v11\codeparrot-github`：下载不完整，并与 X 盘来源重叠。
- `Y:\v11\multilingual-cc`、`arxiv-summarization`：已进入 corpus_v4。
- `zhihu-kol`、`firefly-zh`、`math-shepherd`：抽样确认存在不可逆乱码。
- `NuminaMath-CoT`：下载内存溢出，仅有少量部分输出。
- `.incomplete`、缓存、`temp_hf_dl`：不是语料输入。

## 去重和质量边界

正式可证明的是规范化文本级精确零重复：输出文本在写入前同时查询 corpus_v4 基线哈希和本轮状态库。FineWeb 自身已有去重，但轻微改写、模板变体和语义近重复不可能仅靠精确哈希证明为零；最终报告必须保留这一区分。

Qwen3.5-9B 只处理确定性规则难以判断的文本段落，不能纠正乱码或生成替代正文。模型输出只含段落编号，任何失败或整篇删除建议都保留确定性清洗结果。长文并发压力测试显示 16 篇 6k–10k 字符文本会在 180 秒超时，因此已取消“长且多段即送审”的条件，只保留可疑文本和极低比例抽查。
