# 语料库存对照与缺口清单（2026-08-07）

> 背景：用户计划接 1TB 机械盘专存长对话/长文档开源语料。本文是"已有 vs 缺口"
> 对照，防止重复下载。基线台账：`docs/data/SOURCE_AUDIT_20260725_CN.md`、
> `P:\FLUED_corpus\FLUED_corpus_v4\reports\source_stats.csv`（v4 全量成分）。

## 1. 已在库存（勿重复下载）

| 类别 | 内容 | 位置/体积 |
|---|---|---|
| wiki | enwiki + zhwiki 全量文章 | v4 内，17.7 GiB |
| web 英文 | C4-en、CommonCrawl-WET×2、FineWeb、FineWeb-Edu | v4 内，~130 GiB |
| 多语 web | multilingual-cc | v4 内，16 GiB |
| 合成教育 | cosmopedia | v4 内，10.7 GiB |
| STEM/数学 | proof-pile、OpenWebMath、finemath、arxiv-summarization（摘要级） | v4 内，~32 GiB |
| 代码 | starcoder-data | v4 内，17.2 GiB |
| QA/指令小份 | OASST2、QASC、HotpotQA、SQuAD、Dolly、BoolQ、ARC-Easy | v4 内，0.25 GiB（Y 盘许可层） |
| v5 增量（已审计未落地） | FineWeb 2024-51 全量 362.58 GiB、FineWeb-Edu 2024-51+2025-08 共 57.3 GiB、codeparrot/github-code 许可白名单版 279.87 GiB | **暂存于已拆下的 W/X/Y 盘，深读校验全过；挂盘后逐盘核查 `corpus_v5_sources_20260725.json` 所列路径** |

## 2. 真正缺口（下载清单，按优先级）

| 优先级 | 项 | 规模（约） | 用途 |
|---|---|---:|---|
| P0 | LoCoMo / LongBench / QMSum / NarrativeQA / RULER / InfiniteBench / MSC / Conversation Chronicles | <10 GB | 文档级评测集（D5 考场：状态通道正式判决与翻页曲线的需求前置） |
| P1 | MNBVC 中文分类子集（小说/书籍/杂志/论文/台词/聊天记录） | 200-300 GB | 中文长文档主力（库存最大洞：现有中文料仅 zhwiki + multilingual-cc，无书籍级长文） |
| P1 | Project Gutenberg（或 PG-19） | 30-40 GB | 英文书籍长叙事 |
| P2 | arXiv 全文（现库存为摘要集） | ~100 GB | 长科技文档 |
| P2 | 长对话：LMSYS-Chat-1M / WildChat / UltraChat / MOSS-003 / Belle / COIG-CQIA | 50-80 GB | 会话级序列（档案馆模型/J-Space 轮次级形态原料）；ShareGPT 许可灰色跳过 |

## 3. 施工纪律

1. 与 v3/v4 及全部评测集精确去重（blake2b-128 规范化哈希合同沿用，
   参考库 `state/dedupe_hashes.sqlite3`）；评测集与训练料隔离。
2. 清洗管线**必须保留文档边界标记**——v3/v4 均无边界（规格 §6 记录的坑），
   本批料的最大价值就是带边界。
3. 1TB 盘存 raw，清洗后预计剩 300-500 GB 可用。
4. 挂盘后第一步：逐盘核对 v5 暂存（W/X/Y 三盘），能救回 700GB 级已校验下载。

## 4. 待办挂账

- [ ] W/X/Y 盘逐盘点库（用户挂盘后执行）：核查 v5 暂存完整性，产出
   revival 清单（哪些可直接清洗进新盘）。
- [ ] P0 评测集下载与建集（不等硬盘，周内可先行）。
- [ ] MNBVC 子集选择清单（按分类目录挑小说/书籍/论文等长文类）。
