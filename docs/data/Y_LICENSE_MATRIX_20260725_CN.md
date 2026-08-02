# Y 盘语料许可分层（2026-07-25）

本表用于决定哪些下载完成的数据可以进入 `corpus_v5`。许可判断优先采用官方数据卡、原项目和原始许可；数据集卡上的宽松标签不能自动覆盖底层新闻、网站或上游合成数据权利。此表是工程风险筛查，不是法律意见。

## 当前启用

| 来源 | 许可 | 工程处理 | 官方来源 |
|---|---|---|---|
| OASST2 | Apache-2.0 | 过滤 `deleted=true`；保留数据集级归因 | https://huggingface.co/datasets/OpenAssistant/oasst2 |
| QASC | CC-BY-4.0 | 保留问题、选项、答案和支持事实 | https://huggingface.co/datasets/allenai/qasc |
| HotpotQA | CC-BY-SA-4.0 | 保留嵌套上下文、问题和答案；发布时执行相同方式共享 | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| SQuAD | CC-BY-SA-4.0 | 保留标题、上下文、问题和答案；发布时执行相同方式共享 | https://huggingface.co/datasets/rajpurkar/squad |
| Databricks Dolly 15K | CC-BY-SA-3.0 | 保留 instruction/context/response；发布时执行相同方式共享 | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| BoolQ | CC-BY-SA-3.0 | 保留 passage、question 和布尔答案 | https://huggingface.co/datasets/google/boolq |
| ARC-Easy | CC-BY-SA-4.0 | 保留问题、选项和答案 | https://huggingface.co/datasets/allenai/ai2_arc |

启用来源均在状态库的 `v5_records.source` 中保留来源标识，`source_provenance.json` 保存许可和官方链接。含 CC-BY-SA 的清洗后数据公开时不能统一改标为 Apache/MIT。

## 当前排除

| 来源 | 排除原因 |
|---|---|
| cnn_dailymail | 新闻正文版权未因数据卡 Apache 标签而转让 |
| ice-en | 非商业学术研究许可，禁止再分发 |
| race | 非商业研究且限制派生数据再分发 |
| alpaca-cleaned、no-robots、sciq | CC-BY-NC，禁止商业使用 |
| ag_news、clue、cos_e、webquestions、wsc | 总体或底层内容许可未知/不统一 |
| wikiqa | Microsoft 自定义研究许可，不适合作为严格开放语料 |
| COIG-PC-core | 混合来源和 gated EULA，不能整体放行 |
| capybara、Chinese-alpaca、orca_dpo | 名称或版本不唯一，上游合成数据许可尚未逐源厘清 |
| wikitext-103 | 数据卡中的 CC-BY-SA/GFDL 版本口径冲突，待固定具体版本 |
| OpenR1-Math-220k | Apache 标签之外仍需核对 NuminaMath 和模型生成链路 |

其余已确认下载不完整、已进入 corpus_v4 或存在不可逆乱码的来源，继续按 `corpus_v5_sources_20260725.json` 的 `excluded_sources` 排除。
