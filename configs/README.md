# Configuration Index

## Canonical（默认起点）

| 文件 | 对应实现 | 内容 |
| --- | --- | --- |
| [`canonical_v36.json`](canonical_v36.json) | `flued/v36` + `tools/train/v3_6/` | **v3.6 线当前默认起点**（v36.1-20260731）：S0 预训接管、动态边界、4× KDA 状态、k=1 readout 包 |
| [`canonical_v35.json`](canonical_v35.json) | `flued/v34` + `tools/train/v3_4/` | v3.4 收尾系列旧口径（v35.1-20260717），保留有效 |

## 版本实验矩阵（可复现实验记录，非默认起点）

| 目录 | 对应实现 | 内容 |
| --- | --- | --- |
| [`v3_3/`](v3_3/) | `flued/v33` | v3.3 smoke、memory 分支、2M 消融和 300M 配置 |
| [`v3_4/`](v3_4/) | `flued/v34` | v3.4 位置/AR、边际编码率、emit、5K 全量消融、CBIU 三轮 |
| [`v3_5/`](v3_5/) | `flued/v34`（收尾系列） | L0 codec 20K、emit 退火 20K |
| [`v3_6/`](v3_6/) | `flued/v36` | 空——v3.6 目前只有 canonical，无实验矩阵配置 |
| [`data/`](data/) | `tools/data/` | corpus v5 语料源清单 |

配置文件只描述实验参数。训练入口位于 `tools/train/v3_*/`；漂移守卫见 `tests/test_canonical_sync.py`。
