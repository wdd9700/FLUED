# FLUED v0.4 最终实验计划

> 更新日期: 2026-05-26 | 目标: EMNLP 2026 ARR

---

## 核心原则

1. **公平对比**: 所有架构的 E3 LM 主体参数一致（28L, d=1024, 16h, FFN=4096），仅 embedding 不同
2. **全量数据**: E1 全部在 22GB corpus_v3.txt 上训练；E3 全部在新的更大数据集上训练
3. **统一步数**: E1 → 40K 步, E3 → 30K 步
4. **三种子**: FLUED 所有实验 S1(42)/S2(123)/S3(999) 三个种子

---

## 数据集

| 阶段 | 数据集 | 大小 | 用途 |
|------|--------|------|------|
| E1 | `corpus_v3.txt` | 23.8 GB, 123M 行 | 预训练 autoencoder / tokenizer |
| E3 | **新数据集** (下载中) | 更大 | 下游 LM 训练 |

---

## Phase 1: 基座训练 (当前 → ~3天)

### 1.1 FLUED E1: 三种子 × 22GB 全量

| 种子 | 状态 | 步数 | 配置 |
|------|------|------|------|
| S1 (42) | 🔄 待重训 | 0 → 40K | class300m_48gb, batch=4, grad_accum=8, streaming 22GB |
| S2 (123) | 🟢 运行中 | 24K → 40K | 同上，ckpt 每 2000 |
| S3 (999) | ⏳ 排队 | 24K → 40K | 同上 |

**注意**: S1 旧的 50K 行 checkpoint 直接废弃，用 22GB 全量 streaming 从头训练。

### 1.2 BPE Tokenizer: 三种词表 × 22GB 全量

| 词表 | 状态 | 备注 |
|------|------|------|
| 8K | 🔄 待重训 | 旧版 500K 行废弃，streaming 22GB |
| 16K | 🟡 运行中 | CPU 156min+ |
| 32K | ⏳ 排队 | 16K 完成后启动 |

### 1.3 BLT Autoencoder + ByteLM: 22GB 全量重训

| 组件 | 状态 | 备注 |
|------|------|------|
| BLT Autoencoder | 🔄 待训 | 基于 22GB |
| ByteLM (100M) | 🔄 待训 | 基于 22GB |

---

## Phase 2: E2 重建质量对比 (Phase 1 完成后)

### 2.1 FLUED E2 (3 种子上限)

每个种子加载 E1 checkpoint → eval_only 模式评估：

| 种子 | 指标 |
|------|------|
| S1 (42) | recon_acc, compression, bp_std, per-type bp |
| S2 (123) | 同上 |
| S3 (999) | 同上 |

### 2.2 BPE E2 (3 词表)

每个词表从头训练 BPE autoencoder → eval：

| 词表 | 指标 |
|------|------|
| 8K | recon_acc, compression |
| 16K | 同上 |
| 32K | 同上 |

### 2.3 BLT E2

BLT autoencoder eval。

---

## Phase 3: E3 下游 LM 公平对比 (新数据集, 30K 步)

### 架构约束（严格对齐）

所有 E3 LM 使用相同主体：

```
CausalTransformerLM:
  d_model=1024, nhead=16, dim_feedforward=4096, num_layers=28
  max_seq_len=256, dropout=0.0
```

| 架构 | Embedding | 额外参数 | 主体参数 |
|------|-----------|---------|---------|
| FLUED | byte_embed(257) | 冻结 FLUED encoder | 352.9M |
| BPE 8K | token_embed(8192) | 冻结 BPE encoder | 361.1M |
| BPE 16K | token_embed(16384) | 冻结 BPE encoder | ~370M |
| BPE 32K | token_embed(32768) | 冻结 BPE encoder | ~387M |
| BLT | byte_embed(257) | 冻结 BLT encoder + ByteLM | 352.9M |

**差异仅在于 embedding 表**，主体参数完全一致。

### 3.1 FLUED E3 (3 种子)

| 种子 | E1 来源 | 步数 | 数据 |
|------|---------|------|------|
| S1 (42) | Phase 1.1 | 30K | 新数据集 |
| S2 (123) | Phase 1.1 | 30K | 新数据集 |
| S3 (999) | Phase 1.1 | 30K | 新数据集 |

### 3.2 BPE E3 (3 词表)

| 词表 | E2 来源 | 步数 | 数据 |
|------|---------|------|------|
| 8K | Phase 2.2 | 30K | 新数据集 |
| 16K | Phase 2.2 | 30K | 新数据集 |
| 32K | Phase 2.2 | 30K | 新数据集 |

### 3.3 BLT E3

| 模型 | E2 来源 | 步数 | 数据 |
|------|---------|------|------|
| BLT | Phase 2.3 | 30K | 新数据集 |

### 核心产出表

| Model | Seed/Vocab | E3 bpb | Trainable | LM 主体 |
|-------|-----------|--------|-----------|---------|
| FLUED | S1 (42) | ? | 353.2M | 352.9M |
| FLUED | S2 (123) | ? | 353.2M | 352.9M |
| FLUED | S3 (999) | ? | 353.2M | 352.9M |
| BPE | 8K | ? | ~369M | 352.9M |
| BPE | 16K | ? | ~375M | 352.9M |
| BPE | 32K | ? | ~390M | 352.9M |
| BLT | — | ? | 353.5M | 352.9M |

---

## Phase 4: FLUED 消融实验

| 实验 | 消融项 | 步数 | 数据 |
|------|--------|------|------|
| A1 | λ_type=0 | 10K | 22GB streaming |
| A2 | λ_var=0 | 10K | 22GB streaming |
| A3 | λ_entropy=0 | 10K | 22GB streaming |
| A4 | λ_utf8=0 | 10K | 22GB streaming |
| A5 | 全 λ=0 (仅 recon) | 10K | 22GB streaming |

---

## 实验总数: 22 个

| 类别 | 数量 | 详情 |
|------|------|------|
| FLUED E1 | 3 | S1/S2/S3 |
| FLUED E3 | 3 | S1/S2/S3 |
| BPE Tokenizer | 3 | 8K/16K/32K |
| BPE E2 | 3 | 8K/16K/32K |
| BPE E3 | 3 | 8K/16K/32K |
| BLT E1 | 1 | autoencoder + ByteLM |
| BLT E3 | 1 | downstream |
| Ablation | 5 | A1-A5 |
| **合计** | **22** | |

---

## 执行顺序 & 依赖

```
Phase 1 (并行)
├── S2 E1 ────────┐
├── S3 E1 ────────┤
├── S1 E1 (重训) ─┤──→ Phase 2 FLUED E2
├── BPE 16K ──────┤
├── BPE 32K ──────┤──→ Phase 2 BPE E2
├── BPE 8K (重训) ─┘
└── BLT E1 ───────→ Phase 2 BLT E2

Phase 2 (E2 评估, 快)
├── FLUED S1/S2/S3 eval
├── BPE 8K/16K/32K train+eval
└── BLT eval

Phase 3 (E3, 等新数据集就绪)
├── FLUED ×3 (30K步)
├── BPE ×3 (30K步)
└── BLT ×1 (30K步)

Phase 4 (消融, 可随时并行)
└── A1-A5 (10K步)
```

---

## 代码改动清单

| 文件 | 改动 | 原因 |
|------|------|------|
| `flued/e3_train.py` | `--max-steps` 默认改为 30000 | 统一 E3 步数 |
| `flued/e3_train.py` | 支持新数据集路径 `--data-path` | E3 在新数据上训练 |
| `flued/e3_downstream.py` | 确认 LM 主体参数一致性 | 代码审查 |
| `train_bpe.py` | 已支持 streaming (max-lines=0) | ✅ 无需改动 |
| `run_ablation.ps1` | 改为 streaming 22GB (去掉 max-lines) | 全量数据消融 |

---

## 风险 & 对策

| 风险 | 对策 |
|------|------|
| VRAM OOM | batch_size=4 已验证稳定 |
| BPE 内存爆炸 (42GB+) | 32K 时考虑分批或采样 |
| 磁盘不足 (E:88G, F:40G) | ckpt 每轮保留最后 2 个，旧版及时删 |
| 新数据集下载延迟 | Phase 3 可先用现有 22GB 跑 FLUED E3，后续用新数据重跑 BPE/BLT |
| S1/S2/S3 训练不收敛 | 对比 early step 指标, 必要时调 λ |
