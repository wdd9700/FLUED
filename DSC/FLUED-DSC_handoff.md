
# FLUED-DSC 代码包交接文档
## 可直接交接给 Coding Agent

---

## 文件清单

| 文件 | 说明 | 行数 |
|------|------|------|
| [FLUED-DSC_model.py](sandbox:///mnt/agents/output/FLUED-DSC_model.py) | 核心模型实现 | ~350 |
| [FLUED-DSC_train.py](sandbox:///mnt/agents/output/FLUED-DSC_train.py) | 训练脚本 (两阶段) | ~280 |
| [FLUED-DSC_benchmark.py](sandbox:///mnt/agents/output/FLUED-DSC_benchmark.py) | 对照实验框架 | ~250 |
| [FLUED-DSC_math_derivation.md](sandbox:///mnt/agents/output/FLUED-DSC_math_derivation.md) | 数学推导 | ~200 |

---

## 核心架构

```
Byte Sequence X ∈ R^(B×T)
    ↓
Embedding + PE
    ↓
[CoarseNet] ──→ Z ∈ R^(B×T×d_c)  (语义表示)
    ↓
Z Z^T / τ ──→ S ∈ R^(B×T×T)  (相似度)
    ↓
α-entmax ──→ P ∈ R^(B×T×T)  (稀疏先验, 每行≈k个非零)
    ↓
Top-k ──→ indices ∈ R^(B×T×k)
    ↓
Q, K, V from H (FLUED hidden states)
    ↓
Sparse Attention: o_i = Σ_{j∈N_i} softmax( (q_i^T k_j)/√d + log P_ij - μ_i ) v_j
    ↓
FLUED 原有流程: ΔH → boundary_head → Soft Assignment → DSC⁻¹
```

---

## 关键数学公式

### 1. 标准 Attention (Full)
```
O = D^{-1} A V,  A_{ij} = exp( (q_i^T k_j) / √d )
复杂度: O(T² · d)
```

### 2. DSC-Attention
```
Z = CoarseNet(H) ∈ R^(T×d_c)
S_{ij} = (Z_i · Z_j) / τ
P_{i,:} = α-entmax(S_{i,:})  (稀疏, 可微)
N_i = topk_j(P_{ij}, k)

s_{ij} = (q_i^T k_j)/√d + log P_{ij},  j ∈ N_i
μ_i = (1/k) Σ_{j∈N_i} s_{ij}  (零和中心化, CPA借鉴)
s'_{ij} = s_{ij} - μ_i

o_i = Σ_{j∈N_i} softmax(s'_{ij}) v_j
复杂度: O(T · k · d)
```

### 3. 复杂度对比 (T=8192, d=512, k=512)

| 方案 | 复杂度 | FLOPs | 激活显存 |
|------|--------|-------|----------|
| Full | O(T²d) | 68.7 G | 0.27 GB |
| DSC | O(Tkd) | 4.3 G | 0.02 GB |
| 加速比 | **16x** | **16x** | **16x** |

---

## 超参配置

```python
DSC_CONFIG = {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 24,

    # CoarseNet
    "d_c": 128,           # 语义维度
    "coarse_layers": 2,   # 层数
    "coarse_heads": 2,    # 头数
    "coarse_window": 512, # SWA窗口

    # 稀疏选择
    "alpha": 1.5,         # entmax稀疏度 (1.2→训练初期, 1.5→推荐, 2.0→最稀疏)
    "k_neighbors": 512,   # 邻居数 (T=8192时6.25%)
    "tau": 1.0,           # 温度

    # 零和中心化 (CPA借鉴)
    "use_zero_sum": True,
    "zs_temperature": 1.0,

    # 训练
    "dropout": 0.1,
    "coarse_lr_mult": 2.0,  # CoarseNet学习率倍数
    "warmup_steps": 1000,   # 预热步数
}
```

---

## 对照实验矩阵

| 组别 | 模型 | 复杂度 | 预期时间 | 决策点 |
|------|------|--------|----------|--------|
| A1 | Full FlashAttn | O(T²d) | ~30天 | 基线(可能跑不完) |
| A2 | SWA (w=2048) | O(Twd) | ~7天 | 工业界保底 |
| A3 | DSC (k=512, α=1.5) | O(Tkd) | ~2天 | **主推方案** |
| A4 | DSC (k=256, α=1.5) | O(Tkd) | ~1.5天 | 更激进稀疏 |
| A5 | DSC (k=512, α=1.2) | O(Tkd) | ~2天 | 验证稀疏度影响 |
| A6 | Linear Attention | O(Td²) | ~2天 | 纯O(T)对比 |

**决策规则**:
- A3 BPB < A2 + 1% → 锁定DSC (论文核心卖点)
- A3 BPB > A2 + 3% → 退守SWA
- A4 与 A3 差距 < 1% → 可压缩到 k=256

---

## 安装依赖

```bash
pip install torch entmax
```

---

## 快速验证

```python
# 测试 T=8192 能否在 5080 上跑起来
import torch
from FLUED-DSC_model import DSCAttention

model = DSCAttention(d_model=512, d_c=128, num_heads=8, 
                     alpha=1.5, k_neighbors=512).cuda()
x = torch.randn(1, 8192, 512).cuda()
out = model(x)
print(f"显存: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

---

## 与 ByteFlow 的论文定位

| 维度 | ByteFlow | FLUED-DSC |
|------|----------|-----------|
| 分块信号 | ΔR_t (编码率) | ΔH + Z Z^T (语义相似度) |
| 分块方式 | Hard Top-K | Soft α-entmax |
| 可微分 | 否 (STE) | **是** (完整梯度) |
| 架构 | 4模块层次化 | 单层 + 选择器 |
| 参数 | 600M | 302M + 1.5M |
| 复杂度 | O(T·w + K²) | **O(T·k·d)** |
| 理论 | 率失真 | 稀疏优化 + 零和中心化 |

**核心论点**:
> "ByteFlow 用信息论硬切分序列，不可微；FLUED-DSC 用稀疏优化软选择邻居，
> 保持端到端可微分，且不需要层次化架构即可达到同等长程建模能力。"

---

## 风险与 Fallback

| 风险 | 症状 | 对策 | Fallback |
|------|------|------|----------|
| CoarseNet不收敛 | Loss震荡 | 降LR, 增预热 | 固定P为SWA+锚点 |
| entmax梯度消失 | grad_norm≈0 | α降到1.2 | 用softmax替代 |
| Gather太慢 | 只快2x | 增batch size | 用block-sparse |
| 长程丢失 | 括号匹配<80% | 加全局锚点 | 90%DSC+10%Full |
| Soft Path瓶颈 | 总时间没降 | A矩阵低秩近似 | hard mean-pool |

---

## 执行顺序 (12天)

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | 验证 T=8192 显存, 跑通 model.py | 确认可行性 |
| Day 2-3 | 跑 A2 (SWA) 基线 | BPB锚点 |
| Day 4-6 | 跑 A3 (DSC main) | 核心数据 |
| Day 7-8 | 跑 A4/A5 (ablation) | 稀疏度分析 |
| Day 9 | 跑 A6 (Linear对比) | 纯O(T)数据 |
| Day 10-12 | 论文图表 + 分析 | 可提交结果 |

---

**交接完成。Coding Agent 可直接基于上述文件开始实现。**
