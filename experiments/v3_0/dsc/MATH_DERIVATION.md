
# FLUED-DSC: 完整数学推导与工程实现文档
# 可直接交接给 Coding Agent

---

## 1. 问题定义与符号体系

### 1.1 输入与输出
- 字节序列: $X \in \{0, 1, ..., 256\}^{B \times T}$, 其中 $B$ 为 batch size, $T$ 为序列长度
- 嵌入表示: $H^{(0)} = \text{Embed}(X) + \text{PE} \in \mathbb{R}^{B \times T \times d}$
- 目标: 预测下一个字节 $p(x_{t+1} | x_{\leq t})$

### 1.2 FLUED 原有结构（保持不变）
FLUED 使用 TiedBlock 堆叠:
```
H^{(l+1)} = H^{(l)} + \text{Attn}(\text{LN}(H^{(l)})) + \text{FFN}(\text{LN}(H^{(l)}))
```

边界检测（完全保留）:
```
\Delta H_t = H_{t+1} - H_t \in \mathbb{R}^d
p_t = \sigma(W_b \cdot \Delta H_t) \in [0,1]
```

Soft Assignment（完全保留）:
```
A = \text{softmax}(p / \tau) \in \mathbb{R}^{T \times m}
Z_{\text{soft}} = A^T H \in \mathbb{R}^{m \times d}
```

---

## 2. DSC-Attention 的数学推导

### 2.1 标准 Attention 的瓶颈
标准多头注意力:
```
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right) V
```
其中 $Q, K, V \in \mathbb{R}^{T \times d_h}$, 复杂度 $O(T^2 \cdot d_h)$。

当 $T=8192, d_h=64$ 时:
- $QK^T$ 矩阵大小: $8192 \times 8192 = 67$M 元素
- 每元素 4 bytes (fp32): $256$ MB 激活显存
- 每步计算量: $2 \times T^2 \times d_h = 8.6$ GFLOPs

### 2.2 核心思想: 可微分稀疏选择

**Step 1: 语义粗编码 (CoarseNet)**
```
Z = \text{CoarseNet}(H) \in \mathbb{R}^{B \times T \times d_c}
```
CoarseNet 是一个轻量 2 层 Transformer (SWA, $d_c=128$, heads=2), 将每个位置编码为低维语义向量。

**Step 2: 语义相似度矩阵**
```
S_{ij} = \frac{Z_i \cdot Z_j}{\tau} \in \mathbb{R}
```
$S \in \mathbb{R}^{B \times T \times T}$ 表示所有位置对的语义相关性。

**Step 3: $\alpha$-entmax 稀疏化**
```
P_{i,:} = \alpha\text{-entmax}(S_{i,:}) \in \mathbb{R}^T
```
$\alpha$-entmax 是 softmax 的稀疏推广:
- $\alpha=1$: 退化为 softmax (无稀疏)
- $\alpha=1.5$: 约 5-10% 非零 (推荐)
- $\alpha=2$: sparsemax (最稀疏)

性质:
- $P_{ij} \geq 0$, $\sum_j P_{ij} = 1$
- 大部分 $P_{ij} = 0$ (精确稀疏)
- 对 $S$ 完全可微 (Jacobian 已知)

**Step 4: Top-k 邻居选择**
对每个位置 $i$, 从 $P_{i,:}$ 中选择 top-$k$ 个非零邻居:
```
\mathcal{N}_i = \text{topk}_j(P_{ij}, k) \subseteq \{1, ..., T\}
```
$k$ 为超参 (推荐 512, 即 $T=8192$ 时 6.25% 稀疏度)。

**Step 5: 局部精确 Attention + 零和中心化**
对每个位置 $i$, 只在选中的 $k$ 个邻居上做精确 attention:
```
s_{ij} = \frac{q_i^T k_j}{\sqrt{d_h}} + \log P_{ij}, \quad j \in \mathcal{N}_i
```

**CPA 借鉴: 零和中心化 (Zero-Sum Centering)**
```
\mu_i = \frac{1}{|{\mathcal{N}_i}|} \sum_{j \in \mathcal{N}_i} s_{ij}
s'_{ij} = s_{ij} - \mu_i
```

最终输出:
```
o_i = \sum_{j \in \mathcal{N}_i} \text{softmax}(s'_{ij}) \cdot v_j
```

### 2.3 复杂度分析

| 操作 | 复杂度 | T=8192 数值 |
|---|---|---|
| CoarseNet | $O(T \cdot w \cdot d_c)$ | 0.5 GFLOPs |
| Similarity $S$ | $O(T^2 \cdot d_c)$ | 8.6 GFLOPs |
| $\alpha$-entmax | $O(T^2 \cdot \log T)$ | ~10 GFLOPs |
| Top-k 选择 | $O(T^2)$ | 可忽略 |
| Sparse Attention | $O(T \cdot k \cdot d)$ | 4.3 GFLOPs |
| **总计** | **$O(T^2 d_c + Tkd)$** | **~23 GFLOPs** |
| **标准 Attention** | **$O(T^2 d)$** | **68.7 GFLOPs** |

**注意**: $S$ 和 entmax 的 $O(T^2)$ 只涉及 $d_c=128$ (不是 $d=512$), 且可用 block-wise 优化。
实际工程中, $S$ 不需要全矩阵计算: 可用局部窗口 + 全局锚点的近似。

**纯稀疏版本 (推荐)**:
如果 $S$ 也只在局部窗口 $w$ 内计算:
```
S_{ij} = \begin{cases} Z_i \cdot Z_j / \tau & |i-j| \leq w \text{ or } j \in \text{anchors} \\ -\infty & \text{otherwise} \end{cases}
```
此时总复杂度: $O(T(w + k_{\text{global}})d_c + Tkd) \approx O(T \cdot 1024 \cdot d_c)$, 真线性。

---

## 3. 与 FLUED 的兼容性证明

### 3.1 软边界逻辑不受影响
FLUED 的边界检测:
```
\Delta H_t = H_{t+1} - H_t
p_t = \sigma(W_b \cdot \Delta H_t)
```
输入是 $H$ (hidden states), 不是 Attention 矩阵 $A$。
DSC-Attention 只改变 $H^{(l)} \to H^{(l+1)}$ 的映射方式, 不改变 $H$ 的语义空间。
因此 $\Delta H$ 的计算和物理意义完全保留。

### 3.2 Soft Path 不受影响
Soft Path 的 $A$ 矩阵:
```
A = \text{softmax}(p / \tau) \in \mathbb{R}^{T \times m}
```
与 DSC-Attention 的 $P$ 矩阵独立:
- $A$ 是边界分配矩阵 (压缩维度 $m$)
- $P$ 是注意力选择矩阵 (稀疏维度 $k$)
两者不冲突, 可共存。

### 3.3 梯度流完整性
```
\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial P} \cdot \frac{\partial P}{\partial S} \cdot \frac{\partial S}{\partial Z}
```
- $\frac{\partial P}{\partial S}$: $\alpha$-entmax 的 Jacobian (已知闭式解)
- $\frac{\partial S}{\partial Z}$: $Z Z^T$ 的梯度 (dense, 但 $d_c=128$ 很小)
- 梯度完整流回 CoarseNet, CoarseNet 可学习。

---

## 4. 关键超参与配置

```python
DSC_CONFIG = {
    # CoarseNet
    "d_c": 128,           # 语义维度 (越小越快, 信息损失越大)
    "coarse_layers": 2,   # CoarseNet 层数
    "coarse_heads": 2,    # CoarseNet 头数
    "coarse_window": 512, # CoarseNet SWA 窗口

    # 稀疏选择
    "alpha": 1.5,         # entmax 稀疏度 (1.2=训练初期, 1.5=推荐, 2.0=最稀疏)
    "k_neighbors": 512,   # 每个 query 的邻居数 (T=8192 时 6.25%)
    "tau": 1.0,           # 相似度温度

    # 零和中心化
    "use_zero_sum": True, # 是否启用 CPA 借鉴的零和中心化
    "zs_temperature": 1.0, # 零和后的温度缩放

    # 训练策略
    "coarse_lr_mult": 2.0,  # CoarseNet 学习率倍数
    "warmup_steps": 1000,   # CoarseNet 预热步数
    "alpha_anneal": False,  # 是否退火 alpha
}
```

---

## 5. 对照实验设计

| 组别 | Attention | 复杂度 | 预期时间 | 决策点 |
|------|-----------|--------|----------|--------|
| A1 | Full FlashAttn | $O(T^2 d)$ | ~30天 | 基线 (可能跑不完) |
| A2 | SWA (w=2048) | $O(T w d)$ | ~7天 | 工业界保底 |
| A3 | DSC (k=512, $\alpha$=1.5) | $O(T k d)$ | ~2天 | 主推方案 |
| A4 | DSC (k=256, $\alpha$=1.5) | $O(T k d)$ | ~1.5天 | 更激进稀疏 |
| A5 | DSC (k=512, $\alpha$=1.2) | $O(T k d)$ | ~2天 | 验证稀疏度影响 |
| A6 | Linear Attention | $O(T d^2)$ | ~2天 | 纯 $O(T)$ 对比 |

必测指标:
1. BPB (FineWeb-Edu 子集)
2. 训练 WPS (bytes/sec)
3. 显存峰值 (nvidia-smi)
4. 长程精确召回: 8K bytes 代码/XML 括号匹配
5. 下游任务: 代码补全 + OCR (CTM 数据)
6. P 矩阵分析: 邻居分布热力图、平均距离、局部 vs 全局比例
