"""
FLUED-DSC: Differentiable Selective Coarse Attention
完整模型实现，可直接替换原有 FLUED 的 Attention 层

作者: [Coding Agent]
日期: 2026-06-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect
from typing import Optional, Tuple, Dict
import math


# ============================================================
# 1. CoarseNet: 超小语义编码网络
# ============================================================

class CoarseBlock(nn.Module):
    """
    CoarseNet 的单层 block: SWA + FFN
    只处理局部窗口，参数量极小
    """
    def __init__(self, d_c: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_c = d_c
        self.num_heads = num_heads
        self.head_dim = d_c // num_heads
        self.window_size = window_size

        # QKV 投影
        self.qkv = nn.Linear(d_c, 3 * d_c)
        self.out_proj = nn.Linear(d_c, d_c)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_c, 4 * d_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_c, d_c),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_c)
        self.norm2 = nn.LayerNorm(d_c)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, d_c)
        return: (B, T, d_c)
        """
        B, T, _ = x.shape

        # Pre-norm + SWA
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = sliding_window_attention(q, k, v, self.window_size)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_c)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CoarseNet(nn.Module):
    """
    2层轻量 Transformer，输出语义表示 Z
    输入: H ∈ R^(B,T,d)  (FLUED 的 hidden states)
    输出: Z ∈ R^(B,T,d_c) (低维语义表示)
    """
    def __init__(self, d_model: int = 512, d_c: int = 128,
                 num_layers: int = 2, num_heads: int = 2,
                 window_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_c = d_c
        self.window_size = window_size

        # 投影到粗维度
        self.input_proj = nn.Linear(d_model, d_c)

        # 2层轻量 block
        self.layers = nn.ModuleList([
            CoarseBlock(d_c, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        return: (B, T, d_c)
        """
        z = self.input_proj(x)  # (B, T, d_c)
        for layer in self.layers:
            z = layer(z)
        return self.norm(z)


# ============================================================
# 2. DSC-Attention: 可微分稀疏选择注意力
# ============================================================

class DSCAttention(nn.Module):
    """
    Differentiable Selective Coarse Attention

    核心流程:
        1. CoarseNet 编码语义 -> Z
        2. Z Z^T 计算相似度 -> S
        3. alpha-entmax 稀疏化 -> P (每行约k个非零)
        4. Top-k 选择邻居
        5. 局部精确 Attention + 零和中心化 -> 输出

    复杂度: O(T * k * d) 对比 Full: O(T^2 * d)
    """
    def __init__(self, d_model: int = 512, d_c: int = 128,
                 num_heads: int = 8, alpha: float = 1.5,
                 k_neighbors: int = 512, tau: float = 1.0,
                 use_zero_sum: bool = True,
                 zs_temperature: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.alpha = alpha
        self.k_neighbors = k_neighbors
        self.tau = tau
        self.use_zero_sum = use_zero_sum
        self.zs_temperature = zs_temperature

        # CoarseNet: 语义选择器
        self.coarse_net = CoarseNet(d_model, d_c, num_layers=2,
                                      num_heads=2, window_size=512)

        # Fine Attention 的 QKV 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        return: (B, T, d_model)
        """
        B, T, _ = x.shape

        # === Step 1: Coarse 语义编码 ===
        Z = self.coarse_net(x)  # (B, T, d_c)

        # === Step 2: 语义相似度 S = Z Z^T / tau ===
        # 注意: 这里可以用局部窗口优化，避免 O(T^2)
        # 当前先实现完整版，后续优化为局部窗口 + 全局锚点
        S = torch.matmul(Z, Z.transpose(-2, -1)) / self.tau  # (B, T, T)

        # === Step 3: alpha-entmax 稀疏化 ===
        # P_ij ∈ [0,1], Σ_j P_ij = 1, 大部分为 0
        P = entmax_bisect(S, alpha=self.alpha, dim=-1)  # (B, T, T)

        # === Step 4: Top-k 邻居选择 ===
        k = min(self.k_neighbors, T)
        topk_values, topk_indices = torch.topk(P, k, dim=-1)  # (B, T, k)

        # === Step 5: Fine Attention 投影 ===
        Q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Q,K,V: (B, H, T, D)

        # === Step 6: 稀疏局部 Attention ===
        attn_out = self.sparse_attention(Q, K, V, topk_indices, topk_values)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        attn_out = self.out_proj(attn_out)

        return self.dropout(attn_out)

    def sparse_attention(self, Q, K, V, indices, prior_values):
        """
        核心稀疏 attention 计算

        Q: (B, H, T, D)
        K,V: (B, H, T, D)
        indices: (B, T, k) — 每个 query 的 k 个邻居位置
        prior_values: (B, T, k) — 对应的 P 值

        return: (B, H, T, D)
        """
        B, H, T, D = Q.shape
        k = indices.shape[-1]

        # 扩展 indices 到所有 heads: (B, H, T, k)
        indices = indices.unsqueeze(1).expand(-1, H, -1, -1)
        prior_values = prior_values.unsqueeze(1).expand(-1, H, -1, -1)

        # Gather K 和 V: (B, H, T, k, D)
        # 使用 torch.gather 进行不规则索引
        K_selected = torch.gather(
            K.unsqueeze(3).expand(-1, -1, -1, k, -1),
            dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        )
        V_selected = torch.gather(
            V.unsqueeze(3).expand(-1, -1, -1, k, -1),
            dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        )

        # 计算 scores: Q · K^T
        # Q: (B, H, T, 1, D), K_selected: (B, H, T, k, D)
        scores = torch.matmul(Q.unsqueeze(3), K_selected.transpose(-2, -1)).squeeze(3)
        # scores: (B, H, T, k)
        scores = scores / (D ** 0.5)

        # 加入 log P 先验作为 bias
        log_prior = torch.log(prior_values + 1e-10)
        scores = scores + log_prior

        # === CPA 借鉴: 零和中心化 ===
        if self.use_zero_sum:
            # 在选中的 k 个邻居上减均值
            mean_scores = scores.mean(dim=-1, keepdim=True)  # (B, H, T, 1)
            scores = scores - mean_scores
            # 温度缩放，防止数值过小
            scores = scores / self.zs_temperature

        # Causal mask: 确保不关注未来位置
        pos = torch.arange(T, device=Q.device).view(1, 1, T, 1)
        neighbor_pos = indices  # (B, H, T, k)
        causal_mask = neighbor_pos > pos
        scores.masked_fill_(causal_mask, float('-inf'))

        # Softmax 归一化
        attn = F.softmax(scores, dim=-1)  # (B, H, T, k)
        attn = self.dropout(attn)

        # 加权求和 V
        out = torch.matmul(attn.unsqueeze(3), V_selected).squeeze(3)  # (B, H, T, D)
        return out


# ============================================================
# 3. 工具函数: SWA 实现
# ============================================================

def sliding_window_attention(q, k, v, window_size):
    """
    手动实现的 Sliding Window Attention
    q, k, v: (B, H, T, D)
    return: (B, H, T, D)
    """
    B, H, T, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

    # Causal mask
    causal_mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores.masked_fill_(causal_mask, float('-inf'))

    # Sliding window mask
    positions = torch.arange(T, device=q.device)
    window_mask = (positions.unsqueeze(0) - positions.unsqueeze(1)) > window_size
    scores.masked_fill_(window_mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


# ============================================================
# 4. Drop-in 替换: TiedBlockDSC
# ============================================================

class TiedBlockDSC(nn.Module):
    """
    FLUED TiedBlock 的 DSC 版本
    完全保持原有接口，只替换 Attn 内部
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 d_c: int = 128, alpha: float = 1.5,
                 k_neighbors: int = 512, dropout: float = 0.1,
                 use_zero_sum: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DSCAttention(d_model, d_c, num_heads, alpha,
                                   k_neighbors, use_zero_sum=use_zero_sum,
                                   dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 5. 配置类
# ============================================================

class DSCConfig:
    """FLUED-DSC 配置"""
    # 模型维度
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 24

    # CoarseNet
    d_c: int = 128
    coarse_layers: int = 2
    coarse_heads: int = 2
    coarse_window: int = 512

    # 稀疏选择
    alpha: float = 1.5
    k_neighbors: int = 512
    tau: float = 1.0

    # 零和中心化
    use_zero_sum: bool = True
    zs_temperature: float = 1.0

    # 训练
    dropout: float = 0.1
    coarse_lr_mult: float = 2.0
    warmup_steps: int = 1000

    # 序列
    seq_len: int = 8192
    vocab_size: int = 257  # 256 bytes + PAD


if __name__ == "__main__":
    # 快速测试
    config = DSCConfig()
    model = DSCAttention(config.d_model, config.d_c, config.num_heads,
                         config.alpha, config.k_neighbors).cuda()
    x = torch.randn(2, 4096, config.d_model).cuda()
    out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
    print(f"显存: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
