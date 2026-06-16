"""
FLUED-DSC 对照实验脚本
对比: Full Attention / SWA / DSC / Linear Attention

作者: [Coding Agent]
日期: 2026-06-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from typing import Dict, List
import math

from FLUED-DSC_model import DSCAttention, TiedBlockDSC, DSCConfig


# ============================================================
# 1. 基线模型定义
# ============================================================

class FullAttention(nn.Module):
    """标准 Full Attention (FlashAttention 风格)"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class SWAAttention(nn.Module):
    """Sliding Window Attention"""
    def __init__(self, d_model: int, num_heads: int, window_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal + SWA
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        positions = torch.arange(T, device=x.device)
        window_mask = (positions.unsqueeze(0) - positions.unsqueeze(1)) > self.window_size
        scores.masked_fill_(window_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """Linear Attention (核技巧)"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 核特征映射: elu + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 改变乘法顺序: Q (K^T V) 而不是 (Q K^T) V
        kv = torch.matmul(k.transpose(-2, -1), v)  # (B, H, D, D)
        z = k.sum(dim=-2, keepdim=True).transpose(-2, -1)  # (B, H, D, 1)

        out = torch.matmul(q, kv)  # (B, H, T, D)
        out = out / (z.transpose(-2, -1) + 1e-6)  # 归一化

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


# ============================================================
# 2. 测试框架
# ============================================================

class Benchmark:
    """
    对照实验框架
    测试: 速度、显存、BPB (模拟)
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 seq_len: int = 8192, device: str = 'cuda'):
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device

    def test_model(self, model: nn.Module, name: str, 
                   num_iters: int = 10) -> Dict:
        """
        测试单个模型的速度、显存
        """
        model = model.to(self.device).eval()
        x = torch.randn(1, self.seq_len, self.d_model).to(self.device)

        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # 正式测试
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        avg_time = elapsed / num_iters * 1000  # ms

        # 估算 FLOPs
        if isinstance(model, FullAttention):
            flops = 2 * self.seq_len * self.seq_len * self.d_model / 1e9
        elif isinstance(model, SWAAttention):
            w = model.window_size
            flops = 2 * self.seq_len * w * self.d_model / 1e9
        elif isinstance(model, DSCAttention):
            k = model.k_neighbors
            flops = 2 * self.seq_len * k * self.d_model / 1e9
        elif isinstance(model, LinearAttention):
            flops = 2 * self.seq_len * self.d_model * self.d_model / 1e9
        else:
            flops = 0

        return {
            'name': name,
            'time_ms': avg_time,
            'peak_mem_gb': peak_mem,
            'flops_g': flops,
            'seq_len': self.seq_len
        }

    def run_all(self) -> List[Dict]:
        """
        运行所有对照实验
        """
        results = []

        # A1: Full Attention
        print("测试 Full Attention...")
        full = FullAttention(self.d_model, self.num_heads)
        results.append(self.test_model(full, "Full Attention"))

        # A2: SWA
        print("测试 SWA (w=2048)...")
        swa = SWAAttention(self.d_model, self.num_heads, window_size=2048)
        results.append(self.test_model(swa, "SWA (w=2048)"))

        # A3: DSC (k=512)
        print("测试 DSC (k=512, alpha=1.5)...")
        dsc = DSCAttention(self.d_model, d_c=128, num_heads=self.num_heads,
                          alpha=1.5, k_neighbors=512)
        results.append(self.test_model(dsc, "DSC (k=512)"))

        # A4: DSC (k=256)
        print("测试 DSC (k=256, alpha=1.5)...")
        dsc_small = DSCAttention(self.d_model, d_c=128, num_heads=self.num_heads,
                                alpha=1.5, k_neighbors=256)
        results.append(self.test_model(dsc_small, "DSC (k=256)"))

        # A5: Linear Attention
        print("测试 Linear Attention...")
        linear = LinearAttention(self.d_model, self.num_heads)
        results.append(self.test_model(linear, "Linear Attention"))

        return results

    def print_results(self, results: List[Dict]):
        """打印结果表格"""
        print("
" + "="*80)
        print(f"{'模型':<25} {'时间(ms)':<12} {'显存(GB)':<12} {'FLOPs(G)':<12}")
        print("="*80)

        for r in results:
            print(f"{r['name']:<25} {r['time_ms']:<12.2f} {r['peak_mem_gb']:<12.2f} {r['flops_g']:<12.1f}")

        print("="*80)

        # 计算加速比
        full_time = results[0]['time_ms']
        print(f"
相对 Full Attention 加速比:")
        for r in results[1:]:
            speedup = full_time / r['time_ms']
            print(f"  {r['name']}: {speedup:.2f}x")

        # 保存 JSON
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("
结果已保存到 benchmark_results.json")


# ============================================================
# 3. 长程精确召回测试
# ============================================================

def test_long_range_recall(model: nn.Module, seq_len: int = 8192, 
                           device: str = 'cuda') -> float:
    """
    测试长程精确召回能力
    构造: 8K bytes 的代码，包含成对括号，测试模型能否匹配远距离括号

    简化版: 用 synthetic pattern 测试
    """
    model = model.to(device).eval()

    # 构造测试数据: 在位置 0 放标记 A，在位置 4096 放标记 B
    # 模型需要知道位置 4096 的内容与位置 0 相关
    x = torch.zeros(1, seq_len, 512).to(device)
    x[0, 0, :] = 1.0  # 标记 A
    x[0, 4096, :] = 1.0  # 标记 B

    with torch.no_grad():
        out = model(x)

    # 检查输出中位置 4096 是否保留了位置 0 的信息
    # 简化: 计算位置 4096 的输出与位置 0 的输入的 cosine similarity
    sim = F.cosine_similarity(out[0, 4096], x[0, 0], dim=0)

    return sim.item()


# ============================================================
# 4. 主函数
# ============================================================

def main():
    print("FLUED-DSC 对照实验")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"序列长度: 8192")
    print()

    benchmark = Benchmark(d_model=512, num_heads=8, seq_len=8192)
    results = benchmark.run_all()
    benchmark.print_results(results)

    print("
对照实验完成!")


if __name__ == "__main__":
    main()
