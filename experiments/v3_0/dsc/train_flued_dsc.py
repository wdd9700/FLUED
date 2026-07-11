"""
FLUED-DSC 训练脚本
支持两阶段训练: CoarseNet 预热 + 联合训练

作者: [Coding Agent]
日期: 2026-06-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import json
import time
from pathlib import Path
from typing import Dict, Optional

from flued_dsc_model import DSCAttention, TiedBlockDSC, DSCConfig


# ============================================================
# 1. 数据加载 (示例: 字节级文本)
# ============================================================

class ByteDataset(Dataset):
    """
    字节级数据集
    将文本文件转为字节序列，PAD=0, byte b -> b+1
    """
    def __init__(self, data_path: str, seq_len: int = 8192):
        self.seq_len = seq_len

        # 加载二进制数据
        with open(data_path, 'rb') as f:
            raw_bytes = f.read()

        # 转为 tensor: byte b -> b+1 (保留 0 给 PAD)
        self.data = torch.tensor([b + 1 for b in raw_bytes], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# ============================================================
# 2. 完整 FLUED-DSC 模型 (含 Embedding + DSC Blocks + Head)
# ============================================================

class FLUED_DSC_Model(nn.Module):
    """
    完整 FLUED-DSC 模型
    保留 FLUED 的: Embedding, TiedBlockDSC 堆叠, boundary_head, Soft Path
    """
    def __init__(self, config: DSCConfig):
        super().__init__()
        self.config = config

        # 输入层
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = SinusoidalPE(config.d_model, config.seq_len)

        # DSC TiedBlock 堆叠
        self.blocks = nn.ModuleList([
            TiedBlockDSC(config.d_model, config.num_heads, config.d_c,
                        config.alpha, config.k_neighbors, config.dropout,
                        config.use_zero_sum)
            for _ in range(config.num_layers)
        ])

        # 输出层 (权重绑定)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight  # 权重绑定

        # 边界检测 (FLUED 原有)
        self.boundary_head = nn.Linear(config.d_model, 1)

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, T) 字节索引
        return: dict 包含 logits, boundaries 等
        """
        B, T = x.shape

        # Embedding + PE
        h = self.embedding(x) + self.pos_encoding(x)  # (B, T, d)

        # DSC Blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # 输出 logits
        logits = self.output(h)  # (B, T, vocab_size)

        # 边界检测 (FLUED 原有)
        if T > 1:
            delta_h = h[:, 1:] - h[:, :-1]  # (B, T-1, d)
            boundaries = torch.sigmoid(self.boundary_head(delta_h))  # (B, T-1, 1)
        else:
            boundaries = None

        return {
            'logits': logits,
            'boundaries': boundaries,
            'hidden': h
        }


class SinusoidalPE(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]


# ============================================================
# 3. 训练器: 支持两阶段训练
# ============================================================

class Trainer:
    """
    两阶段训练器:
    Stage 1: CoarseNet 预热 (冻结 Fine Attention QKV)
    Stage 2: 联合训练 (全部解冻)
    """
    def __init__(self, model: FLUED_DSC_Model, config: DSCConfig,
                 device: str = 'cuda', mixed_precision: bool = True):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None

        # 优化器: 主网络 + CoarseNet 高学习率
        self.optimizer = self._create_optimizer()

        self.step = 0
        self.stage = 1  # 当前阶段

    def _create_optimizer(self):
        """
        为不同参数组设置不同学习率
        """
        # 主网络参数
        main_params = []
        coarse_params = []

        for name, param in self.model.named_parameters():
            if 'coarse_net' in name or 'coarse' in name:
                coarse_params.append(param)
            else:
                main_params.append(param)

        return torch.optim.AdamW([
            {'params': main_params, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': coarse_params, 'lr': 1e-4 * self.config.coarse_lr_mult, 
             'weight_decay': 0.01}
        ], betas=(0.9, 0.95))

    def _stage1_warmup(self, batch):
        """
        Stage 1: 只训练 CoarseNet，冻结 Fine Attention
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # 冻结 Fine Attention 的 QKV
        for block in self.model.blocks:
            for param in block.attn.q_proj.parameters():
                param.requires_grad = False
            for param in block.attn.k_proj.parameters():
                param.requires_grad = False
            for param in block.attn.v_proj.parameters():
                param.requires_grad = False
            for param in block.attn.out_proj.parameters():
                param.requires_grad = False

        self.optimizer.zero_grad()

        with autocast(enabled=self.mixed_precision):
            outputs = self.model(x)
            logits = outputs['logits']

            # 标准语言建模损失
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                   y.reshape(-1))

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _stage2_joint(self, batch):
        """
        Stage 2: 联合训练，全部解冻
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # 解冻全部参数
        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer.zero_grad()

        with autocast(enabled=self.mixed_precision):
            outputs = self.model(x)
            logits = outputs['logits']

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1))

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def train_step(self, batch):
        """根据当前阶段选择训练策略"""
        if self.step < self.config.warmup_steps:
            loss = self._stage1_warmup(batch)
            self.stage = 1
        else:
            loss = self._stage2_joint(batch)
            self.stage = 2

        self.step += 1
        return loss

    def train(self, dataloader: DataLoader, total_steps: int = 20000,
              eval_every: int = 1000, save_every: int = 5000):
        """
        主训练循环
        """
        self.model.train()
        losses = []
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if self.step >= total_steps:
                break

            loss = self.train_step(batch)
            losses.append(loss)

            # 日志
            if self.step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                elapsed = time.time() - start_time
                print(f"Step {self.step}/{total_steps} | Stage {self.stage} | "
                      f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

            # 评估
            if self.step % eval_every == 0 and self.step > 0:
                self.evaluate(dataloader)

            # 保存
            if self.step % save_every == 0 and self.step > 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt")

        # 最终保存
        self.save_checkpoint("final_model.pt")
        print(f"训练完成! 总步数: {self.step}")

    def evaluate(self, dataloader: DataLoader):
        """快速评估 BPB"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= 100:  # 只评估 100 个 batch
                    break
                x, y = x.to(self.device), y.to(self.device)

                with autocast(enabled=self.mixed_precision):
                    outputs = self.model(x)
                    logits = outputs['logits']
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           y.reshape(-1), reduction='sum')

                total_loss += loss.item()
                total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        bpb = avg_loss / math.log(2)  # bits per byte
        perplexity = math.exp(avg_loss)

        print(f"[Eval] BPB: {bpb:.4f} | PPL: {perplexity:.2f}")
        self.model.train()

        return {'bpb': bpb, 'ppl': perplexity}

    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'stage': self.stage,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")


# ============================================================
# 4. 主函数
# ============================================================

def main():
    # 配置
    config = DSCConfig()
    config.seq_len = 8192  # 硬刚 ByteFlow
    config.k_neighbors = 512
    config.alpha = 1.5
    config.use_zero_sum = True

    # 数据
    dataset = ByteDataset("data/train.bin", config.seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                           num_workers=4, pin_memory=True)

    # 模型
    model = FLUED_DSC_Model(config)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    coarse_params = sum(p.numel() for p in model.blocks[0].attn.coarse_net.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"CoarseNet 参数量: {coarse_params / 1e6:.2f}M")

    # 训练器
    trainer = Trainer(model, config, device='cuda', mixed_precision=True)

    # 训练
    trainer.train(dataloader, total_steps=20000, eval_every=1000, save_every=5000)


if __name__ == "__main__":
    main()
