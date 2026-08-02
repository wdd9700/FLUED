"""H-Net reproduction (same-scale baseline for FLUED v3.6).

Faithful in mechanism: byte encoder -> routing module (cosine-dissimilarity
boundary probs) -> hard downsample to chunk states -> hierarchical main net
-> dechunk/smoothing decoder -> next-byte logits. Everything causal.

Differences from the paper (disclosed): main net is causal Transformer
instead of Mamba-2 (mamba-ssm does not build on this Windows/torch combo);
dechunk = last COMPLETED chunk output + encoder state (strictly causal),
with routing-probability smoothing. Ratio loss steers compression rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from flued.data import PAD_ID
from flued.v34.model import PlainByteLookup, _rope


class CausalBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, ffn_dim: int, causal: bool = True) -> None:
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Linear(ffn_dim, dim))
        self.nhead = nhead

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(bsz, seq, self.nhead, dim // self.nhead).transpose(1, 2)
        k = k.view(bsz, seq, self.nhead, dim // self.nhead).transpose(1, 2)
        v = v.view(bsz, seq, self.nhead, dim // self.nhead).transpose(1, 2)
        pos = torch.arange(seq, device=x.device)
        q, k = _rope(q, pos), _rope(k, pos)
        pad = valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=pad, is_causal=self.causal)
        y = y.transpose(1, 2).reshape(bsz, seq, dim)
        x = x + self.out(y)
        x = x + self.ff(self.norm2(x))
        return x * valid.unsqueeze(-1).to(x.dtype)


@dataclass
class HNetReproConfig:
    d_model: int = 512
    nhead: int = 8
    ffn_dim: int = 2048
    encoder_layers: int = 2
    main_layers: int = 9
    decoder_layers: int = 3
    ratio_target: float = 0.2
    ratio_weight: float = 0.03
    max_chunks: int = 192
    causal: bool = True
    decoder_skip: bool = True


class HNetRepro(nn.Module):
    def __init__(self, config: HNetReproConfig | None = None) -> None:
        super().__init__()
        self.config = config or HNetReproConfig()
        c = self.config
        self.byte_lookup = PlainByteLookup(c.d_model)
        self.encoder = nn.ModuleList(
            [CausalBlock(c.d_model, c.nhead, c.ffn_dim, c.causal) for _ in range(c.encoder_layers)]
        )
        self.main = nn.ModuleList(
            [CausalBlock(c.d_model, c.nhead, c.ffn_dim, c.causal) for _ in range(c.main_layers)]
        )
        self.decoder = nn.ModuleList(
            [CausalBlock(c.d_model, c.nhead, c.ffn_dim, c.causal) for _ in range(c.decoder_layers)]
        )
        self.chunk_proj = nn.Linear(c.d_model, c.d_model)
        self.out_norm = nn.LayerNorm(c.d_model)

    def forward(self, token_ids: torch.Tensor) -> dict:
        c = self.config
        valid = token_ids.ne(PAD_ID)
        x = self.byte_lookup(token_ids)
        for block in self.encoder:
            x = block(x, valid)
        enc = x
        normed = F.normalize(enc.float(), dim=-1)
        prev = torch.cat([normed[:, :1], normed[:, :-1]], dim=1)
        cos = (normed * prev).sum(dim=-1)
        boundary_prob = (0.5 * (1.0 - cos)).to(x.dtype) * valid.to(x.dtype)
        boundary_hard = (boundary_prob > 0.5) & valid
        first = valid.float().argmax(dim=1)
        boundary_hard[torch.arange(token_ids.size(0), device=token_ids.device), first] = True
        boundary_st = boundary_hard.to(x.dtype) + (boundary_prob - boundary_prob.detach())
        chunk_ids = (torch.cumsum(boundary_hard.to(torch.long), dim=1) - 1).clamp(min=0, max=c.max_chunks - 1)
        n_chunks = chunk_ids.max(dim=1).values + 1
        bsz, seq, dim = enc.shape
        chunk_sum = enc.new_zeros(bsz, c.max_chunks, dim)
        chunk_cnt = enc.new_zeros(bsz, c.max_chunks, 1)
        flat = (torch.arange(bsz, device=x.device).unsqueeze(1) * c.max_chunks + chunk_ids.clamp(min=0))[valid]
        chunk_sum.reshape(-1, dim).index_add_(0, flat, enc[valid])
        chunk_cnt.reshape(-1, 1).index_add_(0, flat, enc.new_ones(flat.numel(), 1))
        chunk_mask = torch.zeros(bsz, c.max_chunks, dtype=torch.bool, device=x.device)
        chunk_mask[torch.arange(bsz, device=x.device).unsqueeze(1), torch.arange(c.max_chunks, device=x.device).unsqueeze(0)] = (
            torch.arange(c.max_chunks, device=x.device).unsqueeze(0) < n_chunks.unsqueeze(1)
        )
        chunk_states = chunk_sum / chunk_cnt.clamp(min=1.0)
        m = self.chunk_proj(chunk_states)
        chunk_valid = chunk_mask
        for block in self.main:
            m = block(m, chunk_valid)
        gather_idx = chunk_ids.clamp(min=0)
        completed = torch.where(
            boundary_hard, gather_idx, (gather_idx - 1).clamp(min=0)
        )
        if not self.config.causal:
            completed = gather_idx
        main_at_byte = torch.gather(
            m, 1, completed.clamp(min=0).unsqueeze(-1).expand(-1, -1, dim)
        )
        smooth = boundary_st.unsqueeze(-1)
        if self.config.decoder_skip:
            h = enc + smooth * main_at_byte
        else:
            h = smooth * main_at_byte
        for block in self.decoder:
            h = block(h, valid)
        h = self.out_norm(h)
        vocab = torch.arange(258, device=x.device)
        table = self.byte_lookup(vocab).to(h.dtype)
        logits = 10.0 * torch.matmul(F.normalize(h, dim=-1), F.normalize(table, dim=-1).transpose(0, 1))
        ratio = boundary_prob[valid].mean() if valid.any() else boundary_prob.mean()
        return {
            "logits": logits,
            "boundary_prob": boundary_prob,
            "ratio_loss": (ratio - c.ratio_target).square(),
            "chunks_per_sample": n_chunks.float().mean(),
            "boundary_rate": ratio,
        }
