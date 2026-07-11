from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .chunk_builder import ChunkBatch


@dataclass
class InterpreterOutput:
    z_content: torch.Tensor
    readout_z: torch.Tensor
    readout_gate: torch.Tensor
    readout_emit: torch.Tensor
    m_write: torch.Tensor | None
    aux: dict


class LatentMemoryInterpreter(nn.Module):
    """One-shot chunk-to-latent interpreter."""

    def __init__(
        self,
        d_model: int,
        d_z: int,
        d_mem: int | None = None,
        memory_rank: int = 0,
        max_readout_vectors: int = 1,
        chunk_mixer: str = "mean",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.d_mem = int(d_mem or d_z)
        self.memory_rank = int(memory_rank)
        self.max_readout_vectors = max(1, int(max_readout_vectors))
        self.chunk_mixer = str(chunk_mixer)
        self.memory_proj = nn.Linear(self.d_mem, self.d_model)
        self.marker_proj = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        if self.chunk_mixer == "delta_lite":
            self.delta_norm = nn.LayerNorm(self.d_model)
            self.delta_gate_scale = nn.Parameter(torch.full((self.d_model,), 0.50))
            self.delta_gate_bias = nn.Parameter(torch.full((self.d_model,), -1.00))
            self.delta_candidate_scale = nn.Parameter(torch.ones(self.d_model))
            self.delta_candidate_bias = nn.Parameter(torch.zeros(self.d_model))
            self.delta_output_scale = nn.Parameter(torch.ones(self.d_model))
            self.delta_scale = nn.Parameter(torch.tensor(0.10))
        elif self.chunk_mixer != "mean":
            raise ValueError(f"unknown chunk_mixer: {self.chunk_mixer}")
        self.content = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.SiLU(),
            nn.Linear(self.d_model * 2, self.d_z),
        )
        self.readout_slot_embed = nn.Parameter(torch.empty(self.max_readout_vectors, self.d_model * 2))
        nn.init.trunc_normal_(self.readout_slot_embed, std=0.02)
        if self.max_readout_vectors > 1:
            self.readout_gate = nn.Sequential(
                nn.LayerNorm(self.d_model * 2),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.max_readout_vectors - 1),
            )
            self.readout_emit = nn.Sequential(
                nn.LayerNorm(self.d_model * 2),
                nn.Linear(self.d_model * 2, self.max_readout_vectors - 1),
            )
            nn.init.zeros_(self.readout_emit[-1].weight)
            nn.init.constant_(self.readout_emit[-1].bias, -2.0)
        else:
            self.readout_gate = None
            self.readout_emit = None
        self.memory_gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
        )
        if self.memory_rank > 0:
            self.write = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.memory_rank * self.d_mem),
            )
        else:
            self.write = None

    def _span_features(self, chunks: ChunkBatch) -> torch.Tensor:
        marker = torch.stack(
            [
                chunks.transition_markers.to(chunks.span_embeddings.dtype),
                chunks.force_continue_markers.to(chunks.span_embeddings.dtype),
            ],
            dim=-1,
        )
        enriched = chunks.span_embeddings + self.marker_proj(marker)
        return enriched * chunks.token_mask.unsqueeze(-1).to(enriched.dtype)

    def _mean_pool(self, chunks: ChunkBatch) -> torch.Tensor:
        span_features = self._span_features(chunks)
        mask = chunks.token_mask.unsqueeze(-1).to(chunks.span_embeddings.dtype)
        denom = mask.sum(dim=2).clamp(min=1.0)
        mean = span_features.sum(dim=2) / denom
        if self.chunk_mixer == "mean":
            return mean

        delta_in = self.delta_norm(span_features)
        gate = torch.sigmoid(delta_in * self.delta_gate_scale + self.delta_gate_bias) * mask
        candidate = torch.tanh(delta_in * self.delta_candidate_scale + self.delta_candidate_bias)
        retention = (1.0 - gate.float()).clamp(min=1.0e-4)
        log_prefix = torch.cumsum(torch.log(retention), dim=2)
        total_log = log_prefix[:, :, -1:, :]
        suffix_after = torch.exp((total_log - log_prefix).clamp(min=-30.0, max=30.0))
        state = (gate.float() * candidate.float() * suffix_after).sum(dim=2).to(mean.dtype)
        mixed = mean + self.delta_scale.to(mean.dtype) * (state * self.delta_output_scale.to(mean.dtype))
        return mixed * chunks.chunk_mask.unsqueeze(-1).to(mixed.dtype)

    def mean_pool(self, chunks: ChunkBatch) -> torch.Tensor:
        return self._mean_pool(chunks)

    def write_memory(self, chunks: ChunkBatch, pooled: torch.Tensor | None = None) -> torch.Tensor | None:
        if self.write is None:
            return None
        pooled = self._mean_pool(chunks) if pooled is None else pooled
        m_write = self.write(pooled).view(pooled.size(0), pooled.size(1), self.memory_rank, self.d_mem)
        return m_write * chunks.chunk_mask.unsqueeze(-1).unsqueeze(-1).to(m_write.dtype)

    def forward(
        self,
        chunks: ChunkBatch,
        memory_read: torch.Tensor | None = None,
        pooled: torch.Tensor | None = None,
        m_write: torch.Tensor | None = None,
    ) -> InterpreterOutput:
        pooled = self._mean_pool(chunks) if pooled is None else pooled
        if memory_read is None:
            memory_h = torch.zeros_like(pooled)
        else:
            memory_h = self.memory_proj(memory_read)
        gate_in = torch.cat([pooled, memory_h], dim=-1)
        memory_gate = torch.sigmoid(self.memory_gate(gate_in))
        gated_memory = memory_gate * memory_h
        readout_base = torch.cat([pooled, gated_memory], dim=-1)
        readout_in = readout_base.unsqueeze(2) + self.readout_slot_embed.view(1, 1, self.max_readout_vectors, -1)
        readout_raw = self.content(readout_in)
        if self.readout_gate is None:
            readout_gate = torch.ones(
                (*pooled.shape[:2], 1),
                dtype=pooled.dtype,
                device=pooled.device,
            )
            readout_emit = readout_gate
        else:
            extra_gate = torch.sigmoid(self.readout_gate(readout_base))
            extra_emit = torch.sigmoid(self.readout_emit(readout_base))
            fallback = torch.ones_like(extra_gate[..., :1])
            readout_gate = torch.cat([fallback, extra_gate], dim=-1)
            readout_emit = torch.cat([fallback, extra_emit], dim=-1)
        readout_gate = readout_gate * chunks.chunk_mask.unsqueeze(-1).to(readout_gate.dtype)
        readout_emit = readout_emit * chunks.chunk_mask.unsqueeze(-1).to(readout_emit.dtype)
        readout_z = readout_raw * readout_gate.unsqueeze(-1).to(readout_raw.dtype)
        gate_sum = readout_gate.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        z_content = readout_z.sum(dim=2) / gate_sum
        z_content = z_content * chunks.chunk_mask.unsqueeze(-1).to(z_content.dtype)
        if m_write is None:
            m_write = self.write_memory(chunks, pooled=pooled)
        return InterpreterOutput(
            z_content=z_content,
            readout_z=readout_z,
            readout_gate=readout_gate,
            readout_emit=readout_emit,
            m_write=m_write,
            aux={
                "readout_units": readout_gate.float().sum(),
                "emit_units": readout_emit.float().sum(),
                "readout_gate_mean": (
                    readout_gate[..., 1:].float().sum() / (chunks.chunk_mask.float().sum() * max(self.max_readout_vectors - 1, 1)).clamp(min=1.0)
                    if self.max_readout_vectors > 1
                    else readout_gate.new_zeros(())
                ),
                "readout_emit_mean": (
                    readout_emit[..., 1:].float().sum() / (chunks.chunk_mask.float().sum() * max(self.max_readout_vectors - 1, 1)).clamp(min=1.0)
                    if self.max_readout_vectors > 1
                    else readout_emit.new_zeros(())
                ),
                "memory_gate": memory_gate.squeeze(-1) * chunks.chunk_mask.to(memory_gate.dtype),
                "chunk_mixer": self.chunk_mixer,
            },
        )
