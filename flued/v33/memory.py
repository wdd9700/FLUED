from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class MemoryState:
    committed: torch.Tensor | None = None
    committed_mask: torch.Tensor | None = None
    committed_key: torch.Tensor | None = None
    committed_value: torch.Tensor | None = None
    provisional: torch.Tensor | None = None
    provisional_mask: torch.Tensor | None = None
    metadata: dict | None = None


class CausalLowRankSequenceMemory(nn.Module):
    """Append-only plus prompt-local memory read/write interface.

    Committed memory stores earlier dialogue / earlier calls.  A forward pass
    may also provide prompt-local writes for all current chunks.  Visibility is
    controlled at read time, so encoder-style prompt processing can read other
    chunks while strict streaming paths can keep causal access.
    """

    def __init__(self, query_dim: int, d_mem: int, top_k: int = 4) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.d_mem = int(d_mem)
        self.top_k = int(top_k)
        self.query = nn.Linear(self.query_dim, self.d_mem, bias=False)
        self.key = nn.Linear(self.d_mem, self.d_mem, bias=False)
        self.value = nn.Linear(self.d_mem, self.d_mem, bias=False)

    def empty_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        committed = torch.zeros((batch_size, 0, self.d_mem), device=device, dtype=dtype)
        committed_mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
        committed_key = torch.zeros((batch_size, 0, self.d_mem), device=device, dtype=dtype)
        committed_value = torch.zeros((batch_size, 0, self.d_mem), device=device, dtype=dtype)
        return MemoryState(
            committed=committed,
            committed_mask=committed_mask,
            committed_key=committed_key,
            committed_value=committed_value,
            metadata={},
        )

    def _committed_kv(self, state: MemoryState, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.committed is None:
            empty = torch.zeros((0,), device=device, dtype=dtype)
            return empty, empty, torch.zeros((0,), device=device, dtype=torch.bool)
        mask = (
            torch.ones((state.committed.size(0), state.committed.size(1)), dtype=torch.bool, device=device)
            if state.committed_mask is None
            else state.committed_mask.to(device=device)
        )
        if (not torch.is_grad_enabled()) and state.committed_key is not None and state.committed_value is not None:
            return (
                state.committed_key.to(device=device, dtype=dtype),
                state.committed_value.to(device=device, dtype=dtype),
                mask,
            )
        committed = state.committed.to(device=device, dtype=dtype)
        return self.key(committed), self.value(committed), mask

    def read(self, z_query: torch.Tensor, state: MemoryState | None) -> torch.Tensor:
        if state is None or state.committed is None or state.committed.size(1) == 0:
            return z_query.new_zeros((*z_query.shape[:-1], self.d_mem))
        q = self.query(z_query)
        k, v, mask = self._committed_kv(state, z_query.device, z_query.dtype)
        scores = torch.matmul(q, k.transpose(1, 2)) / max(self.d_mem, 1) ** 0.5
        scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
        top_k = min(self.top_k, scores.size(-1))
        top_scores, top_idx = torch.topk(scores, k=top_k, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        gathered = torch.gather(v.unsqueeze(1).expand(-1, z_query.size(1), -1, -1), 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_mem))
        return (weights.unsqueeze(-1) * gathered).sum(dim=2)

    def read_with_visibility(
        self,
        z_query: torch.Tensor,
        current_write: torch.Tensor | None,
        chunk_mask: torch.Tensor,
        state: MemoryState | None = None,
        visibility: str = "past_only",
    ) -> Tuple[torch.Tensor, dict]:
        """Read committed memory plus visible prompt-local memory.

        ``past_only`` is the old causal behavior.  ``bidirectional_no_self`` is
        the v3.3 prompt-encoding behavior: every chunk may read the other chunks'
        local memory, including later chunks, but not its own local memory.
        ``all_visible`` is a diagnostic mode and should not be used for codec
        claims because it allows the current chunk to read its own memory.
        """
        if visibility not in {"past_only", "bidirectional_no_self", "all_visible"}:
            raise ValueError(f"unknown memory visibility: {visibility}")
        bsz, chunks, _ = z_query.shape
        device = z_query.device
        dtype = z_query.dtype
        key_parts = []
        value_parts = []
        mask_parts = []
        order_parts = []

        if state is not None and state.committed is not None and state.committed.size(1) > 0:
            committed_k, committed_v, committed_mask = self._committed_kv(state, device, dtype)
            key_parts.append(committed_k)
            value_parts.append(committed_v)
            mask_parts.append(committed_mask)
            order_parts.append(torch.full((state.committed.size(1),), -1, dtype=torch.long, device=device))

        if current_write is not None:
            _, cur_chunks, rank, dim = current_write.shape
            current_slots = current_write.reshape(bsz, cur_chunks * rank, dim)
            current_mask = chunk_mask.repeat_interleave(rank, dim=1)
            current_order = torch.arange(cur_chunks, device=device).repeat_interleave(rank)
            key_parts.append(self.key(current_slots))
            value_parts.append(self.value(current_slots))
            mask_parts.append(current_mask)
            order_parts.append(current_order)

        if not key_parts:
            empty = z_query.new_zeros((bsz, chunks, self.d_mem))
            return empty, {"has_memory": torch.zeros((bsz, chunks), dtype=torch.bool, device=device)}

        k = torch.cat(key_parts, dim=1)
        v = torch.cat(value_parts, dim=1)
        memory_mask = torch.cat(mask_parts, dim=1)
        slot_order = torch.cat(order_parts, dim=0)

        q = self.query(z_query)
        scores = torch.matmul(q, k.transpose(1, 2)) / max(self.d_mem, 1) ** 0.5

        query_order = torch.arange(chunks, device=device).view(1, chunks, 1)
        committed_slot = slot_order.view(1, 1, -1) < 0
        local_order = slot_order.view(1, 1, -1)
        if visibility == "past_only":
            visible = committed_slot | (local_order < query_order)
        elif visibility == "bidirectional_no_self":
            visible = committed_slot | (local_order != query_order)
        else:
            visible = torch.ones((1, chunks, slot_order.numel()), dtype=torch.bool, device=device)

        allowed = memory_mask.unsqueeze(1) & visible & chunk_mask.unsqueeze(-1)
        has_memory = allowed.any(dim=-1)
        scores = scores.masked_fill(~allowed, -1.0e9)

        top_k = min(self.top_k, scores.size(-1))
        top_scores, top_idx = torch.topk(scores, k=top_k, dim=-1)
        top_allowed = torch.gather(allowed, 2, top_idx)
        top_scores = top_scores.masked_fill(~top_allowed, -1.0e9)
        weights = torch.softmax(top_scores, dim=-1)
        weights = torch.where(top_allowed, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)

        gathered = torch.gather(
            v.unsqueeze(1).expand(-1, chunks, -1, -1),
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_mem),
        )
        read = (weights.unsqueeze(-1) * gathered).sum(dim=2)
        read = read * has_memory.unsqueeze(-1).to(read.dtype)
        entropy = -(weights.clamp(min=1.0e-8).log() * weights).sum(dim=-1)
        self_slots = (local_order == query_order) & (local_order >= 0)
        self_allowed = allowed & self_slots
        return read, {
            "has_memory": has_memory,
            "read_entropy": entropy,
            "read_norm": read.norm(dim=-1),
            "self_allowed_count": self_allowed.sum(dim=-1),
            "visible_slots": allowed.sum(dim=-1),
        }

    def read_causal(
        self,
        z_query: torch.Tensor,
        current_write: torch.Tensor | None,
        chunk_mask: torch.Tensor,
        state: MemoryState | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Backward-compatible causal prompt-local memory read."""
        return self.read_with_visibility(
            z_query,
            current_write,
            chunk_mask,
            state=state,
            visibility="past_only",
        )

    def write(self, m_write: torch.Tensor | None, chunk_mask: torch.Tensor, state: MemoryState | None) -> MemoryState:
        if state is None:
            state = self.empty_state(chunk_mask.size(0), chunk_mask.device, m_write.dtype if m_write is not None else torch.float32)
        if m_write is None:
            return state
        bsz, chunks, rank, dim = m_write.shape
        provisional = m_write.reshape(bsz, chunks * rank, dim)
        provisional_mask = chunk_mask.repeat_interleave(rank, dim=1)
        return MemoryState(
            committed=state.committed,
            committed_mask=state.committed_mask,
            committed_key=state.committed_key,
            committed_value=state.committed_value,
            provisional=provisional,
            provisional_mask=provisional_mask,
            metadata=state.metadata or {},
        )

    def commit(self, state: MemoryState, cache_kv: bool | None = None) -> MemoryState:
        if state.provisional is None:
            return state
        cache_kv = (not torch.is_grad_enabled()) if cache_kv is None else bool(cache_kv)
        committed = state.provisional if state.committed is None else torch.cat([state.committed, state.provisional], dim=1)
        committed_mask = state.provisional_mask if state.committed_mask is None else torch.cat([state.committed_mask, state.provisional_mask], dim=1)
        committed_key = None
        committed_value = None
        if cache_kv:
            provisional_key = self.key(state.provisional)
            provisional_value = self.value(state.provisional)
            committed_key = (
                provisional_key
                if state.committed_key is None
                else torch.cat([state.committed_key.to(device=provisional_key.device, dtype=provisional_key.dtype), provisional_key], dim=1)
            )
            committed_value = (
                provisional_value
                if state.committed_value is None
                else torch.cat([state.committed_value.to(device=provisional_value.device, dtype=provisional_value.dtype), provisional_value], dim=1)
            )
        return MemoryState(
            committed=committed,
            committed_mask=committed_mask,
            committed_key=committed_key,
            committed_value=committed_value,
            provisional=None,
            provisional_mask=None,
            metadata=state.metadata or {},
        )
