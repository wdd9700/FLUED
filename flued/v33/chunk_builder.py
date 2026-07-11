from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ChunkBatch:
    span_embeddings: torch.Tensor
    confidence_values: torch.Tensor
    chunk_ids: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor
    chunk_mask: torch.Tensor
    token_mask: torch.Tensor
    transition_markers: torch.Tensor
    force_continue_markers: torch.Tensor
    pack_info: dict


class ChunkBuilder(nn.Module):
    """Pack byte features into executable chunks."""

    def __init__(self, max_chunks: int = 128, max_span: int = 16) -> None:
        super().__init__()
        self.max_chunks = int(max_chunks)
        self.max_span = int(max_span)

    def forward(
        self,
        byte_features: torch.Tensor,
        valid: torch.Tensor,
        hard_cut: torch.Tensor,
        confidence: torch.Tensor | None = None,
        soft_transition: torch.Tensor | None = None,
        force_continue: torch.Tensor | None = None,
    ) -> ChunkBatch:
        bsz, seq_len, dim = byte_features.shape
        device = byte_features.device
        span_embeddings = byte_features.new_zeros((bsz, self.max_chunks, self.max_span, dim))
        confidence_values = byte_features.new_zeros((bsz, self.max_chunks, self.max_span))
        token_mask = torch.zeros((bsz, self.max_chunks, self.max_span), dtype=torch.bool, device=device)
        transition_markers = torch.zeros_like(token_mask)
        force_markers = torch.zeros_like(token_mask)
        offsets = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
        chunk_ids = torch.full((bsz, seq_len), -1, dtype=torch.long, device=device)
        lengths = torch.zeros((bsz, self.max_chunks), dtype=torch.long, device=device)
        chunk_mask = torch.zeros((bsz, self.max_chunks), dtype=torch.bool, device=device)
        truncated_tokens = torch.zeros((bsz,), dtype=torch.long, device=device)

        soft_transition = soft_transition if soft_transition is not None else torch.zeros_like(valid)
        force_continue = force_continue if force_continue is not None else torch.zeros_like(valid)
        confidence = confidence if confidence is not None else byte_features.new_zeros(valid.shape)

        rank = valid.to(torch.long).cumsum(dim=1) - 1
        cuts = hard_cut.bool() & valid
        first = valid.float().argmax(dim=1)
        cuts = cuts.clone()
        cuts[torch.arange(bsz, device=device), first] |= valid.any(dim=1)

        hard_start = torch.cummax(torch.where(cuts, rank, torch.zeros_like(rank)), dim=1).values
        slot_from_hard = rank - hard_start
        span_cut = valid & slot_from_hard.ge(self.max_span) & ((slot_from_hard % self.max_span) == 0)
        cuts = cuts | span_cut

        final_start = torch.cummax(torch.where(cuts, rank, torch.zeros_like(rank)), dim=1).values
        slots = rank - final_start
        chunks_for_tokens = torch.cumsum(cuts.to(torch.long), dim=1) - 1
        within = valid & chunks_for_tokens.ge(0) & chunks_for_tokens.lt(self.max_chunks) & slots.ge(0) & slots.lt(self.max_span)
        truncated_tokens = (valid & ~within).sum(dim=1)

        b_idx, src_idx = within.nonzero(as_tuple=True)
        chunk_idx = chunks_for_tokens[b_idx, src_idx]
        slot_idx = slots[b_idx, src_idx]
        span_embeddings[b_idx, chunk_idx, slot_idx] = byte_features[b_idx, src_idx]
        confidence_values[b_idx, chunk_idx, slot_idx] = confidence[b_idx, src_idx].to(confidence_values.dtype)
        token_mask[b_idx, chunk_idx, slot_idx] = True
        transition_markers[b_idx, chunk_idx, slot_idx] = soft_transition[b_idx, src_idx].bool()
        force_markers[b_idx, chunk_idx, slot_idx] = force_continue[b_idx, src_idx].bool()
        chunk_ids[b_idx, src_idx] = chunk_idx
        offsets[b_idx, src_idx] = slot_idx
        chunk_mask[b_idx, chunk_idx] = True

        flat_chunk = b_idx * self.max_chunks + chunk_idx
        flat_lengths = lengths.reshape(-1)
        flat_lengths.scatter_reduce_(0, flat_chunk, slot_idx + 1, reduce="amax", include_self=True)

        return ChunkBatch(
            span_embeddings=span_embeddings,
            confidence_values=confidence_values,
            chunk_ids=chunk_ids,
            offsets=offsets,
            lengths=lengths,
            chunk_mask=chunk_mask,
            token_mask=token_mask,
            transition_markers=transition_markers,
            force_continue_markers=force_markers,
            pack_info={"max_chunks": self.max_chunks, "max_span": self.max_span, "truncated_tokens": truncated_tokens},
        )
