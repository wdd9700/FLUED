from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from flued.data import PAD_ID
from flued.v33.byte_lookup import StructuredByteLookup
from flued.v33.chunk_builder import ChunkBatch, ChunkBuilder
from flued.v33.segmentor import SegmentorOutput
from flued.v33.threshold_policy import ThresholdPolicyOutput
from flued.v34.rate_emit import CodingRateSelection, MarginalCodingRateSelector, ReadoutEmitController


class PlainByteLookup(nn.Module):
    """Parameter-matched ablation alternative to the structured lookup."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(258, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(token_ids.clamp(min=0, max=257)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = x.size(-1)
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inv = torch.exp(
        torch.arange(0, dim, 2, device=x.device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    phase = positions.float().unsqueeze(-1) * inv
    cos = torch.repeat_interleave(phase.cos(), 2, dim=-1).to(x.dtype)
    sin = torch.repeat_interleave(phase.sin(), 2, dim=-1).to(x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def _sinusoidal_position(seq: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Absolute prompt-level position side channel carried by byte payloads."""
    even_dim = dim - (dim % 2)
    positions = torch.arange(seq, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, even_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(even_dim, 1))
    )
    phase = positions * frequencies.unsqueeze(0)
    encoded = torch.zeros(seq, dim, device=device, dtype=torch.float32)
    encoded[:, 0:even_dim:2] = phase.sin()
    encoded[:, 1:even_dim:2] = phase.cos()
    return encoded.to(dtype)


def _plastic_signed_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Keep [-1, 1] forward values without irreversible tanh saturation."""
    bounded = torch.tanh(logits)
    return logits + (bounded - logits).detach()


def _alibi_slopes(nhead: int, device: torch.device) -> torch.Tensor:
    """Fixed geometric head slopes for bidirectional byte-distance ALiBi."""
    head = torch.arange(1, nhead + 1, device=device, dtype=torch.float32)
    return torch.pow(2.0, -8.0 * head / float(nhead))


def _bidirectional_alibi_bias(
    seq: int,
    nhead: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(seq, device=device, dtype=torch.float32)
    distance = (positions[:, None] - positions[None, :]).abs()
    slopes = _alibi_slopes(nhead, device)
    return (-slopes[:, None, None] * distance[None]).to(dtype)


class ParallelAttention(nn.Module):
    def __init__(self, dim: int, nhead: int, use_rope: bool, use_alibi: bool = False) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dim must be divisible by nhead")
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.use_rope = bool(use_rope)
        self.use_alibi = bool(use_alibi)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.nhead, self.head_dim).transpose(1, 2)
        if self.use_rope:
            pos = torch.arange(seq, device=x.device)
            q = _rope(q, pos)
            k = _rope(k, pos)
        if self.use_alibi:
            mask = _bidirectional_alibi_bias(seq, self.nhead, x.device, q.dtype).unsqueeze(0)
            mask = mask.masked_fill(~valid[:, None, None, :], float("-inf"))
        else:
            mask = valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(bsz, seq, self.dim)
        return self.out(y) * valid.unsqueeze(-1).to(y.dtype)


class DiTStyleBlock(nn.Module):
    """Parallel one-step latent refinement block.

    The scalar noise condition keeps the probe compatible with a later
    multi-step diffusion curriculum; this matrix deliberately evaluates only
    the one-shot deployment form.
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        ffn_dim: int,
        use_rope: bool,
        use_alibi: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ParallelAttention(dim, nhead, use_rope, use_alibi)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ffn_dim * 2)
        self.ff_out = nn.Linear(ffn_dim, dim)
        self.noise_cond = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attn_scale = nn.Parameter(torch.tensor(0.1))
        self.ff_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, valid: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        cond = self.noise_cond(noise_level.view(-1, 1)).unsqueeze(1).to(x.dtype)
        x = x + self.attn_scale.to(x.dtype) * self.attn(self.norm1(x + cond), valid)
        ff_gate, ff_value = self.ff_in(self.norm2(x + cond)).chunk(2, dim=-1)
        ff = self.ff_out(F.silu(ff_gate) * ff_value)
        x = x + self.ff_scale.to(x.dtype) * ff
        return x * valid.unsqueeze(-1).to(x.dtype)

    def inverse_block(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """First-order tied-weight inverse approximation.

        This reverses residual order and subtracts the same FFN/attention
        updates. It deliberately reuses encoder/interpreter parameters rather
        than training a second decoder stack; it is not an exact inverse.
        """
        cond = self.noise_cond(noise_level.view(-1, 1)).unsqueeze(1).to(x.dtype)
        ff_gate, ff_value = self.ff_in(self.norm2(x + cond)).chunk(2, dim=-1)
        ff = self.ff_out(F.silu(ff_gate) * ff_value)
        x = x - self.ff_scale.to(x.dtype) * ff
        x = x - self.attn_scale.to(x.dtype) * self.attn(self.norm1(x + cond), valid)
        return x * valid.unsqueeze(-1).to(x.dtype)


class SoftBoundaryBridge(nn.Module):
    """Hard forward chunks with a differentiable soft-assignment backward path."""

    def __init__(
        self,
        max_chunks: int,
        tau_cut: float,
        temperature: float = 0.35,
        gradient_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_chunks = int(max_chunks)
        self.tau_cut = float(tau_cut)
        self.temperature = float(temperature)
        self.gradient_scale = float(gradient_scale)

    def forward(
        self,
        chunks: ChunkBatch,
        token_features: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        force_continue: torch.Tensor,
        cut_probability: torch.Tensor | None = None,
    ) -> ChunkBatch:
        if not self.training:
            return chunks
        cut_prob = cut_probability.float() if cut_probability is not None else torch.sigmoid(
            (confidence.float() - self.tau_cut) / self.temperature
        )
        cut_prob = cut_prob * valid.float() * (~force_continue).float()
        if cut_prob.size(1):
            cut_prob = cut_prob.clone()
            cut_prob[:, 0] = valid[:, 0].float()
        expected_chunk = torch.cumsum(cut_prob, dim=1) - cut_prob[:, :1]
        centers = torch.arange(self.max_chunks, device=token_features.device, dtype=torch.float32)
        distance = expected_chunk.unsqueeze(-1) - centers.view(1, 1, -1)
        weights = torch.softmax(-distance.square() / self.temperature, dim=-1) * valid.unsqueeze(-1).float()
        soft_sum = torch.einsum("btc,btd->bcd", weights, token_features.float())
        soft_denom = weights.sum(dim=1).unsqueeze(-1).clamp(min=1.0e-6)
        soft_mean = (soft_sum / soft_denom).to(token_features.dtype)

        hard_mask = chunks.token_mask.unsqueeze(-1).to(token_features.dtype)
        hard_denom = hard_mask.sum(dim=2).clamp(min=1.0)
        hard_mean = (chunks.span_embeddings * hard_mask).sum(dim=2) / hard_denom
        straight_through = soft_mean + (hard_mean - soft_mean).detach()
        bridge = self.gradient_scale * (straight_through - hard_mean).unsqueeze(2)
        span_embeddings = chunks.span_embeddings + bridge * chunks.token_mask.unsqueeze(-1).to(bridge.dtype)
        return ChunkBatch(
            span_embeddings=span_embeddings,
            confidence_values=chunks.confidence_values,
            chunk_ids=chunks.chunk_ids,
            offsets=chunks.offsets,
            lengths=chunks.lengths,
            chunk_mask=chunks.chunk_mask,
            token_mask=chunks.token_mask,
            transition_markers=chunks.transition_markers,
            force_continue_markers=chunks.force_continue_markers,
            pack_info={**chunks.pack_info, "soft_boundary_bridge": True},
        )


class FixedDualThresholdPolicy(nn.Module):
    """Executable v3.4 decisions from continuous signed confidence.

    Only the two settled positive thresholds affect segmentation. UTF-8
    continuation bytes are handled separately as a structural hard guard.
    """

    def __init__(self, tau_cut: float = 0.90, tau_trans: float = 0.75) -> None:
        super().__init__()
        if not 0.0 < tau_trans < tau_cut < 1.0:
            raise ValueError("expected 0 < tau_trans < tau_cut < 1")
        self.tau_cut = float(tau_cut)
        self.tau_trans = float(tau_trans)

    def forward(self, confidence: torch.Tensor, valid: torch.Tensor) -> ThresholdPolicyOutput:
        hard_cut = confidence.gt(self.tau_cut) & valid
        soft_transition = confidence.gt(self.tau_trans) & confidence.le(self.tau_cut) & valid
        if valid.ndim == 2:
            first = valid.float().argmax(dim=1)
            hard_cut = hard_cut.clone()
            hard_cut[torch.arange(valid.size(0), device=valid.device), first] = valid.any(dim=1)
        return ThresholdPolicyOutput(
            confidence=confidence,
            hard_cut=hard_cut,
            soft_transition=soft_transition,
            force_continue=torch.zeros_like(valid),
            aux={"tau_cut": self.tau_cut, "tau_trans": self.tau_trans},
        )


class QueryPool(nn.Module):
    """Parallel, order-aware learned-query pooling over each chunk."""

    def __init__(self, dim: int, slots: int, nhead: int, use_position: bool) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dim must be divisible by nhead")
        self.dim = dim
        self.slots = slots
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.use_position = bool(use_position)
        self.query = nn.Parameter(torch.empty(slots, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, spans: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, chunks, width, dim = spans.shape
        flat = bsz * chunks
        x = spans.reshape(flat, width, dim)
        valid = token_mask.reshape(flat, width)
        q = self.query.view(1, self.slots, dim).expand(flat, -1, -1)
        q = self.q_proj(q).view(flat, self.slots, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(flat, width, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(flat, width, self.nhead, self.head_dim).transpose(1, 2)
        if self.use_position:
            q = _rope(q, torch.arange(self.slots, device=x.device))
            k = _rope(k, torch.arange(width, device=x.device))
        mask = valid[:, None, None, :]
        pooled = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        pooled = pooled.transpose(1, 2).reshape(flat, self.slots, dim)
        pooled = self.out(pooled).reshape(bsz, chunks, self.slots, dim)
        active = token_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)
        return pooled * active.to(pooled.dtype)


class DenseNoSelfMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        use_position: bool = True,
        position_mode: str | None = None,
        access_mode: str = "other_only",
    ) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dim must be divisible by nhead")
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.position_mode = position_mode or ("chunk_rope" if use_position else "none")
        if self.position_mode not in {"none", "chunk_rope", "byte_alibi"}:
            raise ValueError("memory position_mode must be none, chunk_rope, or byte_alibi")
        self.use_position = self.position_mode != "none"
        if access_mode not in {"other_only", "all", "current_only", "none"}:
            raise ValueError("memory access_mode must be other_only, all, current_only, or none")
        self.access_mode = access_mode
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        readout: torch.Tensor,
        memory: torch.Tensor,
        chunk_mask: torch.Tensor,
        chunk_anchors: torch.Tensor | None = None,
        diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        bsz, chunks, readouts, dim = readout.shape
        ranks = memory.size(2)
        q = self.q(readout).view(bsz, chunks * readouts, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k(memory).view(bsz, chunks * ranks, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v(memory).view(bsz, chunks * ranks, self.nhead, self.head_dim).transpose(1, 2)
        q_owner = torch.arange(chunks, device=readout.device).repeat_interleave(readouts)
        k_owner = torch.arange(chunks, device=readout.device).repeat_interleave(ranks)
        if self.position_mode == "chunk_rope":
            q = _rope(q, q_owner)
            k = _rope(k, k_owner)
        same_owner = q_owner[:, None].eq(k_owner[None, :])
        if self.access_mode == "other_only":
            owner_allowed = ~same_owner
        elif self.access_mode == "current_only":
            owner_allowed = same_owner
        elif self.access_mode == "all":
            owner_allowed = torch.ones_like(same_owner)
        else:
            owner_allowed = torch.zeros_like(same_owner)
        key_valid = chunk_mask.repeat_interleave(ranks, dim=1)
        query_valid = chunk_mask.repeat_interleave(readouts, dim=1)
        allowed = owner_allowed[None, None] & key_valid[:, None, None, :] & query_valid[:, None, :, None]
        if self.position_mode == "byte_alibi":
            if chunk_anchors is None:
                raise ValueError("chunk_anchors are required for byte_alibi memory position")
            q_anchor = chunk_anchors.repeat_interleave(readouts, dim=1)
            k_anchor = chunk_anchors.repeat_interleave(ranks, dim=1)
            distance = (q_anchor[:, :, None] - k_anchor[:, None, :]).abs().float()
            slopes = _alibi_slopes(self.nhead, readout.device)
            mask = (-slopes[None, :, None, None] * distance[:, None]).to(q.dtype)
            mask = mask.masked_fill(~allowed, float("-inf"))
        else:
            mask = allowed
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = torch.nan_to_num(y)
        y = y.transpose(1, 2).reshape(bsz, chunks, readouts, dim)
        y = self.out(y) * chunk_mask.unsqueeze(-1).unsqueeze(-1).to(y.dtype)
        current_share = y.new_zeros(())
        other_share = y.new_zeros(())
        if diagnostics:
            score = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            if self.position_mode == "byte_alibi":
                # Match the actual attention logits above. Diagnostics must not
                # silently report content-only weights for a distance-biased path.
                score = score + (-slopes[None, :, None, None] * distance[:, None])
            score = score.masked_fill(~allowed, float("-inf"))
            weights = torch.nan_to_num(score.softmax(dim=-1))
            query_weight = query_valid[:, None, :, None].float()
            denom = query_weight.sum() * self.nhead
            current_share = (weights * same_owner[None, None].float() * query_weight).sum() / denom.clamp(min=1.0)
            other_share = (weights * (~same_owner)[None, None].float() * query_weight).sum() / denom.clamp(min=1.0)
        return y, {
            "self_allowed": current_share,
            "memory_attention_current_share": current_share,
            "memory_attention_other_share": other_share,
            "visible_memory_slots": allowed[:, 0].float().sum(dim=-1),
        }


class SmallChunkARCorrection(nn.Module):
    """Small causal correction that still sees the ordered byte slots.

    All chunks remain parallel.  Only the at-most ``max_span`` positions inside
    each chunk are recurrent, so this ablation fairly tests whether a narrow AR
    path can replace explicit positional encoding.
    """

    def __init__(self, dim: int, hidden: int, memory_rank: int, readouts: int, gate_scale: float = 0.1) -> None:
        super().__init__()
        self.gate_scale = float(gate_scale)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.memory_proj = nn.Linear(hidden, memory_rank * dim)
        self.readout_proj = nn.Linear(hidden, readouts * dim)
        self.memory_gate = nn.Linear(hidden, 1)
        self.readout_gate = nn.Linear(hidden, 1)
        self.memory_rank = int(memory_rank)
        self.readouts = int(readouts)
        for proj in (self.memory_proj, self.readout_proj):
            nn.init.normal_(proj.weight, std=0.01)
            nn.init.zeros_(proj.bias)
        nn.init.constant_(self.memory_gate.bias, -2.0)
        nn.init.constant_(self.readout_gate.bias, -2.0)

    def forward(
        self,
        spans: torch.Tensor,
        token_mask: torch.Tensor,
        memory: torch.Tensor,
        readout: torch.Tensor,
        apply_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, chunks, width, dim = spans.shape
        flat = spans.reshape(bsz * chunks, width, dim)
        sequence, _ = self.gru(flat)
        lengths = token_mask.sum(dim=-1).clamp(min=1).reshape(-1)
        index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
        summary = torch.gather(sequence, 1, index).squeeze(1)
        readout_delta = torch.tanh(self.readout_proj(summary)).view(bsz, chunks, self.readouts, dim)
        readout_gate = self.gate_scale * torch.sigmoid(self.readout_gate(summary)).view(bsz, chunks, 1, 1)
        readout_applied = readout_gate * readout_delta
        active = token_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1).to(memory.dtype)
        if apply_memory:
            memory_delta = torch.tanh(self.memory_proj(summary)).view(bsz, chunks, self.memory_rank, dim)
            memory_gate = self.gate_scale * torch.sigmoid(self.memory_gate(summary)).view(bsz, chunks, 1, 1)
            memory_applied = memory_gate * memory_delta
            memory_out = (memory + memory_applied) * active
            memory_ratio = memory_applied.norm(dim=-1).mean() / memory.detach().norm(dim=-1).mean().clamp(min=1.0e-6)
        else:
            memory_out = memory
            memory_ratio = readout_applied.new_zeros(())
        readout_out = (readout + readout_applied) * active
        readout_ratio = readout_applied.norm(dim=-1).mean() / readout.detach().norm(dim=-1).mean().clamp(min=1.0e-6)
        ratio = 0.5 * (memory_ratio + readout_ratio) if apply_memory else readout_ratio
        return memory_out, readout_out, ratio


class SpanDecoder(nn.Module):
    """Historical independently-parameterized v3.4 decoder."""

    def __init__(self, dim: int, hidden: int, max_span: int, byte_lookup: StructuredByteLookup) -> None:
        super().__init__()
        self.max_span = max_span
        self.byte_lookup = byte_lookup
        self.slot = nn.Parameter(torch.empty(max_span, dim))
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.scale = nn.Parameter(torch.tensor(10.0))
        nn.init.trunc_normal_(self.slot, std=0.02)

    def forward(self, readout: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        bsz, chunks, readouts, dim = readout.shape
        q = self.q(self.slot).view(1, 1, self.max_span, dim).expand(bsz, chunks, -1, -1)
        k = self.k(readout)
        v = self.v(readout)
        score = torch.einsum("bcsd,bcrd->bcsr", q, k) / math.sqrt(dim)
        ctx = torch.einsum("bcsr,bcrd->bcsd", score.softmax(dim=-1), v)
        h = self.ff(ctx + q)
        vocab = torch.arange(258, device=h.device)
        table = self.byte_lookup(vocab).to(h.dtype)
        logits = self.scale.clamp(1.0, 100.0) * torch.matmul(
            F.normalize(h, dim=-1), F.normalize(table, dim=-1).transpose(0, 1)
        )
        return logits.masked_fill(~chunk_mask.unsqueeze(-1).unsqueeze(-1), 0.0)


class SharedInverseSpanDecoder(nn.Module):
    """Parameter-free wrapper around the interpreter's tied inverse path.

    Span expansion uses the readout pool's transposed projections plus a fixed
    positional bias. Interpreter blocks then run in reverse order through
    their first-order inverse approximation. The only output basis is the
    encoder's active byte lookup table.
    """

    def __init__(self, dim: int, max_span: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_span = int(max_span)

    def forward(
        self,
        readout: torch.Tensor,
        chunk_mask: torch.Tensor,
        readout_active: torch.Tensor,
        readout_pool: QueryPool,
        interpreter_blocks: nn.ModuleList,
        byte_lookup: nn.Module,
    ) -> torch.Tensor:
        bsz, chunks, readouts, dim = readout.shape
        flat = bsz * chunks
        active = readout_active.reshape(flat, readouts) & chunk_mask.reshape(-1, 1)
        # The fallback slot is a structural invariant and prevents an empty
        # attention row even when all optional readouts remain silent.
        if readouts:
            active = active.clone()
            active[:, 0] = chunk_mask.reshape(-1)

        z = readout.reshape(flat, readouts, dim)
        # Reverse the interpreter at the same R-readout resolution used by the
        # encoder. Expanding to max_span first would be both the wrong inverse
        # order and an O(max_span^2) decoder bottleneck.
        zero_noise = readout.new_zeros(flat)
        for block in reversed(interpreter_blocks):
            z = block.inverse_block(z, active, zero_noise)

        slot_basis = _sinusoidal_position(
            self.max_span, dim, readout.device, readout.dtype
        ).unsqueeze(0).expand(flat, -1, -1)
        q = F.linear(slot_basis, readout_pool.k_proj.weight)
        k = F.linear(z, readout_pool.q_proj.weight)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dim)

        slot_coordinate = torch.linspace(
            0.0,
            float(max(readouts - 1, 0)),
            self.max_span,
            device=readout.device,
            dtype=score.dtype,
        )
        readout_coordinate = torch.arange(readouts, device=readout.device, dtype=score.dtype)
        score = score - (slot_coordinate[:, None] - readout_coordinate[None, :]).abs()
        score = score.masked_fill(~active.unsqueeze(1), float("-inf"))
        weight = torch.nan_to_num(score.softmax(dim=-1))

        # Approximate the reverse of out(V(span)) without introducing decoder
        # parameters. Linear transposes are tied to the encoder pool.
        value = F.linear(z, readout_pool.out.weight.transpose(0, 1))
        expanded = torch.matmul(weight, value)
        expanded = F.linear(expanded, readout_pool.v_proj.weight.transpose(0, 1))
        expanded = expanded.reshape(bsz, chunks, self.max_span, dim)

        vocab = torch.arange(258, device=readout.device)
        table = byte_lookup(vocab).to(expanded.dtype)
        logits = 10.0 * torch.matmul(
            F.normalize(expanded, dim=-1),
            F.normalize(table, dim=-1).transpose(0, 1),
        )
        return logits.masked_fill(~chunk_mask.unsqueeze(-1).unsqueeze(-1), 0.0)


@dataclass
class FLUEDV34ProbeConfig:
    d_model: int = 512
    nhead: int = 8
    ffn_dim: int = 1536
    segmentor_layers: int = 5
    interpreter_layers: int = 3
    memory_rank: int = 4
    readout_vectors: int = 4
    ar_hidden: int = 128
    use_position: bool = True
    position_strategy: str = "layered_rope"
    prompt_position_scale: float = 0.1
    use_prompt_alibi: bool = False
    use_ar: bool = True
    use_structured_lookup: bool = True
    use_memory: bool = True
    use_boundary_bridge: bool = True
    memory_use_position: bool = True
    memory_position_mode: str = "legacy"
    memory_residual_scale: float = 0.1
    memory_context_norm: str = "none"
    memory_scale_mode: str = "fixed"
    memory_scale_max: float = 0.1
    memory_access_mode: str = "other_only"
    current_memory_mode: str = "off"
    current_memory_scale: float = 0.03
    current_memory_scale_max: float = 0.1
    boundary_mode: str = "threshold"
    coding_rate_dim: int = 16
    coding_rate_epsilon: float = 1.0
    coding_rate_temperature: float = 0.15
    coding_rate_mode: str = "exact"
    boundary_blend_alpha: float = 1.0
    fixed_chunk_budget: int = 0
    bytes_per_chunk_budget: int = 16
    use_emit_controller: bool = False
    emit_forward_mode: str = "hard_st"
    emit_initial_probability: float = 0.1
    emit_threshold: float = 0.5
    emit_controller_hidden: int = 0
    emit_controller_slot_embedding: bool = False
    max_chunks: int = 40
    max_span: int = 128
    tau_cut: float = 0.9
    tau_trans: float = 0.75
    boundary_temperature: float = 0.15
    boundary_bridge_gradient_scale: float = 1.0
    noise_scale: float = 0.02
    decoder_mode: str = "legacy_independent"


@dataclass
class V34ProbeOutput:
    byte_logits: torch.Tensor
    readout_z: torch.Tensor
    memory_z: torch.Tensor
    chunks: ChunkBatch
    segmentor: SegmentorOutput
    policy: ThresholdPolicyOutput
    readout_candidates: torch.Tensor
    emit_logits: torch.Tensor
    emit_soft: torch.Tensor
    emit_hard: torch.Tensor
    ar_delta: torch.Tensor
    aux: dict


class FLUEDV34Probe(nn.Module):
    """Scalable FLUED v3.4 codec used by both probes and full models."""

    def __init__(self, config: FLUEDV34ProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or FLUEDV34ProbeConfig()
        c = self.config
        if c.position_strategy not in {"layered_rope", "prompt_additive", "prompt_plus_local_rope", "none"}:
            raise ValueError(
                "position_strategy must be layered_rope, prompt_additive, prompt_plus_local_rope, or none"
            )
        strategy = c.position_strategy if c.use_position else "none"
        self.position_strategy = strategy
        layer_rope = strategy in {"layered_rope", "prompt_plus_local_rope"}
        self.use_prompt_position = strategy in {"prompt_additive", "prompt_plus_local_rope"}
        self.byte_lookup = StructuredByteLookup(c.d_model)
        self.segmentor_blocks = nn.ModuleList(
            [
                DiTStyleBlock(c.d_model, c.nhead, c.ffn_dim, layer_rope, c.use_prompt_alibi)
                for _ in range(c.segmentor_layers)
            ]
        )
        self.segmentor_head = nn.Sequential(nn.LayerNorm(c.d_model), nn.Linear(c.d_model, 1))
        self.policy = FixedDualThresholdPolicy(c.tau_cut, c.tau_trans)
        # Persistent training-state price for batch-level chunk compute. It is
        # deliberately a buffer rather than a learned architecture parameter.
        self.register_buffer("boundary_compute_dual", torch.zeros(()), persistent=True)
        self.chunk_builder = ChunkBuilder(c.max_chunks, c.max_span)
        self.boundary_bridge = SoftBoundaryBridge(
            c.max_chunks,
            c.tau_cut,
            c.boundary_temperature,
            c.boundary_bridge_gradient_scale,
        )
        self.memory_pool = QueryPool(c.d_model, c.memory_rank, c.nhead, layer_rope)
        self.readout_pool = QueryPool(c.d_model, c.readout_vectors, c.nhead, layer_rope)
        memory_position_mode = c.memory_position_mode
        if memory_position_mode == "legacy":
            memory_position_mode = (
                "chunk_rope" if strategy == "layered_rope" and c.memory_use_position else "none"
            )
        self.memory_read = DenseNoSelfMemory(
            c.d_model,
            c.nhead,
            position_mode=memory_position_mode,
            access_mode=c.memory_access_mode,
        )
        self.current_memory_read = DenseNoSelfMemory(
            c.d_model,
            c.nhead,
            position_mode="none",
            access_mode="current_only",
        )
        if c.memory_context_norm not in {"none", "layernorm"}:
            raise ValueError("memory_context_norm must be none or layernorm")
        if c.memory_scale_mode not in {"fixed", "bounded"}:
            raise ValueError("memory_scale_mode must be fixed or bounded")
        if c.memory_scale_max <= 0 or c.current_memory_scale_max <= 0:
            raise ValueError("memory scale maxima must be positive")
        if not 0 <= c.memory_residual_scale <= c.memory_scale_max:
            raise ValueError("memory_residual_scale must be within [0, memory_scale_max]")
        if not 0 <= c.current_memory_scale <= c.current_memory_scale_max:
            raise ValueError("current_memory_scale must be within [0, current_memory_scale_max]")
        # These modules and scalars exist in every ablation so parameter counts
        # stay comparable. LayerNorm is intentionally affine-free: it aligns
        # context scale without adding another learned representation path.
        self.memory_context_normalizer = nn.LayerNorm(c.d_model, elementwise_affine=False)
        self.current_memory_context_normalizer = nn.LayerNorm(c.d_model, elementwise_affine=False)
        self.memory_scale_logit = nn.Parameter(
            torch.tensor(self._scale_to_logit(c.memory_residual_scale, c.memory_scale_max))
        )
        self.current_memory_scale_logit = nn.Parameter(
            torch.tensor(self._scale_to_logit(c.current_memory_scale, c.current_memory_scale_max))
        )
        self.memory_gate = nn.Sequential(nn.LayerNorm(c.d_model * 2), nn.Linear(c.d_model * 2, c.d_model), nn.Sigmoid())
        self.interpreter_blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_model, c.nhead, c.ffn_dim, layer_rope) for _ in range(c.interpreter_layers)]
        )
        # Instantiate in every ablation so parameter counts stay identical.
        self.ar = SmallChunkARCorrection(
            c.d_model,
            c.ar_hidden,
            c.memory_rank,
            c.readout_vectors,
        )
        if c.decoder_mode == "legacy_independent":
            self.decoder = SpanDecoder(c.d_model, c.ffn_dim, c.max_span, self.byte_lookup)
        elif c.decoder_mode == "shared_inverse":
            self.decoder = SharedInverseSpanDecoder(c.d_model, c.max_span)
        else:
            raise ValueError("decoder_mode must be legacy_independent or shared_inverse")
        # Instantiate the ablation table last so enabling the switch does not
        # perturb initialization of the shared architecture.
        self.plain_byte_lookup = PlainByteLookup(c.d_model)
        if not c.use_structured_lookup and c.decoder_mode == "legacy_independent":
            self.decoder.byte_lookup = self.plain_byte_lookup
        # New v3.4 rate/emit modules are initialized last so legacy-mode shared
        # modules retain their historical initialization order.
        self.coding_rate_selector = MarginalCodingRateSelector(
            c.d_model,
            c.coding_rate_dim,
            c.coding_rate_epsilon,
            c.coding_rate_temperature,
            c.coding_rate_mode,
        )
        self.emit_controller = ReadoutEmitController(
            c.d_model,
            c.emit_initial_probability,
            c.emit_threshold,
            hidden_dim=c.emit_controller_hidden,
            max_readouts=c.readout_vectors,
            use_slot_embedding=c.emit_controller_slot_embedding,
        )

    @staticmethod
    def _scale_to_logit(initial: float, maximum: float) -> float:
        ratio = min(max(float(initial) / float(maximum), 1.0e-4), 1.0 - 1.0e-4)
        return math.log(ratio / (1.0 - ratio))

    @staticmethod
    def _effective_scale(
        mode: str,
        fixed: float,
        maximum: float,
        logit: torch.Tensor,
    ) -> torch.Tensor:
        if mode == "bounded":
            return logit.sigmoid() * float(maximum)
        return logit.new_tensor(float(fixed))

    @staticmethod
    def _capacity_safe_cuts(
        requested: torch.Tensor,
        valid: torch.Tensor,
        max_chunks: int,
        max_span: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep executable cuts within lossless chunk-builder capacity."""
        valid_tokens = valid.sum(dim=1)
        forced_base = torch.div(valid_tokens + max_span - 1, max_span, rounding_mode="floor")
        max_requested = (max_chunks - forced_base + 1).clamp(min=1)
        requested_rank = requested.long().cumsum(dim=1)
        executed = requested & requested_rank.le(max_requested.unsqueeze(1))
        overflow = (requested & ~executed).sum(dim=1)
        return executed, overflow

    @staticmethod
    def _uniform_budget_selection(
        valid: torch.Tensor,
        forbidden: torch.Tensor,
        max_chunks: int,
        bytes_per_chunk: int,
        reference: torch.Tensor,
    ) -> CodingRateSelection:
        """Build evenly spaced lossless cuts for pretraining or score anchoring."""
        valid_len = valid.sum(dim=1)
        target = torch.div(
            valid_len + bytes_per_chunk - 1,
            bytes_per_chunk,
            rounding_mode="floor",
        ).clamp(min=1, max=max_chunks)
        positions = torch.arange(valid.size(1), device=valid.device).view(1, -1).expand_as(valid)
        groups = torch.div(
            positions * target.unsqueeze(1),
            valid_len.clamp(min=1).unsqueeze(1),
            rounding_mode="floor",
        ).clamp(max=max_chunks - 1)
        candidates = torch.where(valid & ~forbidden, positions, torch.full_like(positions, valid.size(1)))
        first_per_group = torch.full(
            (valid.size(0), max_chunks), valid.size(1), device=valid.device, dtype=torch.long
        )
        first_per_group.scatter_reduce_(1, groups, candidates, reduce="amin", include_self=True)
        hard = torch.zeros_like(valid)
        b_idx, g_idx = torch.where(
            torch.arange(max_chunks, device=valid.device).view(1, -1) < target.unsqueeze(1)
        )
        t_idx = first_per_group[b_idx, g_idx]
        present = t_idx.lt(valid.size(1))
        hard[b_idx[present], t_idx[present]] = True
        return CodingRateSelection(
            marginal_rate=reference.new_zeros(reference.shape),
            hard_cut=hard,
            soft_cut=hard.float(),
            target_chunks=target,
        )

    def encode(self, token_ids: torch.Tensor) -> V34ProbeOutput:
        c = self.config
        valid = token_ids.ne(PAD_ID)
        active_lookup = self.byte_lookup if c.use_structured_lookup else self.plain_byte_lookup
        x = active_lookup(token_ids)
        if getattr(self, "collect_memory_diagnostics", False) and torch.is_grad_enabled():
            if x.requires_grad:
                x.retain_grad()
                self._diagnostic_byte_input = x
        if self.use_prompt_position:
            prompt_position = _sinusoidal_position(
                token_ids.size(1), c.d_model, x.device, x.dtype
            ).unsqueeze(0)
            x = x + float(c.prompt_position_scale) * prompt_position * valid.unsqueeze(-1).to(x.dtype)
        noise = torch.full((token_ids.size(0),), c.noise_scale if self.training else 0.0, device=x.device)
        if self.training and c.noise_scale > 0:
            x = x + torch.randn_like(x) * c.noise_scale
        h = x
        for block in self.segmentor_blocks:
            h = block(h, valid, noise)
        boundary_logits = self.segmentor_head(h).squeeze(-1)
        confidence = _plastic_signed_confidence(boundary_logits).masked_fill(~valid, 0.0)
        segmentor = SegmentorOutput(confidence, boundary_logits.masked_fill(~valid, 0.0), {"one_shot": True})
        policy = self.policy(confidence, valid)
        raw = (token_ids - 1).clamp(min=0, max=255)
        utf8_cont = raw.ge(0x80) & raw.le(0xBF) & valid
        force_continue = policy.force_continue | utf8_cont
        threshold_policy = ThresholdPolicyOutput(
            confidence=policy.confidence,
            hard_cut=policy.hard_cut & ~utf8_cont,
            soft_transition=policy.soft_transition & ~utf8_cont,
            force_continue=force_continue,
            aux={**policy.aux, "utf8_continuation_hard_guard": True},
        )
        coding_selection = None
        boundary_rate = confidence.new_zeros(confidence.shape)
        if c.boundary_mode in {"marginal_rate_topk", "uniform_l2_blend"}:
            anchor_score = None
            if c.boundary_mode == "uniform_l2_blend":
                anchor_score = self._uniform_budget_selection(
                    valid,
                    utf8_cont,
                    c.max_chunks,
                    c.bytes_per_chunk_budget,
                    confidence,
                ).hard_cut.float()
            coding_selection = self.coding_rate_selector(
                h,
                valid,
                utf8_cont,
                c.max_chunks,
                c.fixed_chunk_budget,
                c.bytes_per_chunk_budget,
                anchor_score=anchor_score,
                blend_alpha=c.boundary_blend_alpha,
            )
            requested_hard_cut = coding_selection.hard_cut
            executable_hard_cut, cut_overflow = self._capacity_safe_cuts(
                requested_hard_cut, valid, c.max_chunks, c.max_span
            )
            boundary_rate = coding_selection.marginal_rate
        elif c.boundary_mode == "uniform_budget":
            coding_selection = self._uniform_budget_selection(
                valid, utf8_cont, c.max_chunks, c.bytes_per_chunk_budget, confidence
            )
            requested_hard_cut = coding_selection.hard_cut
            executable_hard_cut, cut_overflow = self._capacity_safe_cuts(
                requested_hard_cut, valid, c.max_chunks, c.max_span
            )
        elif c.boundary_mode == "uniform_confidence_blend":
            uniform = self._uniform_budget_selection(
                valid, utf8_cont, c.max_chunks, c.bytes_per_chunk_budget, confidence
            )
            boundary_rate = self.coding_rate_selector.marginal_rate(h)
            confidence_soft = torch.sigmoid(
                (confidence.float() - c.tau_cut) / c.boundary_temperature
            )
            confidence_soft = confidence_soft * valid.float() * (~force_continue).float()
            if confidence_soft.size(1):
                confidence_soft = confidence_soft.clone()
                confidence_soft[:, 0] = valid[:, 0].float()
            alpha = min(max(float(c.boundary_blend_alpha), 0.0), 1.0)
            soft_cut = (1.0 - alpha) * uniform.soft_cut + alpha * confidence_soft
            coding_selection = CodingRateSelection(
                marginal_rate=boundary_rate,
                hard_cut=uniform.hard_cut,
                soft_cut=soft_cut,
                target_chunks=uniform.target_chunks,
            )
            requested_hard_cut = uniform.hard_cut
            executable_hard_cut, cut_overflow = self._capacity_safe_cuts(
                requested_hard_cut, valid, c.max_chunks, c.max_span
            )
        elif c.boundary_mode == "confidence_threshold":
            # The fixed signed-confidence threshold owns the executable
            # segmentation. Main-task gradients reach it through the soft
            # boundary bridge; coding rate remains a training/diagnostic signal.
            boundary_rate = self.coding_rate_selector.marginal_rate(h)
            requested_hard_cut = threshold_policy.hard_cut
            executable_hard_cut, cut_overflow = self._capacity_safe_cuts(
                requested_hard_cut, valid, c.max_chunks, c.max_span
            )
        elif c.boundary_mode == "threshold":
            requested_hard_cut = threshold_policy.hard_cut
            executable_hard_cut, cut_overflow = self._capacity_safe_cuts(
                requested_hard_cut, valid, c.max_chunks, c.max_span
            )
        else:
            raise ValueError(f"unknown boundary_mode: {c.boundary_mode}")
        policy = ThresholdPolicyOutput(
            confidence=threshold_policy.confidence,
            hard_cut=executable_hard_cut,
            soft_transition=threshold_policy.soft_transition,
            force_continue=threshold_policy.force_continue,
            aux={
                **threshold_policy.aux,
                "requested_hard_cut": requested_hard_cut,
                "cut_capacity_overflow": cut_overflow,
                "boundary_mode": c.boundary_mode,
            },
        )
        # Segmentation decides spans, but the codec payload starts from the
        # structured byte lookup rather than hidden segmentor activations.
        chunks = self.chunk_builder(
            x,
            valid,
            policy.hard_cut,
            confidence=confidence,
            soft_transition=policy.soft_transition,
            force_continue=policy.force_continue,
        )
        if c.use_boundary_bridge:
            chunks = self.boundary_bridge(
                chunks,
                x,
                confidence,
                valid,
                force_continue,
                cut_probability=coding_selection.soft_cut if coding_selection is not None else None,
            )
        if c.use_memory:
            memory = self.memory_pool(chunks.span_embeddings, chunks.token_mask)
        else:
            memory = chunks.span_embeddings.new_zeros(
                chunks.span_embeddings.size(0), c.max_chunks, c.memory_rank, c.d_model
            )
        interpreter_spans = chunks.span_embeddings
        readout = self.readout_pool(interpreter_spans, chunks.token_mask)
        if c.use_ar:
            memory, readout, ar_delta = self.ar(
                interpreter_spans,
                chunks.token_mask,
                memory,
                readout,
                apply_memory=c.use_memory,
            )
        else:
            ar_delta = readout.new_zeros(())
        if c.use_memory:
            positions = torch.arange(token_ids.size(1), device=token_ids.device).view(1, -1).expand_as(token_ids)
            valid_chunk_id = chunks.chunk_ids.ge(0)
            safe_chunk_id = chunks.chunk_ids.clamp(min=0)
            start = torch.full(
                (token_ids.size(0), c.max_chunks),
                token_ids.size(1),
                device=token_ids.device,
                dtype=torch.long,
            )
            end = torch.full_like(start, -1)
            start.scatter_reduce_(
                1,
                safe_chunk_id,
                torch.where(valid_chunk_id, positions, torch.full_like(positions, token_ids.size(1))),
                reduce="amin",
                include_self=True,
            )
            end.scatter_reduce_(
                1,
                safe_chunk_id,
                torch.where(valid_chunk_id, positions, torch.full_like(positions, -1)),
                reduce="amax",
                include_self=True,
            )
            chunk_anchors = 0.5 * (start.float() + end.clamp(min=0).float())
            diagnostics = bool(getattr(self, "collect_memory_diagnostics", False))
            memory_ctx, mem_aux = self.memory_read(
                readout,
                memory,
                chunks.chunk_mask,
                chunk_anchors=chunk_anchors,
                diagnostics=diagnostics,
            )
            memory_ctx_raw = memory_ctx
            if c.memory_context_norm == "layernorm":
                memory_ctx = self.memory_context_normalizer(memory_ctx)
            memory_scale = self._effective_scale(
                c.memory_scale_mode,
                c.memory_residual_scale,
                c.memory_scale_max,
                self.memory_scale_logit,
            )
            # The controller chooses how strongly each channel reads global
            # context; a bounded residual scale keeps local translation primary.
            gate = self.memory_gate(torch.cat([readout, memory_ctx], dim=-1).detach())
            other_residual = gate * memory_ctx * memory_scale.to(memory_ctx.dtype)
            if c.current_memory_mode in {"separate_detached", "separate_e2e"}:
                current_source = memory.detach() if c.current_memory_mode == "separate_detached" else memory
                current_ctx, current_aux = self.current_memory_read(
                    readout,
                    current_source,
                    chunks.chunk_mask,
                    chunk_anchors=chunk_anchors,
                    diagnostics=diagnostics,
                )
                current_ctx_raw = current_ctx
                if c.memory_context_norm == "layernorm":
                    current_ctx = self.current_memory_context_normalizer(current_ctx)
                current_scale = self._effective_scale(
                    c.memory_scale_mode,
                    c.current_memory_scale,
                    c.current_memory_scale_max,
                    self.current_memory_scale_logit,
                )
                current_residual = current_ctx * current_scale.to(current_ctx.dtype)
            elif c.current_memory_mode == "off":
                current_ctx = torch.zeros_like(readout)
                current_ctx_raw = torch.zeros_like(readout)
                current_residual = torch.zeros_like(readout)
                current_scale = self._effective_scale(
                    c.memory_scale_mode,
                    c.current_memory_scale,
                    c.current_memory_scale_max,
                    self.current_memory_scale_logit,
                )
                current_aux = {
                    "memory_attention_current_share": readout.new_zeros(()),
                    "memory_attention_other_share": readout.new_zeros(()),
                }
            else:
                raise ValueError(f"unknown current_memory_mode: {c.current_memory_mode}")
            memory_residual = other_residual + current_residual
            readout = readout + memory_residual
        else:
            memory_ctx = torch.zeros_like(readout)
            memory_ctx_raw = torch.zeros_like(readout)
            current_ctx = torch.zeros_like(readout)
            current_ctx_raw = torch.zeros_like(readout)
            other_residual = torch.zeros_like(readout)
            current_residual = torch.zeros_like(readout)
            memory_residual = torch.zeros_like(readout)
            memory_scale = self._effective_scale(
                c.memory_scale_mode,
                c.memory_residual_scale,
                c.memory_scale_max,
                self.memory_scale_logit,
            )
            current_scale = self._effective_scale(
                c.memory_scale_mode,
                c.current_memory_scale,
                c.current_memory_scale_max,
                self.current_memory_scale_logit,
            )
            gate = readout.new_zeros(readout.shape)
            mem_aux = {
                "self_allowed": readout.new_zeros(()),
                "memory_attention_current_share": readout.new_zeros(()),
                "memory_attention_other_share": readout.new_zeros(()),
                "visible_memory_slots": readout.new_zeros(
                    readout.size(0), c.max_chunks * c.readout_vectors
                ),
            }
            current_aux = {
                "memory_attention_current_share": readout.new_zeros(()),
                "memory_attention_other_share": readout.new_zeros(()),
            }
        # Interpreter refinement is local to each chunk. Cross-chunk context
        # is available only through the explicit no-self memory path above.
        flat = readout.reshape(readout.size(0) * c.max_chunks, c.readout_vectors, c.d_model)
        flat_valid = chunks.chunk_mask.reshape(-1, 1).expand(-1, c.readout_vectors)
        local_noise = noise.repeat_interleave(c.max_chunks)
        for block in self.interpreter_blocks:
            flat = block(flat, flat_valid, local_noise)
        readout_candidates = flat.reshape_as(readout)
        emit = self.emit_controller(
            readout_candidates,
            chunks.chunk_mask,
            budget_fraction=getattr(self, "emit_budget_override", None),
        )
        if c.use_emit_controller and not getattr(self, "emit_warmup_active", False):
            if c.emit_forward_mode == "hard_st":
                readout = readout_candidates * emit.straight_through.unsqueeze(-1).to(readout_candidates.dtype)
            elif c.emit_forward_mode == "soft":
                readout = readout_candidates * emit.soft.unsqueeze(-1).to(readout_candidates.dtype)
                all_active = chunks.chunk_mask.unsqueeze(-1).expand_as(emit.hard)
                emit = type(emit)(emit.logits, emit.soft, all_active, emit.soft)
            else:
                raise ValueError(f"unknown emit_forward_mode: {c.emit_forward_mode}")
        else:
            readout = readout_candidates
            emit_hard = chunks.chunk_mask.unsqueeze(-1).expand_as(emit.hard)
            emit = type(emit)(emit.logits, emit.soft, emit_hard, emit_hard.to(emit.soft.dtype))
        logits = self.decode(readout, chunks.chunk_mask, emit.hard)
        return V34ProbeOutput(
            byte_logits=logits,
            readout_z=readout,
            memory_z=memory,
            chunks=chunks,
            segmentor=segmentor,
            policy=policy,
            readout_candidates=readout_candidates,
            emit_logits=emit.logits,
            emit_soft=emit.soft,
            emit_hard=emit.hard,
            ar_delta=ar_delta,
            aux={
                **mem_aux,
                "memory_gate_mean": gate[chunks.chunk_mask.unsqueeze(-1).expand_as(gate[..., 0])].mean() if chunks.chunk_mask.any() else gate.new_zeros(()),
                "memory_context_raw_norm": memory_ctx_raw[chunks.chunk_mask].float().norm(dim=-1).mean() if chunks.chunk_mask.any() else gate.new_zeros(()),
                "memory_context_norm": memory_ctx[chunks.chunk_mask].float().norm(dim=-1).mean() if chunks.chunk_mask.any() else gate.new_zeros(()),
                "memory_effective_scale": memory_scale.float(),
                "memory_residual_ratio": (
                    memory_residual[chunks.chunk_mask].float().norm(dim=-1).mean()
                    / readout_candidates[chunks.chunk_mask].detach().float().norm(dim=-1).mean().clamp(min=1.0e-6)
                    if chunks.chunk_mask.any() else gate.new_zeros(())
                ),
                "memory_attention_current_share": mem_aux["memory_attention_current_share"],
                "memory_attention_other_share": mem_aux["memory_attention_other_share"],
                "current_channel_attention_share": current_aux["memory_attention_current_share"],
                "current_memory_context_raw_norm": current_ctx_raw[chunks.chunk_mask].float().norm(dim=-1).mean() if chunks.chunk_mask.any() else gate.new_zeros(()),
                "current_memory_context_norm": current_ctx[chunks.chunk_mask].float().norm(dim=-1).mean() if chunks.chunk_mask.any() else gate.new_zeros(()),
                "current_memory_effective_scale": current_scale.float(),
                "current_memory_readout_cosine": (
                    F.cosine_similarity(current_ctx[chunks.chunk_mask].float(), readout_candidates[chunks.chunk_mask].detach().float(), dim=-1).mean()
                    if chunks.chunk_mask.any() else gate.new_zeros(())
                ),
                "current_memory_contribution_share": (
                    current_residual[chunks.chunk_mask].float().norm(dim=-1).mean()
                    / (
                        current_residual[chunks.chunk_mask].float().norm(dim=-1).mean()
                        + other_residual[chunks.chunk_mask].float().norm(dim=-1).mean()
                    ).clamp(min=1.0e-6)
                    if chunks.chunk_mask.any() else gate.new_zeros(())
                ),
                "requested_hard_cut": requested_hard_cut,
                "cut_capacity_overflow": cut_overflow,
                "marginal_coding_rate": boundary_rate,
                "target_chunks": coding_selection.target_chunks if coding_selection is not None else chunks.chunk_mask.sum(dim=1),
                "boundary_blend_alpha": confidence.new_tensor(float(c.boundary_blend_alpha)),
            },
        )

    def decode(
        self,
        readout: torch.Tensor,
        chunk_mask: torch.Tensor,
        readout_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.decoder_mode == "legacy_independent":
            return self.decoder(readout, chunk_mask)
        if readout_active is None:
            readout_active = chunk_mask.unsqueeze(-1).expand(readout.shape[:-1])
        active_lookup = self.byte_lookup if self.config.use_structured_lookup else self.plain_byte_lookup
        return self.decoder(
            readout,
            chunk_mask,
            readout_active,
            self.readout_pool,
            self.interpreter_blocks,
            active_lookup,
        )

    def forward(self, token_ids: torch.Tensor) -> V34ProbeOutput:
        return self.encode(token_ids)


# Public full-model names share the exact implementation used by probes. Model
# scale is controlled only by FLUEDV34Config fields and training configuration.
FLUEDV34Config = FLUEDV34ProbeConfig
FLUEDV34 = FLUEDV34Probe


def load_v34_state_dict_compatible(
    model: FLUEDV34Probe,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Load an evaluation checkpoint without hiding active-path mismatches.

    P3 added two scalar parameters, and P2 added an optional current-memory
    reader. Older checkpoints may omit them while their corresponding paths are
    disabled. Evaluation may safely retain the freshly initialized inactive
    values; any mismatch on an active path remains a hard error.
    """

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_missing: list[str] = []
    unresolved_missing: list[str] = []
    for name in missing:
        inactive_fixed_scale = (
            model.config.memory_scale_mode == "fixed"
            and name in {"memory_scale_logit", "current_memory_scale_logit"}
        )
        inactive_current_reader = (
            model.config.current_memory_mode == "off"
            and name.startswith("current_memory_read.")
        )
        if inactive_fixed_scale or inactive_current_reader:
            ignored_missing.append(name)
        else:
            unresolved_missing.append(name)
    ignored_unexpected = [name for name in unexpected if name == "logic_transition_prior"]
    unresolved_unexpected = [name for name in unexpected if name not in ignored_unexpected]
    if unresolved_missing or unresolved_unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={unresolved_missing}, unexpected={unresolved_unexpected}"
        )
    return {
        "ignored_missing": ignored_missing,
        "ignored_unexpected": ignored_unexpected,
    }
