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


class ParallelAttention(nn.Module):
    def __init__(self, dim: int, nhead: int, use_rope: bool) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dim must be divisible by nhead")
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.use_rope = bool(use_rope)
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

    def __init__(self, dim: int, nhead: int, ffn_dim: int, use_rope: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ParallelAttention(dim, nhead, use_rope)
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


class SoftBoundaryBridge(nn.Module):
    """Hard forward chunks with a differentiable soft-assignment backward path."""

    def __init__(self, max_chunks: int, tau_cut: float, temperature: float = 0.35) -> None:
        super().__init__()
        self.max_chunks = int(max_chunks)
        self.tau_cut = float(tau_cut)
        self.temperature = float(temperature)

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
        bridge = (straight_through - hard_mean).unsqueeze(2)
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
    def __init__(self, dim: int, nhead: int) -> None:
        super().__init__()
        if dim % nhead:
            raise ValueError("dim must be divisible by nhead")
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        readout: torch.Tensor,
        memory: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        bsz, chunks, readouts, dim = readout.shape
        ranks = memory.size(2)
        q = self.q(readout).view(bsz, chunks * readouts, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k(memory).view(bsz, chunks * ranks, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v(memory).view(bsz, chunks * ranks, self.nhead, self.head_dim).transpose(1, 2)
        q_owner = torch.arange(chunks, device=readout.device).repeat_interleave(readouts)
        k_owner = torch.arange(chunks, device=readout.device).repeat_interleave(ranks)
        no_self = q_owner[:, None].ne(k_owner[None, :])
        key_valid = chunk_mask.repeat_interleave(ranks, dim=1)
        query_valid = chunk_mask.repeat_interleave(readouts, dim=1)
        allowed = no_self[None, None] & key_valid[:, None, None, :] & query_valid[:, None, :, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        y = y.transpose(1, 2).reshape(bsz, chunks, readouts, dim)
        y = self.out(y) * chunk_mask.unsqueeze(-1).unsqueeze(-1).to(y.dtype)
        return y, {
            "self_allowed": torch.zeros((), device=y.device),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, chunks, width, dim = spans.shape
        flat = spans.reshape(bsz * chunks, width, dim)
        sequence, _ = self.gru(flat)
        lengths = token_mask.sum(dim=-1).clamp(min=1).reshape(-1)
        index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
        summary = torch.gather(sequence, 1, index).squeeze(1)
        memory_delta = torch.tanh(self.memory_proj(summary)).view(bsz, chunks, self.memory_rank, dim)
        readout_delta = torch.tanh(self.readout_proj(summary)).view(bsz, chunks, self.readouts, dim)
        memory_gate = self.gate_scale * torch.sigmoid(self.memory_gate(summary)).view(bsz, chunks, 1, 1)
        readout_gate = self.gate_scale * torch.sigmoid(self.readout_gate(summary)).view(bsz, chunks, 1, 1)
        memory_applied = memory_gate * memory_delta
        readout_applied = readout_gate * readout_delta
        active = token_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1).to(memory.dtype)
        memory_out = (memory + memory_applied) * active
        readout_out = (readout + readout_applied) * active
        ratio = 0.5 * (
            memory_applied.norm(dim=-1).mean() / memory.detach().norm(dim=-1).mean().clamp(min=1.0e-6)
            + readout_applied.norm(dim=-1).mean() / readout.detach().norm(dim=-1).mean().clamp(min=1.0e-6)
        )
        return memory_out, readout_out, ratio


class SpanDecoder(nn.Module):
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
    use_ar: bool = True
    use_structured_lookup: bool = True
    use_memory: bool = True
    use_logic_prior: bool = True
    use_boundary_bridge: bool = True
    boundary_mode: str = "threshold"
    coding_rate_dim: int = 16
    coding_rate_epsilon: float = 1.0
    coding_rate_temperature: float = 0.15
    coding_rate_mode: str = "exact"
    fixed_chunk_budget: int = 0
    bytes_per_chunk_budget: int = 16
    use_emit_controller: bool = False
    emit_forward_mode: str = "hard_st"
    emit_initial_probability: float = 0.1
    max_chunks: int = 40
    max_span: int = 128
    tau_cut: float = 0.9
    tau_trans: float = 0.75
    boundary_temperature: float = 0.15
    noise_scale: float = 0.02


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
    """One-tenth-scale v3.4 probe for position x AR experiments."""

    def __init__(self, config: FLUEDV34ProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or FLUEDV34ProbeConfig()
        c = self.config
        self.byte_lookup = StructuredByteLookup(c.d_model)
        self.segmentor_blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_model, c.nhead, c.ffn_dim, c.use_position) for _ in range(c.segmentor_layers)]
        )
        self.segmentor_head = nn.Sequential(nn.LayerNorm(c.d_model), nn.Linear(c.d_model, 1))
        self.policy = FixedDualThresholdPolicy(c.tau_cut, c.tau_trans)
        self.chunk_builder = ChunkBuilder(c.max_chunks, c.max_span)
        self.boundary_bridge = SoftBoundaryBridge(c.max_chunks, c.tau_cut, c.boundary_temperature)
        self.logic_transition_prior = nn.Parameter(torch.empty(c.d_model))
        self.memory_pool = QueryPool(c.d_model, c.memory_rank, c.nhead, c.use_position)
        self.readout_pool = QueryPool(c.d_model, c.readout_vectors, c.nhead, c.use_position)
        self.memory_read = DenseNoSelfMemory(c.d_model, c.nhead)
        self.memory_gate = nn.Sequential(nn.LayerNorm(c.d_model * 2), nn.Linear(c.d_model * 2, c.d_model), nn.Sigmoid())
        self.interpreter_blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_model, c.nhead, c.ffn_dim, c.use_position) for _ in range(c.interpreter_layers)]
        )
        # Instantiate in every ablation so parameter counts stay identical.
        self.ar = SmallChunkARCorrection(
            c.d_model,
            c.ar_hidden,
            c.memory_rank,
            c.readout_vectors,
        )
        self.decoder = SpanDecoder(c.d_model, c.ffn_dim, c.max_span, self.byte_lookup)
        nn.init.trunc_normal_(self.logic_transition_prior, std=0.02)
        # Instantiate the ablation table last so enabling the switch does not
        # perturb initialization of the shared architecture.
        self.plain_byte_lookup = PlainByteLookup(c.d_model)
        if not c.use_structured_lookup:
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
        self.emit_controller = ReadoutEmitController(c.d_model, c.emit_initial_probability)

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

    def encode(self, token_ids: torch.Tensor) -> V34ProbeOutput:
        c = self.config
        valid = token_ids.ne(PAD_ID)
        active_lookup = self.byte_lookup if c.use_structured_lookup else self.plain_byte_lookup
        x = active_lookup(token_ids)
        noise = torch.full((token_ids.size(0),), c.noise_scale if self.training else 0.0, device=x.device)
        if self.training and c.noise_scale > 0:
            x = x + torch.randn_like(x) * c.noise_scale
        h = x
        for block in self.segmentor_blocks:
            h = block(h, valid, noise)
        boundary_logits = self.segmentor_head(h).squeeze(-1)
        confidence = torch.tanh(boundary_logits).masked_fill(~valid, 0.0)
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
        if c.boundary_mode == "marginal_rate_topk":
            coding_selection = self.coding_rate_selector(
                h,
                valid,
                utf8_cont,
                c.max_chunks,
                c.fixed_chunk_budget,
                c.bytes_per_chunk_budget,
            )
            requested_hard_cut = coding_selection.hard_cut
            executable_hard_cut = requested_hard_cut
            cut_overflow = torch.zeros(token_ids.size(0), device=token_ids.device, dtype=torch.long)
        elif c.boundary_mode == "uniform_budget":
            valid_len = valid.sum(dim=1)
            target = torch.div(
                valid_len + c.bytes_per_chunk_budget - 1,
                c.bytes_per_chunk_budget,
                rounding_mode="floor",
            ).clamp(min=1, max=c.max_chunks)
            positions = torch.arange(valid.size(1), device=valid.device).view(1, -1).expand_as(valid)
            groups = torch.div(
                positions * target.unsqueeze(1),
                valid_len.clamp(min=1).unsqueeze(1),
                rounding_mode="floor",
            ).clamp(max=c.max_chunks - 1)
            candidates = torch.where(valid & ~utf8_cont, positions, torch.full_like(positions, valid.size(1)))
            first_per_group = torch.full(
                (valid.size(0), c.max_chunks), valid.size(1), device=valid.device, dtype=torch.long
            )
            first_per_group.scatter_reduce_(1, groups, candidates, reduce="amin", include_self=True)
            requested_hard_cut = torch.zeros_like(valid)
            b_idx, g_idx = torch.where(
                torch.arange(c.max_chunks, device=valid.device).view(1, -1) < target.unsqueeze(1)
            )
            t_idx = first_per_group[b_idx, g_idx]
            present = t_idx.lt(valid.size(1))
            requested_hard_cut[b_idx[present], t_idx[present]] = True
            coding_selection = CodingRateSelection(
                marginal_rate=confidence.new_zeros(confidence.shape),
                hard_cut=requested_hard_cut,
                soft_cut=requested_hard_cut.float(),
                target_chunks=target,
            )
            executable_hard_cut = requested_hard_cut
            cut_overflow = torch.zeros(token_ids.size(0), device=token_ids.device, dtype=torch.long)
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
        memory = self.memory_pool(chunks.span_embeddings, chunks.token_mask)
        logic_prior = (
            chunks.transition_markers.unsqueeze(-1).to(chunks.span_embeddings.dtype)
            * self.logic_transition_prior.view(1, 1, 1, -1)
        )
        interpreter_spans = chunks.span_embeddings + logic_prior * float(c.use_logic_prior)
        readout = self.readout_pool(interpreter_spans, chunks.token_mask)
        if c.use_ar:
            memory, readout, ar_delta = self.ar(
                interpreter_spans,
                chunks.token_mask,
                memory,
                readout,
            )
        else:
            ar_delta = readout.new_zeros(())
        memory_ctx, mem_aux = self.memory_read(readout, memory, chunks.chunk_mask)
        # The usage-range regularizer should specialize the gate controller,
        # not reshape readout or memory merely to satisfy a scalar quota.
        gate = self.memory_gate(torch.cat([readout, memory_ctx], dim=-1).detach())
        readout = readout + gate * memory_ctx * float(c.use_memory)
        flat = readout.reshape(readout.size(0), -1, readout.size(-1))
        flat_valid = chunks.chunk_mask.unsqueeze(-1).expand(-1, -1, c.readout_vectors).reshape(readout.size(0), -1)
        for block in self.interpreter_blocks:
            flat = block(flat, flat_valid, noise)
        readout_candidates = flat.reshape_as(readout)
        emit = self.emit_controller(readout_candidates, chunks.chunk_mask)
        if c.use_emit_controller:
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
        logits = self.decoder(readout, chunks.chunk_mask)
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
                "requested_hard_cut": requested_hard_cut,
                "cut_capacity_overflow": cut_overflow,
                "marginal_coding_rate": coding_selection.marginal_rate if coding_selection is not None else confidence.new_zeros(confidence.shape),
                "target_chunks": coding_selection.target_chunks if coding_selection is not None else chunks.chunk_mask.sum(dim=1),
            },
        )

    def forward(self, token_ids: torch.Tensor) -> V34ProbeOutput:
        return self.encode(token_ids)
