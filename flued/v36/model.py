"""FLUED v3.6: whole-prompt compression to a single readout package via KDA state.

Pipeline: bytes -> byte encoder -> slender segmentor (boundary timetable) ->
chunks -> summarizer (projector, one memory per chunk) -> write head ->
KDA state machine (serial, per-sample state) -> readout package (k queries
read the final state) -> backbone -> global span decoder.

v0 uses a pure-PyTorch KDA recurrence (chunk counts are 32-256, kernel
integration is a later acceleration step).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from flued.data import PAD_ID
from flued.v33.chunk_builder import ChunkBatch, ChunkBuilder
from flued.v34.model import (
    DiTStyleBlock,
    FLUEDV34Probe,
    PlainByteLookup,
    SoftBoundaryBridge,
    _plastic_signed_confidence,
)

try:
    from fla.ops.kda import chunk_kda as _fla_chunk_kda
except Exception:  # fla is optional (tests / non-GPU environments)
    _fla_chunk_kda = None


@dataclass
class V36Config:
    """Defaults track configs/canonical_v36.json (v36.2-20260805)."""

    d_byte: int = 384
    encoder_layers: int = 3
    segmentor_layers: int = 9
    nhead: int = 6
    ffn_dim: int = 1152
    d_mem: int = 512
    summarizer_slots: int = 4
    summarizer_hidden: int = 1024
    kda_heads: int = 4
    kda_head_k: int = 128
    kda_head_v: int = 256
    write_hidden: int = 1024
    kda_tau_max: float = 256.0
    readout_queries: int = 1
    d_pack: int = 1536
    d_backbone: int = 384
    backbone_layers: int = 3
    backbone_nhead: int = 8
    backbone_ffn: int = 1024
    backbone_mode: str = "attn"  # "attn" = cross-chunk attention (default; E30 baseline). "mlp" = per-readout, JUDGED DEAD by E32 (encoder gradient starvation / anchor blowup) -- retained behind the flag for the causal-attention follow-up.
    # "per_chunk" (default, current form): the backbone consumes all C per-chunk
    # readouts. "final" = k=1 backbone interface (user-ruled 2026-08-06): the
    # backbone consumes ONLY the final state's readout (carries 0..C history);
    # per-chunk readouts stay on the decoder side, which is what per-chunk
    # conditioning was always for (task density, E23). This is also the demand
    # structure for the state channel: with "final", a state-less readout only
    # holds the last chunk, so whole-window completion forces the recurrence.
    # "paged" = /n paging (user-ruled 2026-08-07, the only legal nq widening):
    # the window is divided into n sub-pages, the state is read once at each
    # sub-page boundary (each read causal to its position), and the backbone
    # transforms the n read vectors pointwise (sub-page-level causal honesty).
    # n=1 reproduces "final"; n=C reproduces per-chunk.
    backbone_readout: str = "per_chunk"
    paged_reads: int = 4
    decoder_hidden: int = 1024
    decoder_layers: int = 3
    max_chunks: int = 64
    max_span: int = 64
    bytes_per_chunk: int = 16
    tau_cut: float = 0.94
    tau_trans: float = 0.75
    boundary_mode: str = "dynamic"
    boundary_temperature: float = 0.15
    boundary_bridge_gradient_scale: float = 0.1
    max_positions: int = 64
    per_chunk_readout: bool = True
    summarizer_type: str = "dit"
    summarizer_dit_layers: int = 2
    prefix_task: bool = False
    prefix_positions: int = 4
    kda_impl: str = "fla"  # fla chunk_kda fused kernel (default since v36.6: +15% steps/s at 23 chunks, parity-verified; essential at byte-level lengths for R0). "torch" serial loop retained as fallback.
    # R1 relative baseline (spec section 4): when False, each chunk's readout
    # comes straight from its own write (same write head, same realign, same
    # transmitted-scalar budget) -- the serial KDA recurrence is bypassed, so
    # the ONLY difference from the main architecture is the state channel.
    state_channel: bool = True


@dataclass
class V36Output:
    logits_direct: torch.Tensor
    logits_backbone: torch.Tensor
    package: torch.Tensor
    backbone_out: torch.Tensor
    chunks: ChunkBatch
    boundary_confidence: torch.Tensor
    memory: torch.Tensor
    state_norm: torch.Tensor
    aux: dict
    prefix: list | None = None  # [(position_i, logits_direct_i, logits_backbone_i), ...]


class ChunkSummarizer(nn.Module):
    def __init__(self, c: V36Config) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.empty(c.summarizer_slots, c.d_byte))
        nn.init.trunc_normal_(self.slots, std=0.02)
        self.q = nn.Linear(c.d_byte, c.d_byte, bias=False)
        self.k = nn.Linear(c.d_byte, c.d_byte, bias=False)
        self.v = nn.Linear(c.d_byte, c.d_byte, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(c.summarizer_slots * c.d_byte),
            nn.Linear(c.summarizer_slots * c.d_byte, c.summarizer_hidden),
            nn.SiLU(),
            nn.Linear(c.summarizer_hidden, c.d_mem),
        )

    def forward(self, spans: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, chunks, span, dim = spans.shape
        q = self.q(self.slots).view(1, 1, self.slots.size(0), dim).expand(bsz, chunks, -1, -1)
        k = self.k(spans)
        v = self.v(spans)
        score = torch.einsum("bcsd,bctd->bcst", q, k) / (dim ** 0.5)
        score = score.masked_fill(~token_mask.unsqueeze(2), -1.0e4)
        attn = score.softmax(dim=-1)
        pooled = torch.einsum("bcst,bctd->bcsd", attn, v)
        return self.ffn(pooled.reshape(bsz, chunks, -1))


class DiTChunkSummarizer(nn.Module):
    """Literal one-shot DiT summarizer: DiTStyleBlock stack over each chunk's
    span (chunks processed in parallel and independently), masked mean pool,
    FFN to one memory vector per chunk. Matches the user's original mental
    model; `slot` (ChunkSummarizer) remains the canonical default.
    """

    def __init__(self, c: V36Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_byte, c.nhead, c.ffn_dim, True, False) for _ in range(c.summarizer_dit_layers)]
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(c.d_byte),
            nn.Linear(c.d_byte, c.summarizer_hidden),
            nn.SiLU(),
            nn.Linear(c.summarizer_hidden, c.d_mem),
        )

    def forward(self, spans: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, chunks, span, dim = spans.shape
        h = spans.reshape(bsz * chunks, span, dim)
        valid = token_mask.reshape(bsz * chunks, span)
        noise = h.new_zeros(h.size(0))
        for block in self.blocks:
            h = block(h, valid, noise)
        denom = valid.sum(dim=1, keepdim=False).clamp(min=1).unsqueeze(-1).to(h.dtype)
        pooled = (h * valid.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        return self.ffn(pooled).view(bsz, chunks, -1)


class WriteHead(nn.Module):
    def __init__(self, c: V36Config) -> None:
        super().__init__()
        h, hk, hv = c.kda_heads, c.kda_head_k, c.kda_head_v
        self.heads, self.hk, self.hv = h, hk, hv
        self.trunk = nn.Sequential(nn.LayerNorm(c.d_mem), nn.Linear(c.d_mem, c.write_hidden), nn.SiLU())
        self.to_k = nn.Linear(c.write_hidden, h * hk)
        self.to_v = nn.Linear(c.write_hidden, h * hv)
        tau = torch.exp(
            torch.linspace(0.0, 1.0, h * hk) * (torch.log(torch.tensor(c.kda_tau_max)))
        ).clamp(min=1.0 + 1e-3)
        self.alpha_logit = nn.Parameter(torch.log(tau - 1.0))
        self.to_beta = nn.Linear(c.write_hidden, h)
        nn.init.constant_(self.to_beta.bias, -2.0)

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, chunks, _ = memory.shape
        h = self.trunk(memory)
        k = F.normalize(self.to_k(h).view(bsz, chunks, self.heads, self.hk), dim=-1)
        v = self.to_v(h).view(bsz, chunks, self.heads, self.hv)
        beta = torch.sigmoid(self.to_beta(h))
        alpha = torch.sigmoid(self.alpha_logit).view(1, 1, self.heads, self.hk)
        return {"k": k, "v": v, "beta": beta, "alpha": alpha}


class KDAStateMachine(nn.Module):
    def __init__(self, c: V36Config) -> None:
        super().__init__()
        self.c = c
        self.readout_query = nn.Parameter(torch.empty(c.readout_queries, c.kda_heads, c.kda_head_k))
        nn.init.trunc_normal_(self.readout_query, std=0.02)
        self.realign = nn.Sequential(
            nn.LayerNorm(c.kda_heads * c.kda_head_v),
            nn.Linear(c.kda_heads * c.kda_head_v, c.d_pack),
            nn.SiLU(),
            nn.Linear(c.d_pack, c.d_pack),
        )

    def forward(self, gates: dict[str, torch.Tensor], chunk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.c.state_channel:
            # R1 relative baseline: bypass the recurrence entirely -- readout
            # per chunk comes from its own write value only. Padded rows are
            # garbage but masked downstream exactly like the main path's
            # stale-state reads, so the comparison stays clean.
            v = gates["v"]
            read = v.reshape(v.size(0), v.size(1), 1, self.c.kda_heads * self.c.kda_head_v)
            return self.realign(read), v.new_zeros((), dtype=torch.float32)
        if self.c.kda_impl == "fla":
            if _fla_chunk_kda is None:
                warnings.warn(
                    "kda_impl='fla' but fla is unavailable; falling back to the torch recurrence",
                    stacklevel=2,
                )
            else:
                return self._forward_fla(gates, chunk_mask)
        return self._forward_torch(gates, chunk_mask)

    def _forward_fla(self, gates: dict[str, torch.Tensor], chunk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v, beta, alpha = gates["k"], gates["v"], gates["beta"], gates["alpha"]
        keep = chunk_mask.unsqueeze(-1).unsqueeze(-1).to(k.dtype)
        k = k * keep
        v = v * keep
        bsz, chunks, heads, dk = k.shape
        beta = beta * chunk_mask.unsqueeze(-1).to(beta.dtype)
        # g in log space; zero at padded chunks -> decay exp(0)=1 (identity step)
        g = torch.log(alpha.clamp_min(1.0e-6)).to(k.dtype) * keep
        q = self.readout_query
        outs = []
        final_state = None
        for qi in range(q.size(0)):
            o, final_state = _fla_chunk_kda(
                q=q[qi].to(k.dtype).unsqueeze(0).unsqueeze(0).expand(bsz, chunks, -1, -1).contiguous(),
                k=k,
                v=v,
                g=g.expand(bsz, chunks, -1, -1).contiguous(),
                beta=beta,
                scale=1.0,
                use_qk_l2norm_in_kernel=False,
                output_final_state=True,
            )
            outs.append(o)
        o = torch.stack(outs, dim=2)  # (B, C, nq, H, V)
        read = o.reshape(bsz, chunks, q.size(0), heads * v.size(-1))
        if self.c.per_chunk_readout:
            package = self.realign(read)
        else:
            package = self.realign(read[:, -1])
        return package, final_state.float().norm(dim=(-2, -1)).mean()

    def _forward_torch(self, gates: dict[str, torch.Tensor], chunk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v, beta, alpha = gates["k"], gates["v"], gates["beta"], gates["alpha"]
        keep = chunk_mask.unsqueeze(-1).unsqueeze(-1).to(k.dtype)
        k = k * keep
        v = v * keep
        bsz, chunks, heads, dk = k.shape
        dv = v.size(-1)
        state = k.new_zeros(bsz, heads, dk, dv, dtype=torch.float32)
        kf, vf = k.float(), v.float()
        beta = beta.float() * chunk_mask.float().unsqueeze(-1)
        alpha = gates["alpha"][0].unsqueeze(0).unsqueeze(-1)
        mask5 = chunk_mask.float().view(bsz, chunks, 1, 1, 1)
        alpha_eff = 1.0 - mask5 + mask5 * alpha
        q = self.readout_query.float()
        per_chunk_reads = []
        for i in range(chunks):
            ki, vi, bi, ai = kf[:, i], vf[:, i], beta[:, i], alpha_eff[:, i]
            state = ai * state
            kTs = torch.einsum("bhk,bhkv->bhv", ki, state)
            state = state - bi.view(bsz, heads, 1, 1) * ki.unsqueeze(-1) * kTs.unsqueeze(-2)
            state = state + bi.view(bsz, heads, 1, 1) * ki.unsqueeze(-1) * vi.unsqueeze(-2)
            if self.c.per_chunk_readout:
                # Read the state right after consuming chunk i: each chunk gets
                # its own conditioning vector instead of the whole-prompt mean.
                per_chunk_reads.append(torch.einsum("qhk,bhkv->bqhv", q, state))
        if self.c.per_chunk_readout:
            read = torch.stack(per_chunk_reads, dim=1)  # (B, C, q, heads*dv)
            read = read.reshape(bsz, chunks, q.size(0), heads * dv).to(k.dtype)
        else:
            read = torch.einsum("qhk,bhkv->bqhv", q, state)
            read = read.reshape(bsz, q.size(0), heads * dv).to(k.dtype)
        package = self.realign(read)
        return package, state.norm(dim=(-2, -1)).mean()

    def init_stream_state(self, bsz: int, device: torch.device | str) -> torch.Tensor:
        """Zero KDA state for incremental/streaming encoding (inference side)."""
        c = self.c
        return torch.zeros(
            bsz, c.kda_heads, c.kda_head_k, c.kda_head_v, device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def stream_step(
        self, gates_i: dict[str, torch.Tensor], state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume ONE chunk's write gates, read the updated state, carry it on.

        Streaming primitive: encoding chunk-by-chunk with a carried state avoids
        recomputing the recurrence from scratch for every prefix (O(C^2) ->
        O(C)). ``gates_i`` holds one chunk (shapes (B, 1, H, K/V) etc., as
        produced by WriteHead on a single memory row). Returns
        ``(package_i, new_state)`` where ``package_i`` is (B, 1, nq, d_pack),
        matching column i of the batched per-chunk package. Uses the fla
        fused-recurrent kernel when ``kda_impl == "fla"`` (falling back to the
        pure-torch update when fla is unavailable).
        """
        c = self.c
        if c.kda_impl == "fla" and _fla_chunk_kda is not None:
            from fla.ops.kda import fused_recurrent_kda

            k, v, beta, alpha = gates_i["k"], gates_i["v"], gates_i["beta"], gates_i["alpha"]
            bsz = k.size(0)
            g = torch.log(alpha.clamp_min(1.0e-6)).to(k.dtype).expand(bsz, 1, -1, -1).contiguous()
            q = self.readout_query
            outs = []
            new_state = state
            for qi in range(q.size(0)):
                o, new_state = fused_recurrent_kda(
                    q=q[qi].to(k.dtype).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous(),
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    scale=1.0,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                )
                outs.append(o)
            o = torch.stack(outs, dim=2)  # (B, 1, nq, H, V)
            read = o.reshape(bsz, 1, q.size(0), c.kda_heads * c.kda_head_v)
            return self.realign(read), new_state

        # Pure-torch fallback: same update math as _forward_torch, one chunk.
        k, v, beta = gates_i["k"], gates_i["v"], gates_i["beta"]
        bsz = k.size(0)
        heads, dv = c.kda_heads, c.kda_head_v
        kf, vf = k.float(), v.float()
        bi = beta.float()[:, 0]  # (B, H) — a real chunk, mask == 1
        ai = gates_i["alpha"][0].float().unsqueeze(-1)  # (1, H, K, 1)
        ki, vi = kf[:, 0], vf[:, 0]
        state = ai * state
        kTs = torch.einsum("bhk,bhkv->bhv", ki, state)
        state = state - bi.view(bsz, heads, 1, 1) * ki.unsqueeze(-1) * kTs.unsqueeze(-2)
        state = state + bi.view(bsz, heads, 1, 1) * ki.unsqueeze(-1) * vi.unsqueeze(-2)
        q = self.readout_query.float()
        read = torch.einsum("qhk,bhkv->bqhv", q, state)
        read = read.reshape(bsz, 1, q.size(0), heads * dv).to(k.dtype)
        return self.realign(read), state


def paged_boundaries(chunk_mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """/n paging geometry (backbone_readout="paged", user-ruled 2026-08-07).

    Returns (boundary, serve):
    - boundary: (B, n) long -- the chunk index whose state read closes sub-page j
      (each read is causal to its position: it contains chunks 0..boundary).
    - serve: (B, C) long -- chunk i is conditioned on the FIRST read whose
      boundary covers it (serve(b,i) = min{ j : boundary(b,j) >= i }), so a read
      never serves a chunk written after it. n=1 reproduces "final" (the whole
      window served by the last read); n=C reproduces per-chunk.
    """
    device = chunk_mask.device
    c_real = chunk_mask.long().sum(dim=1).clamp(min=1)  # (B,)
    j = torch.arange(n, device=device)
    boundary = ((j + 1).unsqueeze(0) * c_real.unsqueeze(1) // n - 1).clamp(min=0)
    boundary = torch.minimum(boundary, (c_real - 1).unsqueeze(1))  # (B, n)
    i = torch.arange(chunk_mask.size(1), device=device)
    serve = (boundary.unsqueeze(2) < i.view(1, 1, -1)).sum(dim=1).clamp(max=n - 1)  # (B, C)
    return boundary, serve


class CrossReadBackbone(nn.Module):
    """Causal cross-read backbone (s31): the backbone issues one query per chunk
    position and retrieves from the encoder's memory rows STRICTLY causally
    (position i sees memory[0..i] only). The Kimi-Linear lesson mapped onto
    FLUED: the KDA state stays the lossy streaming compressor; exact retrieval
    flows through a thin causal channel over the least-processed store
    (pre-state memory). Distinct from s16 on both axes: causal (not
    bidirectional) and reads memory (not state readouts).
    """

    def __init__(self, c: V36Config) -> None:
        super().__init__()
        self.nhead = c.backbone_nhead
        self.in_proj = nn.Linear(c.d_pack, c.d_backbone) if c.d_pack != c.d_backbone else nn.Identity()
        self.q_proj = nn.Linear(c.d_backbone, c.d_backbone, bias=False)
        self.k_proj = nn.Linear(c.d_mem, c.d_backbone, bias=False)
        self.v_proj = nn.Linear(c.d_mem, c.d_backbone, bias=False)
        self.o_proj = nn.Linear(c.d_backbone, c.d_backbone, bias=False)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(c.d_backbone),
                    nn.Linear(c.d_backbone, c.backbone_ffn),
                    nn.GELU(),
                    nn.Linear(c.backbone_ffn, c.d_backbone),
                )
                for _ in range(c.backbone_layers)
            ]
        )
        self.norm = nn.LayerNorm(c.d_backbone)

    def forward(
        self,
        content: torch.Tensor,
        memory: torch.Tensor,
        chunk_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n, _ = content.shape
        h = self.in_proj(content)
        nh = self.nhead
        hd = h.size(-1) // nh
        q = (self.q_proj(h) + pos).view(bsz, n, nh, hd).transpose(1, 2)
        k = self.k_proj(memory).view(bsz, n, nh, hd).transpose(1, 2)
        v = self.v_proj(memory).view(bsz, n, nh, hd).transpose(1, 2)
        future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=content.device), diagonal=1)
        allowed = (~future).view(1, 1, n, n) & chunk_mask.view(bsz, 1, 1, n)
        # keep the diagonal even on padded rows: a fully-masked SDPA row yields
        # NaN, and 0*NaN poisons downstream (documented padding-NaN pitfall)
        allowed = allowed | torch.eye(n, dtype=torch.bool, device=content.device).view(1, 1, n, n)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        o = o.transpose(1, 2).reshape(bsz, n, nh * hd)
        h = h + self.o_proj(o)
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h)


class PointwiseBackbone(nn.Module):
    """Per-readout backbone (v36.5): each chunk's readout vector is transformed
    independently — cross-chunk sequence modeling belongs to the KDA recurrence,
    not to a second attention stack over the chunk matrix. Completion/prediction
    become strictly causal (no future-chunk leakage), so the training form
    matches streaming generation. The legacy attention backbone remains as
    ``backbone_mode="attn"`` for historical arms. Positional information is
    added downstream (chunk_pos in the caller), so no position embedding here.
    """

    def __init__(self, c: V36Config) -> None:
        super().__init__()
        self.in_proj = nn.Linear(c.d_pack, c.d_backbone) if c.d_pack != c.d_backbone else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(c.d_backbone),
                    nn.Linear(c.d_backbone, c.backbone_ffn),
                    nn.GELU(),
                    nn.Linear(c.backbone_ffn, c.d_backbone),
                )
                for _ in range(c.backbone_layers)
            ]
        )
        self.norm = nn.LayerNorm(c.d_backbone)

    def forward(self, package: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(package)
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h)


class TinyBackbone(nn.Module):
    def __init__(self, c: V36Config) -> None:
        super().__init__()
        self.in_proj = nn.Linear(c.d_pack, c.d_backbone) if c.d_pack != c.d_backbone else nn.Identity()
        self.pos = nn.Embedding(c.max_positions, c.d_backbone)
        layer = nn.TransformerEncoderLayer(
            c.d_backbone,
            c.backbone_nhead,
            c.backbone_ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.backbone_layers)
        nn.init.trunc_normal_(self.pos.weight, std=0.02)

    def forward(self, package: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(package.size(1), device=package.device)
        return self.encoder(self.in_proj(package) + self.pos(positions).unsqueeze(0))


class GlobalSpanDecoder(nn.Module):
    def __init__(self, c: V36Config, byte_lookup: PlainByteLookup) -> None:
        super().__init__()
        self.slot = nn.Parameter(torch.empty(c.max_span, c.d_backbone))
        nn.init.trunc_normal_(self.slot, std=0.02)
        self.blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_backbone, c.backbone_nhead, c.backbone_ffn, True, False) for _ in range(c.decoder_layers)]
        )
        self.byte_lookup = byte_lookup
        # k=1 capacity arms (E38): when d_backbone != d_byte the decoder cannot
        # share the byte table directly -- project decoder states back into the
        # byte-table space with a learned map; the table itself stays the fixed
        # public ruler (E31/E34: decoder/table geometry must not be trainable).
        self.out_proj = (
            nn.Linear(c.d_backbone, c.d_byte, bias=False) if c.d_backbone != c.d_byte else nn.Identity()
        )
        # Gradient-checkpoint gate. The frozen-predict path (functional_call with
        # detached params) must NOT checkpoint: the backward recompute would run
        # outside the param substitution and see the live grad-requiring params
        # (CheckpointError metadata mismatch). train_v36_s1 toggles this off
        # around _frozen_decoder_logits.
        self._ckpt_enabled = True
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, cond: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, chunks, dim = cond.shape
        h = self.slot.view(1, 1, *self.slot.shape).expand(bsz, chunks, -1, -1) + cond.unsqueeze(2)
        h = h.reshape(bsz * chunks, self.slot.size(0), dim)
        valid = token_mask.reshape(bsz * chunks, -1)
        noise = h.new_zeros(h.size(0))
        for block in self.blocks:
            # Gradient checkpointing in training: at d_backbone=1536 the decoder
            # blocks dominate activation memory and push the allocator into the
            # thrash zone (observed 15.5GB / ~14s per step); recompute trades
            # ~30% FLOPs for getting back under the red line. Eval and the
            # frozen-predict ruler path skip it (no graph to save).
            if self.training and self._ckpt_enabled and torch.is_grad_enabled():
                h = _grad_checkpoint(block, h, valid, noise, use_reentrant=False)
            else:
                h = block(h, valid, noise)
        h = self.out_proj(h)
        vocab = torch.arange(258, device=h.device)
        table = self.byte_lookup(vocab).to(h.dtype)
        logits = self.scale.clamp(1.0, 100.0) * torch.matmul(
            F.normalize(h, dim=-1), F.normalize(table, dim=-1).transpose(0, 1)
        )
        logits = logits.view(bsz, chunks, self.slot.size(0), -1)
        return logits.masked_fill(~token_mask.unsqueeze(-1), 0.0)


class FLUEDV36(nn.Module):
    def __init__(self, config: V36Config | None = None) -> None:
        super().__init__()
        self.config = config or V36Config()
        c = self.config
        self.byte_lookup = PlainByteLookup(c.d_byte)
        self.encoder_blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_byte, c.nhead, c.ffn_dim, True, True) for _ in range(c.encoder_layers)]
        )
        self.segmentor_blocks = nn.ModuleList(
            [DiTStyleBlock(c.d_byte, c.nhead, c.ffn_dim, True, False) for _ in range(c.segmentor_layers)]
        )
        self.segmentor_head = nn.Sequential(nn.LayerNorm(c.d_byte), nn.Linear(c.d_byte, 1))
        self.chunk_builder = ChunkBuilder(c.max_chunks, c.max_span)
        self.summarizer = ChunkSummarizer(c) if c.summarizer_type == "slot" else DiTChunkSummarizer(c)
        self.write_head = WriteHead(c)
        self.state_machine = KDAStateMachine(c)
        if c.backbone_mode == "attn":
            self.backbone = TinyBackbone(c)
        elif c.backbone_mode == "xattn":
            self.backbone = CrossReadBackbone(c)
        else:
            self.backbone = PointwiseBackbone(c)
        self.bridge = SoftBoundaryBridge(
            c.max_chunks, c.tau_cut, c.boundary_temperature, c.boundary_bridge_gradient_scale
        )
        self.decoder_in = nn.Linear(c.d_pack, c.d_backbone) if c.d_pack != c.d_backbone else nn.Identity()
        self.decoder = GlobalSpanDecoder(c, self.byte_lookup)
        self.chunk_pos = nn.Embedding(c.max_chunks, c.d_backbone)
        nn.init.trunc_normal_(self.chunk_pos.weight, std=0.02)

    def _front_end_frozen(self) -> bool:
        mods = (self.byte_lookup, *self.encoder_blocks, *self.segmentor_blocks, self.segmentor_head)
        return all(not p.requires_grad for m in mods for p in m.parameters())

    def _encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = token_ids.ne(PAD_ID)
        # When the whole front end (byte lookup + encoder + segmentor) is frozen,
        # keep it out of the autograd graph: downstream modules only need the
        # values, and dropping the 12-layer attention graph saves gigabytes.
        ctx = torch.no_grad() if self._front_end_frozen() else torch.enable_grad()
        with ctx:
            x = self.byte_lookup(token_ids)
            noise = x.new_zeros(token_ids.size(0))
            for block in self.encoder_blocks:
                x = block(x, valid, noise)
            byte_states = x
            for block in self.segmentor_blocks:
                x = block(x, valid, noise)
            logits = self.segmentor_head(x).squeeze(-1)
            confidence = _plastic_signed_confidence(logits)
        return byte_states, confidence, valid, logits

    def encode_boundary_logits(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, valid, logits = self._encode(token_ids)
        return logits, valid

    def _cuts(self, token_ids: torch.Tensor, confidence: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        utf8_cont = token_ids.ge(129) & token_ids.le(192) & valid
        if self.config.boundary_mode == "uniform":
            bsz, seq = valid.shape
            cuts = torch.zeros_like(valid)
            first = valid.float().argmax(dim=1)
            cuts[torch.arange(bsz, device=valid.device), first] = True
            position = valid.to(torch.long).cumsum(dim=1)
            cuts = cuts | (valid & position.gt(1) & ((position - 1) % self.config.bytes_per_chunk == 0))
            return cuts & ~utf8_cont, utf8_cont, cuts.new_zeros(1, dtype=torch.float32)
        requested = confidence.gt(self.config.tau_cut) & valid & ~utf8_cont
        first = valid.float().argmax(dim=1)
        requested[torch.arange(valid.size(0), device=valid.device), first] |= valid.any(dim=1)
        executable, overflow = FLUEDV34Probe._capacity_safe_cuts(
            requested, valid, self.config.max_chunks, self.config.max_span
        )
        return executable, utf8_cont, overflow

    def forward(self, token_ids: torch.Tensor) -> V36Output:
        byte_states, confidence, valid, _logits = self._encode(token_ids)
        hard_cut, utf8_cont, cut_overflow = self._cuts(token_ids, confidence, valid)
        chunks = self.chunk_builder(byte_states, valid, hard_cut, confidence)
        chunks = self.bridge(chunks, byte_states, confidence, valid, utf8_cont)
        memory = self.summarizer(chunks.span_embeddings, chunks.token_mask)
        gates = self.write_head(memory)
        package, state_norm = self.state_machine(gates, chunks.chunk_mask)
        if self.config.per_chunk_readout:
            # package: (B, C, q, d_pack) — one conditioning vector per chunk.
            content = package.mean(dim=2)
            n_chunks = chunks.chunk_mask.size(1)
            pos = self.chunk_pos.weight.unsqueeze(0)[:, :n_chunks]
            cond_direct = self.decoder_in(content) + pos
            if self.config.backbone_mode == "xattn":
                # causal cross-read: queries = per-chunk content, K/V = memory,
                # position i sees memory[0..i] only
                backbone_out = self.backbone(content, memory, chunks.chunk_mask, pos)
                cond_backbone = backbone_out + pos
            elif self.config.backbone_readout == "final":
                if self.config.prefix_task:
                    raise ValueError("prefix_task requires backbone_readout='per_chunk'")
                last = chunks.chunk_mask.long().sum(dim=1).clamp(min=1) - 1
                ar = torch.arange(content.size(0), device=content.device)
                backbone_out = self.backbone(content[ar, last].unsqueeze(1))
                cond_backbone = backbone_out + pos  # (B,1,d) broadcasts over chunks
            elif self.config.backbone_readout == "paged":
                if self.config.prefix_task:
                    raise ValueError("prefix_task requires backbone_readout='per_chunk'")
                boundary, serve = paged_boundaries(chunks.chunk_mask, self.config.paged_reads)
                reads = content.gather(1, boundary.unsqueeze(-1).expand(-1, -1, content.size(-1)))
                paged_out = self.backbone(reads)  # (B, n, d_bb) — pointwise per read
                backbone_out = paged_out
                cond_backbone = paged_out.gather(
                    1, serve.unsqueeze(-1).expand(-1, -1, paged_out.size(-1))
                ) + pos
            else:
                backbone_out = self.backbone(content)
                cond_backbone = backbone_out + pos
            prefix = None
            if self.config.prefix_task:
                # Streaming prefix task: at position i, condition on the readout
                # of S_i and restore ALL bytes of chunks 0..i. Positions are
                # sampled among REAL chunks only (padding rows would just burn
                # memory and trigger allocator thrash).
                real_max = int(chunks.chunk_mask.float().sum(dim=1).max().item())
                real_max = max(real_max, 1)
                if self.training:
                    k = min(self.config.prefix_positions - 1, max(real_max - 1, 0))
                    interior = (
                        torch.randperm(real_max - 1, device=token_ids.device)[:k].add(1).tolist()
                        if k > 0
                        else []
                    )
                    positions = sorted(set(interior + [real_max - 1]))
                else:
                    stride = max(1, real_max // 8)
                    positions = sorted(set(list(range(0, real_max, stride)) + [real_max - 1]))
                prefix = []
                for i in positions:
                    mask_i = chunks.token_mask[:, : i + 1]
                    cond_d_i = self.decoder_in(content[:, i]).unsqueeze(1) + pos[:, : i + 1]
                    cond_b_i = backbone_out[:, i].unsqueeze(1) + pos[:, : i + 1]
                    prefix.append((i, self.decoder(cond_d_i, mask_i), self.decoder(cond_b_i, mask_i)))
        else:
            backbone_out = self.backbone(package)
            n_chunks = chunks.chunk_mask.size(1)
            pos = self.chunk_pos.weight.unsqueeze(0)[:, :n_chunks]
            cond_direct = self.decoder_in(package.mean(dim=1)).unsqueeze(1) + pos
            cond_backbone = backbone_out.mean(dim=1).unsqueeze(1) + pos
        logits_ctx = (
            torch.no_grad()
            if self.config.prefix_task and self.training and self.config.per_chunk_readout
            else torch.enable_grad()
        )
        with logits_ctx:
            # In prefix training these full-path logits are metrics-only, so
            # they must not build an autograd graph (two of the biggest tensors
            # per step).
            logits_direct = self.decoder(cond_direct, chunks.token_mask)
            logits_backbone = self.decoder(cond_backbone, chunks.token_mask)
        aux = {
            "truncated_tokens": chunks.pack_info["truncated_tokens"].float(),
            "cut_capacity_overflow": cut_overflow.float().sum(),
            "chunks_per_sample": chunks.chunk_mask.float().sum(dim=1),
            "boundary_confidence_mean": confidence[valid].mean() if valid.any() else confidence.mean(),
            "hard_cut_fraction": hard_cut.float()[valid].mean() if valid.any() else hard_cut.float().mean(),
            "state_norm": state_norm,
        }
        return V36Output(
            logits_direct=logits_direct,
            logits_backbone=logits_backbone,
            package=package,
            backbone_out=backbone_out,
            chunks=chunks,
            boundary_confidence=confidence,
            memory=memory,
            state_norm=state_norm,
            aux=aux,
            prefix=prefix if self.config.per_chunk_readout else None,
        )
