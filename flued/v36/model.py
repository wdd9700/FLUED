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

import torch
from torch import nn
import torch.nn.functional as F

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
    """Defaults track configs/canonical_v36.json (v36.1-20260731)."""

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
    per_chunk_readout: bool = False
    summarizer_type: str = "slot"
    summarizer_dit_layers: int = 2
    prefix_task: bool = False
    prefix_positions: int = 4
    kda_impl: str = "torch"


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
        if self.c.kda_impl == "fla":
            if _fla_chunk_kda is None:
                raise RuntimeError("kda_impl='fla' requires the fla package")
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
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, cond: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, chunks, dim = cond.shape
        h = self.slot.view(1, 1, *self.slot.shape).expand(bsz, chunks, -1, -1) + cond.unsqueeze(2)
        h = h.reshape(bsz * chunks, self.slot.size(0), dim)
        valid = token_mask.reshape(bsz * chunks, -1)
        noise = h.new_zeros(h.size(0))
        for block in self.blocks:
            h = block(h, valid, noise)
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
        self.backbone = TinyBackbone(c)
        self.bridge = SoftBoundaryBridge(
            c.max_chunks, c.tau_cut, c.boundary_temperature, c.boundary_bridge_gradient_scale
        )
        self.decoder_in = nn.Linear(c.d_pack, c.d_backbone) if c.d_pack != c.d_backbone else nn.Identity()
        if c.d_backbone != c.d_byte:
            raise ValueError("v0 requires d_backbone == d_byte so the decoder shares the encoder byte table")
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
            backbone_out = self.backbone(content)
            n_chunks = chunks.chunk_mask.size(1)
            pos = self.chunk_pos.weight.unsqueeze(0)[:, :n_chunks]
            cond_direct = self.decoder_in(content) + pos
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
