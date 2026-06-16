"""
FLUED E3 — Downstream LM with frozen segmentation backbone.

Architecture (six variants)
-----------------------------
  Input → Frozen Encoder / Tokenizer → segments → Downstream Transformer → logits

  FLUED:       bytes → FLUED.encode() → boundary_probs → soft cumsum → mean-pool
               → CausalTransformerLM → byte logits
  BLT:         bytes → ByteLM → entropy → hard segments → mean-pool
               → CausalTransformerLM → byte logits
  BPE:         bytes → BPE tokenizer → token IDs → CausalTransformerLM → token logits
  FixedPatch:  bytes → fixed N-byte chunks → mean-pool → CausalTransformerLM → byte logits
  Byte:        bytes → CausalTransformerLM → byte logits (no compression baseline)
  PublicTok:   bytes → public tokenizer (Llama/tiktoken) → CausalTransformerLM → token logits

Training objective: causal next-token/byte prediction (cross-entropy).
Evaluation metric: bits-per-byte (bpb) for fair cross-method comparison.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Positional encoding (shared)
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])


import math


# ===========================================================================
# Shared causal Transformer backbone (the ~350M part)
# ===========================================================================

class CausalTransformerLM(nn.Module):
    """Causal Transformer for downstream language modeling.

    Uses TransformerEncoder with causal mask for autoregressive prediction.
    """

    def __init__(
        self,
        vocab_size: int = 257,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] → logits: [B, T, vocab_size]"""
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.pos_enc(x)
        out = self.transformer(h, mask=causal_mask)
        return self.lm_head(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# FLUED downstream wrapper
# ===========================================================================

class FLUEDDownstream(nn.Module):
    """Frozen FLUED encoder → segments → Causal Transformer LM.

    Loads a pre-trained FLUED checkpoint, freezes it, and adds a
    causal Transformer downstream head. The encoder dimensions are
    inferred automatically from the checkpoint.
    """

    def __init__(
        self,
        flued_ckpt: str,
        d_model: int = 1024,             # downstream LM dimension (must match checkpoint)
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 28,            # downstream LM layers (~353M for d=1024)
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        from flued.model import FLUEDAutoencoder

        # Load frozen FLUED — use checkpoint dimensions
        ckpt = torch.load(flued_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        cfg = ckpt.get("model_config", {})
        ckpt_d = state["embedding.weight"].shape[1]
        if "blocks.0.ff_gate.weight" in state:
            ckpt_ff = cfg.get("dim_feedforward", 4096)
            ckpt_swiglu = state["blocks.0.ff_gate.weight"].shape[0]
        else:
            ckpt_ff = state["blocks.0.ff1.weight"].shape[0]
            ckpt_swiglu = None
        ckpt_nhead = cfg.get("nhead", ckpt_d // 64)
        ckpt_layers = cfg.get(
            "num_layers",
            len({k.split(".")[1] for k in state if k.startswith("blocks.")}),
        )

        self.encoder = FLUEDAutoencoder(
            d_model=ckpt_d,
            nhead=ckpt_nhead,
            dim_feedforward=ckpt_ff,
            swiglu_hidden=ckpt_swiglu,
            num_layers=ckpt_layers,
            max_seq_len=cfg.get("max_seq_len", 512),
            assignment_window=cfg.get("assignment_window", 128),
            min_boundary_units=cfg.get("min_boundary_units", 1.0),
            dropout=0.0,
            lambda_var=0.5, lambda_entropy=0.05, lambda_utf8=0.02, lambda_type=0.05,
            compression_weight=cfg.get("compression_weight", 0.1),
            target_compression=cfg.get("target_compression", 0.3),
        )
        self.encoder.load_state_dict(state)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.enc_d_model = ckpt_d

        # Byte embedding — raw byte → encoder dim, zero context (fair vs BPE/BLT).
        # Uses encoder d_model so it shares the same projection as the encoder
        # output path, keeping the LM input dimension consistent.
        # Trainable: learned during E3 alongside the LM.
        self.byte_embed = nn.Embedding(257, ckpt_d)

        # Projection if encoder dim != downstream dim (usually same)
        if ckpt_d != d_model:
            self.proj = nn.Linear(ckpt_d, d_model)
        else:
            self.proj = nn.Identity()

        # Downstream causal LM
        self.lm = CausalTransformerLM(
            vocab_size=257, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            max_seq_len=max_seq_len, dropout=dropout,
        )

    @torch.no_grad()
    def segment(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft segmentation — no hard threshold, fully differentiable in spirit.

        Uses cumulative boundary probability mass as segment IDs,
        then scatter-pools expanded_soft (which already passed through
        the soft assignment matrix A). No binary threshold anywhere.
        """
        pad_mask = src == 0
        expanded, metrics = self.encoder.encode(src, pad_mask, skip_hard=True)

        B, T, d = expanded.shape
        device = src.device
        # Work in float32 for scatter_add stability; autocast may yield float16
        expanded = expanded.float()
        valid = (~pad_mask).float()

        # Soft segment IDs: each position's segment = floor(cumulative bp)
        # No threshold — boundary signal is the continuous bp mass itself
        bp = metrics["boundary_probs"].float()      # [B, T]
        seg_ids = bp.cumsum(dim=1).long()         # [B, T]
        seg_ids = seg_ids - seg_ids[:, :1]          # start from 0
        seg_ids = seg_ids.masked_fill(pad_mask, -1)

        M = int(seg_ids.max().item()) + 1

        # Scatter-based mean pooling (vectorized, same pattern as EntropyPatcher)
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * M
        flat_idx = (seg_ids + batch_offsets).clamp(min=0)  # [B, T]

        seg_sums = torch.zeros(B * M, d, device=device, dtype=expanded.dtype)
        seg_sums.scatter_add_(
            0,
            flat_idx.unsqueeze(-1).expand(-1, -1, d).reshape(-1, d),
            (expanded * valid.unsqueeze(-1)).reshape(-1, d),
        )

        seg_counts = torch.zeros(B * M, device=device, dtype=expanded.dtype)
        seg_counts.scatter_add_(0, flat_idx.reshape(-1), valid.reshape(-1))

        seg_means = (seg_sums / seg_counts.clamp(min=1).unsqueeze(-1)).view(B, M, d)
        seg_lens = seg_counts.long().view(B, M)
        return self.proj(seg_means), seg_lens

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fair E3 forward — encoder used ONLY for segmentation, NOT for representations.

        Pipeline (cf. BLT which uses causal ByteLM for both):
          1. Frozen encoder → boundary_probs (bidirectional, but only for segmentation)
          2. cumsum(bp) → segment IDs
          3. Raw byte embeddings (zero context, like BPE token embeddings)
          4. Scatter-mean pool per segment → broadcast back to byte positions
          5. CausalTransformerLM → byte logits

        The LM input contains no bidirectional encoder context — only per-segment
        pooled raw byte embeddings. This makes the comparison to BLT and BPE fair:
        all three methods feed the LM with representations that do not encode
        future information.
        """
        pad_mask = src == 0
        with torch.no_grad():
            _, metrics = self.encoder.encode(src, pad_mask, skip_hard=True)
            bp = metrics["boundary_probs"]  # [B, T]

        B, T = src.shape
        device = src.device
        valid = (~pad_mask).float()

        # Segment IDs from cumulative boundary probability mass
        seg_ids = bp.cumsum(dim=1).long()            # [B, T]
        seg_ids = seg_ids - seg_ids[:, :1]            # start from 0
        seg_ids = seg_ids.masked_fill(pad_mask, -1)
        M = int(seg_ids.max().item()) + 1

        # Raw byte embeddings — trainable, zero context (fair vs BPE/BLT)
        byte_embeds = self.byte_embed(src)            # [B, T, d_model]
        d_emb = byte_embeds.shape[-1]

        # Scatter-mean pool per segment (same vectorized pattern as BLT EntropyPatcher)
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * M
        flat_idx = (seg_ids + batch_offsets).clamp(min=0)  # [B, T]

        seg_sums = torch.zeros(B * M, d_emb, device=device, dtype=byte_embeds.dtype)
        src_flat = (byte_embeds * valid.unsqueeze(-1)).reshape(B * T, d_emb)
        idx_flat = flat_idx.reshape(B * T, 1).expand(-1, d_emb)
        seg_sums.scatter_add_(0, idx_flat, src_flat)

        seg_counts = torch.zeros(B * M, device=device, dtype=byte_embeds.dtype)
        seg_counts.scatter_add_(0, flat_idx.reshape(-1), valid.reshape(-1))

        seg_means = (seg_sums / seg_counts.clamp(min=1).unsqueeze(-1)).view(B, M, d_emb)

        # Broadcast: each byte position gets its segment's pooled embedding
        idx = seg_ids.unsqueeze(-1).expand(B, T, d_emb).clamp(min=0, max=M - 1)
        expanded = seg_means.gather(1, idx)           # [B, T, d_emb]

        expanded = self.proj(expanded)
        logits = self.lm(expanded)                      # [B, T, 257]

        seg_lens = seg_counts.long().view(B, M)
        return logits, seg_lens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# BLT downstream wrapper
# ===========================================================================

class BLTDownstream(nn.Module):
    """Frozen BLT → segments → Causal Transformer LM."""

    def __init__(
        self,
        blt_ckpt: str,
        bytelm_ckpt: str,
        entropy_theta: float = 0.3,
        d_model: int = 1024,
        local_d_model: int = 512,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        from blt_baseline.model import BLTAutoencoder, ByteLanguageModel

        # Load frozen ByteLM
        lm_ckpt = torch.load(bytelm_ckpt, map_location="cpu", weights_only=False)
        cfg = lm_ckpt.get("config", {})
        self.byte_lm = ByteLanguageModel(
            vocab_size=257,
            d_model=cfg.get("d_model", local_d_model),
            nhead=cfg.get("nhead", 8),
            dim_feedforward=cfg.get("dim_feedforward", 2048),
            num_layers=cfg.get("num_layers", 4),
            max_len=max_seq_len, dropout=cfg.get("dropout", 0.0),
        )
        lm_state = {
            k: v for k, v in lm_ckpt["model"].items()
            if not k.endswith("pos_enc.pe")
        }
        self.byte_lm.load_state_dict(lm_state, strict=False)
        for p in self.byte_lm.parameters():
            p.requires_grad = False
        self.byte_lm.eval()

        # Load frozen BLT (global TF + decoder).
        # Infer architecture from ckpt state_dict so we don't hard-code
        # layer counts (BLT was trained with various global_layers, e.g. 10 or 11).
        blt_ckpt_data = torch.load(blt_ckpt, map_location="cpu", weights_only=False)
        sd = blt_ckpt_data["model"]
        # Count global_transformer layers
        gl_idxs = {int(k.split(".")[2]) for k in sd
                   if k.startswith("global_transformer.layers.")}
        inferred_global_layers = (max(gl_idxs) + 1) if gl_idxs else 10
        # Count decoder layers (BLT decoder uses nn.TransformerEncoder under self.decoder)
        dec_idxs = {int(k.split(".")[2]) for k in sd
                    if k.startswith("decoder.layers.")}
        inferred_decoder_layers = (max(dec_idxs) + 1) if dec_idxs else 12
        # Infer d_model from a known parameter
        # global_transformer.layers.0.linear1.weight: [dim_ff, d_model]
        if "global_transformer.layers.0.linear1.weight" in sd:
            inferred_ff, inferred_d = sd["global_transformer.layers.0.linear1.weight"].shape
        else:
            inferred_ff, inferred_d = dim_feedforward, d_model
        inferred_nhead = max(1, inferred_d // 64)
        import logging as _lg
        _lg.getLogger("e3.blt_downstream").info(
            "BLT ckpt inferred: d_model=%d nhead=%d dim_ff=%d global_layers=%d decoder_layers=%d",
            inferred_d, inferred_nhead, inferred_ff,
            inferred_global_layers, inferred_decoder_layers,
        )
        self.blt = BLTAutoencoder(
            vocab_size=257, d_model=inferred_d, nhead=inferred_nhead,
            dim_feedforward=inferred_ff,
            global_layers=inferred_global_layers, decoder_layers=inferred_decoder_layers,
            local_lm=self.byte_lm, local_lm_d_model=local_d_model,
            patch_mode="entropy", entropy_theta=entropy_theta,
            max_seq_len=max_seq_len, dropout=0.0,
        )
        # Use strict=False because ByteLanguageModel's internal weights may
        # appear under both top-level (loaded above) and BLT.local_lm.* keys.
        sd = {k: v for k, v in sd.items() if not k.endswith("pos_enc.pe")}
        missing, unexpected = self.blt.load_state_dict(sd, strict=False)
        if unexpected:
            _lg.getLogger("e3.blt_downstream").warning(
                "BLT load_state_dict unexpected: %s", unexpected[:5])
        # Override the downstream d_model with what the ckpt actually has
        d_model = inferred_d
        self.blt.freeze_local_lm()
        for p in self.blt.parameters():
            p.requires_grad = False
        self.blt.eval()
        self.local_d_model = local_d_model
        self.seg_threshold = entropy_theta

        # Projection from local d_model to global d_model (if needed)
        if local_d_model != d_model:
            self.patch_proj = nn.Linear(local_d_model, d_model)
        else:
            self.patch_proj = nn.Identity()

        # Downstream LM
        self.lm = CausalTransformerLM(
            vocab_size=257, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            max_seq_len=max_seq_len, dropout=dropout,
        )

    @torch.no_grad()
    def segment(self, src: torch.Tensor):
        """BLT-style segmentation: ByteLM → entropy → mean-pool segments.

        Returns patches [B, M, d_global], seg_lens [B, M], and seg_ids [B, T]
        — the last lets us scatter / gather back to byte-level logits.
        """
        local_repr, entropy = self.byte_lm.compute_entropy(src)
        # Mirror EntropyPatcher's seg-id computation so we can broadcast back.
        boundaries = (entropy > self.seg_threshold).float()
        boundaries[:, 0] = 1.0
        seg_ids_full = boundaries.cumsum(dim=1).long() - 1  # [B, T] in [0, M-1]
        pad_mask = (src != 0).long()
        seg_ids_full = (seg_ids_full * pad_mask).clamp(min=0)

        patches, seg_lens, _ = self.blt.patcher(local_repr, entropy, src)
        patches = self.patch_proj(patches)
        return patches, seg_lens, seg_ids_full

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns byte-level logits [B, T, 257] for fair byte-CE training.

        Pipeline:
            patches [B, M, d] → causal LM [B, M, d] → broadcast to bytes via
            seg_ids → per-byte logits [B, T, 257].
        """
        patches, seg_lens, seg_ids = self.segment(src)
        patch_logits = self.lm(patches)                       # [B, M, 257]
        # Broadcast each patch's logits to all bytes in that segment.
        # gather along dim=1 with index = seg_ids
        B, T = src.shape
        V = patch_logits.size(-1)
        idx = seg_ids.unsqueeze(-1).expand(B, T, V)            # [B, T, V]
        # Clamp idx in case any seg_id exceeds M-1 due to rounding
        idx = idx.clamp(max=patch_logits.size(1) - 1)
        byte_logits = patch_logits.gather(1, idx)              # [B, T, V]
        return byte_logits, seg_lens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# BPE downstream wrapper
# ===========================================================================

class BPEDownstream(nn.Module):
    """BPE tokenizer → Token Embedding → Causal Transformer LM.

    Also exposes ``token_byte_len`` — a buffer [vocab] holding the UTF-8 byte
    length of each token's surface form. This lets bits_per_byte() rescale
    token-level CE to a true per-byte metric for fair cross-method comparison.
    """

    def __init__(
        self,
        tokenizer_path: str,
        vocab_size: int = 8192,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.token_vocab = self.tokenizer.get_vocab_size()
        self.pad_id = 0

        self.embedding = nn.Embedding(self.token_vocab, d_model, padding_idx=self.pad_id)
        self.lm = CausalTransformerLM(
            vocab_size=self.token_vocab, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            max_seq_len=max_seq_len, dropout=dropout,
        )

        # Build token-id → byte-length table.
        # We use tokenizer.id_to_token(i) which returns the surface form, then
        # encode it back to bytes. ByteLevel pre-tokenizer's special chars (Ġ, Ċ, …)
        # decode to whitespace; we count the *decoded* string's UTF-8 bytes to
        # approximate true coverage.
        from tokenizers import decoders as _dec
        # If tokenizer has no decoder configured, decode() may return joined
        # surface form; we accept that as a best-effort byte estimate.
        bl = torch.zeros(self.token_vocab, dtype=torch.long)
        for tid in range(self.token_vocab):
            tok_str = self.tokenizer.id_to_token(tid) or ""
            # Special tokens (<pad>, <bos>, <eos>, <unk>) → 0 bytes
            if tok_str.startswith("<") and tok_str.endswith(">"):
                bl[tid] = 0
                continue
            try:
                decoded = self.tokenizer.decode([tid], skip_special_tokens=False)
            except Exception:
                decoded = tok_str
            bl[tid] = len(decoded.encode("utf-8"))
        self.register_buffer("token_byte_len", bl)

    def tokenize(self, texts: List[str], max_len: int = 512) -> torch.Tensor:
        """Tokenize texts to padded tensor [B, T]."""
        all_ids = []
        for t in texts:
            ids = self.tokenizer.encode(t).ids
            if len(ids) > max_len:
                ids = ids[:max_len]
            all_ids.append(ids)
        T = max(len(ids) for ids in all_ids)
        batch = torch.full((len(all_ids), T), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(all_ids):
            batch[i, :len(ids)] = torch.tensor(ids)
        return batch

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """input_ids: [B, T] BPE token ids → logits: [B, T, vocab].

        Second return is token_byte_lens [B, T] giving each token's UTF-8
        byte length, which bits_per_byte() uses to rescale token CE into
        true per-byte bits.
        """
        x = self.embedding(input_ids)
        logits = self.lm(x)
        token_byte_lens = self.token_byte_len[input_ids]      # [B, T]
        return logits, token_byte_lens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# FixedPatch downstream wrapper
# ===========================================================================

class FixedPatchDownstream(nn.Module):
    """Fixed-size byte patching → mean-pool → Causal Transformer LM.

    Every ``patch_size`` consecutive bytes form one segment (hard, no learning).
    This is the simplest non-trivial compression baseline: it answers whether
    FLUED's dynamic boundaries beat a blind, uniform partition.

    Pipeline:
        bytes [B, T] → byte embedding → fixed-chunk mean pool
        → broadcast back → CausalTransformerLM → byte logits [B, T, 257]
    """

    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        self.byte_embed = nn.Embedding(257, d_model)
        self.lm = CausalTransformerLM(
            vocab_size=257, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            max_seq_len=max_seq_len, dropout=dropout,
        )

    def _make_seg_ids(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create fixed segment IDs: every patch_size bytes → same ID.

        Returns:
            seg_ids:  [B, T] int, -1 for PAD positions
            seg_lens: [B, M] int, bytes in each segment
        """
        B, T = src.shape
        device = src.device
        pad_mask = src == 0
        valid = (~pad_mask).long()

        # Position within valid bytes (cumsum along time)
        pos = valid.cumsum(dim=1) - 1  # 0-indexed within valid bytes
        seg_ids = (pos // self.patch_size).masked_fill(pad_mask, -1)

        M = int(seg_ids.max().item()) + 1 if seg_ids.max().item() >= 0 else 1

        # Count bytes per segment
        seg_lens = torch.zeros(B, M, device=device, dtype=torch.long)
        flat_idx = (seg_ids + torch.arange(B, device=device).unsqueeze(1) * M).clamp(min=0)
        seg_lens.view(-1).scatter_add_(0, flat_idx.reshape(-1), valid.reshape(-1))

        return seg_ids, seg_lens

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """src: [B, T] PAD-offset byte ids → logits [B, T, 257], seg_lens [B, M]."""
        B, T = src.shape
        device = src.device
        pad_mask = src == 0
        valid = (~pad_mask).float()

        seg_ids, seg_lens = self._make_seg_ids(src)
        M = seg_lens.size(1)

        byte_embeds = self.byte_embed(src)      # [B, T, d_model]
        d_emb = byte_embeds.shape[-1]

        # Scatter-mean pool per segment (same pattern as FLUEDDownstream)
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * M
        flat_idx = (seg_ids + batch_offsets).clamp(min=0)  # [B, T]

        seg_sums = torch.zeros(B * M, d_emb, device=device, dtype=byte_embeds.dtype)
        src_flat = (byte_embeds * valid.unsqueeze(-1)).reshape(B * T, d_emb)
        idx_flat = flat_idx.reshape(B * T, 1).expand(-1, d_emb)
        seg_sums.scatter_add_(0, idx_flat, src_flat)

        seg_counts = torch.zeros(B * M, device=device, dtype=byte_embeds.dtype)
        seg_counts.scatter_add_(0, flat_idx.reshape(-1), valid.reshape(-1))

        seg_means = (seg_sums / seg_counts.clamp(min=1).unsqueeze(-1)).view(B, M, d_emb)

        # Broadcast: each byte position gets its segment's pooled embedding
        idx = seg_ids.unsqueeze(-1).expand(B, T, d_emb).clamp(min=0, max=M - 1)
        expanded = seg_means.gather(1, idx)           # [B, T, d_emb]

        logits = self.lm(expanded)                      # [B, T, 257]
        return logits, seg_lens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Byte baseline (no compression)
# ===========================================================================

class ByteDownstream(nn.Module):
    """Raw byte-level causal LM — no segmentation, no compression.

    Each byte is its own "unit". This is the lower bound: any segmentation
    method must beat this to justify its overhead. Conversely, this baseline
    has the highest KV-cache cost per byte.

    Pipeline:
        bytes [B, T] → byte embedding → CausalTransformerLM → byte logits [B, T, 257]
    """

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.byte_embed = nn.Embedding(257, d_model)
        self.lm = CausalTransformerLM(
            vocab_size=257, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            max_seq_len=max_seq_len, dropout=dropout,
        )

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """src: [B, T] → logits [B, T, 257], seg_lens=None.

        seg_lens=None signals bits_per_byte() to use per-position byte counting
        (each non-pad target = 1 byte).
        """
        x = self.byte_embed(src)
        logits = self.lm(x)
        # Return None for seg_lens — bits_per_byte() will use byte-level fallback
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Public tokenizer downstream wrapper (Llama / tiktoken / any HF tokenizer)
# ===========================================================================

class PublicTokenizerDownstream(nn.Module):
    """Public tokenizer → Token Embedding → Causal Transformer LM.

    Serves as an external reference baseline: how well does a widely-used,
    large-scale pre-trained tokenizer perform on our in-domain corpus when
    paired with a from-scratch LM of the same size?

    Supports two backends:
      - ``tiktoken:cl100k_base``  (GPT-4 tokenizer, ~100K vocab, no auth)
      - ``hf:meta-llama/Meta-Llama-3-8B`` (Llama 3 tokenizer, ~128K vocab, needs HF auth)

    Falls back to tiktoken if HF is unavailable.
    """

    def __init__(
        self,
        tokenizer_id: str = "tiktoken:cl100k_base",
        d_model: int = 1024,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tokenizer_id = tokenizer_id
        self.pad_id = 0

        # --- Load tokenizer ---
        if tokenizer_id.startswith("tiktoken:"):
            encoding_name = tokenizer_id.split(":", 1)[1]
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "tiktoken not installed. Install with: pip install tiktoken"
                )
            self._enc = tiktoken.get_encoding(encoding_name)
            self.token_vocab = self._enc.n_vocab
            self._encode = lambda text: self._enc.encode(text)
            self._decode_single = lambda tid: self._enc.decode([tid])
            self._backend = "tiktoken"

        elif tokenizer_id.startswith("hf:"):
            model_name = tokenizer_id.split(":", 1)[1]
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers not installed. Install with: pip install transformers"
                )
            self._hf_tok = AutoTokenizer.from_pretrained(model_name)
            self.token_vocab = self._hf_tok.vocab_size
            self._encode = lambda text: self._hf_tok.encode(
                text, add_special_tokens=False
            )
            self._decode_single = lambda tid: self._hf_tok.decode([tid])
            self._backend = "hf"

        else:
            raise ValueError(
                f"Unknown tokenizer_id format: {tokenizer_id!r}. "
                "Use 'tiktoken:<name>' or 'hf:<model_name>'."
            )

        import logging
        _log = logging.getLogger("e3.public_tok")
        _log.info("PublicTokenizer: backend=%s vocab=%d", self._backend, self.token_vocab)

        # --- Model ---
        self.embedding = nn.Embedding(self.token_vocab, d_model, padding_idx=self.pad_id)
        self.lm = CausalTransformerLM(
            vocab_size=self.token_vocab, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=num_layers,
            max_seq_len=max_seq_len, dropout=dropout,
        )

        # Build token-id → byte-length table.
        bl = torch.zeros(self.token_vocab, dtype=torch.long)
        for tid in range(self.token_vocab):
            try:
                decoded = self._decode_single(tid) or ""
            except Exception:
                decoded = ""
            bl[tid] = len(decoded.encode("utf-8")) if decoded else 0
        self.register_buffer("token_byte_len", bl)

    def encode_text(self, text: str) -> List[int]:
        return self._encode(text)

    def tokenize_batch(self, texts: List[str], max_len: int = 512) -> torch.Tensor:
        """Tokenize texts to padded tensor [B, T]."""
        all_ids = []
        for t in texts:
            ids = self._encode(t)
            if len(ids) > max_len:
                ids = ids[:max_len]
            all_ids.append(ids)
        T = max(len(ids) for ids in all_ids) if all_ids else 1
        batch = torch.full((len(all_ids), T), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(all_ids):
            batch[i, :len(ids)] = torch.tensor(ids)
        return batch

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """input_ids: [B, T] token ids → logits [B, T, vocab], token_byte_lens [B, T]."""
        x = self.embedding(input_ids)
        logits = self.lm(x)
        token_byte_lens = self.token_byte_len[input_ids]
        return logits, token_byte_lens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Metrics
# ===========================================================================

def bits_per_byte(
    logits: torch.Tensor,
    targets: torch.Tensor,
    extra: Optional[torch.Tensor] = None,
    vocab_size: int = 257,
) -> float:
    """Compute bits-per-byte (bpb) — unified across FLUED / BLT / BPE.

    Args:
        logits:  [B, T, V] model output (T = #positions: bytes for FLUED/BLT,
                 tokens for BPE).
        targets: [B, T] ground-truth ids (PAD = 0 → ignored).
        extra:   For FLUED/BLT (byte-level): may be ``seg_lens`` or ``None`` —
                 ignored; each non-pad target counts as 1 byte.
                 For BPE (token-level): ``token_byte_lens`` [B, T] giving each
                 target token's UTF-8 byte length, used as the denominator.
        vocab_size: vocabulary size (currently unused; kept for API stability).

    Returns: bits per byte = (sum_NLL_nats / total_bytes) / ln(2).

    Notes:
        - For BPE: bpb = (Σ CE_token) / (Σ byte_len_target) / ln 2.
          This is the standard "bits per byte" used in PG-19 / The Pile evals.
        - For byte-level: bpb = (Σ CE_byte) / (#non_pad_targets) / ln 2.
    """
    del vocab_size  # kept for backward compat
    mask = targets != 0
    ce_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0,
        reduction="sum",
    )

    # Token-level (BPE) → use real byte counts from extra
    if extra is not None and extra.dim() == 2 and extra.shape == targets.shape:
        # extra is token_byte_lens [B, T]; only count non-pad targets
        byte_counts = extra.masked_fill(~mask, 0).sum().item()
        if byte_counts == 0:
            return 0.0
        return (ce_sum / byte_counts).item() / math.log(2)

    # Byte-level fallback
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return (ce_sum / total).item() / math.log(2)
