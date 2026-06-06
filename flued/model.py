"""
FLUED v0.4 Autoencoder — Dynamic Semantic Compiler (DSC).

Architecture
------------
  src byte ids (PAD=0, byte b → id b+1, MASK=257, vocab_size=258)
    → Embedding + PositionalEncoding
    → TiedTransformerBlock × num_layers          (DSC forward pass; SDPA + SwiGLU)
    → BoundaryScorer  (delta-based hidden-state differences)
    → _compile_semantic_units:
        soft path  → banded assignment matrix A → Z_soft = Aᵀ H → expanded = A Z_soft
        hard path  → threshold → spans → mean-pool Z_hard   (metrics only)
    → TiedTransformerBlock × num_layers reversed (DSC⁻¹ inverse pass)
    → F.linear(inv_hidden, embedding.weight)      (tied output projection)
    → logits [B, T, 258]

Key design decisions
--------------------
* PAD-offset byte encoding: PAD=0, byte 0→1, …, byte 255→256, MASK=257.
  Avoids byte-0 / PAD-id collision from earlier versions.
* Tied weights: encoder and inverse decoder share ALL parameters.
  Inverse is a first-order approximation: x ← x − F(x).
* Soft segmentation (training): differentiable assignment matrix A ensures
  boundary_head receives gradients through both reconstruction loss and
  compression loss.  Hard segmentation is used only for metrics.
* Compression loss: soft boundary density (differentiable) not hard m/n.

.. warning::
   ``assignment_window`` currently only SEMANTICALLY restricts segment
   affiliation windows; the internal ``_soft_assignment()`` still constructs
   a full ``[B, T, T]`` matrix and masks out-of-window positions.  This is
   safe for T ≤ 512 but remains O(T²) in memory.  For T ≥ 2048, the
   soft-assignment step MUST be refactored into a truly banded/streaming
   computation (e.g. ``torch.as_strided`` + custom backward).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID: int = 0
BYTE_OFFSET: int = 1
RAW_BYTE_VOCAB: int = 256
MASK_ID: int = BYTE_OFFSET + RAW_BYTE_VOCAB
VOCAB_SIZE: int = MASK_ID + 1  # PAD + 256 byte values + MASK


def byte_to_id(b: int) -> int:
    """Map a raw byte value (0–255) to a token id (1–256)."""
    return b + BYTE_OFFSET


def id_to_byte(token_id: int) -> int:
    """Map a token id (1–256) to a raw byte value (0–255)."""
    if token_id < BYTE_OFFSET or token_id >= MASK_ID:
        raise ValueError(f"token id {token_id} is not a raw byte id")
    return token_id - BYTE_OFFSET


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model)
        pos = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.pe = self._build_pe(x.size(1)).to(device=x.device, dtype=x.dtype)
        return self.dropout(x + self.pe[:, : x.size(1)])


class SDPAAttention(nn.Module):
    """Exact self-attention using PyTorch scaled_dot_product_attention kernels."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(bsz, seq_len, d_model)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# FLUED v0.4 sub-modules
# ---------------------------------------------------------------------------


class TiedTransformerBlock(nn.Module):
    """Single Transformer block with tied weights used in both encode and inverse-decode.

    Forward pass (DSC encoding):
        h1 = x + MHA(LN(x))
        h2 = h1 + FFN(LN(h1))

    Inverse pass (DSC⁻¹, first-order tied-weight approximation):
        h1 = x  − FFN(LN(x))
        h2 = h1 − MHA(LN(h1))

    The inverse is NOT an exact mathematical inverse.  It is a reasonable
    engineering approximation that works well for reconstruction pretraining.
    Always describe it as "tied-weight inverse approximation".
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = SDPAAttention(d_model, nhead, dropout=dropout)
        self.ff_gate = nn.Linear(d_model, dim_feedforward)
        self.ff_value = nn.Linear(d_model, dim_feedforward)
        self.ff_out = nn.Linear(dim_feedforward, d_model)
        self.ff_drop = nn.Dropout(dropout)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_out(self.ff_drop(F.silu(self.ff_gate(x)) * self.ff_value(x)))

    def forward_block(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DSC forward: encode step."""
        n1 = self.norm1(x)
        h = x + self.attn(n1, key_padding_mask=key_padding_mask)
        h = h + self._ffn(self.norm2(h))
        return h

    def inverse_block(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DSC⁻¹ inverse (approximate): decode step."""
        h = x - self._ffn(self.norm2(x))
        n1 = self.norm1(h)
        h = h - self.attn(n1, key_padding_mask=key_padding_mask)
        return h


# ---------------------------------------------------------------------------
# Main model — FLUEDAutoencoder v0.4
# ---------------------------------------------------------------------------


class FLUEDAutoencoder(nn.Module):
    """FLUED v0.4 Autoencoder with differentiable soft segmentation.

    Training uses the soft path (assignment matrix A) so that boundary_head
    receives gradients through both reconstruction loss and compression loss.
    Inference / reporting uses hard segmentation (threshold on boundary_probs).

    The forward() method returns (logits, metrics_dict) where metrics_dict
    contains compression_loss (a differentiable scalar) alongside non-grad
    diagnostics such as spans, m/n, z_hard.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        swiglu_hidden: Optional[int] = None,
        num_layers: int = 4,
        max_seq_len: int = 512,
        assignment_window: int = 128,
        dropout: float = 0.0,
        boundary_threshold: float = 0.5,
        boundary_temperature: float = 1.0,
        target_compression: float = 0.3,
        compression_weight: float = 0.1,
        min_boundary_units: float = 1.0,
        # Type-conditional boundary regularization (default 0 = disabled)
        lambda_var: float = 0.0,
        lambda_entropy: float = 0.0,
        lambda_utf8: float = 0.0,
        # CJK-specific boundary target (legacy MSE penalty; kept for back-compat)
        lambda_cjk: float = 0.0,
        cjk_target: float = 0.16,
        # Type-conditional BCE prior — per-type target boundary probability.
        # Implements README §"lambda_type × MSE(p, type_target) — type-conditional prior",
        # but uses BCE instead of MSE (non-vanishing gradient near saturation).
        # Active iff lambda_type > 0. type_targets overrides defaults; any subset of
        # the keys below is accepted (missing keys → skipped).
        lambda_type: float = 0.0,
        type_targets: Optional[Dict[str, float]] = None,
        # Legacy keyword arguments — accepted but ignored for backward compat
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
        shallow_layers: Optional[int] = None,
        gate_entropy_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.swiglu_hidden = swiglu_hidden if swiglu_hidden is not None else max(1, dim_feedforward * 3 // 4)
        self.assignment_window = int(assignment_window)
        self.boundary_threshold = boundary_threshold
        self.boundary_temperature = max(boundary_temperature, 1e-3)
        self.target_compression = target_compression
        self.compression_weight = compression_weight
        self.min_boundary_units = min_boundary_units
        self.lambda_var = lambda_var
        self.lambda_entropy = lambda_entropy
        self.lambda_utf8 = lambda_utf8
        self.lambda_cjk = lambda_cjk
        self.cjk_target = cjk_target
        self.lambda_type = lambda_type
        # Default per-type boundary targets (v5e).
        # is_cont REMOVED — utf8 continuation is already handled independently
        # by lambda_utf8.  Keeping it in type_loss wasted gradient budget on
        # the largest byte class (40%) at the expense of op/digit.
        _default_targets = {
            "is_cjk_lead": 0.15,
            "is_alpha":    0.40,
            "is_digit":    0.60,
            "is_operator": 0.80,
        }
        if type_targets:
            _default_targets.update(type_targets)
        self.type_targets: Dict[str, float] = _default_targets

        # num_encoder_layers alias for num_layers (migration support)
        if num_encoder_layers is not None and num_layers == 4:
            num_layers = num_encoder_layers
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(
            d_model, max_len=max_seq_len * 2, dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [
                TiedTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=self.swiglu_hidden,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Boundary scorer: delta of adjacent hidden states → scalar logit
        self.boundary_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------
    # Byte type classification + type-conditional boundary regularization
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_bytes(src: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify each byte position for type-conditional boundary priors.

        Args:
            src: [B, T] token ids (PAD=0, byte b → id b+1)

        Returns:
            Dict of bool masks [B, T], PAD positions already excluded.
        """
        raw = src.long() - 1  # raw byte value; PAD → -1, MASK → 256
        valid = (src != PAD_ID) & (src != MASK_ID)

        is_cont     = valid & (raw >= 0x80) & (raw <= 0xBF)         # UTF-8 continuation
        is_alpha    = valid & (
            ((raw >= 0x61) & (raw <= 0x7A)) |  # a-z
            ((raw >= 0x41) & (raw <= 0x5A))    # A-Z
        )
        is_digit    = valid & (raw >= 0x30) & (raw <= 0x39)
        _op_vals    = torch.tensor(
            [0x21, 0x23, 0x25, 0x26, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D,
             0x2E, 0x2F, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x40, 0x5B, 0x5C,
             0x5D, 0x5E, 0x5F, 0x7B, 0x7C, 0x7D, 0x7E],
            device=src.device,
        )
        is_operator = valid & torch.isin(raw, _op_vals)
        is_cjk_lead = valid & (raw >= 0xE4) & (raw <= 0xE9)  # most CJK lead bytes

        return {
            "is_cont":     is_cont,
            "is_alpha":    is_alpha,
            "is_digit":    is_digit,
            "is_operator": is_operator,
            "is_cjk_lead": is_cjk_lead,
        }

    def _boundary_loss(
        self,
        boundary_probs: torch.Tensor,   # [B, T]
        valid_mask: torch.Tensor,        # [B, T] bool
        byte_types: Optional[Dict],
    ) -> torch.Tensor:
        """Multi-term boundary regularization.

        mean_loss    — global compression budget  (compression_weight)
        var_loss     — encourage prob spread       (lambda_var, negated)
        entropy_loss — polarize toward 0/1        (lambda_entropy)
        utf8_loss    — suppress continuation cuts  (lambda_utf8)
        cjk_loss     — pull CJK-lead bp toward cjk_target via BCE (lambda_cjk)
        type_loss    — per-type BCE prior on boundary probs       (lambda_type)
        """
        eps = 1e-4
        p   = boundary_probs[valid_mask]        # [N_valid]
        # Clamp & entropy compute in FP32: FP16 1.0-1e-4 rounds to 1.0 → log(0)=NaN
        p_f32 = p.float()
        p_c = p_f32.clamp(eps, 1.0 - eps)

        valid_f = valid_mask.float()
        valid_n = valid_f.sum(dim=1).clamp(min=1)
        density = (boundary_probs.float() * valid_f).sum(dim=1) / valid_n
        min_density = torch.full_like(valid_n, float(self.min_boundary_units)) / valid_n
        target = torch.maximum(torch.full_like(density, self.target_compression), min_density).clamp(max=1.0)
        mean_loss    = self.compression_weight * (density - target).pow(2).mean()
        var_loss     = -self.lambda_var * p_f32.var(unbiased=False)
        entropy_loss = self.lambda_entropy * (
            -(p_c * p_c.log() + (1.0 - p_c) * (1.0 - p_c).log())
        ).mean()

        if self.lambda_utf8 > 0 and byte_types is not None:
            cont_mask = byte_types.get("is_cont")
            utf8_loss = (
                self.lambda_utf8 * boundary_probs[cont_mask].mean()
                if (cont_mask is not None and cont_mask.any())
                else mean_loss.new_zeros(())
            )
        else:
            utf8_loss = mean_loss.new_zeros(())

        # ---- CJK BCE prior (replaces former MSE) ----
        # MSE has gradient 2*(p − target) which vanishes at saturation and is
        # too weak to escape the 0.33 plateau observed in retrain v2/v3.
        # BCE: dL/dp = (p − target) / (p (1 − p))  →  large gradient when far away.
        if self.lambda_cjk > 0 and byte_types is not None:
            cjk_mask = byte_types.get("is_cjk_lead")
            if cjk_mask is not None and cjk_mask.any():
                cjk_p = boundary_probs[cjk_mask].float().clamp(eps, 1.0 - eps)
                t = float(self.cjk_target)
                cjk_loss = self.lambda_cjk * (
                    -(t * cjk_p.log() + (1.0 - t) * (1.0 - cjk_p).log())
                ).mean()
            else:
                cjk_loss = mean_loss.new_zeros(())
        else:
            cjk_loss = mean_loss.new_zeros(())

        # ---- Type-conditional BCE prior ----
        # Implements README "lambda_type × per-type prior". Uses BCE (cross-entropy
        # between current p and a constant target) for non-vanishing gradients.
        # NOTE (v5d): sum across types instead of mean — eliminates the ÷5
        # dilution that made per-type gradients 5× too weak at late-stage lr.
        if self.lambda_type > 0 and byte_types is not None:
            type_terms = []
            for mask_name, target in self.type_targets.items():
                m = byte_types.get(mask_name)
                if m is None or not m.any():
                    continue
                p_t = boundary_probs[m].float().clamp(eps, 1.0 - eps)
                tgt = float(target)
                term = -(tgt * p_t.log() + (1.0 - tgt) * (1.0 - p_t).log()).mean()
                type_terms.append(term)
            if type_terms:
                type_loss = self.lambda_type * torch.stack(type_terms).sum()
            else:
                type_loss = mean_loss.new_zeros(())
        else:
            type_loss = mean_loss.new_zeros(())

        return mean_loss + var_loss + entropy_loss + utf8_loss + cjk_loss + type_loss

    # ------------------------------------------------------------------
    # Soft assignment (differentiable, for training)
    # ------------------------------------------------------------------

    def _soft_assignment(self, boundary_probs: torch.Tensor) -> torch.Tensor:
        """Compute soft assignment matrix A[b, i, j].

        A[b, i, j] = P(position i belongs to segment started at j)
                   = p[b,j] * ∏_{k=j+1}^{i} (1 − p[b,k])   for j ≤ i
                   = 0                                         for j >  i

        Computed in log-space for numerical stability, then row-normalised
        via softmax to form a proper probability distribution over segment
        starts for each query position.

        The entire computation is performed in float32 regardless of the
        autocast context (bf16/fp16) to avoid NaN from log1p / cumsum /
        softmax over T=512 positions in low precision.  The result is cast
        back to the caller's dtype before return.

        Args:
            boundary_probs: [B, T] sigmoid activations in (0, 1)

        Returns:
            soft_A: [B, T, T] row-stochastic assignment matrix
        """
        orig_dtype = boundary_probs.dtype
        # Upcast to float32 for the entire log-space computation
        p = boundary_probs.float()
        B, T = p.shape

        # eps=1e-4: safe for fp32; 1e-6 is too tight for bf16-origin inputs
        eps = 1e-4
        p = p.clamp(min=eps, max=1.0 - eps)

        # log(1 − p)
        log_no_boundary = torch.log1p(-p)  # [B, T]

        # cumlog[b, i] = Σ_{k=0}^{i} log(1-p[k])
        cumlog = torch.cumsum(log_no_boundary, dim=1)  # [B, T]

        # log_stay[b, i, j] = cumlog[b,i] − cumlog[b,j]
        #                    = log( ∏_{k=j+1}^{i} (1-p[k]) )
        cumlog_i = cumlog.unsqueeze(2)   # [B, T, 1]
        cumlog_j = cumlog.unsqueeze(1)   # [B, 1, T]
        log_stay = cumlog_i - cumlog_j   # [B, T, T]

        # Mask upper triangle: j > i is impossible
        # Use -1e4 (not -1e9) to stay well within fp32 range and avoid
        # softmax instability from extreme values.
        idx = torch.arange(T, device=p.device)
        impossible = idx.view(1, -1) > idx.view(-1, 1)
        if self.assignment_window > 0:
            impossible = impossible | ((idx.view(-1, 1) - idx.view(1, -1)) > self.assignment_window)
        log_stay = log_stay.masked_fill(impossible.unsqueeze(0), -1e4)

        # Add log(p[j]) broadcast over query-position dimension i
        log_bp = torch.log(p)                       # [B, T], already clamped
        log_w = log_stay + log_bp.unsqueeze(1)       # [B, T, T]

        # Row-normalise → soft assignment probabilities, then downcast
        soft_A = torch.softmax(log_w, dim=2)         # [B, T, T] float32
        return soft_A.to(orig_dtype)

    # ------------------------------------------------------------------
    # Hard segmentation (non-differentiable, for metrics / inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _hard_segmentation(
        self,
        boundary_probs: torch.Tensor,
    ) -> Tuple[List[List[Tuple[int, int]]], torch.Tensor]:
        """Convert boundary probabilities to hard spans.

        Returns:
            spans:    list[list[(start, end)]] exclusive end, covering [0, T)
            span_ids: [B, T] long tensor — span index per position
        """
        B, T = boundary_probs.shape
        hard = boundary_probs > self.boundary_threshold  # [B, T]
        hard[:, 0] = True  # first position is always a boundary

        spans: List[List[Tuple[int, int]]] = []
        span_ids = torch.zeros(B, T, dtype=torch.long, device=boundary_probs.device)
        for b in range(B):
            sample_spans: List[Tuple[int, int]] = []
            sid, start = 0, 0
            for t in range(T):
                if hard[b, t] and t > 0:
                    sample_spans.append((start, t))
                    sid += 1
                    start = t
                span_ids[b, t] = sid
            sample_spans.append((start, T))
            spans.append(sample_spans)
        return spans, span_ids

    # ------------------------------------------------------------------
    # Compile semantic units (soft + hard)
    # ------------------------------------------------------------------

    def _compile_semantic_units(
        self,
        hidden: torch.Tensor,
        boundary_scores: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        byte_types: Optional[Dict] = None,
        skip_hard: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Produce both soft (training) and hard (metrics) segmentations.

        Soft path (always active):
            A  = _soft_assignment(boundary_probs)  [B, T, T]
            Z  = Aᵀ H                              [B, T, d]  (pooling)
            Ẑ  = A Z                               [B, T, d]  (expanding)

        Hard path (skip_hard=False, for metrics only):
            threshold → spans → mean-pool → Z_hard

        When skip_hard=True: skips all CPU for-loops (hard seg, z_hard pool,
        sweep). Keeps type_bp monitoring.
        """
        B, T, d = hidden.shape
        boundary_probs = torch.sigmoid(boundary_scores / self.boundary_temperature)  # [B, T]
        # Avoid in-place on tensor with grad: replace col 0 via mask
        mask0 = torch.zeros_like(boundary_probs)
        mask0[:, 0] = 1.0
        boundary_probs = boundary_probs * (1.0 - mask0) + mask0  # col0=1.0, rest unchanged

        # ---- soft path (always) ----
        soft_A = self._soft_assignment(boundary_probs)          # [B, T, T]
        z_soft = torch.bmm(soft_A.transpose(1, 2), hidden)      # [B, T, d]
        expanded_soft = torch.bmm(soft_A, z_soft)               # [B, T, d]

        # ---- valid mask + soft m/n ----
        if src_key_padding_mask is not None:
            valid_mask = ~src_key_padding_mask
        else:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=hidden.device)
        valid_n = valid_mask.float().sum(dim=1)

        # Soft m/n via cumsum(bp) — no threshold
        soft_m_over_n = (
            (boundary_probs * valid_mask.float()).sum(dim=1)
            / valid_n.clamp(min=1)
        ).mean()

        # ---- multi-term boundary regularization loss ----
        compression_loss = self._boundary_loss(boundary_probs, valid_mask, byte_types)

        # ---- per-type boundary prob means (keep even in skip_hard) ----
        type_means: Dict[str, float] = {}
        if byte_types is not None:
            with torch.no_grad():
                bp_d = boundary_probs.detach()
                for key, label in (
                    ("is_cont",     "utf8_cont"),
                    ("is_alpha",    "ascii"),
                    ("is_cjk_lead", "cjk"),
                    ("is_operator", "op"),
                    ("is_digit",    "digit"),
                ):
                    mask = byte_types.get(key)
                    type_means[f"{label}_bp_mean"] = (
                        bp_d[mask].mean().item()
                        if (mask is not None and mask.any())
                        else float("nan")
                    )

        # ---- hard path (skip if not needed) ----
        if not skip_hard:
            spans, span_ids = self._hard_segmentation(boundary_probs)
            hard_m = torch.tensor(
                [len(s) for s in spans], dtype=torch.float, device=hidden.device
            )
            hard_m_over_n = (hard_m / valid_n.clamp(min=1)).mean()
            num_units = hard_m.mean().item()

            max_units = max(len(s) for s in spans)
            z_hard = torch.zeros(B, max_units, d, device=hidden.device)
            for b in range(B):
                for j, (start, end) in enumerate(spans[b]):
                    z_hard[b, j] = hidden[b, start:end].mean(dim=0)

            with torch.no_grad():
                bp_det = boundary_probs.detach()
                sweep: Dict[float, float] = {}
                for thr in (0.50, 0.55, 0.60, 0.65):
                    h_mask = (bp_det > thr).float()
                    h_mask[:, 0] = 1.0
                    m_thr = h_mask.sum(dim=1)
                    sweep[thr] = (m_thr / valid_n.clamp(min=1)).mean().item()
        else:
            # Soft-only: use cumsum(bp) for segment count
            soft_m = boundary_probs.sum(dim=1).mean().item()
            hard_m_over_n = soft_m_over_n.item() if isinstance(soft_m_over_n, torch.Tensor) else float(soft_m_over_n)
            num_units = soft_m
            z_hard = z_soft  # placeholder
            spans = []
            sweep = {}

        metrics: Dict = {
            "m_over_n": hard_m_over_n.item() if not skip_hard else hard_m_over_n,
            "hard_m_over_n": hard_m_over_n.item() if not skip_hard else hard_m_over_n,
            "soft_m_over_n": soft_m_over_n,
            "num_units": num_units,
            "num_units_tensor": torch.tensor(num_units, device=hidden.device),
            "spans": spans,
            "span_ids": torch.zeros(B, T, dtype=torch.long, device=hidden.device) if skip_hard else span_ids,
            "boundary_probs": boundary_probs,
            "compression_loss": compression_loss,
            "expanded": expanded_soft,
            "z": z_hard,
            "hard_mn_sweep": sweep,
            **type_means,
        }
        return expanded_soft, metrics

    # ------------------------------------------------------------------
    # Encode / decode interface
    # ------------------------------------------------------------------

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        skip_hard: bool = True,
        boundary_src: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Encode byte ids to a differentiable soft-expanded latent.

        Args:
            src:                  [B, T] byte token ids
            src_key_padding_mask: [B, T] bool (True = padding)
            skip_hard:            skip hard segmentation for-loops (faster, pure soft)
        """
        if boundary_src is None:
            boundary_src = src
        h = self.pos_enc(self.embedding(src))
        for block in self.blocks:
            h = block.forward_block(h, key_padding_mask=src_key_padding_mask)

        delta = torch.zeros_like(h)
        delta[:, 1:] = h[:, 1:] - h[:, :-1]
        boundary_scores = self.boundary_head(delta).squeeze(-1)

        byte_types = self._classify_bytes(boundary_src)
        return self._compile_semantic_units(h, boundary_scores, src_key_padding_mask, byte_types, skip_hard=skip_hard)

    def decode(
        self,
        z_expanded: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inverse decode (DSC⁻¹): soft-expanded latent → byte logits.

        Uses tied weights in reverse order with approximate inverse blocks.

        Returns:
            logits: [B, T, vocab_size]
        """
        x = z_expanded
        for block in reversed(self.blocks):
            x = block.inverse_block(x, key_padding_mask=src_key_padding_mask)
        return F.linear(x, self.embedding.weight)  # tied output projection

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        skip_hard: bool = True,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        boundary_src: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Full autoencoder forward.

        Args:
            src: [B, T] byte token ids (PAD=0, byte b → b+1)
            tgt: unused (kept for API compat)
            skip_hard: skip hard segmentation for-loops (pure soft, faster)

        Returns:
            logits:  [B, T, vocab_size]
            metrics: dict
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = src == PAD_ID
        if boundary_src is None:
            boundary_src = tgt if tgt is not None else src
        expanded_soft, metrics = self.encode(src, src_key_padding_mask, skip_hard=skip_hard, boundary_src=boundary_src)
        logits = self.decode(expanded_soft, src_key_padding_mask)
        return logits, metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Removed in v0.4 (kept as empty stubs to avoid import errors in old scripts).
# These classes are NOT used by FLUEDAutoencoder v0.4.


class ShallowEncoder(nn.Module):
    """Removed in v0.4."""
    pass


class SGLGatingModule(nn.Module):
    """Removed in v0.4."""
    pass


class DynamicLatentEncoder(nn.Module):
    """Removed in v0.4."""
    pass


