"""
FLUED v0.4 Autoencoder — Dynamic Semantic Compiler (DSC).

Architecture
------------
  src byte ids (PAD=0, byte b → id b+1, vocab_size=257)
    → Embedding + PositionalEncoding
    → TiedTransformerBlock × num_layers          (DSC forward pass)
    → BoundaryScorer  (delta-based hidden-state differences)
    → _compile_semantic_units:
        soft path  → assignment matrix A → Z_soft = Aᵀ H → expanded = A Z_soft
        hard path  → threshold → spans → mean-pool Z_hard   (metrics only)
    → TiedTransformerBlock × num_layers reversed (DSC⁻¹ inverse pass)
    → F.linear(inv_hidden, embedding.weight)      (tied output projection)
    → logits [B, T, 257]

Key design decisions
--------------------
* PAD-offset byte encoding: PAD=0, byte 0→1, …, byte 255→256.
  Avoids byte-0 / PAD-id collision from earlier versions.
* Tied weights: encoder and inverse decoder share ALL parameters.
  Inverse is a first-order approximation: x ← x − F(x).
* Soft segmentation (training): differentiable assignment matrix A ensures
  boundary_head receives gradients through both reconstruction loss and
  compression loss.  Hard segmentation is used only for metrics.
* Compression loss: soft boundary density (differentiable) not hard m/n.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID: int = 0
VOCAB_SIZE: int = 257  # PAD + 256 byte values


def byte_to_id(b: int) -> int:
    """Map a raw byte value (0–255) to a token id (1–256)."""
    return b + 1


def id_to_byte(token_id: int) -> int:
    """Map a token id (1–256) to a raw byte value (0–255)."""
    return token_id - 1


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


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
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.ff_drop = nn.Dropout(dropout)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(self.ff_drop(F.gelu(self.ff1(x))))

    def forward_block(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DSC forward: encode step."""
        n1 = self.norm1(x)
        h = x + self.attn(n1, n1, n1, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
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
        h = h - self.attn(n1, n1, n1, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
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
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        boundary_threshold: float = 0.5,
        target_compression: float = 0.3,
        compression_weight: float = 0.1,
        # Legacy keyword arguments — accepted but ignored for backward compat
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
        shallow_layers: Optional[int] = None,
        gate_entropy_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.boundary_threshold = boundary_threshold
        self.target_compression = target_compression
        self.compression_weight = compression_weight

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
                    dim_feedforward=dim_feedforward,
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

        Args:
            boundary_probs: [B, T] sigmoid activations in (0, 1)

        Returns:
            soft_A: [B, T, T] row-stochastic assignment matrix
        """
        B, T = boundary_probs.shape
        eps = 1e-6

        # log(1 − p) — clamped for safety
        log_no_boundary = torch.log1p(
            -(boundary_probs.clamp(min=eps, max=1.0 - eps))
        )  # [B, T]

        # cumlog[b, i] = Σ_{k=0}^{i} log(1-p[k])
        cumlog = torch.cumsum(log_no_boundary, dim=1)  # [B, T]

        # log_stay[b, i, j] = cumlog[b,i] − cumlog[b,j]
        #                    = log( ∏_{k=j+1}^{i} (1-p[k]) )
        cumlog_i = cumlog.unsqueeze(2)   # [B, T, 1]
        cumlog_j = cumlog.unsqueeze(1)   # [B, 1, T]
        log_stay = cumlog_i - cumlog_j   # [B, T, T]

        # Mask upper triangle: j > i is impossible
        upper = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=boundary_probs.device),
            diagonal=1,
        )
        log_stay = log_stay.masked_fill(upper.unsqueeze(0), -1e9)

        # Add log(p[j]) broadcast over query-position dimension i
        log_bp = torch.log(boundary_probs.clamp(min=eps))  # [B, T]
        log_w = log_stay + log_bp.unsqueeze(1)              # [B, T, T]

        # Row-normalise → soft assignment probabilities
        return torch.softmax(log_w, dim=2)  # [B, T, T]

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
    ) -> Tuple[torch.Tensor, Dict]:
        """Produce both soft (training) and hard (metrics) segmentations.

        Soft path:
            A  = _soft_assignment(boundary_probs)  [B, T, T]
            Z  = Aᵀ H                              [B, T, d]  (pooling)
            Ẑ  = A Z                               [B, T, d]  (expanding)

        Hard path (no grad):
            threshold → spans → mean-pool → Z_hard

        Compression loss (differentiable):
            soft_m_over_n = mean(boundary_probs[valid]) ≈ m/n
            loss = compression_weight × (soft_m_over_n − target_compression)²

        Returns:
            expanded_soft: [B, T, d_model]  differentiable latent for decode
            metrics:       dict
        """
        B, T, d = hidden.shape
        boundary_probs = torch.sigmoid(boundary_scores)  # [B, T]
        boundary_probs = boundary_probs.clone()
        boundary_probs[:, 0] = 1.0  # first token is always a boundary (soft/hard aligned)

        # ---- soft path ----
        soft_A = self._soft_assignment(boundary_probs)          # [B, T, T]
        z_soft = torch.bmm(soft_A.transpose(1, 2), hidden)      # [B, T, d]
        expanded_soft = torch.bmm(soft_A, z_soft)               # [B, T, d]

        # ---- differentiable compression metric ----
        if src_key_padding_mask is not None:
            valid_mask = ~src_key_padding_mask                   # [B, T]
            valid_n = valid_mask.float().sum(dim=1)              # [B]
            soft_m_over_n = (
                (boundary_probs * valid_mask.float()).sum(dim=1)
                / valid_n.clamp(min=1)
            ).mean()
        else:
            soft_m_over_n = boundary_probs.mean()

        compression_loss = self.compression_weight * (
            soft_m_over_n - self.target_compression
        ) ** 2

        # ---- hard path (no grad) ----
        spans, span_ids = self._hard_segmentation(boundary_probs)

        hard_m = torch.tensor(
            [len(s) for s in spans], dtype=torch.float, device=hidden.device
        )
        if src_key_padding_mask is not None:
            valid_n_hard = (~src_key_padding_mask).float().sum(dim=1)
        else:
            valid_n_hard = torch.full(
                (B,), T, dtype=torch.float, device=hidden.device
            )
        hard_m_over_n = (hard_m / valid_n_hard.clamp(min=1)).mean()

        # Mean-pool hard spans → z_hard (interpretability / logging)
        max_units = max(len(s) for s in spans)
        z_hard = torch.zeros(B, max_units, d, device=hidden.device)
        for b in range(B):
            for j, (start, end) in enumerate(spans[b]):
                z_hard[b, j] = hidden[b, start:end].mean(dim=0)

        metrics: Dict = {
            "m_over_n": hard_m_over_n.item(),
            "hard_m_over_n": hard_m_over_n.item(),
            "soft_m_over_n": soft_m_over_n,
            "num_units": hard_m.mean().item(),
            "num_units_tensor": hard_m.mean(),
            "spans": spans,
            "span_ids": span_ids,
            "boundary_probs": boundary_probs,
            "compression_loss": compression_loss,
            "z": z_hard,
        }
        return expanded_soft, metrics

    # ------------------------------------------------------------------
    # Encode / decode interface
    # ------------------------------------------------------------------

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Encode byte ids to a differentiable soft-expanded latent.

        Args:
            src:                  [B, T] byte token ids (PAD=0, byte b → b+1)
            src_key_padding_mask: [B, T] bool (True = padding)

        Returns:
            expanded_soft: [B, T, d_model]
            metrics:       dict with boundary_probs, spans, m/n, etc.
        """
        h = self.pos_enc(self.embedding(src))  # [B, T, d]
        for block in self.blocks:
            h = block.forward_block(h, key_padding_mask=src_key_padding_mask)

        # Boundary scoring: delta between adjacent hidden states
        delta = torch.zeros_like(h)
        delta[:, 1:] = h[:, 1:] - h[:, :-1]
        boundary_scores = self.boundary_head(delta).squeeze(-1)  # [B, T]

        return self._compile_semantic_units(h, boundary_scores, src_key_padding_mask)

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
    ) -> Tuple[torch.Tensor, Dict]:
        """Full autoencoder forward.

        Args:
            src: [B, T] byte token ids (PAD=0, byte b → b+1)
            tgt: unused (kept for API compat); reconstruction always targets src

        Returns:
            logits:  [B, T, vocab_size]
            metrics: dict — includes differentiable compression_loss
        """
        src_key_padding_mask = src == PAD_ID
        expanded_soft, metrics = self.encode(src, src_key_padding_mask)
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


