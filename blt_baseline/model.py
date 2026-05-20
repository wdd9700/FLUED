"""
BLT (Byte Latent Transformer) style Autoencoder baseline.

Reference: Pagnoni et al. (2024) "Byte Latent Transformer: Patches Scale
           Better Than Tokens"  (https://arxiv.org/abs/2412.09871)

Architecture
------------
  1. LocalEncoder  — shallow Transformer on raw byte embeddings; produces
                     per-byte hidden representations.
  2. Patcher       — groups bytes into fixed-size patches (the full paper
                     uses entropy-based dynamic patching; we implement a
                     fixed-size stub and note where to swap in the
                     entropy-based variant).
  3. GlobalTransformer — large Transformer backbone operating on the
                         shorter patch sequence (sequence length ÷ patch_size).
  4. LocalDecoder  — expands patch representations back to byte-level
                     and projects to output logits.

The key efficiency claim: the global Transformer sees patch_size× fewer
positions, so O(n²) attention cost is reduced by patch_size².
For Stage A, all components are trained end-to-end for reconstruction.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding (shared helper)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# BLT sub-modules
# ---------------------------------------------------------------------------


class LocalEncoder(nn.Module):
    """Shallow byte-level Transformer encoder.

    Produces a per-byte hidden representation that is subsequently grouped
    into patches by the Patcher module.

    In the full BLT paper the local encoder also uses a cross-attention
    mechanism to inject patch-boundary information; here we keep it as a
    standard encoder for simplicity.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.byte_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len * 2, dropout=dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, T] byte ids  →  [B, T, d_model]"""
        h = self.pos_enc(self.byte_embed(x))
        return self.encoder(h, src_key_padding_mask=padding_mask)


class Patcher(nn.Module):
    """Fixed-size patch aggregator.

    Groups consecutive byte representations into non-overlapping patches of
    exactly *patch_size* bytes by concatenating their hidden vectors and
    projecting down to d_model.

    In a production BLT implementation this module would be replaced by an
    entropy-based patcher that places boundaries where the next-byte
    prediction entropy of a small language model exceeds a threshold θ.
    That variant produces variable-length patches — to swap it in, replace
    the forward() method here with one that returns a padded patch tensor
    and the corresponding patch lengths.
    """

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        # Learned linear aggregation: patch_size * d_model → d_model
        self.patch_agg = nn.Linear(patch_size * d_model, d_model)

    def forward(self, local_repr: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Aggregate local representations into fixed-size patches.

        Args:
            local_repr: [B, T, d_model]

        Returns:
            patches:      [B, num_patches, d_model]
            original_len: T (needed by decoder to trim back to original length)
        """
        B, T, d = local_repr.shape
        original_len = T

        # Pad T to the next multiple of patch_size using zero-padding on the
        # time axis (F.pad pads with 0.0 by default).  Zero-padded positions
        # contribute 0-vector representations to the patch aggregation linear
        # layer; since patches are learned embeddings (not raw bytes), the
        # effect on reconstruction quality is minimal and the decoder trims
        # the output back to original_len before computing logits.
        remainder = T % self.patch_size
        if remainder != 0:
            pad = self.patch_size - remainder
            local_repr = F.pad(local_repr, (0, 0, 0, pad))

        T_padded = local_repr.size(1)
        num_patches = T_padded // self.patch_size

        # Reshape → [B, num_patches, patch_size * d_model]
        patches_in = local_repr.view(B, num_patches, self.patch_size * d)
        patches = self.patch_agg(patches_in)  # [B, num_patches, d_model]
        return patches, original_len


class LocalDecoder(nn.Module):
    """Byte-level decoder that expands patch representations back to bytes.

    Each patch vector is expanded to patch_size byte vectors by a learned
    linear layer, then refined by a shallow Transformer, and finally
    projected to output logits over the byte vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        patch_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Expand one patch vector to patch_size byte vectors
        self.patch_expand = nn.Linear(d_model, patch_size * d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        # A standard encoder used auto-regressively (no memory cross-attention
        # needed here since patches already carry the encoded information)
        self.refine = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, patches: torch.Tensor, target_len: int) -> torch.Tensor:
        """Expand patches back to byte-level logits.

        Args:
            patches:    [B, num_patches, d_model]
            target_len: original sequence length T to trim the output

        Returns:
            logits: [B, T, vocab_size]
        """
        B, num_patches, _ = patches.shape

        # Expand: [B, num_patches, patch_size * d_model] → [B, T_padded, d_model]
        expanded = self.patch_expand(patches)  # [B, num_patches, patch_size * d]
        expanded = expanded.view(B, num_patches * self.patch_size, self.d_model)

        expanded = self.pos_enc(expanded)
        out = self.refine(expanded)            # [B, T_padded, d_model]
        out = out[:, :target_len, :]           # trim to original length
        return self.output_proj(out)           # [B, T, vocab_size]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class BLTAutoencoder(nn.Module):
    """BLT-style Byte Latent Transformer Autoencoder.

    Full forward pass:
      bytes → LocalEncoder → Patcher → GlobalTransformer
            → LocalDecoder → logits

    The global Transformer operates on patch_size× fewer positions than
    the raw byte sequence, concentrating model capacity on patch-level
    (morpheme / subword-granularity) semantic reasoning.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        local_layers: int = 2,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.patch_size = patch_size

        # Allocate encoder layers: local_layers for LocalEncoder,
        # remaining for the GlobalTransformer
        enc_local = max(1, local_layers)
        enc_global = max(1, num_encoder_layers - enc_local)
        dec_local = max(1, num_decoder_layers)

        # 1. Local encoder
        self.local_encoder = LocalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=enc_local,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # 2. Patcher (fixed-size; entropy-based patching is noted as future work)
        self.patcher = Patcher(patch_size=patch_size, d_model=d_model)

        # 3. Global Transformer (the main capacity allocation)
        self.global_pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=enc_global,
            norm=nn.LayerNorm(d_model),
        )

        # 4. Local decoder
        self.local_decoder = LocalDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=dec_local,
            patch_size=patch_size,
            dropout=dropout,
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass for autoencoder reconstruction.

        Args:
            src: [B, T] byte ids (0–255)
            tgt: unused — included for API consistency with FLUED/BPE

        Returns:
            logits:   [B, T, vocab_size]
            aux_loss: tensor(0.0) — no auxiliary loss for BLT baseline
        """
        # 1. Local byte encoding
        local_repr = self.local_encoder(src)        # [B, T, d_model]

        # 2. Patch grouping
        patches, original_len = self.patcher(local_repr)  # [B, num_patches, d_model]

        # 3. Global Transformer on patch sequence
        patches = self.global_pos_enc(patches)
        global_repr = self.global_transformer(patches)    # [B, num_patches, d_model]

        # 4. Local decoder → byte-level logits
        logits = self.local_decoder(global_repr, target_len=original_len)

        aux_loss = torch.zeros(1, device=src.device).squeeze()
        return logits, aux_loss

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
