"""
BLT (Byte Latent Transformer) — Full Paper Reproduction.

Reference: Pagnoni et al. (2024) "Byte Latent Transformer: Patches Scale
           Better Than Tokens"  (https://arxiv.org/abs/2412.09871)

Architecture (faithful to paper)
---------------------------------
  Stage 1 — Pre-train small byte-level LM (ByteLanguageModel):
      bytes → causal Transformer → next-byte logits
      Loss: cross-entropy (next-byte prediction)

  Stage 2 — BLT autoencoder with FROZEN local LM:
      bytes → Frozen ByteLM → hidden states [B,T,d_local]
      Frozen ByteLM → next-byte logits → entropy → boundaries
      hidden states → EntropyPatcher → mean-pooled patches [B,N,d_local]
      patches → Linear(d_local→d_global) → GlobalTransformer → Decoder
      Loss: reconstruction cross-entropy (global+decoder only)

Key differences from our earlier joint-training version:
  - Local encoder is pre-trained as a causal LM (not jointly trained)
  - Entropy comes from the LM's own predictions (not a separate head)
  - Local LM is FROZEN during stage 2 — only global TF + decoder trained
"""

import math
from typing import Dict, Optional, Tuple

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


# ===========================================================================
# Stage 1: Byte Language Model (pre-trained, then frozen)
# ===========================================================================

class ByteLanguageModel(nn.Module):
    """Small autoregressive byte-level LM — pre-trained in Stage 1.

    Uses a causal Transformer encoder to predict next-byte probabilities.
    After pre-training, the model is frozen and used to:
      (a) extract per-byte hidden representations → fed to patcher
      (b) compute next-byte entropy → determines patch boundaries
    """

    def __init__(
        self,
        vocab_size: int = 257,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.byte_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len * 2, dropout=dropout)

        # Causal Transformer (encoder + causal mask = autoregressive)
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

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both hidden states and logits.

        Args:
            x: [B, T] byte ids (PAD-offset)

        Returns:
            hidden: [B, T, d_model]  — per-byte representations
            logits: [B, T, vocab_size] — next-byte predictions
        """
        T = x.size(1)
        device = x.device

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Embed
        h = self.pos_enc(self.byte_embed(x))

        # Causal encoder
        out = self.transformer(h, mask=causal_mask, is_causal=False)  # [B, T, d_model]

        logits = self.lm_head(out)  # [B, T, vocab_size]
        return out, logits

    @torch.no_grad()
    def compute_entropy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-position next-byte entropy (for patching).

        FP32 forced to avoid FP16 underflow: softmax over 257 classes
        can underflow to exact 0, causing log(0) = -inf, 0*(-inf) = NaN.

        Returns:
            hidden:  [B, T, d_model]
            entropy: [B, T]
        """
        hidden, logits = self.forward(x)
        # Force FP32 for numerical stability (same fix as FLUED boundary_loss)
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)  # [B, T]
        return hidden, entropy


class FixedPatcher(nn.Module):
    """Fixed-size patch aggregator (simple baseline)."""

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_agg = nn.Linear(patch_size * d_model, d_model)

    def forward(self, local_repr: torch.Tensor, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, T, d = local_repr.shape
        remainder = T % self.patch_size
        if remainder != 0:
            pad = self.patch_size - remainder
            local_repr = F.pad(local_repr, (0, 0, 0, pad))
        T_padded = local_repr.size(1)
        num_patches = T_padded // self.patch_size
        patches_in = local_repr.view(B, num_patches, self.patch_size * d)
        patches = self.patch_agg(patches_in)
        # segment_lengths: each patch covers exactly patch_size bytes
        seg_lens = torch.full((B, num_patches), self.patch_size, device=local_repr.device, dtype=torch.long)
        seg_lens[:, -1] = self.patch_size - (T_padded - T)  # adjust last if padded
        return patches, seg_lens, num_patches


class EntropyPatcher(nn.Module):
    """Entropy-based dynamic patcher — the core BLT innovation.

    Takes pre-computed per-position next-byte entropy (from a frozen
    ByteLanguageModel) and places patch boundaries where entropy > θ.

    Reference: Pagnoni et al. (2024), Section 3.2
    """

    def __init__(self, theta: float = 3.5) -> None:
        super().__init__()
        self.theta = theta

    def forward(self, local_repr: torch.Tensor, entropy: torch.Tensor, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Dynamic patching via scatter-based mean pooling — VECTORIZED.

        Args:
            local_repr: [B, T, d_model]  — hidden states from frozen ByteLM
            entropy:    [B, T]           — per-position next-byte entropy
            src:        [B, T]           — byte ids (for padding mask)

        Returns:
            patches:        [B, max_segments, d_model]  (padded)
            segment_lengths: [B, max_segments]  (padded, 0 for padding)
            avg_segments:   average number of segments
        """
        B, T, d = local_repr.shape
        device = local_repr.device

        # Build boundaries: segment starts where entropy > theta
        boundaries = (entropy > self.theta).float()      # [B, T]
        boundaries[:, 0] = 1.0

        # Segment IDs via cumsum
        seg_ids = boundaries.cumsum(dim=1).long() - 1    # [B, T]

        # Mask padding
        pad_mask = (src != 0).float()
        seg_ids = seg_ids * pad_mask.long()

        max_seg = int(seg_ids.max().item()) + 1

        # Scatter-based mean pooling
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * max_seg
        flat_idx = (seg_ids + batch_offsets).long()
        flat_idx = flat_idx * pad_mask.long()

        flat_src = local_repr * pad_mask.unsqueeze(-1)
        seg_sums = torch.zeros(B * max_seg, d, device=device, dtype=local_repr.dtype)
        seg_sums.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, -1, d).reshape(-1, d),
                              flat_src.reshape(-1, d))

        seg_counts = torch.zeros(B * max_seg, device=device, dtype=local_repr.dtype)
        seg_counts.scatter_add_(0, flat_idx.reshape(-1), pad_mask.reshape(-1))

        seg_counts = seg_counts.clamp(min=1).unsqueeze(-1)
        seg_means = (seg_sums / seg_counts).view(B, max_seg, d)

        # Segment lengths
        seg_lens = torch.zeros(B * max_seg, device=device, dtype=torch.long)
        ones = torch.ones_like(flat_idx, dtype=torch.long) * pad_mask.long()
        seg_lens.scatter_add_(0, flat_idx.reshape(-1), ones.reshape(-1))
        seg_lens = seg_lens.view(B, max_seg)

        valid_seg_counts = (seg_lens > 0).sum(dim=1).float()
        avg_seg = valid_seg_counts.mean().item()

        return seg_means, seg_lens, avg_seg


class LocalDecoder(nn.Module):
    """Byte-level decoder that expands patch representations back to bytes.

    Each patch vector is broadcast to its original segment length,
    then refined by a shallow Transformer, and finally projected to
    output logits over the byte vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.refine = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, patches: torch.Tensor, segment_lengths: torch.Tensor, original_len: int = 0) -> torch.Tensor:
        """Expand patches back to byte-level logits via vectorized broadcast.

        Uses repeat_interleave instead of Python for-loops.
        """
        B, num_seg, d = patches.shape
        device = patches.device

        # Broadcast each patch to its segment length: [B, T_total, d]
        # repeat_interleave on dim=1: for each batch, repeat patch[s] seg_lens[b,s] times
        expanded_parts = []
        for b in range(B):
            valid_mask = segment_lengths[b] > 0
            valid_lens = segment_lengths[b][valid_mask]
            valid_patches = patches[b][valid_mask]
            if valid_lens.numel() == 0:
                expanded_parts.append(torch.zeros(1, d, device=device))
            else:
                expanded_parts.append(valid_patches.repeat_interleave(valid_lens, dim=0))
            # Note: this ignores zero-length (padding) segments

        # Pad to max T
        max_t = max(max(p.shape[0] for p in expanded_parts), original_len)
        padded = []
        for p in expanded_parts:
            if p.shape[0] < max_t:
                pad = torch.zeros(max_t - p.shape[0], d, device=device)
                p = torch.cat([p, pad], dim=0)
            padded.append(p)
        expanded = torch.stack(padded)  # [B, T_max, d]

        expanded = self.pos_enc(expanded)
        out = self.refine(expanded)            # [B, T_max, d]
        return self.output_proj(out)           # [B, T_max, vocab_size]


# ===========================================================================
# Stage 2: BLT Autoencoder (global TF + decoder, local LM frozen)
# ===========================================================================

class BLTAutoencoder(nn.Module):
    """BLT Autoencoder — paper reproduction.

    Stage 2 training: local ByteLM is frozen; only GlobalTransformer
    and LocalDecoder are trained.

    Full forward pass:
      bytes → Frozen ByteLM → hidden [B,T,d_local] + entropy [B,T]
      hidden + entropy → Patcher → patches [B,N,d_local]
      patches → Linear(d_local→d_global) → GlobalTransformer → Decoder → logits

    Constructor accepts pre-trained ByteLM via ``local_lm`` parameter.
    """

    def __init__(
        self,
        vocab_size: int = 257,
        d_model: int = 1024,                     # global transformer dim
        nhead: int = 16,
        dim_feedforward: int = 4096,
        global_layers: int = 10,
        decoder_layers: int = 12,
        # Pre-trained local LM (frozen)
        local_lm: Optional[ByteLanguageModel] = None,
        local_lm_d_model: int = 512,             # local LM output dim (must match checkpoint)
        # Patch config
        patch_mode: str = "entropy",             # "entropy" | "fixed"
        entropy_theta: float = 3.5,
        fixed_patch_size: int = 4,
        # Shared
        max_seq_len: int = 512,
        dropout: float = 0.0,
        # Legacy compat
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
        local_layers: int = 0,                   # ignored when local_lm is provided
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.local_d_model = local_lm_d_model
        self.patch_mode = patch_mode

        # Resolve legacy args
        if num_encoder_layers is not None:
            global_layers = max(1, num_encoder_layers - 2)
        if num_decoder_layers is not None:
            decoder_layers = num_decoder_layers

        # 1. Local byte LM (pre-trained, will be frozen)
        if local_lm is not None:
            self.local_lm = local_lm
        else:
            # Fallback: create a small ByteLM for joint training (ablation mode)
            self.local_lm = ByteLanguageModel(
                vocab_size=vocab_size, d_model=local_lm_d_model,
                nhead=max(4, nhead // 2), dim_feedforward=dim_feedforward // 2,
                num_layers=max(1, local_layers) if local_layers > 0 else 2,
                max_len=max_seq_len, dropout=dropout,
            )
        self._local_lm_frozen = False

        # 2. Patcher
        if patch_mode == "entropy":
            self.patcher = EntropyPatcher(theta=entropy_theta)
        else:
            self.patcher = FixedPatcher(patch_size=fixed_patch_size, d_model=local_lm_d_model)

        # 3. Projection: local LM dim → global transformer dim (if different)
        if local_lm_d_model != d_model:
            self.patch_proj = nn.Linear(local_lm_d_model, d_model)
        else:
            self.patch_proj = nn.Identity()

        # 4. Global Transformer
        self.global_pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=global_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 5. Local decoder
        self.local_decoder = LocalDecoder(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, num_layers=decoder_layers,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Collect all parameters that belong to the pre-trained ByteLM subtree
        local_lm_params = set()
        if self.local_lm is not None:
            local_lm_params = set(self.local_lm.parameters())
        for module in self.modules():
            # Skip any module whose parameters overlap with ByteLM
            mod_params = set(module.parameters(recurse=False))
            if mod_params and mod_params.issubset(local_lm_params):
                continue
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def freeze_local_lm(self) -> None:
        """Freeze the ByteLM parameters (call before Stage 2 training)."""
        if self.local_lm is not None:
            for p in self.local_lm.parameters():
                p.requires_grad = False
            self.local_lm.eval()
            self._local_lm_frozen = True

    # ------------------------------------------------------------------

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Full forward pass.

        Args:
            src: [B, T] byte ids (PAD-offset)

        Returns:
            logits:      [B, T_out, vocab_size]
            metrics_dict: {"avg_num_patches": float}
        """
        # 1. Local LM: hidden states + entropy
        if self._local_lm_frozen:
            with torch.no_grad():
                local_repr, entropy = self.local_lm.compute_entropy(src)
        else:
            # Fallback: joint training (no frozen LM)
            local_repr, logits = self.local_lm(src)
            # Force FP32 for entropy to avoid underflow
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)

        # 2. Patch grouping
        if self.patch_mode == "entropy":
            patches, seg_lens, avg_seg = self.patcher(local_repr, entropy, src)
        else:
            patches, seg_lens, avg_seg = self.patcher(local_repr, src)

        # 3. Project to global dim
        patches = self.patch_proj(patches)  # [B, N, d_global]

        # 4. Global Transformer on patches
        patches = self.global_pos_enc(patches)
        global_repr = self.global_transformer(patches)  # [B, N, d_global]

        # 5. Local decoder → byte logits
        logits = self.local_decoder(global_repr, seg_lens, original_len=src.size(1))

        return logits, {"avg_num_patches": float(avg_seg)}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameters(self) -> int:
        """Count only trainable (non-frozen) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
