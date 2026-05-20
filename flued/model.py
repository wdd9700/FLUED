"""
FLUED Stage A Autoencoder — Dynamic Semantic Compiler.

Architecture summary
--------------------
  1. Byte embedding + sinusoidal positional encoding
  2. ShallowEncoder (DSC front-end): 2–4 Transformer layers that also
     accumulate cross-layer Attention Residuals (AttenRes).
  3. SGLGatingModule: maps (hidden, attenres) → three gate signals per
     position: γ_compress, γ_expand, γ_bridge.
  4. DynamicLatentEncoder: differentiable soft-pooling driven by γ_compress,
     producing a same-length latent that encodes variable-granularity spans.
  5. DeepEncoder: additional Transformer layers to refine the latent.
  6. TransformerDecoder: reconstructs the byte sequence from the latent.

Auxiliary loss: SGL gate entropy regularisation (prevents gate collapse).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared sub-modules
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

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
        # Register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]"""
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# FLUED-specific sub-modules
# ---------------------------------------------------------------------------


class ShallowEncoder(nn.Module):
    """DSC front-end: shallow Transformer encoder that also computes AttenRes.

    AttenRes (Attention Residual) is defined as the weighted sum of
    per-layer hidden-state differences:

        R_l = H^{l+1} - H^l                           (layer-wise change)
        AttenRes = Σ_l  softmax(w)_l · R_l            (weighted aggregate)

    The learnable weights w_l let the network attend to shallow (syntactic)
    vs. deep (semantic) layer signals.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,  # Pre-LN for training stability
                )
                for _ in range(num_layers)
            ]
        )
        # Learnable per-layer aggregation weights for AttenRes
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                    [B, T, d_model] embedded input
            src_key_padding_mask: [B, T] bool mask (True = padding position)

        Returns:
            final_hidden: [B, T, d_model] hidden states after all layers
            attenres:     [B, T, d_model] weighted aggregate of layer residuals
        """
        h = x
        residuals = []
        for layer in self.layers:
            h_next = layer(h, src_key_padding_mask=src_key_padding_mask)
            residuals.append(h_next - h)
            h = h_next

        # Weighted sum of residuals (softmax normalises the weights)
        weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers]
        attenres = sum(w * r for w, r in zip(weights, residuals))  # [B, T, d]
        return h, attenres  # type: ignore[return-value]


class SGLGatingModule(nn.Module):
    """Self-Gating Logic (SGL) module.

    Takes the shallow encoder's hidden states and AttenRes signal and
    produces three sigmoid gate values per sequence position:

      γ_compress ∈ [0,1] — high → merge this position into the preceding span
      γ_expand   ∈ [0,1] — high → force a semantic boundary here
      γ_bridge   ∈ [0,1] — high → write a bridge potential for long-range links

    Also computes the bridge potential vector P_i = γ_bridge_i · MLP(h_i).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Maps concatenated [hidden ‖ attenres] → 3 gate logits
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        # Projection for bridge potential vectors
        self.bridge_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        attenres: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden:   [B, T, d_model]
            attenres: [B, T, d_model]

        Returns:
            gamma_compress:   [B, T]
            gamma_expand:     [B, T]
            gamma_bridge:     [B, T]
            bridge_potential: [B, T, d_model]
        """
        combined = torch.cat([hidden, attenres], dim=-1)  # [B, T, 2d]
        gates = torch.sigmoid(self.gate_mlp(combined))    # [B, T, 3]

        gamma_compress = gates[..., 0]   # [B, T]
        gamma_expand   = gates[..., 1]   # [B, T]
        gamma_bridge   = gates[..., 2]   # [B, T]

        # Bridge potential: gated linear projection of hidden state
        bridge_potential = gamma_bridge.unsqueeze(-1) * self.bridge_proj(hidden)
        return gamma_compress, gamma_expand, gamma_bridge, bridge_potential


class DynamicLatentEncoder(nn.Module):
    """Differentiable soft-pooling driven by the compress gate.

    Instead of hard span segmentation (not differentiable), we implement a
    soft exponential running average: each position t blends its own hidden
    state with the accumulated context from the left, weighted by γ_compress.

        acc_0 = h_0
        acc_t = γ_compress_t · acc_{t-1} + (1 − γ_compress_t) · h_t

    When γ_compress ≈ 1 the accumulator carries the left context (merge);
    when γ_compress ≈ 0 the accumulator resets to the current position (new span).

    The accumulated vector is then concatenated with AttenRes and refined by
    a small MLP to produce the final latent — carrying a "compressed fingerprint"
    of the span's internal structure.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.span_refine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attenres: torch.Tensor,
        gamma_compress: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden:         [B, T, d_model]
            attenres:       [B, T, d_model]
            gamma_compress: [B, T] ∈ [0, 1]

        Returns:
            latent: [B, T, d_model] soft-pooled latent representation
        """
        B, T, d = hidden.shape

        # Sequential soft accumulation over time.
        # NOTE: This loop is intentionally sequential (each step depends on the
        # previous accumulator), which prevents parallelisation. For Stage A
        # correctness this is acceptable; a production implementation should
        # replace this with a parallel scan using cumulative products:
        #   log-space cumprod of γ_compress, then weighted prefix sums.
        acc = hidden[:, 0, :]                   # [B, d]
        accumulated = [acc]
        for t in range(1, T):
            g = gamma_compress[:, t].unsqueeze(-1)  # [B, 1]
            acc = g * acc + (1.0 - g) * hidden[:, t, :]
            accumulated.append(acc)

        # Stack back to [B, T, d]
        accumulated_t = torch.stack(accumulated, dim=1)

        # Refine: blend with AttenRes fingerprint
        latent = self.span_refine(torch.cat([accumulated_t, attenres], dim=-1))
        return latent


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class FLUEDAutoencoder(nn.Module):
    """FLUED Stage A Autoencoder (Dynamic Semantic Compiler).

    Full forward pass:
      src → embed → ShallowEncoder → SGLGating → DynamicLatentEncoder
          → DeepEncoder → TransformerDecoder (teacher-forced) → logits

    Loss = cross-entropy reconstruction + SGL gate entropy regularisation.
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
        shallow_layers: int = 2,
        gate_entropy_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.gate_entropy_weight = gate_entropy_weight

        # --- Input representation ---
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len * 2, dropout=dropout)

        # --- DSC front-end ---
        # Clamp shallow_layers so that:
        #   (a) shallow_layers ≥ 1 (ShallowEncoder needs at least one layer), and
        #   (b) num_encoder_layers - shallow_layers ≥ 1 (deep encoder needs at least one).
        # Using min(shallow_layers, max(1, num_encoder_layers - 1)) satisfies both
        # when num_encoder_layers ≥ 2; when num_encoder_layers == 1 shallow=1 and
        # deep_layers is clamped to 1 below.
        self.shallow_layers = min(max(1, shallow_layers), max(1, num_encoder_layers - 1))
        self.shallow_enc = ShallowEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=self.shallow_layers,
            dropout=dropout,
        )

        # --- SGL gating ---
        self.sgl = SGLGatingModule(d_model)

        # --- Dynamic latent encoder ---
        self.latent_enc = DynamicLatentEncoder(d_model)

        # --- Deep encoder (refines the latent) ---
        deep_layers = max(1, num_encoder_layers - self.shallow_layers)
        self.deep_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=deep_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Decoder ---
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection: latent → logits over byte vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
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
    # Encode / decode interface
    # ------------------------------------------------------------------

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode a byte sequence to a latent representation.

        Args:
            src:                  [B, T] long tensor of byte ids (0–255)
            src_key_padding_mask: [B, T] bool (True = padding)

        Returns:
            latent:    [B, T, d_model]
            gate_info: dict with keys
                         "gamma_compress", "gamma_expand", "gamma_bridge",
                         "bridge_potential"
        """
        # 1. Embed + positional encoding
        x = self.pos_enc(self.embedding(src))  # [B, T, d]

        # 2. Shallow encoder → hidden states + AttenRes
        shallow_out, attenres = self.shallow_enc(x, src_key_padding_mask)

        # 3. SGL gating
        gamma_compress, gamma_expand, gamma_bridge, bridge_potential = self.sgl(
            shallow_out, attenres
        )

        # 4. Differentiable dynamic latent encoding
        latent_raw = self.latent_enc(shallow_out, attenres, gamma_compress)

        # 5. Deep encoder refinement
        latent = self.deep_enc(latent_raw, src_key_padding_mask=src_key_padding_mask)

        gate_info = {
            "gamma_compress": gamma_compress,
            "gamma_expand": gamma_expand,
            "gamma_bridge": gamma_bridge,
            "bridge_potential": bridge_potential,
        }
        return latent, gate_info

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced decode from encoder memory.

        Args:
            tgt:    [B, T] long tensor of byte ids (teacher-forcing input)
            memory: [B, T, d_model] encoder output

        Returns:
            logits: [B, T, vocab_size]
        """
        tgt_emb = self.pos_enc(self.embedding(tgt))
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(dec_out)

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full autoencoder forward (teacher-forced reconstruction).

        Args:
            src: [B, T] input byte ids
            tgt: [B, T] target byte ids for teacher forcing;
                 if None, uses src (same-sequence reconstruction).

        Returns:
            logits:   [B, T, vocab_size]
            aux_loss: scalar — SGL gate entropy regularisation
        """
        if tgt is None:
            tgt = src

        T = src.size(1)
        # Causal mask so the decoder cannot attend to future positions
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)

        memory, gate_info = self.encode(src)
        logits = self.decode(tgt, memory, tgt_mask=tgt_mask)
        aux_loss = self._sgl_entropy_loss(
            gate_info["gamma_compress"],
            gate_info["gamma_expand"],
            gate_info["gamma_bridge"],
        )
        return logits, aux_loss

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def _sgl_entropy_loss(self, *gates: torch.Tensor) -> torch.Tensor:
        """Binary-entropy regularisation to prevent gate collapse.

        Maximises the entropy of each gate distribution so gates
        do not trivially saturate at 0 or 1.

        L_sgl = -Σ_k [ γ_k·log(γ_k) + (1−γ_k)·log(1−γ_k) ]
        """
        eps = 1e-6
        loss = torch.zeros(1, device=gates[0].device)
        for g in gates:
            entropy = -(
                g * (g + eps).log() + (1.0 - g) * (1.0 - g + eps).log()
            )
            # Minimise negative entropy → maximise entropy
            loss = loss - entropy.mean()
        return self.gate_entropy_weight * loss.squeeze()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
