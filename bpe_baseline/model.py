"""
BPE-Transformer Autoencoder baseline (~300M class-A backbone).

A vanilla Transformer encoder–decoder operating on BPE token sequences.
Serves as the primary tokenisation-based comparison for FLUED Stage A.

Input  : BPE token ids  [B, T]
Output : logits over BPE vocab  [B, T, vocab_size]

Training objective: teacher-forced reconstruction (cross-entropy).
No auxiliary loss — clean baseline with no extra inductive biases.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional encoding (shared helper — identical in all three models)
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
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BPETransformerAutoencoder(nn.Module):
    """Vanilla BPE-Transformer Autoencoder.

    Architecture:
      Embedding(bpe_vocab) + PE → TransformerEncoder → TransformerDecoder
      → Linear(vocab_size)

    The encoder produces a full-length memory tensor; the decoder attends
    to it with a causal mask (teacher-forcing).  This is the simplest
    possible autoencoder baseline and deliberately avoids any bespoke
    inductive biases so that differences in evaluation can be attributed
    to the FLUED / BLT architectural choices.
    """

    def __init__(
        self,
        vocab_size: int = 65536,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding — padding_idx=0 matches SimpleBPE.PAD_ID
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len * 2, dropout=dropout)

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Decoder (attends to encoder memory with causal self-attention mask)
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

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

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

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a BPE token sequence to a memory tensor.

        Args:
            src:                  [B, T] long tensor of BPE token ids
            src_key_padding_mask: [B, T] bool (True = padding)

        Returns:
            memory: [B, T, d_model]
        """
        x = self.pos_enc(self.embedding(src))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode from memory with teacher forcing.

        Args:
            tgt:    [B, T] long tensor of target BPE token ids
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

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full teacher-forced autoencoder forward pass.

        Args:
            src: [B, T] BPE token ids
            tgt: [B, T] target ids; if None, same as src

        Returns:
            logits:   [B, T, vocab_size]
            aux_loss: tensor(0.0) — no auxiliary loss for BPE baseline
        """
        if tgt is None:
            tgt = src

        T = src.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)

        memory = self.encode(src)
        logits = self.decode(tgt, memory, tgt_mask=tgt_mask)
        aux_loss = torch.zeros(1, device=src.device).squeeze()
        return logits, aux_loss

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
