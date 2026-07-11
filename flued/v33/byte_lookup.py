from __future__ import annotations

import torch
from torch import nn

from flued.data import BYTE_OFFSET, MASK_ID, PAD_ID


class StructuredByteLookup(nn.Module):
    """Factorized byte seed used by FLUED v3.3.

    Raw byte identity is represented by high/low 4-bit coordinates plus a small
    byte-type embedding.  This keeps the byte entry structured without treating
    each byte as a semantic token.
    """

    TYPE_PAD = 0
    TYPE_MASK = 1
    TYPE_ASCII_LETTER = 2
    TYPE_ASCII_DIGIT = 3
    TYPE_ASCII_SPACE = 4
    TYPE_ASCII_PUNCT = 5
    TYPE_UTF8_CONT = 6
    TYPE_UTF8_START2 = 7
    TYPE_UTF8_START3 = 8
    TYPE_UTF8_START4 = 9
    TYPE_OTHER = 10
    NUM_TYPES = 11

    def __init__(self, d_model: int, type_count: int | None = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.row_embed = nn.Embedding(16, self.d_model)
        self.col_embed = nn.Embedding(16, self.d_model)
        self.type_embed = nn.Embedding(int(type_count or self.NUM_TYPES), self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    @staticmethod
    def byte_types(token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp(min=PAD_ID, max=MASK_ID)
        valid_byte = token_ids.ge(BYTE_OFFSET) & token_ids.lt(MASK_ID)
        raw = (token_ids - BYTE_OFFSET).clamp(min=0, max=255)
        types = torch.full_like(token_ids, StructuredByteLookup.TYPE_OTHER)
        types = torch.where(token_ids.eq(PAD_ID), torch.full_like(types, StructuredByteLookup.TYPE_PAD), types)
        types = torch.where(token_ids.eq(MASK_ID), torch.full_like(types, StructuredByteLookup.TYPE_MASK), types)

        upper = raw.ge(ord("A")) & raw.le(ord("Z"))
        lower = raw.ge(ord("a")) & raw.le(ord("z"))
        ascii_letter = upper | lower
        ascii_digit = raw.ge(ord("0")) & raw.le(ord("9"))
        ascii_space = (raw == 9) | (raw == 10) | (raw == 13) | (raw == 32)
        ascii_printable = raw.ge(32) & raw.le(126)
        ascii_punct = ascii_printable & ~(ascii_letter | ascii_digit | ascii_space)
        utf8_cont = raw.ge(0x80) & raw.le(0xBF)
        utf8_start2 = raw.ge(0xC2) & raw.le(0xDF)
        utf8_start3 = raw.ge(0xE0) & raw.le(0xEF)
        utf8_start4 = raw.ge(0xF0) & raw.le(0xF4)

        types = torch.where(valid_byte & ascii_letter, torch.full_like(types, StructuredByteLookup.TYPE_ASCII_LETTER), types)
        types = torch.where(valid_byte & ascii_digit, torch.full_like(types, StructuredByteLookup.TYPE_ASCII_DIGIT), types)
        types = torch.where(valid_byte & ascii_space, torch.full_like(types, StructuredByteLookup.TYPE_ASCII_SPACE), types)
        types = torch.where(valid_byte & ascii_punct, torch.full_like(types, StructuredByteLookup.TYPE_ASCII_PUNCT), types)
        types = torch.where(valid_byte & utf8_cont, torch.full_like(types, StructuredByteLookup.TYPE_UTF8_CONT), types)
        types = torch.where(valid_byte & utf8_start2, torch.full_like(types, StructuredByteLookup.TYPE_UTF8_START2), types)
        types = torch.where(valid_byte & utf8_start3, torch.full_like(types, StructuredByteLookup.TYPE_UTF8_START3), types)
        types = torch.where(valid_byte & utf8_start4, torch.full_like(types, StructuredByteLookup.TYPE_UTF8_START4), types)
        return types

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp(min=PAD_ID, max=MASK_ID)
        valid_byte = token_ids.ge(BYTE_OFFSET) & token_ids.lt(MASK_ID)
        raw = (token_ids - BYTE_OFFSET).clamp(min=0, max=255)
        hi = torch.where(valid_byte, raw >> 4, torch.zeros_like(raw))
        lo = torch.where(valid_byte, raw & 15, torch.zeros_like(raw))
        h = self.row_embed(hi) + self.col_embed(lo) + self.type_embed(self.byte_types(token_ids))
        return self.norm(h)
