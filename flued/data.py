"""
Data pipeline for FLUED Stage A experiments.

Provides:
  - UTF-8 byte / Unicode codepoint encoding utilities
  - SimpleBPE: minimal byte-pair encoding trained from scratch
  - ByteTextDataset: chunked dataset of raw UTF-8 bytes (FLUED, BLT)
  - BPETextDataset:  chunked dataset of BPE token ids (BPE baseline)
  - get_dataloader:  convenience wrapper around DataLoader
  - Dynamic span utilities used by the FLUED encoder at inference time
"""

from collections import Counter
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Tiny stub corpus — used when no external data path is provided
# ---------------------------------------------------------------------------

STUB_CORPUS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "FLUED is a fluid language unified embedding-matrix discrete-continuous converter.",
    "Natural language processing enables machines to understand human language.",
    "Transformers have revolutionized the field of natural language processing.",
    "Stage A autoencoder pretraining with dynamic semantic compilation.",
    "UTF-8 encoding supports all Unicode characters including 中文、日本語、한국어.",
    "Byte-level tokenization handles any input without out-of-vocabulary issues.",
    "The dynamic segmenter learns context-sensitive span boundaries.",
    "BPE (byte-pair encoding) learns a subword vocabulary from data statistics.",
    "BLT (byte latent transformer) operates directly on raw byte sequences.",
    "Self-gating logic lets the model decide its own segmentation boundaries.",
    "AttenRes measures the change in hidden states between transformer layers.",
    "The bridge potential creates long-range semantic associations.",
    "Autoencoder pretraining provides a strong initialisation for downstream tasks.",
    "Reconstruction accuracy is the primary metric for Stage A evaluation.",
]


# ---------------------------------------------------------------------------
# UTF-8 / codepoint utilities
# ---------------------------------------------------------------------------


def text_to_bytes(text: str) -> List[int]:
    """Encode a Unicode string to a list of UTF-8 byte values (0–255)."""
    return list(text.encode("utf-8"))


def bytes_to_text(byte_seq: List[int]) -> str:
    """Decode a list of byte values to a Unicode string.

    Invalid byte sequences are replaced with the Unicode replacement character.
    """
    return bytes(byte_seq).decode("utf-8", errors="replace")


def text_to_char_ids(text: str) -> List[int]:
    """Encode a string as a list of Unicode codepoints (clipped to 0xFFFF).

    **Limitation**: Characters outside the Basic Multilingual Plane (BMP),
    i.e. those with codepoint > U+FFFF (emoji U+1F000–U+1FFFF, rare CJK
    extensions U+20000+, historic scripts, etc.), are silently clipped to
    0xFFFF.  This loses the distinction between different non-BMP characters.
    Use text_to_bytes (byte-level UTF-8) when non-BMP character identity must
    be preserved.

    Used as an alternative to byte-level encoding for CJK-heavy text where
    UTF-8 produces 3 bytes per character, inflating sequence length.
    """
    return [min(ord(c), 0xFFFF) for c in text]


def char_ids_to_text(ids: List[int]) -> str:
    """Decode a codepoint list back to a Python string."""
    return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# Minimal BPE tokenizer
# ---------------------------------------------------------------------------


class SimpleBPE:
    """Minimal byte-pair encoding tokenizer trained on a raw text corpus.

    Operates on UTF-8 bytes so it handles arbitrary Unicode input.
    Intended as a structural stub for the 64k BPE baseline — not optimised
    for production throughput.

    Special token IDs:
      PAD_ID = 0, BOS_ID = 1, EOS_ID = 2, UNK_ID = 3
      Byte tokens start at id 4 (i.e. byte b maps to id b + 4).
    """

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3
    _SPECIAL: int = 4  # first real (non-special) token id

    def __init__(self, vocab_size: int = 65536) -> None:
        self.vocab_size = vocab_size
        # merge table: (left_id, right_id) -> merged_id
        self.merges: dict = {}
        # id -> tuple of raw byte values (0–255) that this token represents
        self.id_to_bytes: List[Tuple[int, ...]] = []
        self._build_base_vocab()

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def _build_base_vocab(self) -> None:
        """Initialise the vocabulary with 256 single-byte tokens."""
        # Placeholder entries for the 4 special tokens
        self.id_to_bytes = [() for _ in range(self._SPECIAL)]
        # Single-byte tokens: id = byte + _SPECIAL
        for b in range(256):
            self.id_to_bytes.append((b,))

    def train(self, texts: List[str], num_merges: Optional[int] = None) -> None:
        """Learn BPE merges from a list of raw text strings.

        Args:
            texts: list of training strings (any Unicode).
            num_merges: how many merge rules to learn; defaults to fill
                        up to self.vocab_size.
        """
        if num_merges is None:
            num_merges = self.vocab_size - len(self.id_to_bytes)
        num_merges = max(0, min(num_merges, self.vocab_size - len(self.id_to_bytes)))

        # Encode corpus as sequences of token ids (initially single-byte ids)
        seqs: List[Tuple[int, ...]] = []
        for t in texts:
            seqs.append(tuple(b + self._SPECIAL for b in t.encode("utf-8")))

        for _ in range(num_merges):
            # Count all adjacent token pairs
            pair_counts: Counter = Counter()
            for seq in seqs:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_id = len(self.id_to_bytes)
            self.merges[best_pair] = new_id
            # Store the byte content of the new merged token
            merged_bytes = self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            self.id_to_bytes.append(merged_bytes)
            # Apply the merge across all corpus sequences
            seqs = [self._apply_merge(seq, best_pair, new_id) for seq in seqs]

    @staticmethod
    def _apply_merge(
        seq: Tuple[int, ...], pair: Tuple[int, int], new_id: int
    ) -> Tuple[int, ...]:
        """Replace all non-overlapping occurrences of *pair* with *new_id*."""
        out: List[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode a string to a list of token ids using greedy merge."""
        seq: Tuple[int, ...] = tuple(b + self._SPECIAL for b in text.encode("utf-8"))
        # Repeatedly scan and apply the highest-priority applicable merge
        changed = True
        while changed:
            changed = False
            out: List[int] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1:
                    pair = (seq[i], seq[i + 1])
                    if pair in self.merges:
                        out.append(self.merges[pair])
                        i += 2
                        changed = True
                        continue
                out.append(seq[i])
                i += 1
            seq = tuple(out)
        return list(seq)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back to a UTF-8 string."""
        byte_list: List[int] = []
        for tid in ids:
            if tid < self._SPECIAL:
                continue  # skip special tokens
            if tid < len(self.id_to_bytes):
                byte_list.extend(self.id_to_bytes[tid])
        return bytes(byte_list).decode("utf-8", errors="replace")

    @property
    def current_vocab_size(self) -> int:
        """Number of tokens currently in the vocabulary."""
        return len(self.id_to_bytes)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ByteTextDataset(Dataset):
    """Dataset of fixed-length UTF-8 byte chunks.

    Used by the FLUED and BLT models which operate at byte level.

    Each sample is a pair (src, tgt) where:
      src = bytes[start : start + seq_len]
      tgt = bytes[start + 1 : start + seq_len + 1]   (next-byte prediction)
    """

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        seq_len: int = 128,
        stride: int = 64,
    ) -> None:
        self.seq_len = seq_len

        if texts is None and file_path is None:
            # Use stub corpus repeated to produce a non-trivial number of chunks
            texts = STUB_CORPUS * 50
        elif file_path is not None:
            with open(file_path, encoding="utf-8") as fh:
                texts = fh.readlines()

        # Concatenate all texts into a single byte stream
        all_bytes: List[int] = []
        for t in texts:  # type: ignore[union-attr]
            all_bytes.extend(text_to_bytes(t.rstrip("\n")))
            all_bytes.append(10)  # newline as separator

        self.data: torch.Tensor = torch.tensor(all_bytes, dtype=torch.long)

        # Slide a window of (seq_len + 1) bytes with given stride
        self.chunks: List[torch.Tensor] = []
        for start in range(0, len(self.data) - seq_len, stride):
            self.chunks.append(self.data[start : start + seq_len + 1])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        return chunk[: self.seq_len], chunk[1 : self.seq_len + 1]


class BPETextDataset(Dataset):
    """Dataset of fixed-length BPE token chunks.

    Used by the BPE-Transformer baseline model.

    Each sample is a pair (src, tgt) where:
      src = ids[start : start + seq_len]
      tgt = ids[start + 1 : start + seq_len + 1]
    """

    def __init__(
        self,
        bpe: SimpleBPE,
        texts: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        seq_len: int = 128,
        stride: int = 64,
    ) -> None:
        self.seq_len = seq_len
        self.bpe = bpe

        if texts is None and file_path is None:
            texts = STUB_CORPUS * 50
        elif file_path is not None:
            with open(file_path, encoding="utf-8") as fh:
                texts = fh.readlines()

        all_ids: List[int] = []
        for t in texts:  # type: ignore[union-attr]
            all_ids.extend(bpe.encode(t.rstrip("\n")))
            all_ids.append(SimpleBPE.EOS_ID)

        self.data: torch.Tensor = torch.tensor(all_ids, dtype=torch.long)

        self.chunks: List[torch.Tensor] = []
        for start in range(0, len(self.data) - seq_len, stride):
            self.chunks.append(self.data[start : start + seq_len + 1])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        return chunk[: self.seq_len], chunk[1 : self.seq_len + 1]


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience wrapper that creates a DataLoader with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Dynamic span utilities (used by FLUED encoder at inference / analysis time)
# ---------------------------------------------------------------------------


def compute_dynamic_spans(
    gate_compress: torch.Tensor,
    gate_expand: torch.Tensor,
    threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """Convert per-position SGL gate signals into non-overlapping spans.

    Decision rules (applied left-to-right):
      - expand gate high  → always close the current span here (semantic boundary)
      - compress gate low → close span (position is not merging with predecessor)
      - compress gate high → continue accumulating the current span

    Args:
        gate_compress: 1-D float tensor of shape [seq_len] ∈ [0, 1].
                       High value → merge with previous span.
        gate_expand:   1-D float tensor of shape [seq_len] ∈ [0, 1].
                       High value → force a span boundary.
        threshold:     Binary threshold applied to the gate values.

    Returns:
        List of (start, end) pairs with exclusive end, covering [0, seq_len).
    """
    seq_len = gate_compress.size(0)
    compress = (gate_compress > threshold).tolist()
    expand = (gate_expand > threshold).tolist()

    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(seq_len):
        if expand[i] and i > start:
            spans.append((start, i))
            start = i
        elif not compress[i] and i > start:
            spans.append((start, i))
            start = i

    if start < seq_len:
        spans.append((start, seq_len))

    return spans


def pool_spans(
    hidden: torch.Tensor,
    spans: List[Tuple[int, int]],
) -> torch.Tensor:
    """Average-pool hidden states within each span.

    Args:
        hidden: [seq_len, d_model] tensor of hidden states.
        spans:  list of (start, end) tuples returned by compute_dynamic_spans.

    Returns:
        [num_spans, d_model] tensor of per-span latent vectors.
    """
    latents = [hidden[s:e].mean(dim=0) for s, e in spans]
    return torch.stack(latents, dim=0)
