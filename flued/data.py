"""Data utilities for FLUED v0.4 local experiments.

Provides:
  - UTF-8 byte / Unicode codepoint encoding utilities
  - SimpleBPE: minimal byte-pair encoding trained from scratch
  - ByteTextDataset: chunked dataset of raw UTF-8 bytes (FLUED, BLT)
  - ByteReconstructionDataset: src == tgt with PAD-offset encoding (E1)
  - BPETextDataset:  chunked dataset of BPE token ids (BPE baseline)
  - get_dataloader:  convenience wrapper around DataLoader
  - safe_train_eval_split: robust train/eval split that handles tiny datasets
  - Dynamic span utilities used by the FLUED encoder at inference time
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

PAD_ID = 0
BYTE_OFFSET = 1
BYTE_VOCAB_SIZE = 257  # PAD + 256 bytes


STUB_CORPUS: List[str] = [
    "春眠不觉晓，处处闻啼鸟。",
    "The quick brown fox jumps over the lazy dog.",
    "FLUED v0.4 compiles dynamic semantic units during prefill.",
    "Decode is modeled as a tied-weight inverse approximation.",
    "中文 few-shot cloze tasks are included for local smoke tests.",
    "这是一条用于重建实验的样本文本。",
]


# ---------------------------------------------------------------------------
# Byte PAD-offset helpers
# ---------------------------------------------------------------------------


def text_to_byte_ids(text: str) -> List[int]:
    """Encode UTF-8 bytes with PAD offset: raw bytes 0..255 -> ids 1..256."""
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


def byte_ids_to_text(token_ids: Sequence[int]) -> str:
    """Decode PAD-offset byte ids back to text (ignores PAD ids)."""
    byte_values: List[int] = []
    for tid in token_ids:
        if tid == PAD_ID:
            continue
        if tid < BYTE_OFFSET or tid >= BYTE_VOCAB_SIZE:
            raise ValueError(f"Invalid byte token id: {tid}")
        byte_values.append(tid - BYTE_OFFSET)
    return bytes(byte_values).decode("utf-8", errors="replace")


def text_to_bytes(text: str) -> List[int]:
    """Compatibility helper used by tests."""
    return list(text.encode("utf-8"))


def bytes_to_text(byte_seq: List[int]) -> str:
    """Compatibility helper used by tests."""
    return bytes(byte_seq).decode("utf-8", errors="replace")


def text_to_char_ids(text: str) -> List[int]:
    return [ord(c) for c in text]


def char_ids_to_text(ids: List[int]) -> str:
    return "".join(chr(i) for i in ids)


def _load_texts(texts: Optional[List[str]], file_path: Optional[str]) -> List[str]:
    if texts is not None:
        return texts
    if file_path:
        with open(file_path, encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]
    return STUB_CORPUS * 40


class ByteReconstructionDataset(Dataset):
    """Byte-level reconstruction dataset: src == tgt."""

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        seq_len: int = 128,
        stride: int = 64,
    ) -> None:
        self.seq_len = seq_len
        stream: List[int] = []
        for text in _load_texts(texts, file_path):
            stream.extend(text_to_byte_ids(text))
            stream.append(ord("\n") + BYTE_OFFSET)

        self.data = torch.tensor(stream, dtype=torch.long)
        self.chunks: List[torch.Tensor] = []
        if self.data.numel() == 0:
            self.chunks.append(torch.full((seq_len,), PAD_ID, dtype=torch.long))
            return

        for start in range(0, max(1, len(self.data) - seq_len + 1), max(1, stride)):
            chunk = self.data[start : start + seq_len]
            if chunk.numel() < seq_len:
                pad = torch.full((seq_len - chunk.numel(),), PAD_ID, dtype=torch.long)
                chunk = torch.cat([chunk, pad], dim=0)
            self.chunks.append(chunk)

        if not self.chunks:
            pad = torch.full((seq_len,), PAD_ID, dtype=torch.long)
            self.chunks = [pad]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.chunks[idx]
        return src, src.clone()


# Backward compatibility name.
ByteTextDataset = ByteReconstructionDataset


class SimpleBPE:
    """Minimal byte-level BPE tokenizer for local baselines."""

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    _SPECIAL = 4

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = max(vocab_size, self._SPECIAL + 256)
        self.merges: dict[Tuple[int, int], int] = {}
        self.id_to_bytes: List[Tuple[int, ...]] = []
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        self.id_to_bytes = [() for _ in range(self._SPECIAL)]
        for b in range(256):
            self.id_to_bytes.append((b,))

    def train(self, texts: List[str], num_merges: Optional[int] = None) -> None:
        if num_merges is None:
            num_merges = self.vocab_size - len(self.id_to_bytes)
        num_merges = max(0, min(num_merges, self.vocab_size - len(self.id_to_bytes)))

        seqs: List[Tuple[int, ...]] = [tuple(b + self._SPECIAL for b in t.encode("utf-8")) for t in texts]
        for _ in range(num_merges):
            pair_counts: Counter = Counter()
            for seq in seqs:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1
            if not pair_counts:
                break
            pair = pair_counts.most_common(1)[0][0]
            new_id = len(self.id_to_bytes)
            self.merges[pair] = new_id
            self.id_to_bytes.append(self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]])
            seqs = [self._apply_merge(seq, pair, new_id) for seq in seqs]

    @staticmethod
    def _apply_merge(seq: Tuple[int, ...], pair: Tuple[int, int], new_id: int) -> Tuple[int, ...]:
        out: List[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return tuple(out)

    def encode(self, text: str) -> List[int]:
        seq: Tuple[int, ...] = tuple(b + self._SPECIAL for b in text.encode("utf-8"))
        changed = True
        while changed:
            changed = False
            out: List[int] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) in self.merges:
                    out.append(self.merges[(seq[i], seq[i + 1])])
                    i += 2
                    changed = True
                else:
                    out.append(seq[i])
                    i += 1
            seq = tuple(out)
        return list(seq)

    def decode(self, ids: List[int]) -> str:
        byte_list: List[int] = []
        for tid in ids:
            if self._SPECIAL <= tid < len(self.id_to_bytes):
                byte_list.extend(self.id_to_bytes[tid])
        return bytes(byte_list).decode("utf-8", errors="replace")

    @property
    def current_vocab_size(self) -> int:
        return len(self.id_to_bytes)


class BPETextDataset(Dataset):
    """BPE reconstruction dataset: src == tgt."""

    def __init__(
        self,
        bpe: SimpleBPE,
        texts: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        seq_len: int = 128,
        stride: int = 64,
    ) -> None:
        self.seq_len = seq_len
        all_ids: List[int] = []
        for text in _load_texts(texts, file_path):
            ids = bpe.encode(text)
            if not ids:
                continue
            all_ids.extend(ids)
            all_ids.append(SimpleBPE.EOS_ID)

        self.data = torch.tensor(all_ids or [SimpleBPE.PAD_ID], dtype=torch.long)
        self.chunks: List[torch.Tensor] = []
        for start in range(0, max(1, len(self.data) - seq_len + 1), max(1, stride)):
            chunk = self.data[start : start + seq_len]
            if chunk.numel() < seq_len:
                pad = torch.full((seq_len - chunk.numel(),), SimpleBPE.PAD_ID, dtype=torch.long)
                chunk = torch.cat([chunk, pad], dim=0)
            self.chunks.append(chunk)
        if not self.chunks:
            self.chunks = [torch.full((seq_len,), SimpleBPE.PAD_ID, dtype=torch.long)]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.chunks[idx]
        return src, src.clone()


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# v0.4 additions: ByteReconstructionDataset and safe_train_eval_split
# ---------------------------------------------------------------------------

# PAD=0, byte b (0-255) -> token id b+1 (1-256).  Matches flued.model.PAD_ID.
_BYTE_OFFSET: int = 1


class ByteReconstructionDataset(Dataset):
    """Dataset for strict byte reconstruction (src == tgt) with PAD-offset encoding.

    Used by E1 Stage A: the model must reconstruct the exact input sequence.

    Token encoding:
        PAD = 0
        byte b (0-255) -> token id  b + 1   (i.e. 1-256)

    Each sample is a pair (src, tgt) where src == tgt - the model is not
    doing next-token prediction; it is autoencoding.
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
            texts = STUB_CORPUS * 50
        elif file_path is not None:
            with open(file_path, encoding="utf-8") as fh:
                texts = fh.readlines()

        # Concatenate texts -> single byte stream with PAD-offset encoding
        all_ids: List[int] = []
        for t in texts:  # type: ignore[union-attr]
            all_ids.extend(b + _BYTE_OFFSET for b in t.rstrip("\n").encode("utf-8"))
            all_ids.append(10 + _BYTE_OFFSET)  # newline separator, also offset

        self.data: torch.Tensor = torch.tensor(all_ids, dtype=torch.long)

        self.chunks: List[torch.Tensor] = []
        for start in range(0, len(self.data) - seq_len + 1, stride):
            self.chunks.append(self.data[start : start + seq_len])

        if not self.chunks and len(self.data) > 0:
            # Fallback: single padded chunk so tiny corpora don't crash
            chunk = self.data[: seq_len]
            if len(chunk) < seq_len:
                pad = torch.zeros(seq_len - len(chunk), dtype=torch.long)
                chunk = torch.cat([chunk, pad])
            self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (src, tgt) where src == tgt (strict reconstruction)."""
        chunk = self.chunks[idx]
        return chunk, chunk


def safe_train_eval_split(
    dataset: Dataset,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and eval subsets, handling tiny corpora.

    If len(dataset) < 2, both subsets point to the full dataset (no copy).
    Otherwise performs a standard random_split with at least 1 eval sample.
    """
    n = len(dataset)  # type: ignore[arg-type]
    if n < 2:
        return dataset, dataset
    n_eval = max(1, int(n * eval_fraction))
    n_train = n - n_eval
    return random_split(
        dataset,
        [n_train, n_eval],
        generator=torch.Generator().manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# Dynamic span utilities (used by FLUED encoder at inference / analysis time)
# ---------------------------------------------------------------------------


def tiny_chinese_logic_samples() -> Dict[str, List[ClozeItem]]:
    """Tiny built-in smoke data for E2 few-shot style tasks."""
    return {
        "idiom_cloze": [
            ClozeItem("他学习很努力，成绩___。", ["一落千丈", "蒸蒸日上"], 1),
            ClozeItem("这次准备不足，结果___。", ["马到成功", "一败涂地"], 1),
        ],
        "connective": [
            ClozeItem("虽然下雨了，___我们还是出发了。", ["但是", "因为"], 0),
            ClozeItem("他迟到了，___路上堵车。", ["所以", "因为"], 1),
        ],
        "coref_cloze": [
            ClozeItem("小王告诉小李，___明天会到。", ["他", "她"], 0),
            ClozeItem("小红看见小张后，对___挥手。", ["她", "他"], 1),
        ],
    }


