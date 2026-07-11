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
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

PAD_ID = 0
BYTE_OFFSET = 1
MASK_ID = BYTE_OFFSET + 256
BYTE_VOCAB_SIZE = MASK_ID + 1  # PAD + 256 bytes + MASK


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
        if tid in (PAD_ID, MASK_ID):
            continue
        if tid < BYTE_OFFSET or tid >= MASK_ID:
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


class StreamingReconstructionDataset(torch.utils.data.IterableDataset):
    """mmap-based streaming dataset for arbitrarily large corpora.

    Randomly samples byte chunks from the raw file via memory-mapping.
    Each worker independently mmaps the file and generates infinite
    random chunks — no full-file load into RAM, no pre-processing,
    and DataLoader's ``num_workers`` provides natural async prefetch.

    Returns ``(src, tgt)`` where both are PAD-offset byte-id tensors
    of length ``seq_len`` (tgt = src shifted by 1, like causal LM data
    but used for reconstruction here).
    """

    def __init__(
        self,
        file_path: str,
        seq_len: int = 512,
        samples_per_worker: int = 2000,
        seed: int = 42,
    ):
        import os
        super().__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.file_size = os.path.getsize(file_path)
        self.samples_per_worker = samples_per_worker
        self.seed = seed

    def __iter__(self):
        import mmap, random
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id * 10000
        rng = random.Random(worker_seed)

        # Each worker opens its own mmap handle.
        with open(self.file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            max_start = max(0, self.file_size - self.seq_len - 2)
            for _ in range(self.samples_per_worker):
                start = rng.randint(0, max_start)
                # Prefer natural text starts. Random mmap offsets often land in
                # the middle of UTF-8 codepoints or lines, which creates a
                # train/eval segmentation distribution mismatch for codec
                # experiments even though the raw bytes are valid training data.
                search_from = max(0, start - 256)
                line_start = mm.rfind(b"\n", search_from, start + 1)
                if line_start >= 0 and line_start + 1 <= max_start:
                    start = line_start + 1
                for _adjust in range(4):
                    if start >= max_start:
                        break
                    b0 = mm[start]
                    if not (0x80 <= b0 <= 0xBF):
                        break
                    start += 1
                raw = mm[start : start + self.seq_len + 1]
                if len(raw) < self.seq_len + 1:
                    continue
                # PAD-offset: byte b → b + 1 (0 = PAD)
                buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long() + 1
                src = buf[:self.seq_len]
                tgt = buf[1:self.seq_len + 1]
                yield src, tgt


class ShardedStreamingReconstructionDataset(torch.utils.data.IterableDataset):
    """Streaming reconstruction dataset over a text-file shard manifest.

    The manifest contains one UTF-8 text shard path per line.  Sampling is
    weighted by shard byte size, so large shards contribute proportionally
    without concatenating the corpus into one 200GB+ file.
    """

    def __init__(
        self,
        manifest_path: str,
        seq_len: int = 512,
        samples_per_worker: int = 2000,
        seed: int = 42,
    ) -> None:
        import os
        super().__init__()
        self.manifest_path = manifest_path
        self.seq_len = int(seq_len)
        self.samples_per_worker = int(samples_per_worker)
        self.seed = int(seed)
        root = []
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                path = line.strip()
                if not path:
                    continue
                size = os.path.getsize(path)
                if size > self.seq_len + 2:
                    root.append((path, size))
        if not root:
            raise ValueError(f"No usable shards found in manifest: {manifest_path}")
        self.shards = root
        self.total_size = sum(size for _path, size in self.shards)

    def __iter__(self):
        import bisect
        import mmap
        import random

        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id * 10000
        rng = random.Random(worker_seed)
        cumulative = []
        total = 0
        for _path, size in self.shards:
            total += size
            cumulative.append(total)

        handles = {}

        def get_mmap(index: int):
            if index not in handles:
                path, size = self.shards[index]
                fh = open(path, "rb")
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                handles[index] = (fh, mm, size)
            return handles[index][1], handles[index][2]

        try:
            for _ in range(self.samples_per_worker):
                shard_idx = bisect.bisect_right(cumulative, rng.randrange(total))
                mm, size = get_mmap(shard_idx)
                max_start = max(0, size - self.seq_len - 2)
                start = rng.randint(0, max_start)
                search_from = max(0, start - 256)
                line_start = mm.rfind(b"\n", search_from, start + 1)
                if line_start >= 0 and line_start + 1 <= max_start:
                    start = line_start + 1
                for _adjust in range(4):
                    if start >= max_start:
                        break
                    b0 = mm[start]
                    if not (0x80 <= b0 <= 0xBF):
                        break
                    start += 1
                raw = mm[start : start + self.seq_len + 1]
                if len(raw) < self.seq_len + 1:
                    continue
                buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long() + 1
                src = buf[: self.seq_len]
                tgt = buf[1 : self.seq_len + 1]
                yield src, tgt
        finally:
            for fh, mm, _size in handles.values():
                mm.close()
                fh.close()


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

