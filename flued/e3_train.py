"""
FLUED E3 — Downstream Language Model Training.

Trains a causal Transformer LM on top of a frozen segmentation encoder.
Compares FLUED vs BLT vs BPE on next-byte perplexity (bits-per-byte).

Usage
-----
    # FLUED downstream
    python -m flued.e3_train --model flued \
        --flued-ckpt checkpoints/e1_step50000.pt \
        --data-path corpus.txt --max-lines 50000

    # BLT downstream
    python -m flued.e3_train --model blt \
        --blt-ckpt checkpoints/blt_step40000.pt \
        --bytelm-ckpt checkpoints/bytel_m_latest.pt \
        --data-path corpus.txt

    # BPE downstream
    python -m flued.e3_train --model bpe \
        --tokenizer-path checkpoints/bpe_tokenizer/tokenizer.json \
        --data-path corpus.txt
"""

import argparse
import logging
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("e3.train")

from flued.e3_downstream import (
    FLUEDDownstream, BLTDownstream, BPEDownstream,
    FixedPatchDownstream, ByteDownstream, PublicTokenizerDownstream,
    bits_per_byte,
)


# ===========================================================================
# FLOPs estimation (analytic, for fair cross-method comparison)
# ===========================================================================

def estimate_transformer_flops(
    d_model: int,
    dim_ff: int,
    num_layers: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """Estimate FLOPs for one Transformer forward pass (matmuls only).

    Returns total FLOPs as a float. Each matmul A·B where A is [M,K] and B is
    [K,N] counts as 2*M*K*N FLOPs (multiply-add).

    Per-layer breakdown:
      - MHA QKV: 3 × (d²·T) × 2
      - MHA out:  d²·T × 2
      - Attention: T²·d × 2
      - FFN up+down: 2 × (d·ff·T) × 2
    """
    T = seq_len
    d = d_model
    ff = dim_ff
    L = num_layers
    # Per layer
    mha_qkv = 3 * d * d * T * 2
    mha_out = d * d * T * 2
    attn    = T * T * d * 2
    ffn     = 2 * d * ff * T * 2
    per_layer = mha_qkv + mha_out + attn + ffn
    # LM head
    lm_head = d * vocab_size * T * 2
    total = (L * per_layer + lm_head) * batch_size
    return total


def format_flops(x: float) -> str:
    if x >= 1e12:
        return f"{x/1e12:.2f} TFLOPs"
    if x >= 1e9:
        return f"{x/1e9:.2f} GFLOPs"
    return f"{x/1e6:.2f} MFLOPs"


# ===========================================================================
# Presets
# ===========================================================================

PRESETS = {
    "smoke": {
        "num_layers": 4, "d_model": 256, "nhead": 4, "dim_feedforward": 512,
        "max_seq_len": 128, "batch_size": 8, "max_steps": 100, "lr": 3e-4,
        "warmup_steps": 10, "grad_accum_steps": 1, "device": "cpu",
    },
    "small": {
        "num_layers": 8, "d_model": 512, "nhead": 8, "dim_feedforward": 2048,
        "max_seq_len": 256, "batch_size": 4, "max_steps": 2000, "lr": 1e-4,
        "warmup_steps": 200, "grad_accum_steps": 2, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
    "500m": {
        "num_layers": 28, "d_model": 1024, "nhead": 16, "dim_feedforward": 4096,
        "max_seq_len": 2048, "batch_size": 1, "max_steps": 20000, "lr": 3e-5,
        "warmup_steps": 200, "grad_accum_steps": 8, "device": "cuda",
        "amp": True, "amp_dtype": "fp16",
    },
}


# ===========================================================================
# Data
# ===========================================================================

# Threshold (bytes) — above this, use streaming dataset instead of in-memory.
# 22GB corpus on 61GB RAM would OOM if loaded as Python str + int64 tensor
# (22GB UTF-8 → ~176GB int64 tensor + ~50GB Python str overhead).
_STREAMING_THRESHOLD = 256 * 1024 * 1024  # 256 MB


def load_texts(data_path: str, max_lines: Optional[int] = None):
    """Load entire corpus into memory. Only safe for small files / max_lines.

    For large corpora, use StreamingByteLMDataset / StreamingTokenLMDataset
    directly, which mmap the file and don't require full in-memory load.
    """
    with open(data_path, encoding="utf-8") as fh:
        if max_lines:
            texts = []
            for i, line in enumerate(fh):
                if i >= max_lines: break
                line = line.rstrip("\n")
                if line.strip(): texts.append(line)
            return texts
        return [line.rstrip("\n") for line in fh if line.strip()]


class ByteLMDataset(torch.utils.data.Dataset):
    """Next-byte prediction dataset for FLUED/BLT downstream (in-memory)."""
    def __init__(self, texts, seq_len=512, stride=256):
        self.seq_len = seq_len
        all_bytes = []
        for text in texts:
            if not text: continue
            all_bytes.extend(b + 1 for b in text.encode("utf-8"))
            all_bytes.append(0)
        if not all_bytes: all_bytes = [0]
        self.data = torch.tensor(all_bytes, dtype=torch.long)
        self.chunks = []
        for start in range(0, max(1, len(self.data) - seq_len), max(1, stride)):
            chunk = self.data[start:start + seq_len + 1]
            if chunk.numel() < seq_len + 1:
                chunk = torch.cat([chunk, torch.zeros(seq_len + 1 - chunk.numel(), dtype=torch.long)])
            self.chunks.append(chunk)
        if not self.chunks: self.chunks = [torch.zeros(seq_len + 1, dtype=torch.long)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return c[:self.seq_len], c[1:self.seq_len + 1]


class StreamingByteLMDataset(torch.utils.data.Dataset):
    """Memory-mapped byte-level dataset for arbitrarily large corpora.

    Reads the raw UTF-8 file directly via mmap and converts each chunk to
    PAD-offset byte ids on demand (b + 1, with PAD=0). The corpus is treated
    as one contiguous byte stream without per-line buffering — this matches
    ByteLMDataset semantics minus the explicit PAD-separator between lines.

    Memory footprint: ~16 KB per chunk view, O(1) regardless of corpus size.
    """
    def __init__(self, data_path: str, seq_len: int = 512, stride: int = 256):
        import os
        self.seq_len = seq_len
        self.stride = stride
        self.data_path = data_path
        self.file_size = os.path.getsize(data_path)
        # Open with mmap; keep handle for lifetime of dataset.
        # Worker subprocesses will re-mmap independently (lazy in __getitem__).
        self._mm = None
        self._fp = None
        # Number of chunks — each chunk needs seq_len + 1 bytes
        if self.file_size < seq_len + 1:
            self.n_chunks = 1
        else:
            usable = self.file_size - (seq_len + 1)
            self.n_chunks = max(1, usable // stride + 1)

    def _ensure_mmap(self):
        import mmap
        if self._mm is None:
            self._fp = open(self.data_path, "rb")
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        self._ensure_mmap()
        start = idx * self.stride
        end = start + self.seq_len + 1
        if end > self.file_size:
            # Tail chunk: read what we have, pad with 0
            raw = self._mm[start:self.file_size]
            buf = torch.zeros(self.seq_len + 1, dtype=torch.long)
            # PAD-offset: byte b → b + 1 (0 reserved for PAD)
            if raw:
                buf[:len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long() + 1
            return buf[:self.seq_len], buf[1:self.seq_len + 1]
        raw = self._mm[start:end]
        # Convert bytes → int64 tensor with PAD offset in one shot
        buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long() + 1
        return buf[:self.seq_len], buf[1:self.seq_len + 1]

    def __getstate__(self):
        # mmap handles are not picklable; recreate in worker.
        state = self.__dict__.copy()
        state["_mm"] = None
        state["_fp"] = None
        return state


class TokenLMDataset(torch.utils.data.Dataset):
    """Next-token prediction dataset for BPE downstream (in-memory)."""
    def __init__(self, tokenizer, texts, seq_len=256, stride=128):
        self.seq_len = seq_len
        all_ids = []
        for text in texts:
            if not text: continue
            ids = tokenizer.encode(text).ids
            all_ids.extend(ids)
            all_ids.append(tokenizer.token_to_id("<eos>") or 2)
        if not all_ids: all_ids = [0]
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.chunks = []
        for start in range(0, max(1, len(self.data) - seq_len), max(1, stride)):
            chunk = self.data[start:start + seq_len + 1]
            if chunk.numel() < seq_len + 1:
                chunk = torch.cat([chunk, torch.zeros(seq_len + 1 - chunk.numel(), dtype=torch.long)])
            self.chunks.append(chunk)
        if not self.chunks: self.chunks = [torch.zeros(seq_len + 1, dtype=torch.long)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return c[:self.seq_len], c[1:self.seq_len + 1]


class StreamingTokenLMDataset(torch.utils.data.Dataset):
    """Memory-mapped BPE-token dataset.

    Pre-tokenizes corpus once into a uint32 binary cache file on disk,
    then mmap-reads the cache. The cache stores raw token ids with line
    breaks marked by inserting <eos> (id from tokenizer, or fallback 2).

    For 22GB UTF-8 → ~8GB cache (3:1 BPE compression × 4 bytes/token).
    Cache is reused across runs (keyed by data_path + tokenizer file mtime).
    """
    def __init__(self, tokenizer, data_path: str, seq_len: int = 256, stride: int = 128,
                 cache_path: Optional[str] = None, line_batch: int = 4096):
        import os
        import struct
        self.seq_len = seq_len
        self.stride = stride
        self.tokenizer = tokenizer
        # Per-id container: tokens.json vocab fits in uint32 (we use 4 bytes)
        if cache_path is None:
            cache_path = data_path + ".bpe_ids.u32"
        self.cache_path = cache_path
        # Build cache if missing or stale
        if not os.path.exists(cache_path) or os.path.getmtime(cache_path) < os.path.getmtime(data_path):
            self._build_cache(data_path, cache_path, line_batch)
        self.n_tokens = os.path.getsize(cache_path) // 4
        if self.n_tokens < seq_len + 1:
            self.n_chunks = 1
        else:
            usable = self.n_tokens - (seq_len + 1)
            self.n_chunks = max(1, usable // stride + 1)
        self._mm = None
        self._fp = None

    def _build_cache(self, data_path: str, cache_path: str, line_batch: int):
        import time
        import logging
        log = logging.getLogger("e3.streaming_bpe")
        eos = self.tokenizer.token_to_id("<eos>")
        if eos is None:
            eos = 2
        log.info("Building BPE token cache: %s → %s", data_path, cache_path)
        t0 = time.time()
        n_lines = 0
        n_toks = 0
        BATCH_BYTES = 0
        # Use 32-bit little-endian unsigned ints; tokenizer vocab is ~8192,
        # well within uint16 — but uint32 is safer if user retrains larger.
        with open(data_path, encoding="utf-8") as fin, open(cache_path, "wb") as fout:
            batch = []
            for line in fin:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                batch.append(line)
                if len(batch) >= line_batch:
                    encs = self.tokenizer.encode_batch(batch)
                    for enc in encs:
                        ids = enc.ids
                        n_toks += len(ids) + 1
                        ids_with_eos = ids + [eos]
                        fout.write(_ids_to_uint32_bytes(ids_with_eos))
                    n_lines += len(batch)
                    batch.clear()
                    if n_lines % (line_batch * 25) == 0:
                        log.info("  cache: %d lines, %d tokens (%.1fs)",
                                 n_lines, n_toks, time.time() - t0)
            if batch:
                encs = self.tokenizer.encode_batch(batch)
                for enc in encs:
                    ids = enc.ids
                    n_toks += len(ids) + 1
                    ids_with_eos = ids + [eos]
                    fout.write(_ids_to_uint32_bytes(ids_with_eos))
                n_lines += len(batch)
        log.info("Cache built: %d lines, %d tokens (%.1fs)", n_lines, n_toks, time.time() - t0)

    def _ensure_mmap(self):
        import mmap
        if self._mm is None:
            self._fp = open(self.cache_path, "rb")
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        self._ensure_mmap()
        start_tok = idx * self.stride
        end_tok = start_tok + self.seq_len + 1
        start_b = start_tok * 4
        end_b = min(end_tok * 4, self.n_tokens * 4)
        if end_b - start_b < (self.seq_len + 1) * 4:
            raw = self._mm[start_b:end_b]
            ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
            buf = torch.zeros(self.seq_len + 1, dtype=torch.long)
            buf[:ids.numel()] = ids
            return buf[:self.seq_len], buf[1:self.seq_len + 1]
        raw = self._mm[start_b:end_b]
        ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        return ids[:self.seq_len], ids[1:self.seq_len + 1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        state["_fp"] = None
        return state


class OnTheFlyTokenLMDataset(torch.utils.data.Dataset):
    """Streaming BPE dataset without a full token-id cache.

    Each item reads a raw byte window from the source file, decodes it with
    replacement for partial UTF-8 boundaries, tokenizes that window, and pads
    or truncates to seq_len + 1 tokens. This keeps disk use O(1), which matters
    on AutoDL 50GB data disks where a full 22GB corpus plus BPE cache can fill
    the volume.
    """

    def __init__(
        self,
        tokenizer,
        data_path: str,
        seq_len: int = 256,
        stride: int = 128,
        bytes_per_token: int = 6,
    ):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.seq_len = seq_len
        self.stride = stride
        self.file_size = os.path.getsize(data_path)
        self.bytes_per_chunk = max((seq_len + 1) * bytes_per_token, 4096)
        self.byte_stride = max(stride * bytes_per_token, 1024)
        if self.file_size <= self.bytes_per_chunk:
            self.n_chunks = 1
        else:
            self.n_chunks = max(1, (self.file_size - self.bytes_per_chunk) // self.byte_stride + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = min(idx * self.byte_stride, max(0, self.file_size - 1))
        with open(self.data_path, "rb") as fh:
            fh.seek(start)
            raw = fh.read(self.bytes_per_chunk)
        text = raw.decode("utf-8", errors="replace")
        ids = self.tokenizer.encode(text).ids
        if len(ids) < self.seq_len + 1 and start + self.bytes_per_chunk < self.file_size:
            with open(self.data_path, "rb") as fh:
                fh.seek(start + self.bytes_per_chunk)
                text += fh.read(self.bytes_per_chunk).decode("utf-8", errors="replace")
            ids = self.tokenizer.encode(text).ids
        buf = torch.zeros(self.seq_len + 1, dtype=torch.long)
        if ids:
            ids_t = torch.tensor(ids[: self.seq_len + 1], dtype=torch.long)
            buf[: ids_t.numel()] = ids_t
        return buf[: self.seq_len], buf[1 : self.seq_len + 1]


def _ids_to_uint32_bytes(ids):
    import struct
    return struct.pack(f"<{len(ids)}I", *ids)


def _should_stream(data_path: Optional[str], max_lines: Optional[int]) -> bool:
    """Decide whether to use streaming dataset based on file size and max_lines."""
    import os
    if data_path is None or not os.path.exists(data_path):
        return False
    if max_lines is not None and max_lines <= 200000:
        return False
    return os.path.getsize(data_path) > _STREAMING_THRESHOLD


# ---------------------------------------------------------------------------
# Public tokenizer datasets (tiktoken / HF)
# ---------------------------------------------------------------------------

class PublicTokenDataset(torch.utils.data.Dataset):
    """In-memory token dataset for public tokenizers.

    Tokenizes texts on construction using the model's ``encode_text()`` method.
    """
    def __init__(self, model, texts, seq_len=256, stride=128):
        self.seq_len = seq_len
        all_ids = []
        for text in texts:
            if not text:
                continue
            ids = model.encode_text(text)
            all_ids.extend(ids)
            all_ids.append(model.pad_id)  # use pad_id as separator
        if not all_ids:
            all_ids = [model.pad_id]
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.chunks = []
        for start in range(0, max(1, len(self.data) - seq_len), max(1, stride)):
            chunk = self.data[start:start + seq_len + 1]
            if chunk.numel() < seq_len + 1:
                chunk = torch.cat([chunk, torch.zeros(seq_len + 1 - chunk.numel(), dtype=torch.long)])
            self.chunks.append(chunk)
        if not self.chunks:
            self.chunks = [torch.zeros(seq_len + 1, dtype=torch.long)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        return c[:self.seq_len], c[1:self.seq_len + 1]


class PublicTokenCacheDataset(torch.utils.data.Dataset):
    """Memory-mapped token cache for public tokenizers.

    Pre-tokenizes the corpus once into a uint32 binary cache file, then
    mmap-reads it. Same pattern as StreamingTokenLMDataset but uses the
    model's own ``encode_text()`` method.
    """
    def __init__(self, model, data_path, cache_path, seq_len=256, stride=128):
        import os
        self.seq_len = seq_len
        self.stride = stride
        self.cache_path = cache_path
        self.n_tokens = os.path.getsize(cache_path) // 4
        if self.n_tokens < seq_len + 1:
            self.n_chunks = 1
        else:
            usable = self.n_tokens - (seq_len + 1)
            self.n_chunks = max(1, usable // stride + 1)
        self._mm = None
        self._fp = None

    def _ensure_mmap(self):
        import mmap
        if self._mm is None:
            self._fp = open(self.cache_path, "rb")
            self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        self._ensure_mmap()
        start_tok = idx * self.stride
        end_tok = start_tok + self.seq_len + 1
        start_b = start_tok * 4
        end_b = min(end_tok * 4, self.n_tokens * 4)
        if end_b - start_b < (self.seq_len + 1) * 4:
            raw = self._mm[start_b:end_b]
            ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
            buf = torch.zeros(self.seq_len + 1, dtype=torch.long)
            buf[:ids.numel()] = ids
            return buf[:self.seq_len], buf[1:self.seq_len + 1]
        raw = self._mm[start_b:end_b]
        ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        return ids[:self.seq_len], ids[1:self.seq_len + 1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        state["_fp"] = None
        return state


def _build_public_token_cache(model, data_path, seq_len=256, stride=128):
    """Build uint32 cache from public tokenizer, return PublicTokenCacheDataset."""
    import os, time, struct
    cache_path = data_path + f".pubtok_{model.tokenizer_id.replace(':', '_')}.u32"
    if not os.path.exists(cache_path) or os.path.getmtime(cache_path) < os.path.getmtime(data_path):
        logger.info("Building public token cache: %s → %s", data_path, cache_path)
        t0 = time.time()
        n_lines, n_toks = 0, 0
        eos = model.pad_id  # use pad_id as line separator
        with open(data_path, encoding="utf-8") as fin, open(cache_path, "wb") as fout:
            batch = []
            for line in fin:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                batch.append(line)
                if len(batch) >= 4096:
                    for text in batch:
                        ids = model.encode_text(text)
                        n_toks += len(ids) + 1
                        fout.write(struct.pack(f"<{len(ids) + 1}I", *(ids + [eos])))
                    n_lines += len(batch)
                    batch.clear()
                    if n_lines % 100000 == 0:
                        logger.info("  cache: %d lines, %d tokens (%.1fs)", n_lines, n_toks, time.time() - t0)
            for text in batch:
                ids = model.encode_text(text)
                n_toks += len(ids) + 1
                fout.write(struct.pack(f"<{len(ids) + 1}I", *(ids + [eos])))
            n_lines += len(batch)
        logger.info("Public token cache built: %d lines, %d tokens (%.1fs)", n_lines, n_toks, time.time() - t0)
    return PublicTokenCacheDataset(model, data_path, cache_path, seq_len=seq_len, stride=stride)


# ===========================================================================
# Helpers
# ===========================================================================

def cosine_schedule(optimizer, warmup: int, total: int):
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# Training
# ===========================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    logger.info("Seed: %s", args.seed)

    # --- Build model ---
    if args.model == "flued":
        model = FLUEDDownstream(
            flued_ckpt=args.flued_ckpt,
            d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers, max_seq_len=args.max_seq_len,
        ).to(device)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("FLUED downstream: frozen=%s trainable=%s", f"{n_frozen:,}", f"{n_train:,}")

    elif args.model == "blt":
        model = BLTDownstream(
            blt_ckpt=args.blt_ckpt, bytelm_ckpt=args.bytelm_ckpt,
            entropy_theta=args.blt_entropy_theta,
            d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers, max_seq_len=args.max_seq_len,
        ).to(device)
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("BLT downstream: frozen=%s trainable=%s", f"{n_frozen:,}", f"{n_train:,}")

    elif args.model == "bpe":
        model = BPEDownstream(
            tokenizer_path=args.tokenizer_path,
            d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers, max_seq_len=args.max_seq_len,
        ).to(device)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("BPE downstream: trainable=%s vocab=%d", f"{n_train:,}", model.token_vocab)

    elif args.model == "fixed_patch":
        model = FixedPatchDownstream(
            patch_size=args.patch_size,
            d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers, max_seq_len=args.max_seq_len,
        ).to(device)
        n_train = model.count_parameters()
        logger.info("FixedPatch (size=%d) downstream: trainable=%s", args.patch_size, f"{n_train:,}")

    elif args.model == "byte":
        model = ByteDownstream(
            d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers, max_seq_len=args.max_seq_len,
        ).to(device)
        n_train = model.count_parameters()
        logger.info("Byte (no compression) downstream: trainable=%s", f"{n_train:,}")

    elif args.model == "public_tok":
        model = PublicTokenizerDownstream(
            tokenizer_id=args.public_tokenizer,
            d_model=args.d_model, nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers, max_seq_len=args.max_seq_len,
        ).to(device)
        n_train = model.count_parameters()
        logger.info("PublicTokenizer (%s) downstream: trainable=%s vocab=%d",
                    args.public_tokenizer, f"{n_train:,}", model.token_vocab)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # --- Data ---
    streaming = _should_stream(args.data_path, args.max_lines)
    if streaming:
        logger.info("Using STREAMING dataset (file > 256MB, max_lines unbounded). "
                    "No full in-memory load.")

    if args.model in ("flued", "blt", "fixed_patch", "byte"):
        seq_len = args.max_seq_len
        stride = max(1, seq_len // 2)
        vocab_size = 257
        if streaming:
            dataset = StreamingByteLMDataset(args.data_path, seq_len=seq_len, stride=stride)
            logger.info("StreamingByteLMDataset: file=%s, %d chunks (mmap)",
                        args.data_path, len(dataset))
        else:
            texts = load_texts(args.data_path, args.max_lines) if args.data_path else ["hello world"]
            logger.info("Loaded %d lines", len(texts))
            dataset = ByteLMDataset(texts, seq_len=seq_len, stride=stride)
    elif args.model == "public_tok":
        # Public tokenizer: use model's own tokenizer to build cache.
        vocab_size = model.token_vocab
        seq_len = args.max_seq_len
        stride = max(1, seq_len // 2)
        if streaming:
            dataset = _build_public_token_cache(
                args.data_path, model, seq_len=seq_len, stride=stride)
            logger.info("PublicTokenCacheDataset: %d chunks", len(dataset))
        else:
            texts = load_texts(args.data_path, args.max_lines) if args.data_path else ["hello world"]
            logger.info("Loaded %d lines", len(texts))
            dataset = PublicTokenDataset(model, texts, seq_len=seq_len, stride=stride)
    else:
        # BPE (in-domain trained tokenizer)
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        seq_len = args.max_seq_len
        stride = max(1, seq_len // 2)
        if streaming:
            dataset = OnTheFlyTokenLMDataset(tokenizer, args.data_path,
                                             seq_len=seq_len, stride=stride)
            logger.info("OnTheFlyTokenLMDataset: file=%s, %d chunks (no disk cache)",
                        args.data_path, len(dataset))
        else:
            texts = load_texts(args.data_path, args.max_lines) if args.data_path else ["hello world"]
            logger.info("Loaded %d lines", len(texts))
            dataset = TokenLMDataset(tokenizer, texts, seq_len=seq_len, stride=stride)

    n_eval = max(1, int(len(dataset) * 0.1))
    n_train_ds = max(1, len(dataset) - n_eval)
    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(dataset), generator=gen).tolist()
    train_ds = torch.utils.data.Subset(dataset, idx[:n_train_ds])
    eval_ds = torch.utils.data.Subset(dataset, idx[n_train_ds:])
    logger.info("Dataset: %d train / %d eval", n_train_ds, n_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, pin_memory=(device.type == "cuda"),
                              num_workers=2 if streaming else 0,
                              persistent_workers=streaming)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=(device.type == "cuda"),
                             num_workers=2 if streaming else 0,
                             persistent_workers=streaming)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )
    scheduler = cosine_schedule(optimizer, args.warmup_steps, args.max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # --- AMP ---
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    accum = args.grad_accum_steps

    # --- Checkpoint helpers ---
    ckpt_dir = getattr(args, "ckpt_dir", "checkpoints")
    ckpt_every = int(getattr(args, "ckpt_every", 1000) or 1000)
    ckpt_prefix = f"e3_{args.model}"
    os.makedirs(ckpt_dir, exist_ok=True)

    def _save_ckpt(step: int):
        # Only persist the TRAINABLE part — frozen encoders are loaded
        # from their own ckpts at startup, so saving them again wastes disk.
        trainable_state = {
            k: v for k, v in model.state_dict().items()
            # Keep all params that show up in optimizer.param_groups (i.e. trainable).
            # Simpler heuristic: keep things outside 'encoder.' / 'byte_lm.' / 'blt.'
            # which are the frozen submodule prefixes.
            if not (k.startswith("encoder.") or k.startswith("byte_lm.") or k.startswith("blt."))
        }
        state = {
            "global_step": step,
            "model_trainable": trainable_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "amp_dtype": getattr(args, "amp_dtype", "fp16"),
        }
        latest = os.path.join(ckpt_dir, f"{ckpt_prefix}_latest.pt")
        stepf  = os.path.join(ckpt_dir, f"{ckpt_prefix}_step{step:06d}.pt")
        tmp = latest + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, latest)
        if getattr(args, "save_step_ckpts", False):
            tmp = stepf + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, stepf)
        logger.info("Checkpoint saved → %s", stepf)

    # --- Resume ---
    global_step = 0
    resume_path = getattr(args, "resume", None)
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        # Load trainable submodules only (frozen encoders were already loaded).
        missing, unexpected = model.load_state_dict(ck["model_trainable"], strict=False)
        unexpected = [k for k in unexpected
                      if not (k.startswith("encoder.") or k.startswith("byte_lm.")
                              or k.startswith("blt."))]
        if unexpected:
            logger.warning("Resume: unexpected keys: %s", unexpected[:5])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if "scaler" in ck and use_amp and amp_dtype == torch.float16:
            scaler.load_state_dict(ck["scaler"])
        global_step = int(ck.get("global_step", 0))
        logger.info("Resumed from %s at step %d", resume_path, global_step)

    # --- Training loop ---
    model.train()
    train_iter = iter(train_loader)
    running_loss, running_bpb = 0.0, 0.0
    running_bytes = 0     # total target bytes (non-pad) for throughput
    skipped, grad_step = 0, 0
    optimizer.zero_grad()
    t0 = time.time()

    while global_step < args.max_steps:
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt = next(train_iter)
        src, tgt = src.to(device), tgt.to(device)

        with torch.autocast(device.type, amp_dtype, enabled=use_amp):
            logits, seg_lens = model(src)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1)) / accum
        metric_extra = seg_lens
        target_byte_counts = (tgt != 0).sum().item()
        if args.model in ("bpe", "public_tok") and hasattr(model, "token_byte_len"):
            metric_extra = model.token_byte_len[tgt]
            target_byte_counts = metric_extra.masked_fill(tgt == 0, 0).sum().item()

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_step += 1
        running_loss += loss.item() * accum
        running_bpb += bits_per_byte(logits.detach(), tgt, metric_extra, vocab_size)
        # Track real byte count for throughput (non-pad targets)
        running_bytes += target_byte_counts

        if grad_step < accum:
            continue

        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp and amp_dtype == torch.float16:
            s_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < s_before:
                skipped += 1; optimizer.zero_grad(); grad_step = 0; continue
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        grad_step = 0
        global_step += 1

        if global_step % 50 == 0:
            n = 50 * accum
            elapsed = time.time() - t0
            bytes_per_sec = running_bytes / max(1, elapsed)
            logger.info("step=%5d  loss=%.4f  bpb=%.3f  lr=%.2e  skip=%d  "
                        "%.1f step/min  %.1f KB/s",
                        global_step, running_loss / n, running_bpb / n,
                        scheduler.get_last_lr()[0], skipped,
                        global_step / max(1, elapsed) * 60,
                        bytes_per_sec / 1024)
            running_loss = running_bpb = 0.0
            running_bytes = 0
            skipped = 0

        if ckpt_every > 0 and global_step % ckpt_every == 0:
            _save_ckpt(global_step)

    # Always save final ckpt at end of training
    _save_ckpt(global_step)

    # --- Eval ---
    model.eval()
    total_loss, total_bpb, n_eval = 0.0, 0.0, 0
    total_kv_units = 0      # sum of segment/token counts (KV cache length proxy)
    total_eval_bytes = 0    # sum of original byte counts
    with torch.no_grad():
        for src, tgt in eval_loader:
            if n_eval >= 50: break
            src, tgt = src.to(device), tgt.to(device)
            with torch.autocast(device.type, amp_dtype, enabled=use_amp):
                logits, seg_lens = model(src)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            metric_extra = seg_lens
            eval_bytes = (tgt != 0).sum().item()
            if args.model in ("bpe", "public_tok") and hasattr(model, "token_byte_len"):
                metric_extra = model.token_byte_len[tgt]
                eval_bytes = metric_extra.masked_fill(tgt == 0, 0).sum().item()
            total_loss += loss.item()
            total_bpb += bits_per_byte(logits, tgt, metric_extra, vocab_size)
            # KV cache: for byte-level models, seg_lens gives #segments;
            # for token-level, it's #tokens (each token = 1 KV slot)
            if seg_lens is not None:
                if seg_lens.dim() == 2 and seg_lens.shape[1] < src.shape[1]:
                    # seg_lens [B, M] — byte-level segments
                    total_kv_units += (seg_lens > 0).sum().item()
                else:
                    # token_byte_lens [B, T] — token-level
                    total_kv_units += (tgt != 0).sum().item()
            else:
                total_kv_units += (tgt != 0).sum().item()
            total_eval_bytes += eval_bytes
            n_eval += 1

    kv_per_1k = (total_kv_units / max(1, total_eval_bytes)) * 1000 if total_eval_bytes > 0 else 0
    # FLOPs estimate for one forward pass
    flops_per_fwd = estimate_transformer_flops(
        d_model=args.d_model, dim_ff=args.dim_feedforward,
        num_layers=args.num_layers, vocab_size=vocab_size,
        seq_len=seq_len, batch_size=1,
    )
    logger.info("Eval: loss=%.4f  bpb=%.4f  steps=%d  time=%.1f min  KV/1KB=%.1f  FLOPs/fwd=%s",
                total_loss / max(1, n_eval), total_bpb / max(1, n_eval),
                global_step, (time.time() - t0) / 60, kv_per_1k,
                format_flops(flops_per_fwd))


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FLUED E3 Downstream LM Training")
    parser.add_argument("--model", choices=["flued", "blt", "bpe", "fixed_patch", "byte", "public_tok"],
                        required=True)
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")

    # Model
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)

    # FixedPatch / PublicTokenizer specific
    parser.add_argument("--patch-size", type=int, default=4,
                        help="Fixed patch size in bytes (for --model fixed_patch).")
    parser.add_argument("--public-tokenizer", type=str, default="tiktoken:cl100k_base",
                        help="Public tokenizer ID (for --model public_tok). "
                        "Format: 'tiktoken:<name>' or 'hf:<model_name>'.")

    # Checkpoints
    parser.add_argument("--flued-ckpt", default="checkpoints/e1_latest.pt")
    parser.add_argument("--blt-ckpt", default="checkpoints/blt_latest.pt")
    parser.add_argument("--bytelm-ckpt", default="checkpoints/bytel_m_latest.pt")
    parser.add_argument("--blt-entropy-theta", type=float, default=0.3,
                        help="Entropy threshold for BLT patch boundaries.")
    parser.add_argument("--tokenizer-path", default="checkpoints/bpe_tokenizer/tokenizer.json")

    # Training
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="fp16")

    # Data
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model init and training shuffle.")

    # Checkpoint persistence
    parser.add_argument("--ckpt-dir", default="checkpoints",
                        help="Directory to write e3_<model>_latest.pt + step ckpts.")
    parser.add_argument("--ckpt-every", type=int, default=1000,
                        help="Save a step checkpoint every N optimizer steps. Set 0 to disable.")
    parser.add_argument("--save-step-ckpts", action="store_true",
                        help="Also keep numbered step checkpoints. Latest is always saved.")
    parser.add_argument("--resume", default=None,
                        help="Path to an e3_<model>_*.pt to resume from (loads only trainable params).")

    args = parser.parse_args()
    defaults = PRESETS.get(args.preset, PRESETS["smoke"]).copy()
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


if __name__ == "__main__":
    train(parse_args())
