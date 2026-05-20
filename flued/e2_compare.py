"""E2 local comparison runner: FLUED vs sentencepiece/tiktoken token baselines."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bpe_baseline.model import BPETransformerAutoencoder
from flued.config import ModelConfig
from flued.data import (
    BYTE_VOCAB_SIZE,
    PAD_ID,
    STUB_CORPUS,
    ByteReconstructionDataset,
    ClozeItem,
    get_dataloader,
    text_to_byte_ids,
    tiny_chinese_logic_samples,
)
from flued.model import FLUEDAutoencoder

logger = logging.getLogger("flued.e2_compare")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


E2_PRESETS = {
    "smoke": dict(d_model=128, nhead=4, dim_feedforward=256, num_layers=2, seq_len=96, batch_size=2),
    "small": dict(d_model=256, nhead=8, dim_feedforward=1024, num_layers=4, seq_len=256, batch_size=2),
    "class300m_16gb": dict(d_model=896, nhead=14, dim_feedforward=3584, num_layers=12, seq_len=256, batch_size=1),
}


class OptionalDependencyError(RuntimeError):
    pass


class BaseAdapter:
    name: str = "base"

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError


class ByteAdapter(BaseAdapter):
    name = "flued-bytes"

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE

    def encode(self, text: str) -> List[int]:
        return text_to_byte_ids(text)


class SentencePieceAdapter(BaseAdapter):
    name = "sentencepiece"

    def __init__(self, vocab_size: int, texts: List[str], model_prefix: Optional[str] = None) -> None:
        try:
            self.spm = importlib.import_module("sentencepiece")
        except ImportError as exc:
            raise OptionalDependencyError(
                "sentencepiece baseline selected but package is missing. Install with: pip install sentencepiece"
            ) from exc

        self._vocab_size = vocab_size
        # If model_prefix points to an existing model, training corpus texts are ignored.
        tmp_dir = tempfile.mkdtemp(prefix="flued_spm_")
        self._tmp_dir: Optional[str] = tmp_dir if model_prefix is None else None
        safe_prefix = model_prefix or os.path.join(tmp_dir, "spm")
        self.model_file = f"{safe_prefix}.model"
        if not os.path.exists(self.model_file):
            input_path = os.path.join(tmp_dir, "corpus.txt")
            with open(input_path, "w", encoding="utf-8") as fh:
                for line in texts:
                    fh.write(line + "\n")
            self.spm.SentencePieceTrainer.Train(
                input=input_path,
                model_prefix=self.model_file[:-6],
                vocab_size=vocab_size,
                model_type="bpe",
                bos_id=-1,
                eos_id=-1,
                pad_id=0,
                unk_id=1,
            )
        self.processor = self.spm.SentencePieceProcessor(model_file=self.model_file)

    def __del__(self) -> None:
        if self._tmp_dir and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str) -> List[int]:
        return list(self.processor.encode(text, out_type=int))


class TikTokenAdapter(BaseAdapter):
    name = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            tiktoken = importlib.import_module("tiktoken")
        except ImportError as exc:
            raise OptionalDependencyError(
                "tiktoken baseline selected but package is missing. Install with: pip install tiktoken"
            ) from exc
        self.encoding = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return int(self.encoding.n_vocab)

    def encode(self, text: str) -> List[int]:
        return list(self.encoding.encode(text))


class TokenIdReconstructionDataset(Dataset):
    def __init__(self, tokenized: List[List[int]], seq_len: int, pad_id: int = 0) -> None:
        stream: List[int] = []
        for seq in tokenized:
            stream.extend(seq)
            stream.append(pad_id)
        if not stream:
            stream = [pad_id]

        self.seq_len = seq_len
        self.pad_id = pad_id
        data = torch.tensor(stream, dtype=torch.long)
        self.chunks: List[torch.Tensor] = []
        stride = max(1, seq_len // 2)
        for start in range(0, max(1, len(data) - seq_len + 1), stride):
            chunk = data[start : start + seq_len]
            if chunk.numel() < seq_len:
                pad = torch.full((seq_len - chunk.numel(),), pad_id, dtype=torch.long)
                chunk = torch.cat([chunk, pad], dim=0)
            self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        src = self.chunks[idx]
        return src, src.clone()


@dataclass
class ResultRow:
    model: str
    status: str
    perplexity: Optional[float]
    m_over_n: Optional[float]
    cloze_accuracy: Optional[float]
    detail: str


def build_e2_preset_config(preset: str = "smoke") -> dict:
    if preset not in E2_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    return dict(E2_PRESETS[preset])


def build_adapter(model_name: str, vocab_size: int, texts: List[str], tiktoken_encoding: str) -> BaseAdapter:
    if model_name == "flued":
        return ByteAdapter()
    if model_name == "sentencepiece":
        return SentencePieceAdapter(vocab_size=vocab_size, texts=texts)
    if model_name == "tiktoken":
        return TikTokenAdapter(encoding_name=tiktoken_encoding)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_model_for_adapter(model_name: str, cfg: dict, adapter: BaseAdapter) -> torch.nn.Module:
    if model_name == "flued":
        return FLUEDAutoencoder(
            vocab_size=BYTE_VOCAB_SIZE,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["seq_len"],
            dropout=0.0,
        )
    return BPETransformerAutoencoder(
        vocab_size=adapter.vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        num_encoder_layers=cfg["num_layers"],
        num_decoder_layers=cfg["num_layers"],
        max_seq_len=cfg["seq_len"],
        dropout=0.0,
    )


def build_dataset_for_adapter(model_name: str, adapter: BaseAdapter, texts: List[str], seq_len: int) -> Dataset:
    if model_name == "flued":
        return ByteReconstructionDataset(texts=texts, seq_len=seq_len)
    tokenized = [adapter.encode(t) for t in texts]
    return TokenIdReconstructionDataset(tokenized, seq_len=seq_len)


def evaluate_perplexity(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20
) -> Tuple[float, Optional[float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    total_mn = 0.0
    count = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            logits, metrics = model(src, tgt)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += float(loss.item())
            if isinstance(metrics, dict) and "m_over_n" in metrics:
                total_mn += float(metrics["m_over_n"].mean().item())
            count += 1

    if count == 0:
        return float("inf"), None
    avg_loss = total_loss / count
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    avg_mn = (total_mn / count) if total_mn else None
    return ppl, avg_mn


def score_cloze(model: torch.nn.Module, adapter: BaseAdapter, item: ClozeItem, device: torch.device) -> int:
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    losses: List[float] = []
    model.eval()
    for option in item.options:
        text = item.text.replace("___", option)
        ids = adapter.encode(text)
        if not ids:
            losses.append(float("inf"))
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x, x)
            loss = criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        losses.append(float(loss.item()))
    return int(min(range(len(losses)), key=lambda i: losses[i]))


def evaluate_logic_tasks(
    model: torch.nn.Module, adapter: BaseAdapter, task_items: Dict[str, List[ClozeItem]], device: torch.device
) -> float:
    total = 0
    correct = 0
    for items in task_items.values():
        for item in items:
            pred = score_cloze(model, adapter, item, device)
            correct += int(pred == item.answer)
            total += 1
    return (correct / total) if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2 FLUED vs sentencepiece/tiktoken local comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=list(E2_PRESETS.keys()), default="smoke")
    parser.add_argument("--models", default="flued,sentencepiece,tiktoken")
    parser.add_argument("--data-path", default=None, help="Optional corpus text file")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--sentencepiece-vocab-size", type=int, default=8192)
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_e2_preset_config(args.preset)

    texts = STUB_CORPUS
    if args.data_path:
        with open(args.data_path, encoding="utf-8") as fh:
            texts = [line.rstrip("\n") for line in fh if line.strip()]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        logger.warning("CUDA requested but unavailable, using CPU.")

    task_items = tiny_chinese_logic_samples()
    rows: List[ResultRow] = []

    for model_name in models:
        try:
            adapter = build_adapter(
                model_name=model_name,
                vocab_size=args.sentencepiece_vocab_size,
                texts=texts,
                tiktoken_encoding=args.tiktoken_encoding,
            )
        except OptionalDependencyError as exc:
            rows.append(
                ResultRow(
                    model=model_name,
                    status="skipped_missing_dependency",
                    perplexity=None,
                    m_over_n=None,
                    cloze_accuracy=None,
                    detail=str(exc),
                )
            )
            continue

        dataset = build_dataset_for_adapter(model_name, adapter, texts, seq_len=cfg["seq_len"])
        loader = get_dataloader(dataset, batch_size=cfg["batch_size"], shuffle=False)
        model = build_model_for_adapter(model_name, cfg, adapter).to(device)

        ppl, m_over_n = evaluate_perplexity(model, loader, device, max_batches=args.max_batches)
        cloze_acc = evaluate_logic_tasks(model, adapter, task_items, device)
        rows.append(
            ResultRow(
                model=model_name,
                status="ok",
                perplexity=ppl,
                m_over_n=m_over_n,
                cloze_accuracy=cloze_acc,
                detail=f"vocab_size={adapter.vocab_size}",
            )
        )

    result_dicts = [row.__dict__ for row in rows]
    print(json.dumps(result_dicts, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result_dicts, fh, ensure_ascii=False, indent=2)

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(result_dicts[0].keys()) if result_dicts else ["model"])
            writer.writeheader()
            writer.writerows(result_dicts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
