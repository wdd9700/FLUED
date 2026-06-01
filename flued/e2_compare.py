"""
FLUED E2 Comparison Runner — dynamic semantic units vs. fixed tokenizers.

Goal
----
Compare FLUED's dynamic segmentation against external tokenizer baselines
on perplexity and (optionally) cloze accuracy.

    Models: flued | sentencepiece | tiktoken | blt

Usage
-----
    # Quick smoke test (random-init eval, no training)
    python -m flued.e2_compare --preset smoke --models flued,sentencepiece,tiktoken

    # With optional deps missing, those models are skipped gracefully
    python -m flued.e2_compare --models flued

    # Train before eval
    python -m flued.e2_compare \\
        --models flued \\
        --mode train_eval \\
        --train-steps 1000 \\
        --data-path corpus.txt

    # Load checkpoint
    python -m flued.e2_compare \\
        --models flued \\
        --checkpoint flued_ckpt.pt \\
        --mode eval_only

    # Save results
    python -m flued.e2_compare --models flued,blt \\
        --output-json e2_results.json \\
        --output-csv  e2_results.csv

Notes
-----
* sentencepiece uses hard_vocab_limit=False to handle small corpora safely (P1-3).
* BLT uses vocab_size=257 (PAD-offset compatible with FLUED v0.4).
* E2 cloze scoring uses next-token log-probability of the masked token given
  the prefix — this is a reconstruction-based approximation, not conditional
  sampling.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("flued.e2")

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict] = {
    "smoke": {
        "d_model": 64,
        "nhead": 4,
        "dim_feedforward": 128,
        "num_layers": 2,
        "max_seq_len": 32,
        "dropout": 0.0,
        "batch_size": 4,
        "seq_len": 32,
        "stride": 16,
        "train_steps": 100,
        "lr": 3e-4,
        "sentencepiece_vocab_size": 256,
        "blt_patch_size": 4,
    },
    "medium": {
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 1024,
        "num_layers": 4,
        "max_seq_len": 256,
        "dropout": 0.0,
        "batch_size": 16,
        "seq_len": 128,
        "stride": 64,
        "train_steps": 2000,
        "lr": 1e-4,
        "sentencepiece_vocab_size": 8192,
        "blt_patch_size": 4,
    },
}

# ---------------------------------------------------------------------------
# Tiny Chinese logic samples for cloze testing
# ---------------------------------------------------------------------------

def tiny_chinese_logic_samples() -> List[Dict]:
    """Return a small set of cloze-style test items.

    Each item is:
        prefix: str   — context before the masked token
        target: str   — the correct completion (single token / word)
        category: str — label for grouping results
    """
    return [
        # Chengyu / semantic fill-in
        {"prefix": "马到", "target": "成功", "category": "chengyu"},
        {"prefix": "一石", "target": "二鸟", "category": "chengyu"},
        {"prefix": "半途", "target": "而废", "category": "chengyu"},
        # Connective prediction
        {"prefix": "虽然天气不好，", "target": "但是", "category": "connective"},
        {"prefix": "不仅如此，", "target": "而且", "category": "connective"},
        # Anaphora
        {"prefix": "小明喜欢足球，", "target": "他", "category": "anaphora"},
        {"prefix": "这本书很有趣，", "target": "它", "category": "anaphora"},
        # Simple English fill-in
        {"prefix": "The quick brown fox ", "target": "jumps", "category": "english"},
        {"prefix": "Language models predict the next ", "target": "token", "category": "english"},
    ]


# ---------------------------------------------------------------------------
# Perplexity computation (model-agnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    loader: DataLoader,
    vocab_size: int,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Compute per-token cross-entropy perplexity.

    Works for any model that accepts (src,) or (src, tgt) and returns
    (logits, _) where logits is [B, T, V].
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        src = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        result = model(src)
        logits = result[0]
        loss = criterion(logits.view(-1, logits.size(-1)), src.view(-1))
        total_loss += loss.item()
        total_tokens += (src != 0).sum().item()

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 20))  # clip to avoid overflow


# ---------------------------------------------------------------------------
# Cloze accuracy (prefix → target, reconstruction-based approximation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cloze_accuracy(
    model: nn.Module,
    samples: List[Dict],
    encode_fn,
    vocab_size: int,
    device: torch.device,
    max_seq_len: int = 32,
    byte_level: bool = True,
) -> float:
    """Compute reconstruction-based cloze accuracy.

    For each sample, encodes `prefix + target` as a byte sequence, runs the
    autoencoder, and checks whether the highest-probability prediction at
    the last byte position of `prefix` matches the first byte of `target`.

    .. warning::
        **BYTE-LEVEL MODELS ONLY** (FLUED, BLT).
        This metric is *not comparable* across model types.
        For sentencepiece / tiktoken, token boundaries do not align with byte
        positions, so the prefix-end byte position has no meaningful
        relationship to the first token of `target` in those vocabularies.
        Pass ``byte_level=False`` to suppress computation and get ``nan``.

    This is a byte-level reconstruction approximation, not exact cloze scoring.
    """
    if not byte_level:
        logger.warning(
            "compute_cloze_accuracy: skipped for non-byte-level model "
            "(token boundaries don’t align with byte positions — "
            "results would be an artifact, not a fair comparison)."
        )
        return float("nan")

    model.eval()
    correct = 0
    total = 0

    for sample in samples:
        prefix_bytes = list((sample["prefix"]).encode("utf-8"))
        target_bytes = list((sample["target"]).encode("utf-8"))
        full_bytes = (prefix_bytes + target_bytes)[:max_seq_len]
        if not full_bytes:
            continue

        ids = encode_fn(full_bytes)
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        result = model(src)
        logits = result[0]  # [1, T, V]

        pos = len(prefix_bytes) - 1
        if pos < 0 or pos >= logits.size(1):
            continue

        pred_id = logits[0, pos].argmax().item()
        target_id = encode_fn([target_bytes[0]])[0]
        if pred_id == target_id:
            correct += 1
        total += 1

    return correct / max(1, total)


# ---------------------------------------------------------------------------
# FLUED adapter
# ---------------------------------------------------------------------------

def _make_flued_adapter(d_model, nhead, dim_feedforward, num_layers,
                         max_seq_len, dropout, device):
    from flued.model import FLUEDAutoencoder, VOCAB_SIZE

    model = FLUEDAutoencoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    def encode_fn(raw_bytes):
        return [b + 1 for b in raw_bytes]  # PAD-offset

    return model, encode_fn, VOCAB_SIZE


# ---------------------------------------------------------------------------
# BLT adapter
# ---------------------------------------------------------------------------

def _make_blt_adapter(d_model, nhead, dim_feedforward, num_layers,
                       max_seq_len, dropout, blt_patch_size, device):
    from blt_baseline.model import BLTAutoencoder
    VOCAB = 257

    model = BLTAutoencoder(
        vocab_size=VOCAB,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
        patch_mode="entropy",
        entropy_theta=3.5,
    ).to(device)

    def encode_fn(raw_bytes):
        return [b + 1 for b in raw_bytes]  # PAD-offset (matches FLUED v0.4)

    return model, encode_fn, VOCAB


# ---------------------------------------------------------------------------
# sentencepiece adapter
# ---------------------------------------------------------------------------

def _make_sentencepiece_adapter(vocab_size, d_model, nhead, dim_feedforward,
                                  num_layers, max_seq_len, dropout, texts, device):
    try:
        import sentencepiece as spm
    except ImportError:
        return None, None, None, "sentencepiece not installed (pip install sentencepiece)"

    from bpe_baseline.model import BPETransformerAutoencoder

    # Train a tiny SPM model on a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     encoding="utf-8", delete=False) as tmp:
        tmp.write("\n".join(texts))
        tmp_path = tmp.name

    model_prefix = tmp_path + ".spm"
    try:
        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            hard_vocab_limit=False,   # P1-3: safe for small corpora
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        sp = spm.SentencePieceProcessor()
        sp.Load(model_prefix + ".model")
        actual_vocab = sp.GetPieceSize()
    except Exception as exc:
        return None, None, None, f"sentencepiece training failed: {exc}"
    finally:
        os.unlink(tmp_path)

    model = BPETransformerAutoencoder(
        vocab_size=actual_vocab,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    def encode_fn(raw_bytes):
        text = bytes(raw_bytes).decode("utf-8", errors="replace")
        return sp.EncodeAsIds(text)

    return model, encode_fn, actual_vocab, None


# ---------------------------------------------------------------------------
# tiktoken adapter
# ---------------------------------------------------------------------------

def _make_tiktoken_adapter(d_model, nhead, dim_feedforward, num_layers,
                             max_seq_len, dropout, device):
    try:
        import tiktoken
    except ImportError:
        return None, None, None, "tiktoken not installed (pip install tiktoken)"

    from bpe_baseline.model import BPETransformerAutoencoder

    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab

    model = BPETransformerAutoencoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    def encode_fn(raw_bytes):
        text = bytes(raw_bytes).decode("utf-8", errors="replace")
        return enc.encode(text)

    return model, encode_fn, vocab_size, None


# ---------------------------------------------------------------------------
# Short training loop
# ---------------------------------------------------------------------------

def _train_model(model, loader, train_steps, lr, device):
    """Run a short training loop on reconstruction objective."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model.train()
    train_iter = iter(loader)
    for step in range(train_steps):
        try:
            src, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            src, tgt = next(train_iter)
        src = src.to(device)
        optimizer.zero_grad()
        result = model(src)
        logits = result[0]
        aux = result[1]
        if isinstance(aux, dict):
            aux_loss = aux.get("compression_loss", torch.tensor(0.0, device=device))
        else:
            aux_loss = aux
        loss = criterion(logits.view(-1, logits.size(-1)), src.view(-1)) + aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            logger.info("  train step %d/%d  loss=%.4f", step + 1, train_steps, loss.item())


# ---------------------------------------------------------------------------
# Main compare runner
# ---------------------------------------------------------------------------

def run_e2(args: argparse.Namespace) -> List[Dict]:
    """Execute E2 comparison.  Returns list of result dicts."""
    from flued.data import ByteReconstructionDataset, safe_train_eval_split, STUB_CORPUS

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # --- Load text corpus ---
    texts: Optional[List[str]] = None
    if args.data_path:
        with open(args.data_path, encoding="utf-8") as fh:
            texts = fh.readlines()
        logger.info("Loaded %d lines from %s", len(texts), args.data_path)
    else:
        texts = STUB_CORPUS

    # --- Dataset (byte-reconstruction, PAD-offset) ---
    dataset = ByteReconstructionDataset(
        texts=texts,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.1, seed=42)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=len(ds) > args.batch_size,
            pin_memory=(device_str == "cuda"),
        )

    train_loader = _loader(train_ds, shuffle=True)
    eval_loader = _loader(eval_ds, shuffle=False)

    cloze_samples = tiny_chinese_logic_samples()
    requested = [m.strip() for m in args.models.split(",")]
    results: List[Dict] = []

    for model_name in requested:
        logger.info("=== %s ===", model_name)
        row: Dict = {"model": model_name, "status": "ok"}

        # --- Build adapter ---
        error_msg = None
        if model_name == "flued":
            model, encode_fn, vocab_size = _make_flued_adapter(
                args.d_model, args.nhead, args.dim_feedforward, args.num_layers,
                args.max_seq_len, args.dropout, device,
            )
        elif model_name == "blt":
            model, encode_fn, vocab_size = _make_blt_adapter(
                args.d_model, args.nhead, args.dim_feedforward, args.num_layers,
                args.max_seq_len, args.dropout, args.blt_patch_size, device,
            )
        elif model_name == "sentencepiece":
            model, encode_fn, vocab_size, error_msg = _make_sentencepiece_adapter(
                args.sentencepiece_vocab_size, args.d_model, args.nhead,
                args.dim_feedforward, args.num_layers, args.max_seq_len,
                args.dropout, texts, device,
            )
        elif model_name == "tiktoken":
            model, encode_fn, vocab_size, error_msg = _make_tiktoken_adapter(
                args.d_model, args.nhead, args.dim_feedforward, args.num_layers,
                args.max_seq_len, args.dropout, device,
            )
        else:
            row["status"] = "error"
            row["detail"] = f"Unknown model name: {model_name}"
            results.append(row)
            continue

        if error_msg or model is None:
            row["status"] = "skipped"
            row["detail"] = error_msg or "adapter returned None"
            logger.warning("Skipping %s: %s", model_name, row["detail"])
            results.append(row)
            continue

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  %s params=%s", model_name, f"{n_params:,}")

        # --- Optional checkpoint load ---
        if args.checkpoint and model_name == "flued":
            try:
                ckpt = torch.load(args.checkpoint, map_location=device)
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state)
                logger.info("  Loaded checkpoint: %s", args.checkpoint)
            except Exception as exc:
                logger.warning("  Failed to load checkpoint: %s", exc)

        # --- Optional training ---
        if args.mode == "train_eval" and args.train_steps > 0:
            logger.info("  Training for %d steps …", args.train_steps)
            _train_model(model, train_loader, args.train_steps, args.lr, device)

        # --- Save checkpoint ---
        if args.save_checkpoint and model_name == "flued":
            torch.save({"model_state_dict": model.state_dict()}, args.save_checkpoint)
            logger.info("  Checkpoint saved to %s", args.save_checkpoint)

        # --- Evaluation ---
        ppl = compute_perplexity(model, eval_loader, vocab_size, device)
        byte_level = model_name in ("flued", "blt")
        cloze = compute_cloze_accuracy(
            model, cloze_samples, encode_fn, vocab_size, device, args.max_seq_len,
            byte_level=byte_level,
        )

        m_over_n = None
        if model_name == "flued":
            model.eval()
            with torch.no_grad():
                for src, _ in eval_loader:
                    src = src.to(device)
                    _, metrics = model(src)
                    m_over_n = metrics.get("m_over_n")
                    break

        row.update({
            "perplexity": round(ppl, 4),
            "m_over_n": round(m_over_n, 4) if m_over_n is not None else None,
            "cloze_accuracy": None if math.isnan(cloze) else round(cloze, 4),
            "n_params": n_params,
        })
        _cloze_str = "N/A (non-byte-level)" if math.isnan(cloze) else f"{cloze:.4f}"
        logger.info(
            "  ppl=%.2f  cloze=%s  m/n=%s",
            ppl, _cloze_str, f"{m_over_n:.4f}" if m_over_n is not None else "N/A",
        )
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUED E2 — compare dynamic segmentation vs. tokenizer baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default=None,
        help="Named preset (values overridden by explicit flags)",
    )
    parser.add_argument(
        "--models", default="flued",
        help="Comma-separated list: flued,sentencepiece,tiktoken,blt",
    )
    parser.add_argument(
        "--mode", choices=["eval_only", "train_eval"], default="eval_only",
        help="eval_only: just evaluate; train_eval: train then evaluate",
    )

    # Model architecture
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--blt-patch-size", type=int, default=None)

    # Training
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None)

    # Data
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--device", default="cpu")

    # Baseline-specific
    parser.add_argument("--sentencepiece-vocab-size", type=int, default=None)

    # Checkpoint
    parser.add_argument("--checkpoint", default=None, help="Load model checkpoint (FLUED only)")
    parser.add_argument("--save-checkpoint", default=None, help="Save FLUED checkpoint after training")

    # Output
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)

    args = parser.parse_args()

    # Apply preset
    preset_name = args.preset or "smoke"
    defaults = PRESETS.get(preset_name, PRESETS["smoke"]).copy()
    for key, val in defaults.items():
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    # Hard fallbacks
    for attr, val in [
        ("d_model", 64), ("nhead", 4), ("dim_feedforward", 128),
        ("num_layers", 2), ("max_seq_len", 32), ("dropout", 0.0),
        ("batch_size", 4), ("seq_len", 32), ("stride", 16),
        ("train_steps", 0), ("blt_patch_size", 4),
        ("sentencepiece_vocab_size", 256),
    ]:
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    return args


def main() -> None:
    args = _parse_args()
    results = run_e2(args)

    # Print summary table
    print("\n=== E2 Results ===")
    header = f"{'Model':<20} {'Status':<10} {'PPL':>10} {'m/n':>8} {'Cloze':>8} {'Params':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<20} {r.get('status',''):<10} "
            f"{r.get('perplexity', 'N/A'):>10} "
            f"{str(r.get('m_over_n', 'N/A')):>8} "
            f"{str(r.get('cloze_accuracy', 'N/A')):>8} "
            f"{str(r.get('n_params', 'N/A')):>12}"
        )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("JSON written to %s", args.output_json)

    if args.output_csv:
        if results:
            with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logger.info("CSV written to %s", args.output_csv)


if __name__ == "__main__":
    main()
