"""
eval_segmentation.py — Unified segmentation / compression evaluation.

Stage 1 of the FLUED experimental pipeline: measures the "compressor" quality
independent of downstream LM performance.

Methods evaluated
------------------
  flued         FLUED dynamic boundary detection (soft cumsum)
  bpe_8k        In-domain BPE tokenizer, 8K vocab
  bpe_16k       In-domain BPE tokenizer, 16K vocab
  bpe_32k       In-domain BPE tokenizer, 32K vocab
  fixed_N       Fixed N-byte patches (N ∈ {2, 4, 6, 8})
  public_tok    Public tokenizer (tiktoken cl100k_base or HF)
  blt           BLT entropy-based patching (requires ByteLM ckpt)

Metrics (per method, aggregated over eval corpus)
--------------------------------------------------
  total_bytes        Total UTF-8 bytes processed
  total_units        Total segments / tokens / patches
  bytes_per_unit     Mean bytes per unit (↑ = better compression)
  units_per_byte     Mean units per byte (↓ = fewer units)
  compression_ratio  = total_bytes / total_units
  utf8_violation     Fraction of units that split a multi-byte UTF-8 character
  cjk_bytes_pu       Mean CJK bytes per unit
  ascii_bytes_pu     Mean ASCII bytes per unit
  code_bytes_pu      Mean code (operators + digits) bytes per unit

Usage
-----
  python eval_segmentation.py \
    --data-path corpus.txt --max-lines 5000 \
    --methods flued,bpe_8k,bpe_32k,fixed_4,fixed_8,public_tok \
    --flued-ckpt checkpoints/e1_latest.pt \
    --output-json seg_results.json --output-csv seg_results.csv
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("eval.seg")

# ---------------------------------------------------------------------------
# Byte classification
# ---------------------------------------------------------------------------

def _byte_type(b: int) -> str:
    """Classify a single UTF-8 byte (0-255) into a coarse type."""
    # UTF-8 leading byte ranges
    if b < 0x80:
        # ASCII
        c = chr(b)
        if c.isdigit():
            return "digit"
        if c in "+-*/%=<>!&|^~@#$\\:;.,?()[]{}`\"'":
            return "op"
        if c.isspace():
            return "space"
        return "ascii"
    # UTF-8 continuation byte (10xxxxxx)
    if 0x80 <= b < 0xC0:
        return "utf8_cont"
    # UTF-8 leading bytes for multi-byte sequences
    if 0xC0 <= b < 0xE0:
        return "utf8_lead2"
    if 0xE0 <= b < 0xF0:
        return "utf8_lead3"  # CJK lives here
    if 0xF0 <= b < 0xF8:
        return "utf8_lead4"
    return "other"


def _is_cjk_lead(b: int) -> bool:
    """Check if byte starts a CJK character (U+4E00–U+9FFF CJK Unified)."""
    # CJK Unified Ideographs: E4 B8 80 – E9 BE BF in UTF-8
    # Lead byte range: 0xE4–0xE9
    return 0xE4 <= b <= 0xE9


def _classify_bytes(byte_seq: List[int]) -> Dict[str, int]:
    """Count bytes by type in a unit (segment/token/patch)."""
    counts = defaultdict(int)
    for b in byte_seq:
        bt = _byte_type(b)
        counts[bt] += 1
    # Merge subtypes
    cjk = 0
    for i, b in enumerate(byte_seq):
        if _is_cjk_lead(b):
            # A CJK char is 3 bytes in UTF-8 (lead + 2 continuation)
            cjk += 3
    counts["cjk"] = cjk
    return dict(counts)


def _detect_utf8_violation(byte_seq: List[int]) -> bool:
    """Check if a unit splits a multi-byte UTF-8 character.

    A violation occurs when:
      - A continuation byte (10xxxxxx) appears without a preceding lead byte
        within the same unit, OR
      - A lead byte indicates N continuation bytes but the unit ends before
        all continuation bytes arrive.
    """
    expected_cont = 0
    for b in byte_seq:
        if expected_cont > 0:
            if 0x80 <= b < 0xC0:
                expected_cont -= 1
            else:
                return True  # missing continuation byte
        else:
            if 0xC0 <= b < 0xE0:
                expected_cont = 1
            elif 0xE0 <= b < 0xF0:
                expected_cont = 2
            elif 0xF0 <= b < 0xF8:
                expected_cont = 3
            elif 0x80 <= b < 0xC0:
                return True  # stray continuation byte
    return expected_cont > 0  # True if truncated


# ---------------------------------------------------------------------------
# Method: FLUED
# ---------------------------------------------------------------------------

def eval_flued(ckpt_path: str, raw_bytes_list: List[bytes],
               device: torch.device) -> Dict:
    """Evaluate FLUED segmentation on a list of raw UTF-8 byte strings."""
    from flued.model import FLUEDAutoencoder

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_d = ckpt["model"]["embedding.weight"].shape[1]
    ckpt_ff = ckpt["model"]["blocks.0.ff1.weight"].shape[0]
    ckpt_nhead = ckpt_d // 64

    model = FLUEDAutoencoder(
        d_model=ckpt_d, nhead=ckpt_nhead, dim_feedforward=ckpt_ff,
        num_layers=24, max_seq_len=512, dropout=0.0,
        lambda_var=0.5, lambda_entropy=0.05, lambda_utf8=0.02, lambda_type=0.05,
        compression_weight=0.1, target_compression=0.3,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    total_bytes = 0
    total_units = 0
    utf8_violations = 0
    cjk_bytes = 0
    ascii_bytes = 0
    code_bytes = 0  # op + digit

    with torch.no_grad():
        for raw in raw_bytes_list:
            if len(raw) < 2:
                continue
            # Truncate to 512 bytes
            raw_trunc = raw[:512]
            # PAD-offset encoding
            ids = torch.tensor([b + 1 for b in raw_trunc], dtype=torch.long,
                               device=device).unsqueeze(0)  # [1, T]
            pad_mask = torch.zeros_like(ids, dtype=torch.bool)

            _, metrics = model.encode(ids, pad_mask, skip_hard=True)
            bp = metrics["boundary_probs"].float().squeeze(0)  # [T]

            # Soft cumsum segmentation
            seg_ids = bp.cumsum(dim=0).long()
            seg_ids = seg_ids - seg_ids[0]
            M = int(seg_ids.max().item()) + 1
            total_units += M

            # Per-segment analysis
            for seg_idx in range(M):
                mask = seg_ids == seg_idx
                indices = mask.nonzero(as_tuple=True)[0].cpu().tolist()
                if not indices:
                    continue
                seg_bytes = [raw_trunc[i] for i in indices]
                total_bytes += len(seg_bytes)
                if _detect_utf8_violation(seg_bytes):
                    utf8_violations += 1
                counts = _classify_bytes(seg_bytes)
                cjk_bytes += counts.get("cjk", 0)
                ascii_bytes += counts.get("ascii", 0)
                code_bytes += counts.get("op", 0) + counts.get("digit", 0)

    n_units = max(1, total_units)
    return {
        "method": "flued",
        "total_bytes": total_bytes,
        "total_units": total_units,
        "bytes_per_unit": total_bytes / n_units,
        "units_per_byte": total_units / max(1, total_bytes),
        "compression_ratio": total_bytes / n_units,
        "utf8_violation_rate": utf8_violations / n_units,
        "cjk_bytes_per_unit": cjk_bytes / n_units,
        "ascii_bytes_per_unit": ascii_bytes / n_units,
        "code_bytes_per_unit": code_bytes / n_units,
    }


# ---------------------------------------------------------------------------
# Method: BPE (in-domain trained)
# ---------------------------------------------------------------------------

def eval_bpe(tokenizer_path: str, raw_bytes_list: List[bytes]) -> Dict:
    """Evaluate in-domain BPE tokenizer."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    total_bytes = 0
    total_units = 0
    utf8_violations = 0
    cjk_bytes = 0
    ascii_bytes = 0
    code_bytes = 0

    for raw in raw_bytes_list:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        encoding = tok.encode(text)
        ids = encoding.ids
        # Get offsets to find byte spans
        offsets = encoding.offsets  # List of (start, end) in original string

        for (start, end) in offsets:
            if start >= len(raw) or end > len(raw):
                continue
            seg_bytes = list(raw[start:end])
            if not seg_bytes:
                continue
            total_bytes += len(seg_bytes)
            total_units += 1
            if _detect_utf8_violation(seg_bytes):
                utf8_violations += 1
            counts = _classify_bytes(seg_bytes)
            cjk_bytes += counts.get("cjk", 0)
            ascii_bytes += counts.get("ascii", 0)
            code_bytes += counts.get("op", 0) + counts.get("digit", 0)

    n_units = max(1, total_units)
    vocab_size = tok.get_vocab_size()
    label = f"bpe_{vocab_size // 1000}k" if vocab_size >= 1000 else f"bpe_{vocab_size}"
    return {
        "method": label,
        "total_bytes": total_bytes,
        "total_units": total_units,
        "bytes_per_unit": total_bytes / n_units,
        "units_per_byte": total_units / max(1, total_bytes),
        "compression_ratio": total_bytes / n_units,
        "utf8_violation_rate": utf8_violations / n_units,
        "cjk_bytes_per_unit": cjk_bytes / n_units,
        "ascii_bytes_per_unit": ascii_bytes / n_units,
        "code_bytes_per_unit": code_bytes / n_units,
    }


# ---------------------------------------------------------------------------
# Method: Fixed Patch
# ---------------------------------------------------------------------------

def eval_fixed_patch(patch_size: int, raw_bytes_list: List[bytes]) -> Dict:
    """Evaluate fixed N-byte patching."""
    total_bytes = 0
    total_units = 0
    utf8_violations = 0
    cjk_bytes = 0
    ascii_bytes = 0
    code_bytes = 0

    for raw in raw_bytes_list:
        for i in range(0, len(raw), patch_size):
            seg_bytes = list(raw[i:i + patch_size])
            if not seg_bytes:
                continue
            total_bytes += len(seg_bytes)
            total_units += 1
            if _detect_utf8_violation(seg_bytes):
                utf8_violations += 1
            counts = _classify_bytes(seg_bytes)
            cjk_bytes += counts.get("cjk", 0)
            ascii_bytes += counts.get("ascii", 0)
            code_bytes += counts.get("op", 0) + counts.get("digit", 0)

    n_units = max(1, total_units)
    return {
        "method": f"fixed_{patch_size}",
        "total_bytes": total_bytes,
        "total_units": total_units,
        "bytes_per_unit": total_bytes / n_units,
        "units_per_byte": total_units / max(1, total_bytes),
        "compression_ratio": total_bytes / n_units,
        "utf8_violation_rate": utf8_violations / n_units,
        "cjk_bytes_per_unit": cjk_bytes / n_units,
        "ascii_bytes_per_unit": ascii_bytes / n_units,
        "code_bytes_per_unit": code_bytes / n_units,
    }


# ---------------------------------------------------------------------------
# Method: Public tokenizer
# ---------------------------------------------------------------------------

def eval_public_tok(tokenizer_id: str, raw_bytes_list: List[bytes]) -> Dict:
    """Evaluate public tokenizer (tiktoken or HF)."""
    from flued.e3_downstream import PublicTokenizerDownstream
    # Create a throwaway model just for tokenization (no GPU needed)
    model = PublicTokenizerDownstream(
        tokenizer_id=tokenizer_id, d_model=64, nhead=2,
        dim_feedforward=128, num_layers=1, max_seq_len=128,
    )

    total_bytes = 0
    total_units = 0
    utf8_violations = 0
    cjk_bytes = 0
    ascii_bytes = 0
    code_bytes = 0

    for raw in raw_bytes_list:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        ids = model.encode_text(text)
        # Decode each token individually to get its byte span
        pos = 0
        for tid in ids:
            decoded = model._decode_single(tid)
            dec_bytes = decoded.encode("utf-8")
            n = len(dec_bytes)
            if n == 0:
                continue
            # Find the span in the original bytes
            # This is approximate: match decoded bytes against raw[pos:]
            if pos + n <= len(raw) and raw[pos:pos + n] == dec_bytes:
                seg_bytes = list(raw[pos:pos + n])
                pos += n
            else:
                # Fallback: use decoded bytes directly
                seg_bytes = list(dec_bytes)
            if not seg_bytes:
                continue
            total_bytes += len(seg_bytes)
            total_units += 1
            if _detect_utf8_violation(seg_bytes):
                utf8_violations += 1
            counts = _classify_bytes(seg_bytes)
            cjk_bytes += counts.get("cjk", 0)
            ascii_bytes += counts.get("ascii", 0)
            code_bytes += counts.get("op", 0) + counts.get("digit", 0)

    n_units = max(1, total_units)
    short_name = tokenizer_id.split(":")[-1][:20]
    return {
        "method": f"public_{short_name}",
        "total_bytes": total_bytes,
        "total_units": total_units,
        "bytes_per_unit": total_bytes / n_units,
        "units_per_byte": total_units / max(1, total_bytes),
        "compression_ratio": total_bytes / n_units,
        "utf8_violation_rate": utf8_violations / n_units,
        "cjk_bytes_per_unit": cjk_bytes / n_units,
        "ascii_bytes_per_unit": ascii_bytes / n_units,
        "code_bytes_per_unit": code_bytes / n_units,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_corpus(data_path: str, max_lines: int) -> List[bytes]:
    """Load raw UTF-8 bytes for each line."""
    lines = []
    with open(data_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line.encode("utf-8"))
    logger.info("Loaded %d lines from %s", len(lines), data_path)
    return lines


def evaluate_all(methods: List[str], raw_bytes_list: List[bytes],
                 args) -> List[Dict]:
    """Run all requested evaluations and return list of result dicts."""
    results = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_gpu = (device.type == "cuda")

    for method in methods:
        logger.info("=== Evaluating: %s ===", method)
        try:
            if method == "flued":
                r = eval_flued(args.flued_ckpt, raw_bytes_list, device)
            elif method.startswith("bpe_"):
                # bpe_8k, bpe_16k, bpe_32k
                vocab_str = method.split("_", 1)[1]
                tok_dir = args.bpe_dir or "checkpoints/bpe_tokenizer"
                tok_path = os.path.join(tok_dir, f"bpe_{vocab_str}", "tokenizer.json")
                if not os.path.exists(tok_path):
                    # Fallback: single tokenizer file
                    tok_path = os.path.join(tok_dir, "tokenizer.json")
                r = eval_bpe(tok_path, raw_bytes_list)
                r["method"] = method  # override label
            elif method.startswith("fixed_"):
                patch_size = int(method.split("_")[1])
                r = eval_fixed_patch(patch_size, raw_bytes_list)
            elif method == "public_tok":
                r = eval_public_tok(args.public_tokenizer, raw_bytes_list)
            elif method == "blt":
                logger.warning("BLT evaluation not yet implemented (needs ByteLM)")
                continue
            else:
                logger.warning("Unknown method: %s", method)
                continue
            results.append(r)
            logger.info("  → %s", {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in r.items() if k != "method"})
        except Exception as e:
            logger.error("  FAILED: %s", e, exc_info=True)

    return results


def print_table(results: List[Dict]):
    """Print a formatted table of results."""
    if not results:
        return
    keys = ["method", "total_bytes", "total_units", "bytes_per_unit",
            "utf8_violation_rate", "cjk_bytes_per_unit"]
    header = f"{'Method':<18} {'Bytes':>10} {'Units':>8} {'B/U':>8} {'UTF8err':>8} {'CJK/U':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['method']:<18} {r['total_bytes']:>10} {r['total_units']:>8} "
              f"{r['bytes_per_unit']:>8.2f} {r['utf8_violation_rate']:>8.4f} "
              f"{r['cjk_bytes_per_unit']:>8.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FLUED Segmentation/Compression Evaluation (Stage 1)")
    parser.add_argument("--data-path", required=True,
                        help="Path to corpus text file (UTF-8, one doc per line).")
    parser.add_argument("--max-lines", type=int, default=5000,
                        help="Max lines to evaluate (default: 5000).")
    parser.add_argument("--methods", type=str, required=True,
                        help="Comma-separated list: flued,bpe_8k,fixed_4,public_tok,...")
    parser.add_argument("--device", default="cuda",
                        help="Device for FLUED/BLT inference (cpu, cuda).")

    # Checkpoint / tokenizer paths
    parser.add_argument("--flued-ckpt", default="checkpoints/e1_latest.pt",
                        help="Path to FLUED E1 checkpoint.")
    parser.add_argument("--bpe-dir", default="checkpoints/bpe_tokenizer",
                        help="Directory containing BPE tokenizer(s).")
    parser.add_argument("--public-tokenizer", default="tiktoken:cl100k_base",
                        help="Public tokenizer ID (tiktoken:<name> or hf:<model>).")

    # Output
    parser.add_argument("--output-json", default=None,
                        help="Save results as JSON.")
    parser.add_argument("--output-csv", default=None,
                        help="Save results as CSV.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    logger.info("Methods: %s", methods)

    raw_bytes_list = load_corpus(args.data_path, args.max_lines)
    results = evaluate_all(methods, raw_bytes_list, args)
    print_table(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Saved JSON → %s", args.output_json)

    if args.output_csv and results:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved CSV → %s", args.output_csv)
