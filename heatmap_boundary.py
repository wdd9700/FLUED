"""
C-class: Linguistic alignment — boundary heatmap & type breakdown.

Generates:
  1. Per-type boundary probability table (utf8 / cjk / ascii / op / digit)
  2. Boundary heatmap visualizations for sample sentences:
     - Chinese sentence
     - English sentence
     - Code / digit-mixed snippet

Usage:
  python heatmap_boundary.py --flued-ckpt checkpoints/e1_seed42/e1_latest.pt
"""

import argparse
import logging
import sys

import torch
import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("heatmap")

# ---------------------------------------------------------------------------
# Sample sentences
# ---------------------------------------------------------------------------

SAMPLES = {
    "chinese": (
        "zh",
        "今天天气很好，我们决定去公园散步。春天的风吹过来，带着花香。"
    ),
    "english": (
        "en",
        "The transformer architecture has become the backbone of modern "
        "language models, enabling unprecedented few-shot learning capabilities."
    ),
    "code_mixed": (
        "code",
        "def fibonacci(n: int) -> int:\n"
        '    """Return the n-th Fibonacci number."""\n'
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b"
    ),
    "digits": (
        "digit",
        "The model achieves 94.7% accuracy on test set with p < 0.001. "
        "Training took 12,450 steps at lr=3e-5, batch=256, costing $42.80."
    ),
}

# ---------------------------------------------------------------------------
# Byte classification (same as eval_segmentation.py)
# ---------------------------------------------------------------------------

def _byte_type(b: int) -> str:
    if b < 0x80:
        c = chr(b)
        if c.isdigit():
            return "digit"
        if c in "+-*/%=<>!&|^~@#$\\:;.,?()[]{}`\"'":
            return "op"
        if c.isspace():
            return "space"
        return "ascii"
    if 0x80 <= b < 0xC0:
        return "utf8_cont"
    if 0xC0 <= b < 0xE0:
        return "utf8_lead2"
    if 0xE0 <= b < 0xF0:
        return "utf8_lead3"
    if 0xF0 <= b < 0xF8:
        return "utf8_lead4"
    return "other"


# ---------------------------------------------------------------------------
# Heatmap renderer (ASCII art in terminal + HTML optional)
# ---------------------------------------------------------------------------

def render_heatmap(text: str, bp_values: np.ndarray, out_path: str = None):
    """Render a text + boundary probability heatmap.

    Each character is colored by its boundary probability:
      ░ = low bp (continuation)
      █ = high bp (boundary)

    If out_path ends with .html, produces an HTML heatmap.
    Otherwise prints to terminal.
    """
    chars = list(text)
    n = len(bp_values)

    if out_path and out_path.endswith(".html"):
        _render_html(text, bp_values, out_path)
        return

    # ---- Terminal ASCII heatmap ----
    # Top row: boundary probability bars
    print("\n" + "─" * 80)
    print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Bytes: {n}")

    # Boundary bar (0-1 scale, 50 chars wide)
    bar = ""
    for i in range(min(n, 120)):
        p = bp_values[i]
        if p > 0.7:
            bar += "█"
        elif p > 0.4:
            bar += "▓"
        elif p > 0.15:
            bar += "▒"
        else:
            bar += "░"
    print(f"  bp:    {bar}")

    # Byte-level detail (first 40 bytes)
    detail = "  detail: "
    for i in range(min(n, 40)):
        b_val = ord(text[i].encode("utf-8")[0:1] or b'\x00') if i < len(text) else 0
        bt = _byte_type(b_val) if i < len(bytes(text, "utf-8")[i:i+1] or b'\x00') else "?"
        p = bp_values[i]
        marker = "▌" if p > 0.5 else "·"
        detail += marker
    print(detail)

    # Per-type summary
    # NOTE: heatmap type statistics are SAMPLE-LEVEL qualitative illustrations.
    # They differ from the E1 log's corpus-wide `type_bp` metrics in two ways:
    #   1. Sample size: a few dozen bytes vs. millions in the full corpus.
    #   2. CJK definition: here we use precise CJK Unified range (U+4E00-U+9FFF,
    #      UTF-8 lead 0xE4-0xE9), matching the E1 log. The broader "utf8_lead3"
    #      category (0xE0-0xEF) includes non-CJK scripts and is NOT used.
    # Always cite the corpus-wide table (A-class) for quantitative claims;
    # heatmaps are qualitative evidence of structural boundary placement.
    type_bp = {"utf8_cont": [], "cjk": [], "ascii": [], "op": [], "digit": [], "space": []}
    raw_bytes = text.encode("utf-8")
    for i, b in enumerate(raw_bytes):
        if i >= n:
            break
        bt = _byte_type(b)
        # Precise CJK: only U+4E00-U+9FFF (UTF-8 lead bytes 0xE4-0xE9)
        if 0xE4 <= b <= 0xE9:
            type_bp["cjk"].append(bp_values[i])
        elif bt in type_bp:
            type_bp[bt].append(bp_values[i])

    print("  Type bp means (SAMPLE-LEVEL, not corpus-wide):")
    for t in ["ascii", "cjk", "digit", "op", "utf8_cont"]:
        vals = type_bp.get(t, [])
        if vals:
            print(f"    {t:<12}: mean={np.mean(vals):.3f}  n={len(vals)}")
    print("─" * 80)


def _render_html(text: str, bp_values: np.ndarray, out_path: str):
    """Render an HTML heatmap with color-coded characters."""
    raw_bytes = text.encode("utf-8")
    n = min(len(bp_values), len(raw_bytes))

    # Build HTML with per-character coloring
    # Each byte gets a background color based on boundary probability
    html_parts = ['<html><head><meta charset="utf-8"><style>',
                  'body{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px}',
                  '.char{display:inline-block;padding:1px 0;margin:0}',
                  '.legend{display:flex;gap:4px;margin:10px 0;font-size:12px}',
                  '.legend div{padding:2px 8px;border-radius:3px}',
                  '</style></head><body><h3>FLUED Boundary Heatmap</h3>']

    # Legend
    html_parts.append(f'<p>{text[:100]}</p>')
    html_parts.append('<div class="legend">')
    for pct, color in [(0.0, "#0f3460"), (0.25, "#533483"), (0.5, "#e94560"), (0.75, "#f5a623"), (1.0, "#00ff88")]:
        html_parts.append(f'<div style="background:{color}">p={pct:.2f}</div>')
    html_parts.append('</div>')

    # Character heatmap
    html_parts.append('<div style="line-height:1.8;max-width:900px">')
    char_idx = 0
    byte_idx = 0
    while byte_idx < n and char_idx < len(text):
        ch = text[char_idx]
        ch_bytes = ch.encode("utf-8")
        n_ch_bytes = len(ch_bytes)
        # Use the lead byte's boundary probability
        p = bp_values[byte_idx] if byte_idx < n else 0.0
        # Color interpolation: blue (0) → red (0.5) → green (1)
        if p < 0.5:
            r = int(p * 2 * 233)
            g = int(p * 2 * 69)
            b_col = int(96 + p * 2 * (53 - 96))
        else:
            r = int(233 + (p - 0.5) * 2 * (0 - 233))
            g = int(69 + (p - 0.5) * 2 * (255 - 69))
            b_col = int(53 + (p - 0.5) * 2 * (136 - 53))
        color = f"rgb({r},{g},{b_col})"
        html_parts.append(
            f'<span class="char" style="background:{color}" '
            f'title="byte={byte_idx} bp={p:.3f}">{ch}</span>'
        )
        char_idx += 1
        byte_idx += n_ch_bytes

    html_parts.append('</div></body></html>')

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html_parts))
    logger.info("HTML heatmap saved → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FLUED Boundary Heatmap (C-class)")
    parser.add_argument("--flued-ckpt", default="checkpoints/e1_seed42/e1_latest.pt",
                        help="Path to FLUED E1 checkpoint.")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda / cpu).")
    parser.add_argument("--output-html", default=None,
                        help="Optional output directory for HTML heatmaps.")
    parser.add_argument("--samples", default="chinese,english,code_mixed",
                        help="Comma-separated sample keys to render.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    from flued.model import FLUEDAutoencoder
    ckpt = torch.load(args.flued_ckpt, map_location="cpu", weights_only=False)
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
    logger.info("Model loaded: d=%d ff=%d nhead=%d", ckpt_d, ckpt_ff, ckpt_nhead)

    sample_keys = [k.strip() for k in args.samples.split(",")]

    with torch.no_grad():
        for key in sample_keys:
            if key not in SAMPLES:
                logger.warning("Unknown sample key: %s", key)
                continue
            tag, text = SAMPLES[key]
            raw = text.encode("utf-8")[:512]
            ids = torch.tensor([b + 1 for b in raw], dtype=torch.long,
                               device=device).unsqueeze(0)
            pad_mask = torch.zeros_like(ids, dtype=torch.bool)

            _, metrics = model.encode(ids, pad_mask, skip_hard=True)
            bp = metrics["boundary_probs"].float().squeeze(0).cpu().numpy()

            logger.info("Sample '%s': %d bytes, bp_mean=%.3f bp_std=%.3f",
                        key, len(raw), float(bp.mean()), float(bp.std()))

            html_path = None
            if args.output_html:
                import os
                os.makedirs(args.output_html, exist_ok=True)
                html_path = os.path.join(args.output_html, f"heatmap_{key}.html")

            render_heatmap(text, bp, html_path)


if __name__ == "__main__":
    main()
