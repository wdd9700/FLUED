#!/bin/bash
# setup_5090.sh — AutoDL RTX 5090 environment setup for FLUED
#
# Usage on AutoDL:
#   bash setup_5090.sh
#
# Prerequisites: AutoDL base image with CUDA 12.8+ and Python 3.12+

set -e

echo "=== FLUED 5090 Setup ==="
echo "Date: $(date)"

# ---- PyTorch nightly with CUDA 12.8+ (Blackwell support) ----
echo ">> Installing PyTorch 2.11+ for Blackwell (RTX 5090)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---- Core deps ----
echo ">> Installing Python dependencies..."
pip install numpy tiktoken tokenizers transformers pytest

# ---- Verify GPU ----
echo ">> Verifying GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} ({props.total_mem/1e9:.1f} GB)')
"

# ---- Prepare data directory ----
echo ">> Creating directories..."
mkdir -p data checkpoints

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload corpus data to ./data/"
echo "  2. Upload v2 FLUED checkpoints to ./checkpoints/"
echo "  3. Run: python tools/launcher/run_d1_bpb.py"
echo "  4. Run: python tools/launcher/run_e3_downstream.py"
