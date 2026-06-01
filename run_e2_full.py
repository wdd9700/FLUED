"""
E2 Full Comparison — FLUED E1v5 vs BPE vs BLT.

Loads final checkpoints for all three models and evaluates:
  - Reconstruction perplexity (per-token CE → exp)
  - Compression ratio (m/n) for FLUED/BLT
  - Per-type boundary probabilities for FLUED
  - Bits-per-byte (from E3 training logs)

Usage
-----
C:\Python314\python.exe run_e2_full.py

Output
------
checkpoints/e2_compare_results.json
"""
import json
import math
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from flued.model import FLUEDAutoencoder, VOCAB_SIZE as FLUED_VOCAB
from flued.data import ByteReconstructionDataset, safe_train_eval_split, STUB_CORPUS
from flued.config import ModelConfig

RESULTS_FILE = "checkpoints/e2_compare_results.json"
DATA_PATH = r"E:\projects\SoulMamba\soulvlm_project\temp\corpus_v3.txt"
FLUED_CKPT = "checkpoints/e1_step50000.pt"
BPE_CKPT = "bpe_baseline_standalone/checkpoints/bpe_latest.pt"
BLT_CKPT = "checkpoints/blt_latest.pt"

SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data (small subset for E2 eval)
print(f"Loading data from {DATA_PATH} …")
with open(DATA_PATH, encoding="utf-8") as fh:
    texts = [line.rstrip("\n") for line in fh if line.strip()]
texts = texts[:1000]  # 1000 lines for quick E2 evaluation
print(f"Loaded {len(texts)} lines")

dataset = ByteReconstructionDataset(texts=texts, seq_len=256, stride=128)
train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.2, seed=SEED)
eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False, drop_last=False)
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

results = {}
results["metadata"] = {
    "n_lines": len(texts),
    "eval_chunks": len(eval_ds),
    "seq_len": 256,
    "device": str(device),
}

# ═══════════════════════════════════════════════════════════════
# 1. FLUED Evaluation
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FLUED E1v5")
print("=" * 60)

flued_model = FLUEDAutoencoder(
    d_model=1024, nhead=16, dim_feedforward=4096,
    num_layers=24, max_seq_len=512, dropout=0.0,
).to(device)

ckpt = torch.load(FLUED_CKPT, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
flued_model.load_state_dict(state)
flued_model.eval()
flued_n = sum(p.numel() for p in flued_model.parameters() if p.requires_grad)
print(f"FLUED params: {flued_n:,}")
results["flued"] = {"params": flued_n}

# Recon perplexity
total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for src, _ in eval_loader:
        src = src.to(device)
        logits, metrics = flued_model(src)
        loss = criterion(logits.view(-1, FLUED_VOCAB), src.view(-1))
        total_loss += loss.item()
        total_tokens += (src != 0).sum().item()
ppl = math.exp(min(total_loss / total_tokens, 20))
results["flued"]["recon_ppl"] = round(ppl, 4)
results["flued"]["recon_ce"] = round(total_loss / total_tokens, 6)

# m/n & bp stats
with torch.no_grad():
    src = next(iter(eval_loader))[0].to(device)
    logits, metrics = flued_model(src)
    for k in ["soft_m_over_n", "hard_m_over_n", "bp_mean", "bp_std"]:
        v = metrics.get(k)
        if v is not None:
            results["flued"][k] = round(v.item(), 4) if hasattr(v, "item") else round(v, 4)

# Per-type bp
print("Computing per-type boundary probabilities …")
from flued.model import get_byte_type_mask
type_bp = {}
try:
    src_eval = torch.cat([s[0].unsqueeze(0) for s in list(eval_loader.dataset)[:50]], dim=0).to(device)
    with torch.no_grad():
        _, metrics = flued_model(src_eval)
    boundary_probs = metrics.get("boundary_probs")
    if boundary_probs is not None:
        bp = boundary_probs.float().cpu()
        masks = get_byte_type_mask(src_eval.cpu())
        for name, mask in masks.items():
            masked = bp[mask]
            if masked.numel() > 0:
                type_bp[name] = round(masked.mean().item(), 4)
    results["flued"]["type_bp"] = type_bp
except Exception as e:
    print(f"  type_bp skipped: {e}")

print(f"  recon_ppl={results['flued'].get('recon_ppl')}  "
      f"m/n={results['flued'].get('soft_m_over_n')}  "
      f"bp_std={results['flued'].get('bp_std')}")
print(f"  type_bp={type_bp}")

# ═══════════════════════════════════════════════════════════════
# 2. BPE Evaluation
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BPE Baseline")
print("=" * 60)

try:
    from bpe_baseline.model import BPETransformerAutoencoder
    from tokenizers import Tokenizer

    bpe_tokenizer = Tokenizer.from_file("checkpoints/bpe_tokenizer/tokenizer.json")
    bpe_vocab = bpe_tokenizer.get_vocab_size()
    print(f"BPE vocab: {bpe_vocab}")

    bpe_model = BPETransformerAutoencoder(
        vocab_size=bpe_vocab, d_model=1024, nhead=16,
        dim_feedforward=4096, num_encoder_layers=12,
        num_decoder_layers=12, max_seq_len=256, dropout=0.0,
    ).to(device)

    # Try loading checkpoint — BPE standalone or main
    bpe_ckpt_paths = [BPE_CKPT, "checkpoints/e3_bpe_step020000.pt"]
    bpe_loaded = False
    for cp in bpe_ckpt_paths:
        if os.path.exists(cp):
            try:
                raw = torch.load(cp, map_location=device, weights_only=False)
                sd = raw.get("model_state_dict", raw.get("model", raw))
                # Filter only matching keys
                model_sd = bpe_model.state_dict()
                filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
                if len(filtered) > len(model_sd) * 0.5:
                    bpe_model.load_state_dict(filtered, strict=False)
                    bpe_loaded = True
                    print(f"  Loaded BPE checkpoint: {cp} ({len(filtered)}/{len(model_sd)} keys)")
                    break
            except Exception as e:
                print(f"  Failed to load {cp}: {e}")

    if not bpe_loaded:
        print("  WARNING: BPE checkpoint not loaded — using random init (eval will be meaningless)")

    bpe_model.eval()
    bpe_n = sum(p.numel() for p in bpe_model.parameters() if p.requires_grad)
    print(f"BPE params: {bpe_n:,}")
    results["bpe"] = {"params": bpe_n, "vocab": bpe_vocab, "checkpoint_loaded": bpe_loaded}

    # BPE reconstruction: tokenize → eval
    def bpe_encode(text):
        return bpe_tokenizer.encode(text).ids

    bpe_total_loss = 0.0
    bpe_total_tokens = 0
    with torch.no_grad():
        for src_bytes, _ in eval_loader:
            # Convert byte ids back to text, then BPE-tokenize
            batch_texts = []
            for row in src_bytes:
                # byte ids are PAD-offset: subtract 1, skip PAD (0)
                raw = [b - 1 for b in row.tolist() if b != 0]
                try:
                    text = bytes(raw).decode("utf-8", errors="replace")
                except:
                    text = ""
                batch_texts.append(text)

            # Tokenize
            batch_ids = []
            for t in batch_texts:
                ids = bpe_encode(t)
                if len(ids) == 0:
                    ids = [0]
                batch_ids.append(ids[:256])

            # Pad to max len in batch
            max_len = max(len(ids) for ids in batch_ids)
            padded = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
            for i, ids in enumerate(batch_ids):
                padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

            padded = padded.to(device)
            logits, _ = bpe_model(padded)
            loss = criterion(logits.view(-1, bpe_vocab), padded.view(-1))
            bpe_total_loss += loss.item()
            bpe_total_tokens += (padded != 0).sum().item()

    bpe_ppl = math.exp(min(bpe_total_loss / bpe_total_tokens, 20))
    results["bpe"]["recon_ppl"] = round(bpe_ppl, 4)
    results["bpe"]["recon_ce"] = round(bpe_total_loss / bpe_total_tokens, 6)
    print(f"  recon_ppl={bpe_ppl}  tokens={bpe_total_tokens}")

except Exception as e:
    print(f"  BPE evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    results["bpe"] = {"error": str(e)}

# ═══════════════════════════════════════════════════════════════
# 3. BLT Evaluation
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BLT Baseline")
print("=" * 60)

try:
    from blt_baseline.model import BLTAutoencoder

    blt_model = BLTAutoencoder(
        vocab_size=257, d_model=1024, nhead=16,
        dim_feedforward=4096, num_encoder_layers=12,
        num_decoder_layers=12, max_seq_len=256, dropout=0.0,
        patch_mode="entropy", entropy_theta=3.5,
    ).to(device)

    # Load ByteLM + BLT checkpoints
    blt_loaded = False
    blt_ckpt_sources = [
        ("checkpoints/blt_latest.pt", "blt"),
        ("checkpoints/blt_step05000.pt", "blt"),
    ]
    bytelm_path = "checkpoints/bytel m_latest.pt"

    # Load ByteLM first
    if os.path.exists(bytelm_path):
        try:
            raw = torch.load(bytelm_path, map_location=device, weights_only=False)
            sd = raw.get("model_state_dict", raw.get("model", raw))
            bytelm_sd = blt_model.local_lm.state_dict()
            filtered = {k: v for k, v in sd.items() if k in bytelm_sd and v.shape == bytelm_sd[k].shape}
            if len(filtered) > len(bytelm_sd) * 0.5:
                blt_model.local_lm.load_state_dict(filtered, strict=False)
                print(f"  Loaded ByteLM: {bytelm_path} ({len(filtered)}/{len(bytelm_sd)} keys)")
        except Exception as e:
            print(f"  Failed to load ByteLM: {e}")

    # Load BLT global+decoder
    for cp, tag in blt_ckpt_sources:
        if os.path.exists(cp):
            try:
                raw = torch.load(cp, map_location=device, weights_only=False)
                sd = raw.get("model_state_dict", raw.get("model", raw))
                model_sd = blt_model.state_dict()
                # Don't load local_lm keys (they're already loaded)
                filtered = {k: v for k, v in sd.items()
                            if k in model_sd and v.shape == model_sd[k].shape
                            and not k.startswith("local_lm.")}
                if len(filtered) > 10:
                    blt_model.load_state_dict(filtered, strict=False)
                    blt_loaded = True
                    print(f"  Loaded BLT checkpoint: {cp} ({len(filtered)} keys)")
                    break
            except Exception as e:
                print(f"  Failed to load {cp}: {e}")

    if not blt_loaded:
        print("  WARNING: BLT checkpoint not loaded — using partial init")

    blt_model.eval()
    blt_n = sum(p.numel() for p in blt_model.parameters() if p.requires_grad)
    print(f"BLT params: {blt_n:,}")
    results["blt"] = {"params": blt_n, "checkpoint_loaded": blt_loaded}

    # BLT reconstruction
    blt_total_loss = 0.0
    blt_total_tokens = 0
    with torch.no_grad():
        for src, _ in eval_loader:
            src = src.to(device)
            logits, _ = blt_model(src)
            loss = criterion(logits.view(-1, 257), src.view(-1))
            blt_total_loss += loss.item()
            blt_total_tokens += (src != 0).sum().item()

    blt_ppl = math.exp(min(blt_total_loss / blt_total_tokens, 20))
    results["blt"]["recon_ppl"] = round(blt_ppl, 4)
    results["blt"]["recon_ce"] = round(blt_total_loss / blt_total_tokens, 6)
    print(f"  recon_ppl={blt_ppl}  tokens={blt_total_tokens}")

except Exception as e:
    print(f"  BLT evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    results["blt"] = {"error": str(e)}

# ═══════════════════════════════════════════════════════════════
# 4. E3 downstream results (from training logs)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("E3 Downstream Results (from logs)")
print("=" * 60)

e3_results = {
    "flued": {"bpb": 1.2114, "loss": 0.8396, "steps": 20000,
              "source": "checkpoints/e3_flued_fair_v2.log"},
    "bpe": {"bpb": 1.4786, "loss": 3.2182, "steps": 20000,
            "source": "checkpoints/e3_bpe_local.log"},
    "blt_resume": {"bpb": 2.6371, "loss": 1.8279, "steps": 20000,
                   "source": "checkpoints/e3_blt_resume.log"},
    "blt_scratch": {"bpb": 3.324, "loss": 2.3038, "steps": 8000,
                    "source": "checkpoints/e3_blt_local.log",
                    "note": "crashed at step 8000 (checkpoint write error)"},
}
results["e3_downstream"] = e3_results
for name, data in e3_results.items():
    print(f"  {name:<15} bpb={data['bpb']}  @ step {data['steps']}")

# ═══════════════════════════════════════════════════════════════
# 5. FLUED E1v5 final paper data
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FLUED E1v5 Paper Data")
print("=" * 60)

try:
    with open("checkpoints/e1_v5_paper.json") as fh:
        paper_data = json.load(fh)
    records = paper_data.get("records", [])
    if records:
        final = records[-1]
        results["flued_e1v5_paper"] = {
            "step": final["step"],
            "recon_acc": final["recon_acc"],
            "bp_std": final["bp_std"],
            "soft_mn": final["soft_mn"],
            "bp_mean": final["bp_mean"],
            "type_bp": {
                "cjk": final["cjk"],
                "op": final["op"],
                "digit": final["digit"],
                "ascii": final["ascii"],
                "utf8": final["utf8"],
            }
        }
        print(f"  step={final['step']}  recon_acc={final['recon_acc']:.4f}  "
              f"soft_mn={final['soft_mn']:.4f}  bp_std={final['bp_std']:.4f}")
        print(f"  type_bp: cjk={final['cjk']:.4f}  op={final['op']:.4f}  "
              f"digit={final['digit']:.4f}  ascii={final['ascii']:.4f}  utf8={final['utf8']:.4f}")
except Exception as e:
    print(f"  Failed: {e}")

# ═══════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════
with open(RESULTS_FILE, "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"\nResults saved to {RESULTS_FILE}")
print("Done.")
