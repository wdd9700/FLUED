"""
E2 CPU Evaluation — FLUED E1v5 (50k) vs BLT (40k) vs BPE (40k).
All three with comparable training steps on CPU.

Output: checkpoints/e2_cpu_result.json
"""
import json, math, sys, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flued.model import FLUEDAutoencoder
from flued.data import ByteReconstructionDataset, safe_train_eval_split

RESULTS_FILE = "checkpoints/e2_cpu_result.json"
device = torch.device("cpu")
print(f"Device: {device}")

# ── Data ──
DATA_PATH = r"data/corpus.txt"
with open(DATA_PATH, encoding="utf-8") as fh:
    texts = [line.rstrip("\n") for line in fh if line.strip()]
texts = texts[:200]
print(f"Loaded {len(texts)} lines")

dataset = ByteReconstructionDataset(texts=texts, seq_len=128, stride=64)
train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.3, seed=42)
eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False, drop_last=False)
print(f"Eval chunks: {len(eval_ds)}")

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
results = {"metadata": {"n_lines": len(texts), "seq_len": 128, "n_chunks": len(eval_ds)}}

# ═══════════════════════════════════════════════════════════════
# 1. FLUED E1v5 (50,000 steps)
# ═══════════════════════════════════════════════════════════════
print("\n=== FLUED E1v5 (50k steps) ===")
t0 = time.time()
model = FLUEDAutoencoder(
    d_model=1024, nhead=16, dim_feedforward=4096,
    num_layers=24, max_seq_len=512, dropout=0.0,
).to(device)
ckpt = torch.load("checkpoints/e1_step50000.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params:,}  (load: {time.time()-t0:.1f}s)")

total_loss = total_tok = 0
bpm_sum = bpm_n = 0
t0 = time.time()
with torch.no_grad():
    for src, _ in eval_loader:
        src = src.to(device)
        logits, metrics = model(src)
        total_loss += criterion(logits.view(-1, 257), src.view(-1)).item()
        total_tok += (src != 0).sum().item()
        mn = metrics.get("soft_m_over_n")
        if mn is not None:
            bpm_sum += mn.item() * src.size(0)
            bpm_n += src.size(0)

flued = {
    "steps": 50000,
    "params": n_params,
    "recon_ce": round(total_loss / total_tok, 6),
    "recon_ppl": round(math.exp(min(total_loss / total_tok, 20)), 4),
    "m_over_n": round(bpm_sum / bpm_n, 4) if bpm_n > 0 else None,
    "eval_time_s": round(time.time() - t0, 1),
}
print(f"  ce={flued['recon_ce']}  ppl={flued['recon_ppl']}  m/n={flued['m_over_n']}  time={flued['eval_time_s']}s")
results["flued"] = flued

# ═══════════════════════════════════════════════════════════════
# 2. BLT Baseline (40,000 steps)
# ═══════════════════════════════════════════════════════════════
print("\n=== BLT (40k steps) ===")
t0 = time.time()
from blt_baseline.model import BLTAutoencoder

blt_model = BLTAutoencoder(
    vocab_size=257, d_model=1024, nhead=16,
    dim_feedforward=4096,
    global_layers=11, decoder_layers=12,  # match checkpoint
    local_lm_d_model=512, local_layers=4,
    max_seq_len=2048, dropout=0.0,
    patch_mode="entropy", entropy_theta=3.5,
).to(device)

ckpt = torch.load("checkpoints/blt_step40000.pt", map_location=device, weights_only=False)
sd = ckpt["model"]
model_sd = blt_model.state_dict()
filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
blt_model.load_state_dict(filtered, strict=False)
blt_model.eval()
print(f"  Params: {sum(p.numel() for p in blt_model.parameters()):,}  "
      f"keys: {len(filtered)}/{len(model_sd)}  (load: {time.time()-t0:.1f}s)")

total_loss = total_tok = 0
t0 = time.time()
with torch.no_grad():
    for src, _ in eval_loader:
        src = src.to(device)
        logits, _ = blt_model(src)
        total_loss += criterion(logits.view(-1, 257), src.view(-1)).item()
        total_tok += (src != 0).sum().item()

blt = {
    "steps": 40000,
    "params": sum(p.numel() for p in blt_model.parameters()),
    "recon_ce": round(total_loss / total_tok, 6),
    "recon_ppl": round(math.exp(min(total_loss / total_tok, 20)), 4),
    "keys_matched": f"{len(filtered)}/{len(model_sd)}",
    "eval_time_s": round(time.time() - t0, 1),
}
print(f"  ce={blt['recon_ce']}  ppl={blt['recon_ppl']}  time={blt['eval_time_s']}s")
results["blt"] = blt

# ═══════════════════════════════════════════════════════════════
# 3. BPE Baseline (40,000 steps) — token-level
# ═══════════════════════════════════════════════════════════════
print("\n=== BPE (40k steps) ===")
t0 = time.time()
from bpe_baseline.model import BPETransformerAutoencoder
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("checkpoints/bpe_tokenizer/tokenizer.json")
vocab = tokenizer.get_vocab_size()
print(f"  Vocab: {vocab}")

bpe_model = BPETransformerAutoencoder(
    vocab_size=vocab, d_model=1024, nhead=16,
    dim_feedforward=4096, num_encoder_layers=10,
    num_decoder_layers=10, max_seq_len=256, dropout=0.0,
).to(device)

ckpt = torch.load("checkpoints/bpe_step40000.pt", map_location=device, weights_only=False)
sd = ckpt["model"]
model_sd = bpe_model.state_dict()
filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
bpe_model.load_state_dict(filtered, strict=False)
bpe_model.eval()
print(f"  Params: {sum(p.numel() for p in bpe_model.parameters()):,}  "
      f"keys: {len(filtered)}/{len(model_sd)}  (load: {time.time()-t0:.1f}s)")

total_loss = total_tok = 0
t0 = time.time()
with torch.no_grad():
    for src_bytes, _ in eval_loader:
        batch_ids = []
        for row in src_bytes:
            raw = bytes([b - 1 for b in row.tolist() if b != 0])
            try:
                text = raw.decode("utf-8", errors="replace")
                ids = tokenizer.encode(text).ids
            except:
                ids = []
            if not ids:
                ids = [0]
            batch_ids.append(ids[:256])

        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded = padded.to(device)

        logits, _ = bpe_model(padded)
        total_loss += criterion(logits.view(-1, vocab), padded.view(-1)).item()
        total_tok += (padded != 0).sum().item()

bpe = {
    "steps": 40000,
    "params": sum(p.numel() for p in bpe_model.parameters()),
    "vocab": vocab,
    "recon_ce": round(total_loss / total_tok, 6) if total_tok else None,
    "recon_ppl": round(math.exp(min(total_loss / total_tok, 20)), 4) if total_tok else None,
    "keys_matched": f"{len(filtered)}/{len(model_sd)}",
    "eval_time_s": round(time.time() - t0, 1),
    "note": "Token-level CE (not byte-level — not directly comparable to FLUED/BLT)",
}
print(f"  ce={bpe['recon_ce']}  ppl={bpe['recon_ppl']}  time={bpe['eval_time_s']}s")
results["bpe"] = bpe

# ═══════════════════════════════════════════════════════════════
# 4. E1v5 paper + E3 downstream
# ═══════════════════════════════════════════════════════════════
print("\n=== Summary ===")
with open("checkpoints/e1_v5_paper.json") as fh:
    paper = json.load(fh)
final = paper["records"][-1]
results["e1v5_paper_final"] = {
    "step": final["step"], "recon_acc": round(final["recon_acc"], 5),
    "bp_std": round(final["bp_std"], 5), "soft_mn": round(final["soft_mn"], 5),
    "type_bp": {k: round(final[k], 5) for k in ["cjk","op","digit","ascii","utf8"]},
}
results["e3_downstream"] = {
    "flued": {"bpb": 1.2114, "steps": 20000},
    "bpe": {"bpb": 1.4786, "steps": 20000},
    "blt_resume": {"bpb": 2.6371, "steps": 20000},
}

print(f"FLUED: ce={flued['recon_ce']} ppl={flued['recon_ppl']} m/n={flued['m_over_n']}")
print(f"BLT:   ce={blt['recon_ce']} ppl={blt['recon_ppl']}")
print(f"BPE:   ce={bpe['recon_ce']} ppl={bpe['recon_ppl']} (token-level)")
print(f"E3 bpb: FLUED=1.21  BPE=1.48  BLT=2.64")

with open(RESULTS_FILE, "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"\nSaved to {RESULTS_FILE}")
