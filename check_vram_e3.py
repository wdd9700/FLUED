"""VRAM assessment for E3 downstream LM training."""
import torch
from flued.e3_downstream import FLUEDDownstream

print("Loading FLUED 500M downstream model...")
m = FLUEDDownstream(flued_ckpt='checkpoints/e1_step36500.pt', num_layers=28).cuda()
m.eval()
for p in m.encoder.parameters():
    p.requires_grad = False

n_frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Frozen: {n_frozen/1e6:.0f}M, Trainable: {n_train/1e6:.0f}M")

torch.cuda.reset_peak_memory_stats()

# Step 1: Forward (frozen encoder + LM)
src = torch.randint(1, 256, (1, 512), device='cuda')
src[:, -64:] = 0

with torch.no_grad():
    seg_repr, seg_lens = m.segment(src)
M = seg_repr.shape[1]
mem_fwd = torch.cuda.max_memory_allocated() // 1024**2
print(f"Segments: {M}, d_model: {seg_repr.shape[2]}")
print(f"Memory after encoder forward: {mem_fwd} MB")

torch.cuda.reset_peak_memory_stats()

# Step 2: Forward + backward on LM only
seg_repr = seg_repr.detach().requires_grad_(False)
lm_out = m.lm(seg_repr)
tgt = torch.randint(1, 256, (1, M), device='cuda')
loss = torch.nn.functional.cross_entropy(
    lm_out.view(-1, 257), tgt.view(-1), ignore_index=0)
loss.backward()

mem_bwd = torch.cuda.max_memory_allocated() // 1024**2
print(f"Memory after fwd+bwd on LM: {mem_bwd} MB")

# Estimate total with optimizer states
# AdamW: fp32 param copy (4 bytes) + momentum (4) + variance (4) = 12 bytes per param
opt_mem_gb = n_train * 12 / 1024**3
params_fp16_gb = (n_frozen + n_train) * 2 / 1024**3
total_est = params_fp16_gb + opt_mem_gb + (mem_bwd - mem_fwd) / 1024

print(f"\n=== VRAM Estimate ===")
print(f"Params FP16:           {params_fp16_gb:.1f} GB")
print(f"Optimizer (AdamW):     {opt_mem_gb:.1f} GB")
print(f"Activations (peak):    {(mem_bwd - mem_fwd) / 1024:.1f} GB")
print(f"TOTAL estimated:       {total_est:.1f} GB")
print(f"")
print(f"5060 Laptop 8GB:       {'FITS' if total_est < 7.0 else 'TIGHT (< 8GB)' if total_est < 8.0 else 'NO'}")
print(f"RTX 5080 16GB:          FITS")
print(f"")

# Check if gradient checkpointing helps
print(f"Note: Gradient checkpointing can reduce activation memory ~30-40%")
print(f"      With grad_ckpt:  ~{params_fp16_gb + opt_mem_gb + (mem_bwd - mem_fwd) * 0.6 / 1024:.1f} GB")
