"""Profile BLT bottlenecks with CUDA events for accurate timing."""
import time
import torch
from blt_baseline.model import BLTAutoencoder

device = 'cuda'
m = BLTAutoencoder(
    vocab_size=257, d_model=1024, nhead=16, dim_feedforward=4096,
    local_layers=2, global_layers=10, decoder_layers=12,
    max_seq_len=512, dropout=0.0, patch_mode='entropy', entropy_theta=3.5,
).to(device)
m.train()
print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")

src = torch.randint(1, 256, (1, 512), device=device)
src[:, -64:] = 0

# Warmup
for _ in range(5):
    with torch.autocast('cuda', torch.float16):
        logits, metrics = m(src)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
    loss.backward()
    m.zero_grad()

torch.cuda.synchronize()

N = 30
results = []

for i in range(N):
    m.zero_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.autocast('cuda', torch.float16):
        logits, metrics = m(src)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
    loss.backward()
    
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    results.append(dt)

results.sort()
avg = sum(results) / len(results)
med = results[len(results)//2]
p10 = results[len(results)//10]
p90 = results[len(results)*9//10]

print(f"\nFull fwd+bwd over {N} iters (batch=1, seq=512, FP16):")
print(f"  avg:  {avg:.1f} ms")
print(f"  med:  {med:.1f} ms")
print(f"  p10:  {p10:.1f} ms")
print(f"  p90:  {p90:.1f} ms")
print(f"  avg_segments: {metrics['avg_num_patches']:.1f}")
print(f"  steps/sec: {1000/avg:.1f}")
print(f"  effective steps/min (grad_accum=16): {1000/avg*60/16:.1f}")

# Now profile sub-components using CUDA events (more accurate)
print("\n--- Sub-component timing (CUDA events) ---")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Local encoder
m.zero_grad()
local_repr = m.local_encoder(src)  # pre-compute once for patcher+transformer tests

# Patcher
start.record()
patches, seg_lens, avg_seg = m.patcher(local_repr, src)
end.record()
torch.cuda.synchronize()
t_patcher = start.elapsed_time(end)

# Global Transformer (on patches)
start.record()
p = m.global_pos_enc(patches)
global_repr = m.global_transformer(p)
end.record()
torch.cuda.synchronize()
t_global = start.elapsed_time(end)

# Decoder
start.record()
logits = m.local_decoder(global_repr, seg_lens, original_len=src.size(1))
end.record()
torch.cuda.synchronize()
t_decoder = start.elapsed_time(end)

# Backward only
m.zero_grad()
with torch.autocast('cuda', torch.float16):
    logits, metrics = m(src)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
start.record()
loss.backward()
end.record()
torch.cuda.synchronize()
t_bwd = start.elapsed_time(end)

print(f"  Patcher (entropy+pool):  {t_patcher:7.2f} ms")
print(f"  Global Transformer:      {t_global:7.2f} ms")
print(f"  Decoder (broadcast):     {t_decoder:7.2f} ms")
print(f"  Backward:                {t_bwd:7.2f} ms")
total = t_patcher + t_global + t_decoder + t_bwd
print(f"  SUM (fwd sub+bwd):       {total:7.2f} ms")
print(f"  Patcher ratio:           {t_patcher/total*100:5.1f}%")
print(f"  Decoder ratio:           {t_decoder/total*100:5.1f}%")
print(f"  Global TF ratio:         {t_global/total*100:5.1f}%")
