"""Debug BLT v2 — variable-length segment broadcast."""
import torch
from blt_baseline.model import BLTAutoencoder

for mode, theta in [("entropy", 3.5), ("entropy", 0.5), ("fixed", None)]:
    label = f"{mode}" + (f" theta={theta}" if theta else "")
    print(f"\n=== {label} ===")
    m = BLTAutoencoder(
        vocab_size=257, d_model=64, nhead=4, dim_feedforward=128,
        local_layers=1, global_layers=1, decoder_layers=1,
        max_seq_len=64, dropout=0.0, patch_mode=mode,
        entropy_theta=theta or 3.5, fixed_patch_size=4,
    )
    src = torch.randint(1, 256, (2, 64))
    src[:, -16:] = 0
    logits, metrics = m(src)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  logits: {logits.shape}, segs: {metrics['avg_num_patches']:.1f}, params: {n_params:,}")
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
    loss.backward()
    print(f"  loss: {loss.item():.4f}, backward OK")

print("\n=== 300M entropy ===")
m = BLTAutoencoder(
    vocab_size=257, d_model=1024, nhead=16, dim_feedforward=4096,
    local_layers=2, global_layers=10, decoder_layers=12,
    max_seq_len=512, dropout=0.0, patch_mode='entropy', entropy_theta=3.5,
)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"  params: {n:,}")
m = m.cuda()
src = torch.randint(1, 256, (2, 512), device='cuda')
src[:, -64:] = 0
with torch.autocast('cuda', torch.float16):
    logits, metrics = m(src)
print(f"  logits: {logits.shape}, segs: {metrics['avg_num_patches']:.1f}")
print(f"  GPU mem: {torch.cuda.max_memory_allocated() // 1024**2} MB")
