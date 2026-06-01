"""Check BLT 300m params and VRAM before training."""
import torch
from blt_baseline.model import BLTAutoencoder

m = BLTAutoencoder(
    vocab_size=257, d_model=1024, nhead=16, dim_feedforward=4096,
    local_layers=2, global_layers=10, decoder_layers=12,
    max_seq_len=512, dropout=0.0, patch_mode='entropy', entropy_theta=0.5,
)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"BLT 300m params: {n:,}")

m = m.cuda()
src = torch.randint(1, 256, (2, 512), device='cuda')
src[:, -64:] = 0
with torch.autocast('cuda', torch.float16):
    logits, metrics = m(src)
print(f"logits: {logits.shape}, avg_patches: {metrics['avg_num_patches']:.1f}")
print(f"GPU mem peak: {torch.cuda.max_memory_allocated() // 1024**2} MB")
torch.cuda.reset_peak_memory_stats()
