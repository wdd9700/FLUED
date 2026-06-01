"""Smoke test new BLT architecture (Stage 1 + Stage 2)."""
import torch
from blt_baseline.model import ByteLanguageModel, BLTAutoencoder

print("=== ByteLM ===")
lm = ByteLanguageModel(vocab_size=257, d_model=128, nhead=4, dim_feedforward=512, num_layers=2, max_len=64)
src = torch.randint(1, 256, (2, 64))
src[:, -16:] = 0
hidden, logits = lm(src)
print(f"  hidden: {hidden.shape}, logits: {logits.shape}")
loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
loss.backward()
print(f"  loss: {loss.item():.4f}, backward OK")

# Entropy
hidden2, entropy = lm.compute_entropy(src)
print(f"  entropy mean: {entropy[src!=0].mean().item():.3f}")
print(f"  entropy max: {entropy[src!=0].max().item():.3f}")

print("\n=== BLTAutoencoder (frozen LM) ===")
blt = BLTAutoencoder(
    vocab_size=257, d_model=256, nhead=4, dim_feedforward=512,
    global_layers=2, decoder_layers=2,
    local_lm=lm, local_lm_d_model=128,
    patch_mode='entropy', entropy_theta=3.5, max_seq_len=64,
)
blt.freeze_local_lm()
n_trainable = sum(p.numel() for p in blt.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in blt.parameters())
print(f"  trainable: {n_trainable:,} / total: {n_total:,}")

logits, metrics = blt(src)
print(f"  logits: {logits.shape}, avg_patches: {metrics['avg_num_patches']:.1f}")
loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
loss.backward()
print(f"  loss: {loss.item():.4f}, backward OK")

print("\n=== BLTAutoencoder (joint, no frozen LM) ===")
blt2 = BLTAutoencoder(
    vocab_size=257, d_model=128, nhead=4, dim_feedforward=512,
    global_layers=1, decoder_layers=1,
    local_lm=None, local_lm_d_model=128, max_seq_len=64,
)
logits2, metrics2 = blt2(src)
print(f"  logits: {logits2.shape}, avg_patches: {metrics2['avg_num_patches']:.1f}")
loss2 = torch.nn.functional.cross_entropy(logits2.view(-1, 257), src.view(-1), ignore_index=0)
loss2.backward()
print(f"  loss: {loss2.item():.4f}, backward OK")

print("\nALL TESTS PASSED")
