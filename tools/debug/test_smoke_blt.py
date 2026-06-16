"""Quick smoke tests for BLT model and training script."""
import torch
from blt_baseline.model import BLTAutoencoder, EntropyPatcher, FixedPatcher

def test_entropy_patcher():
    print("=== Test 1: Entropy Patcher ===")
    m = BLTAutoencoder(
        vocab_size=257, d_model=64, nhead=4, dim_feedforward=128,
        local_layers=1, global_layers=1, decoder_layers=1,
        max_seq_len=32, dropout=0.0, patch_mode='entropy', entropy_theta=0.5,
    )
    src = torch.randint(1, 256, (2, 32))
    src[:, -8:] = 0
    logits, metrics = m(src)
    print(f"  logits: {logits.shape}, avg_patches: {metrics['avg_num_patches']:.1f}")
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}")
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
    loss.backward()
    print(f"  loss: {loss.item():.4f}, backward OK")
    return True

def test_fixed_patcher():
    print("=== Test 2: Fixed Patcher ===")
    m = BLTAutoencoder(
        vocab_size=257, d_model=64, nhead=4, dim_feedforward=128,
        local_layers=1, global_layers=1, decoder_layers=1,
        max_seq_len=32, dropout=0.0, patch_mode='fixed', fixed_patch_size=4,
    )
    src = torch.randint(1, 256, (2, 32))
    src[:, -8:] = 0
    logits, metrics = m(src)
    print(f"  logits: {logits.shape}, avg_patches: {metrics['avg_num_patches']:.1f}")
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 257), src.view(-1), ignore_index=0)
    loss.backward()
    print(f"  loss: {loss.item():.4f}, backward OK")
    return True

def test_legacy_api():
    print("=== Test 3: Legacy API (e2_compare, train.py compat) ===")
    m = BLTAutoencoder(
        vocab_size=257, d_model=64, nhead=4, dim_feedforward=128,
        num_encoder_layers=4, num_decoder_layers=4,
        max_seq_len=32, dropout=0.0,
    )
    src = torch.randint(1, 256, (2, 32))
    src[:, -8:] = 0
    logits, metrics = m(src)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  logits: {logits.shape}, params: {n_params:,}")
    return True

def test_e3_import():
    print("=== Test 4: E3 ablation imports ===")
    from tools.analysis.e3_ablation import (
        AblationConfig, AblationResult, build_loss_ablations,
        build_compression_sweep, build_target_sweep,
    )
    ablations = build_loss_ablations()
    print(f"  Loss ablations: {len(ablations)} ({', '.join(a.name for a in ablations)})")
    sweep = build_compression_sweep([0.05, 0.1, 0.2])
    print(f"  Compression sweep: {len(sweep)} ({', '.join(a.name for a in sweep)})")
    return True

if __name__ == "__main__":
    ok = all([
        test_entropy_patcher(),
        test_fixed_patcher(),
        test_legacy_api(),
        test_e3_import(),
    ])
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
