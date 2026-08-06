"""R0 KDA-LM smoke tests (require fla + CUDA; skip otherwise)."""

import pytest
import torch

pytest.importorskip("fla", reason="fla not installed")

from flued.v36.kda_lm import KDALM, KDALMConfig  # noqa: E402


def _device():
    if not torch.cuda.is_available():
        pytest.skip("cuda required (fla KDA kernels are GPU-only)")
    return "cuda"


def tiny_lm(**overrides):
    base = dict(d_model=64, n_layers=4, kda_head_dim=16, attn_nhead=4, ffn_dim=128, max_seq=32)
    base.update(overrides)
    return KDALMConfig(**base)


def test_forward_shapes_and_finite():
    torch.manual_seed(0)
    model = KDALM(tiny_lm()).to(_device())
    ids = torch.randint(1, 258, (2, 24), device="cuda")
    logits = model(ids)
    assert logits.shape == (2, 24, 258)
    assert torch.isfinite(logits).all()


def test_backward_reaches_all_blocks():
    torch.manual_seed(0)
    model = KDALM(tiny_lm()).to(_device())
    ids = torch.randint(1, 258, (2, 24), device="cuda")
    loss, bpb = model.loss_bpb(ids)
    assert torch.isfinite(loss) and torch.isfinite(bpb)
    loss.backward()
    for name, p in model.named_parameters():
        if "embed" in name or "mixer" in name or "ff_in" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_logits_are_causal():
    """Next-byte logits at position t must not depend on tokens after t --
    guards both the KDA recurrence direction and the attention mask."""
    torch.manual_seed(0)
    model = KDALM(tiny_lm()).to(_device()).eval()
    ids = torch.randint(1, 258, (2, 24), device="cuda")
    ids2 = ids.clone()
    ids2[:, 12:] = torch.randint(1, 258, (2, 12), device="cuda")
    with torch.no_grad():
        l1 = model(ids)
        l2 = model(ids2)
    assert torch.allclose(l1[:, :12], l2[:, :12], atol=1e-5)
    assert not torch.allclose(l1[:, 12:], l2[:, 12:], atol=1e-3)


def test_param_count_matches_flued_budget():
    """R0 is a same-params comparison: arm A must land within ~3% of the
    FLUED v3.6 full stack (47.2M)."""
    model = KDALM(KDALMConfig())
    n = sum(p.numel() for p in model.parameters())
    assert abs(n - 47_203_112) / 47_203_112 < 0.03
