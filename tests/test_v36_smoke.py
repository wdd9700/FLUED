import torch

from flued.v36 import FLUEDV36, V36Config

PAD_ID = 0
MASK_ID = 257


def tiny_config(**overrides):
    base = dict(
        d_byte=64,
        encoder_layers=2,
        segmentor_layers=2,
        nhead=4,
        ffn_dim=128,
        d_mem=96,
        summarizer_slots=2,
        summarizer_hidden=128,
        kda_heads=2,
        kda_head_k=16,
        kda_head_v=32,
        write_hidden=128,
        readout_queries=1,
        d_pack=64,
        d_backbone=64,
        backbone_layers=2,
        backbone_nhead=4,
        backbone_ffn=128,
        decoder_hidden=128,
        decoder_layers=2,
        max_chunks=4,
        max_span=8,
        bytes_per_chunk=8,
        max_positions=8,
    )
    base.update(overrides)
    return V36Config(**base)


def test_forward_shapes_and_finite():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.logits_direct.shape == (2, 4, 8, 258)
    assert out.logits_backbone.shape == (2, 4, 8, 258)
    assert out.package.shape == (2, 1, 64)
    assert torch.isfinite(out.logits_direct).all()
    assert torch.isfinite(out.logits_backbone).all()
    assert torch.isfinite(out.package).all()
    assert out.aux["truncated_tokens"].sum().item() == 0


def test_backward_reaches_core_modules():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    (out.logits_direct.square().mean() + out.logits_backbone.square().mean()).backward()
    expected = ["byte_lookup", "encoder_blocks", "segmentor_blocks", "summarizer", "write_head", "state_machine", "backbone", "decoder"]
    for prefix in expected:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), prefix


def test_batch_position_independence():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config()).eval()
    sample = torch.randint(1, 258, (1, 32))
    other = torch.randint(1, 258, (1, 32))
    with torch.no_grad():
        a = model(torch.cat([sample, other], dim=0)).package[0]
        b = model(torch.cat([other, sample], dim=0)).package[1]
    assert torch.allclose(a, b, atol=1e-5)


def test_padded_tail_no_nan():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    ids = torch.randint(1, 258, (2, 32))
    ids[:, 20:] = PAD_ID
    out = model(ids)
    assert torch.isfinite(out.logits_direct).all()
    assert torch.isfinite(out.logits_backbone).all()
    assert torch.isfinite(out.package).all()


def test_masked_input_no_nan():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    ids = torch.randint(1, 258, (2, 32))
    ids[:, 5:9] = MASK_ID
    out = model(ids)
    assert torch.isfinite(out.logits_direct).all()
    assert torch.isfinite(out.package).all()


def test_truncation_guard_reports_overflow():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(max_chunks=2))
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.aux["truncated_tokens"].sum().item() > 0


def test_k_queries_package_width():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(readout_queries=4, max_positions=8))
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.package.shape == (2, 4, 64)
    assert out.backbone_out.shape == (2, 4, 64)


def test_per_chunk_readout_shapes_and_finite():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.package.shape == (2, 4, 1, 64)  # (B, C, q, d_pack)
    assert out.backbone_out.shape == (2, 4, 64)
    assert out.logits_direct.shape == (2, 4, 8, 258)
    assert out.logits_backbone.shape == (2, 4, 8, 258)
    assert torch.isfinite(out.logits_direct).all()
    assert torch.isfinite(out.logits_backbone).all()
    assert torch.isfinite(out.package).all()
    assert out.aux["truncated_tokens"].sum().item() == 0


def test_per_chunk_readout_backward_reaches_core():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    (out.logits_direct.square().mean() + out.logits_backbone.square().mean()).backward()
    for prefix in ["summarizer", "write_head", "state_machine", "backbone", "decoder"]:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), prefix
