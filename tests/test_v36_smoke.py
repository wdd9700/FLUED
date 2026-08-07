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
        # legacy path defaults (canonical defaults moved to per-chunk+dit in v36.2)
        per_chunk_readout=False,
        summarizer_type="slot",
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


def test_dit_summarizer_shapes_finite_backward():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(summarizer_type="dit", summarizer_dit_layers=2))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.memory.shape == (2, 4, 96)  # (B, C, d_mem)
    assert out.logits_direct.shape == (2, 4, 8, 258)
    assert torch.isfinite(out.logits_direct).all()
    assert torch.isfinite(out.memory).all()
    (out.logits_direct.square().mean() + out.logits_backbone.square().mean()).backward()
    grads = [p.grad for n, p in model.named_parameters() if n.startswith("summarizer")]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_dit_summarizer_per_chunk_readout_combo():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(summarizer_type="dit", per_chunk_readout=True))
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.package.shape == (2, 4, 1, 64)
    assert torch.isfinite(out.logits_backbone).all()


def test_prefix_task_positions_and_shapes():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True, prefix_task=True, prefix_positions=3))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    assert out.prefix is not None
    positions = [i for i, _ld, _lb in out.prefix]
    assert positions[-1] == 3  # final position always included (max_chunks-1 = 3)
    assert len(positions) <= 3
    for i, ld, lb in out.prefix:
        assert ld.shape == (2, i + 1, 8, 258)
        assert lb.shape == (2, i + 1, 8, 258)
        assert torch.isfinite(ld).all() and torch.isfinite(lb).all()
    model.eval()
    out = model(ids)
    assert out.prefix[-1][0] == 3


def test_prefix_task_backward_reaches_core():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True, prefix_task=True, prefix_positions=3))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = model(ids)
    loss = sum(ld.square().mean() + lb.square().mean() for _i, ld, lb in out.prefix)
    loss.backward()
    for prefix in ["summarizer", "write_head", "state_machine", "backbone", "decoder"]:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), prefix


def test_kda_fla_parity():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    try:
        import fla.ops.kda  # noqa: F401
    except Exception:
        pytest.skip("fla not installed")
    torch.manual_seed(0)
    base = FLUEDV36(tiny_config(per_chunk_readout=True)).cuda()
    twin = FLUEDV36(tiny_config(per_chunk_readout=True, kda_impl="fla")).cuda()
    twin.load_state_dict(base.state_dict())
    ids = torch.randint(1, 258, (2, 32), device="cuda")
    with torch.no_grad():
        out_ref = base(ids)
        out_fla = twin(ids)
    assert torch.allclose(out_ref.package, out_fla.package, atol=5e-2, rtol=5e-2)
    assert abs(out_ref.state_norm.item() - out_fla.state_norm.item()) < 0.1


def test_pointwise_backbone_is_per_readout():
    """mlp mode (E32: judged dead, retained behind flag): output row i must
    depend only on input row i, so chunk permutation permutes outputs."""
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(backbone_mode="mlp"))
    from flued.v36.model import PointwiseBackbone, TinyBackbone

    assert isinstance(model.backbone, PointwiseBackbone)
    assert isinstance(FLUEDV36(tiny_config()).backbone, TinyBackbone)  # default = attn (E32)
    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        out = model.backbone(x)
        perm = [2, 0, 3, 1]
        out_perm = model.backbone(x[:, perm])
    assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


def _stream_columns(model, gates, n_chunks):
    sm = model.state_machine
    state = sm.init_stream_state(gates["k"].size(0), gates["k"].device)
    cols = []
    for i in range(n_chunks):
        gi = {k: (v[:, i : i + 1] if k != "alpha" else v) for k, v in gates.items()}
        pkg_i, state = sm.stream_step(gi, state)
        cols.append(pkg_i)
    return torch.cat(cols, dim=1)


def test_backbone_readout_final_is_k1():
    """k=1 backbone interface (user-ruled 2026-08-06): in final mode the
    backbone output is a single conditioning vector per sample (the transform
    of the final state's readout), while decoding still covers every chunk."""
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True, backbone_readout="final"))
    ids = torch.randint(1, 258, (2, 32))
    with torch.no_grad():
        out = model(ids)
    assert out.backbone_out.size(1) == 1
    assert out.logits_backbone.shape == (2, 4, 8, 258)
    assert torch.isfinite(out.logits_backbone).all()
    # default stays per-chunk (C readouts)
    model2 = FLUEDV36(tiny_config(per_chunk_readout=True))
    with torch.no_grad():
        out2 = model2(ids)
    assert out2.backbone_out.size(1) == out2.chunks.span_embeddings.size(1)


def test_state_channel_off_is_chunk_local():
    """R1 relative-baseline form (spec section 4): with state_channel=False
    each package column depends ONLY on its own chunk -- permuting chunks
    permutes columns. The default (state channel on) must break that property."""
    torch.manual_seed(0)
    cfg = tiny_config(per_chunk_readout=True, state_channel=False)
    model = FLUEDV36(cfg)
    memory = torch.randn(2, 3, cfg.d_mem)
    chunk_mask = torch.ones(2, 3, dtype=torch.bool)
    perm = [2, 0, 1]
    with torch.no_grad():
        gates = model.write_head(memory)
        package, state_norm = model.state_machine(gates, chunk_mask)
        gates_p = {k: (v[:, perm] if k in ("k", "v", "beta") else v) for k, v in gates.items()}
        package_p, _ = model.state_machine(gates_p, chunk_mask)
    assert state_norm.item() == 0.0
    assert torch.allclose(package_p, package[:, perm], atol=1e-5)
    # default: serial channel on -> later columns carry earlier-chunk history
    model2 = FLUEDV36(tiny_config(per_chunk_readout=True))
    with torch.no_grad():
        gates2 = model2.write_head(memory)
        p2, sn2 = model2.state_machine(gates2, chunk_mask)
        gates2_p = {k: (v[:, perm] if k in ("k", "v", "beta") else v) for k, v in gates2.items()}
        p2p, _ = model2.state_machine(gates2_p, chunk_mask)
    assert sn2.item() > 0
    assert not torch.allclose(p2p, p2[:, perm], atol=1e-5)


def test_kda_stream_step_matches_batched_torch():
    """Streaming chunk-by-chunk encoding (carried state) must reproduce the
    batched per-chunk readout column-for-column -- this is the O(C^2) -> O(C)
    inference primitive for the paging/generation line."""
    torch.manual_seed(0)
    cfg = tiny_config(per_chunk_readout=True)
    model = FLUEDV36(cfg)
    memory = torch.randn(2, 3, cfg.d_mem)
    gates = model.write_head(memory)
    chunk_mask = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        package, _ = model.state_machine(gates, chunk_mask)
        streamed = _stream_columns(model, gates, 3)
    assert torch.allclose(package, streamed, atol=1e-5)


def test_kda_stream_step_fla_matches_batched():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    try:
        import fla.ops.kda  # noqa: F401
    except Exception:
        pytest.skip("fla not installed")
    torch.manual_seed(0)
    cfg = tiny_config(per_chunk_readout=True, kda_impl="fla")
    model = FLUEDV36(cfg).cuda()
    memory = torch.randn(2, 3, cfg.d_mem, device="cuda")
    gates = model.write_head(memory)
    chunk_mask = torch.ones(2, 3, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        package, _ = model.state_machine(gates, chunk_mask)
        streamed = _stream_columns(model, gates, 3)
    assert torch.allclose(package, streamed, atol=5e-2, rtol=5e-2)
