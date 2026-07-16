from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flued.data import text_to_byte_ids
from flued.v34.model import (
    DenseNoSelfMemory,
    FLUEDV34Probe,
    FLUEDV34ProbeConfig,
    SpanDecoder,
    _bidirectional_alibi_bias,
    _plastic_signed_confidence,
    _sinusoidal_position,
    load_v34_state_dict_compatible,
)
from flued.v34.rate_emit import MarginalCodingRateSelector, ReadoutEmitController
from tools.train.v3_4.train_v34_pos_ar_probe import (
    apply_boundary_curriculum,
    boundary_compute_budget_loss,
    boundary_threshold_calibration_loss,
    boundary_threshold_density_loss,
    boundary_threshold_positive_margin_loss,
    boundary_rate_minimum_ratio_loss,
)


def _tiny_model() -> FLUEDV34Probe:
    return FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_position=True,
            use_ar=False,
            max_chunks=8,
            max_span=8,
            noise_scale=0.0,
        )
    )


def test_v34_main_codec_loss_reaches_segmentor() -> None:
    torch.manual_seed(7)
    model = _tiny_model().train()
    ids = torch.tensor([text_to_byte_ids("order matters.")])
    out = model(ids)
    chunk_ids = out.chunks.chunk_ids[0]
    offsets = out.chunks.offsets[0]
    valid = chunk_ids.ge(0)
    targets = torch.zeros_like(out.byte_logits[..., 0], dtype=torch.long)
    targets[0, chunk_ids[valid], offsets[valid]] = ids[0, valid]
    active = out.chunks.token_mask
    loss = F.cross_entropy(out.byte_logits[active].float(), targets[active])
    loss.backward()
    grad = model.segmentor_head[-1].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0


def test_v34_utf8_continuations_cannot_cut() -> None:
    model = _tiny_model().eval()
    ids = torch.tensor([text_to_byte_ids("中文")])
    out = model(ids)
    raw = ids - 1
    continuation = raw.ge(0x80) & raw.le(0xBF)
    assert continuation.any()
    assert not out.policy.hard_cut[continuation].any()
    assert out.policy.force_continue[continuation].all()


def test_v34_policy_is_exactly_two_thresholds() -> None:
    model = _tiny_model()
    confidence = torch.tensor([[-0.95, 0.80, 0.95]])
    valid = torch.ones_like(confidence, dtype=torch.bool)
    out = model.policy(confidence, valid)
    assert out.soft_transition.tolist() == [[False, True, False]]
    assert out.hard_cut.tolist() == [[True, False, True]]
    assert not out.force_continue.any()
    assert set(out.aux) == {"tau_cut", "tau_trans"}


def test_v34_decoder_has_no_memory_argument() -> None:
    assert list(inspect.signature(SpanDecoder.forward).parameters) == ["self", "readout", "chunk_mask"]


def test_v34_dense_memory_excludes_current_chunk() -> None:
    model = _tiny_model().eval()
    ids = torch.tensor([text_to_byte_ids("abcdefghijklmno")])
    out = model(ids)
    assert out.aux["self_allowed"].item() == 0.0


def test_v34_cut_capacity_never_drops_valid_bytes() -> None:
    valid = torch.ones((2, 512), dtype=torch.bool)
    requested = torch.ones_like(valid)
    executed, overflow = FLUEDV34Probe._capacity_safe_cuts(requested, valid, 64, 128)
    assert executed.sum(dim=1).tolist() == [61, 61]
    assert overflow.tolist() == [451, 451]


def test_v34_plain_lookup_is_shared_with_decoder() -> None:
    config = _tiny_model().config
    config.use_structured_lookup = False
    model = FLUEDV34Probe(config)
    assert model.decoder.byte_lookup is model.plain_byte_lookup


def test_v34_marginal_rate_topk_obeys_budget_and_forbidden_positions() -> None:
    torch.manual_seed(3)
    selector = MarginalCodingRateSelector(16, rate_dim=4, mode="exact")
    features = torch.randn(2, 32, 16)
    valid = torch.ones((2, 32), dtype=torch.bool)
    forbidden = torch.zeros_like(valid)
    forbidden[:, 5:9] = True
    out = selector(features, valid, forbidden, max_chunks=8, fixed_chunks=6)
    assert out.hard_cut.sum(dim=1).tolist() == [6, 6]
    assert not out.hard_cut[forbidden].any()
    assert out.soft_cut.requires_grad
    assert torch.isfinite(out.marginal_rate).all()


def test_v34_emit_controller_keeps_fallback_and_straight_through_gradient() -> None:
    controller = ReadoutEmitController(16, initial_extra_probability=0.1)
    candidates = torch.randn(2, 3, 5, 16, requires_grad=True)
    chunk_mask = torch.tensor([[True, True, False], [True, False, False]])
    out = controller(candidates, chunk_mask)
    assert out.hard[..., 0].equal(chunk_mask)
    assert not out.hard[..., 1:][chunk_mask.unsqueeze(-1).expand(-1, -1, 4)].any()
    (out.straight_through.sum()).backward()
    assert controller.head.bias.grad is not None
    assert controller.head.bias.grad.abs().item() > 0


def test_v34_emit_threshold_monotonically_reduces_extra_readouts() -> None:
    torch.manual_seed(17)
    controller = ReadoutEmitController(16, initial_extra_probability=0.5, threshold=0.2)
    candidates = torch.randn(2, 3, 5, 16)
    chunk_mask = torch.ones(2, 3, dtype=torch.bool)

    low = controller(candidates, chunk_mask)
    controller.threshold = 0.8
    high = controller(candidates, chunk_mask)

    assert high.hard[..., 0].equal(chunk_mask)
    assert int(high.hard[..., 1:].sum()) <= int(low.hard[..., 1:].sum())


def test_v34_rate_emit_main_loss_reaches_both_controllers() -> None:
    torch.manual_seed(11)
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=4,
            ar_hidden=8,
            use_ar=False,
            boundary_mode="marginal_rate_topk",
            coding_rate_dim=4,
            fixed_chunk_budget=3,
            use_emit_controller=True,
            max_chunks=5,
            max_span=16,
            noise_scale=0.0,
        )
    ).train()
    ids = torch.tensor([text_to_byte_ids("ordered bytes matter")])
    out = model(ids)
    targets = torch.zeros_like(out.byte_logits[..., 0], dtype=torch.long)
    valid = out.chunks.chunk_ids[0].ge(0)
    targets[0, out.chunks.chunk_ids[0, valid], out.chunks.offsets[0, valid]] = ids[0, valid]
    loss = F.cross_entropy(out.byte_logits[out.chunks.token_mask].float(), targets[out.chunks.token_mask])
    loss.backward()
    rate_grad = model.coding_rate_selector.proj.weight.grad
    emit_grad = model.emit_controller.head.bias.grad
    assert rate_grad is not None and rate_grad.abs().sum().item() > 0
    assert emit_grad is not None and emit_grad.abs().sum().item() > 0


def test_v34_uniform_budget_is_lossless_and_avoids_utf8_continuations() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=4,
            ar_hidden=8,
            use_ar=False,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=12,
            max_span=16,
            noise_scale=0.0,
        )
    ).eval()
    ids = torch.tensor([text_to_byte_ids("中文 mixed bytes and words")])
    out = model(ids)
    raw = ids - 1
    continuation = raw.ge(0x80) & raw.le(0xBF)
    assert not out.policy.hard_cut[continuation].any()
    assert out.chunks.pack_info["truncated_tokens"].sum().item() == 0


def test_v34_capacity_guard_applies_to_clustered_coding_rate_cuts() -> None:
    requested = torch.zeros(1, 64, dtype=torch.bool)
    requested[:, :12] = True
    valid = torch.ones_like(requested)

    executed, overflow = FLUEDV34Probe._capacity_safe_cuts(
        requested,
        valid,
        max_chunks=8,
        max_span=16,
    )

    # Four chunks may still be forced by max_span, so only five requested
    # starts are safe (the first start overlaps the first forced span).
    assert executed.sum().item() == 5
    assert overflow.item() == 7


def test_v34_boundary_curriculum_switches_once_at_requested_step() -> None:
    model = _tiny_model()
    model.config.boundary_mode = "uniform_budget"
    model.coding_rate_selector.mode = "l2"
    args = Namespace(
        boundary_curriculum_switch_step=3,
        boundary_curriculum_mode="marginal_rate_topk",
        boundary_curriculum_coding_rate_mode="l2",
    )
    assert not apply_boundary_curriculum(model, args, 2)
    assert model.config.boundary_mode == "uniform_budget"
    assert apply_boundary_curriculum(model, args, 3)
    assert model.config.boundary_mode == "marginal_rate_topk"
    assert model.coding_rate_selector.mode == "l2"
    assert args.boundary_mode == "marginal_rate_topk"
    assert args.boundary_coding_rate_mode == "l2"
    assert not apply_boundary_curriculum(model, args, 4)


def test_v34_boundary_curriculum_blends_scores_before_final_l2() -> None:
    model = _tiny_model()
    model.config.boundary_mode = "uniform_budget"
    args = Namespace(
        boundary_curriculum_switch_step=3,
        boundary_curriculum_transition_steps=2,
        boundary_curriculum_mode="marginal_rate_topk",
        boundary_curriculum_coding_rate_mode="l2",
    )
    assert apply_boundary_curriculum(model, args, 3)
    assert model.config.boundary_mode == "uniform_l2_blend"
    assert model.config.boundary_blend_alpha == 0.0
    assert args.boundary_mode == "uniform_l2_blend"
    assert args.boundary_blend_alpha == 0.0
    assert not apply_boundary_curriculum(model, args, 4)
    assert model.config.boundary_blend_alpha == pytest.approx(0.5)
    assert apply_boundary_curriculum(model, args, 5)
    assert model.config.boundary_mode == "marginal_rate_topk"
    assert model.config.boundary_blend_alpha == 1.0


def test_v34_zero_blend_matches_uniform_hard_boundaries() -> None:
    model = _tiny_model().eval()
    model.config.boundary_mode = "uniform_budget"
    model.config.coding_rate_mode = "l2"
    model.coding_rate_selector.mode = "l2"
    ids = torch.tensor([text_to_byte_ids("中文 mixed bytes and ordered_words")])
    with torch.no_grad():
        uniform = model(ids).policy.hard_cut
        model.config.boundary_mode = "uniform_l2_blend"
        model.config.boundary_blend_alpha = 0.0
        blended = model(ids).policy.hard_cut
    assert torch.equal(uniform, blended)


def test_v34_zero_blend_preserves_uniform_soft_bridge() -> None:
    selector = MarginalCodingRateSelector(dim=8, rate_dim=4, mode="l2")
    features = torch.randn(1, 12, 8)
    valid = torch.ones(1, 12, dtype=torch.bool)
    forbidden = torch.zeros_like(valid)
    anchor = torch.zeros(1, 12)
    anchor[:, [0, 4, 8]] = 1.0
    out = selector(
        features,
        valid,
        forbidden,
        max_chunks=3,
        fixed_chunks=3,
        anchor_score=anchor,
        blend_alpha=0.0,
    )
    assert torch.equal(out.hard_cut, anchor.bool())
    assert torch.equal(out.soft_cut, anchor)


def test_v34_coding_rate_budget_includes_mandatory_first_cut() -> None:
    selector = MarginalCodingRateSelector(dim=8, rate_dim=4, mode="l2")
    features = torch.randn(2, 12, 8)
    valid = torch.ones(2, 12, dtype=torch.bool)
    forbidden = torch.zeros_like(valid)
    out = selector(features, valid, forbidden, max_chunks=5, fixed_chunks=3)
    assert out.hard_cut[:, 0].all()
    assert torch.equal(out.hard_cut.sum(dim=1), torch.tensor([3, 3]))


def test_v34_coding_rate_single_chunk_budget_only_selects_first() -> None:
    selector = MarginalCodingRateSelector(dim=8, rate_dim=4, mode="l2")
    features = torch.randn(1, 12, 8)
    valid = torch.ones(1, 12, dtype=torch.bool)
    forbidden = torch.zeros_like(valid)
    out = selector(features, valid, forbidden, max_chunks=5, fixed_chunks=1)
    assert out.hard_cut.sum().item() == 1
    assert out.hard_cut[0, 0]
    assert out.soft_cut.sum().item() == 1.0


def _memory_locality_model(use_memory: bool) -> FLUEDV34Probe:
    return FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_ar=False,
            use_memory=use_memory,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=2,
            max_span=8,
            noise_scale=0.0,
        )
    ).eval()


def test_v34_no_memory_keeps_other_chunks_out_of_readout() -> None:
    model = _memory_locality_model(use_memory=False)
    first = "abcdefgh"
    ids_a = torch.tensor([text_to_byte_ids(first + "ijklmnop")])
    ids_b = torch.tensor([text_to_byte_ids(first + "QRSTUVWX")])
    with torch.no_grad():
        out_a = model(ids_a)
        out_b = model(ids_b)
    assert torch.equal(out_a.readout_z[:, 0], out_b.readout_z[:, 0])
    assert out_a.memory_z.count_nonzero().item() == 0
    assert out_a.aux["memory_gate_mean"].item() == 0.0


def test_v34_memory_is_the_only_cross_chunk_readout_path() -> None:
    model = _memory_locality_model(use_memory=True)
    first = "abcdefgh"
    ids_a = torch.tensor([text_to_byte_ids(first + "ijklmnop")])
    ids_b = torch.tensor([text_to_byte_ids(first + "QRSTUVWX")])
    with torch.no_grad():
        readout_a = model(ids_a).readout_z[:, 0]
        readout_b = model(ids_b).readout_z[:, 0]
    assert not torch.equal(readout_a, readout_b)


def test_v34_logic_transition_prior_has_no_active_parameter() -> None:
    names = dict(_memory_locality_model(use_memory=True).named_parameters())
    assert "logic_transition_prior" not in names


def test_v34_prompt_position_replaces_layered_rope() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            position_strategy="prompt_additive",
            max_chunks=2,
            max_span=8,
        )
    )
    assert model.position_strategy == "prompt_additive"
    assert model.segmentor_blocks[0].attn.use_rope is False
    assert model.readout_pool.use_position is False
    assert model.memory_read.use_position is False
    assert model.interpreter_blocks[0].attn.use_rope is False


def test_v34_prompt_position_side_channel_distinguishes_offsets() -> None:
    encoded = _sinusoidal_position(4, 32, torch.device("cpu"), torch.float32)
    assert encoded.shape == (4, 32)
    assert not torch.equal(encoded[0], encoded[1])


def test_v34_prompt_plus_local_rope_keeps_local_order_without_chunk_rope() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            position_strategy="prompt_plus_local_rope",
            memory_use_position=True,
            max_chunks=2,
            max_span=8,
        )
    )
    assert model.use_prompt_position is True
    assert model.segmentor_blocks[0].attn.use_rope is True
    assert model.readout_pool.use_position is True
    assert model.interpreter_blocks[0].attn.use_rope is True
    assert model.memory_read.use_position is False


def test_v34_bidirectional_alibi_is_symmetric_and_distance_monotonic() -> None:
    bias = _bidirectional_alibi_bias(8, 4, torch.device("cpu"), torch.float32)
    assert torch.equal(bias, bias.transpose(-1, -2))
    assert torch.all(bias[:, 0, 0] > bias[:, 0, 1])
    assert torch.all(bias[:, 0, 1] > bias[:, 0, 7])


def test_v34_byte_alibi_diagnostics_include_distance_bias() -> None:
    memory_read = DenseNoSelfMemory(
        dim=4,
        nhead=1,
        position_mode="byte_alibi",
        access_mode="all",
    )
    torch.nn.init.zeros_(memory_read.q.weight)
    torch.nn.init.zeros_(memory_read.k.weight)
    readout = torch.zeros(1, 2, 1, 4)
    memory = torch.zeros(1, 2, 1, 4)
    chunk_mask = torch.ones(1, 2, dtype=torch.bool)
    _, aux = memory_read(
        readout,
        memory,
        chunk_mask,
        chunk_anchors=torch.tensor([[0.0, 100.0]]),
        diagnostics=True,
    )
    assert aux["memory_attention_current_share"].item() > 0.5


def test_v34_legacy_loader_allows_only_inactive_missing_paths() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            memory_scale_mode="fixed",
            current_memory_mode="off",
        )
    )
    state = model.state_dict()
    removed = {
        "memory_scale_logit",
        "current_memory_scale_logit",
        *{name for name in state if name.startswith("current_memory_read.")},
    }
    legacy = {name: tensor for name, tensor in state.items() if name not in removed}
    report = load_v34_state_dict_compatible(model, legacy)
    assert set(report["ignored_missing"]) == removed

    active_current = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            memory_scale_mode="fixed",
            current_memory_mode="separate_e2e",
        )
    )
    with pytest.raises(RuntimeError, match="checkpoint/model mismatch"):
        load_v34_state_dict_compatible(active_current, legacy)


def test_v34_prompt_alibi_only_affects_segmentor_attention() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_prompt_alibi=True,
            max_chunks=4,
            max_span=16,
        )
    )
    assert model.segmentor_blocks[0].attn.use_alibi
    assert not model.interpreter_blocks[0].attn.use_alibi


@pytest.mark.parametrize("mode", ["none", "chunk_rope", "byte_alibi"])
def test_v34_memory_position_modes_are_finite(mode: str) -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_memory=True,
            memory_position_mode=mode,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=4,
            max_span=16,
            noise_scale=0.0,
        )
    ).eval()
    ids = torch.tensor([text_to_byte_ids("abcdefghABCDEFGH")])
    with torch.no_grad():
        out = model(ids)
    assert torch.isfinite(out.readout_z).all()
    assert model.memory_read.position_mode == mode


@pytest.mark.parametrize(
    ("access_mode", "current_mode"),
    [
        ("other_only", "off"),
        ("all", "off"),
        ("other_only", "separate_detached"),
        ("other_only", "separate_e2e"),
        ("none", "separate_e2e"),
    ],
)
def test_v34_current_memory_paths_are_finite(access_mode: str, current_mode: str) -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_memory=True,
            memory_access_mode=access_mode,
            current_memory_mode=current_mode,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=4,
            max_span=16,
            noise_scale=0.0,
        )
    ).eval()
    ids = torch.tensor([text_to_byte_ids("abcdefghABCDEFGH")])
    with torch.no_grad():
        out = model(ids)
    assert torch.isfinite(out.readout_z).all()


def test_v34_current_memory_modules_keep_parameter_count_constant() -> None:
    configs = [
        FLUEDV34ProbeConfig(memory_access_mode="other_only", current_memory_mode="off"),
        FLUEDV34ProbeConfig(memory_access_mode="all", current_memory_mode="off"),
        FLUEDV34ProbeConfig(memory_access_mode="other_only", current_memory_mode="separate_detached"),
        FLUEDV34ProbeConfig(memory_access_mode="other_only", current_memory_mode="separate_e2e"),
        FLUEDV34ProbeConfig(memory_access_mode="none", current_memory_mode="separate_e2e"),
    ]
    counts = {sum(parameter.numel() for parameter in FLUEDV34Probe(config).parameters()) for config in configs}
    assert len(counts) == 1


@pytest.mark.parametrize(
    ("context_norm", "scale_mode"),
    [("none", "fixed"), ("layernorm", "fixed"), ("layernorm", "bounded")],
)
def test_v34_memory_scale_paths_are_finite(context_norm: str, scale_mode: str) -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_memory=True,
            memory_access_mode="other_only",
            current_memory_mode="separate_detached",
            memory_context_norm=context_norm,
            memory_scale_mode=scale_mode,
            memory_residual_scale=0.03 if scale_mode == "bounded" else 0.1,
            current_memory_scale=0.03,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=4,
            max_span=16,
            noise_scale=0.0,
        )
    )
    ids = torch.tensor([text_to_byte_ids("abcdefghABCDEFGH")])
    out = model(ids)
    assert torch.isfinite(out.readout_z).all()
    assert -1.0e-6 <= out.aux["memory_effective_scale"].item() <= 0.100001
    assert -1.0e-6 <= out.aux["current_memory_effective_scale"].item() <= 0.100001
    if context_norm == "layernorm":
        assert out.aux["memory_context_norm"].item() == pytest.approx(32**0.5, rel=0.05)
        assert out.aux["current_memory_context_norm"].item() == pytest.approx(32**0.5, rel=0.05)


def test_v34_bounded_memory_scales_receive_gradients() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=2,
            ar_hidden=8,
            use_memory=True,
            current_memory_mode="separate_detached",
            memory_context_norm="layernorm",
            memory_scale_mode="bounded",
            memory_residual_scale=0.03,
            current_memory_scale=0.03,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=4,
            max_span=16,
            noise_scale=0.0,
        )
    )
    ids = torch.tensor([text_to_byte_ids("abcdefghABCDEFGH")])
    model(ids).byte_logits.float().square().mean().backward()
    assert model.memory_scale_logit.grad is not None
    assert model.current_memory_scale_logit.grad is not None


def test_v34_memory_scale_ablation_parameter_count_is_constant() -> None:
    configs = [
        FLUEDV34ProbeConfig(memory_context_norm="none", memory_scale_mode="fixed"),
        FLUEDV34ProbeConfig(memory_context_norm="layernorm", memory_scale_mode="fixed"),
        FLUEDV34ProbeConfig(
            memory_context_norm="layernorm",
            memory_scale_mode="bounded",
            memory_residual_scale=0.03,
        ),
    ]
    counts = {sum(parameter.numel() for parameter in FLUEDV34Probe(config).parameters()) for config in configs}
    assert len(counts) == 1


def test_v34_diagonal_rate_is_prefix_marginal_not_pointwise_energy() -> None:
    torch.manual_seed(42)
    selector = MarginalCodingRateSelector(dim=8, rate_dim=4, epsilon=1.0, mode="diag")
    features = torch.randn(2, 7, 8)
    actual = selector.marginal_rate(features)
    normalized = selector.norm(features).float()
    prefix = normalized.square().cumsum(dim=1)
    total = 0.5 * torch.log1p(prefix).mean(dim=-1)
    expected = total - torch.cat([torch.zeros_like(total[:, :1]), total[:, :-1]], dim=1)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_v34_confidence_threshold_main_loss_reaches_boundary_head() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=1,
            memory_rank=2,
            readout_vectors=4,
            ar_hidden=8,
            use_ar=False,
            use_memory=False,
            boundary_mode="confidence_threshold",
            coding_rate_mode="diag",
            max_chunks=8,
            max_span=16,
            noise_scale=0.0,
        )
    ).train()
    ids = torch.tensor([text_to_byte_ids("confidence must own the differentiable boundary path")[:48]])
    model(ids).readout_z.float().square().mean().backward()
    grad = model.segmentor_head[-1].weight.grad
    assert grad is not None and grad.abs().sum().item() > 0


def test_v34_shared_inverse_decoder_reuses_interpreter_without_parameters() -> None:
    model = FLUEDV34Probe(
        FLUEDV34ProbeConfig(
            d_model=32,
            nhead=4,
            ffn_dim=64,
            segmentor_layers=1,
            interpreter_layers=2,
            memory_rank=2,
            readout_vectors=4,
            ar_hidden=8,
            use_memory=False,
            boundary_mode="uniform_budget",
            bytes_per_chunk_budget=8,
            max_chunks=8,
            max_span=16,
            noise_scale=0.0,
            decoder_mode="shared_inverse",
        )
    ).train()
    ids = torch.tensor([text_to_byte_ids("shared inverse decoder")])
    out = model(ids)
    assert out.byte_logits.shape[-2:] == (16, 258)
    assert sum(parameter.numel() for parameter in model.decoder.parameters()) == 0
    out.byte_logits.float().square().mean().backward()
    grad = model.interpreter_blocks[-1].ff_out.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0


def test_v34_canonical_configs_enable_corrected_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "v34_default_38m_20k.json",
        "v34_full_333m_backbone_107m_4096.json",
    ):
        config = json.loads((root / "configs" / "v3_4" / name).read_text(encoding="utf-8"))
        assert config["boundary_curriculum_mode"] == "confidence_threshold"
        assert config["boundary_curriculum_coding_rate_mode"] == "diag"
        assert config["completion_mask_granularity"] == "readout"
        assert config["decoder_mode"] == "shared_inverse"
        assert config["boundary_target_bytes_per_chunk"] == 16
        assert config["boundary_bridge_gradient_scale"] == 0.1
        assert config["boundary_rate_alignment_weight"] == 0.2


def test_v34_boundary_compute_budget_prices_dense_soft_cuts() -> None:
    valid = torch.ones(2, 32, dtype=torch.bool)
    force_continue = torch.zeros_like(valid)
    dense = torch.full((2, 32), 0.99, requires_grad=True)
    loss, constraint, density, target, soft_density = boundary_compute_budget_loss(
        dense,
        valid,
        force_continue,
        tau_cut=0.9,
        temperature=0.15,
        bytes_per_chunk=16,
        dual_value=torch.tensor(1.0),
        augmented_weight=1.0,
    )
    assert constraint.item() > 0
    assert density.item() > target.item()
    assert soft_density.item() > target.item()
    loss.backward()
    assert dense.grad is not None
    assert dense.grad.mean().item() > 0

    sparse = torch.full((2, 32), -0.5)
    sparse_loss, sparse_constraint, sparse_density, _, _ = boundary_compute_budget_loss(
        sparse,
        valid,
        force_continue,
        tau_cut=0.9,
        temperature=0.15,
        bytes_per_chunk=16,
        dual_value=torch.tensor(0.0),
        augmented_weight=1.0,
    )
    assert sparse_constraint.item() < 0
    assert sparse_density.item() < target.item()
    assert sparse_loss.item() == 0.0

    just_below = torch.full((2, 32), 0.89)
    _, below_constraint, priced_density, _, continuous_density = boundary_compute_budget_loss(
        just_below,
        valid,
        force_continue,
        tau_cut=0.9,
        temperature=0.15,
        bytes_per_chunk=16,
        dual_value=torch.tensor(1.0),
        augmented_weight=1.0,
    )
    assert below_constraint.item() < 0
    assert priced_density.item() < target.item()
    assert continuous_density.item() > target.item()


def test_v34_signed_confidence_remains_plastic_at_both_poles() -> None:
    logits = torch.tensor([-20.0, 20.0], requires_grad=True)
    confidence = _plastic_signed_confidence(logits)
    assert confidence.tolist() == [-1.0, 1.0]
    confidence.sum().backward()
    assert torch.equal(logits.grad, torch.ones_like(logits))


def test_v34_boundary_calibration_is_continuous_and_directional() -> None:
    confidence = torch.tensor([[0.2, 0.2, 0.2]], requires_grad=True)
    target = torch.tensor([[0.0, 0.98, -0.5]])
    token_ids = torch.tensor([[98, 99, 100]])
    valid = torch.ones_like(token_ids, dtype=torch.bool)
    loss, gap = boundary_threshold_calibration_loss(
        confidence,
        target,
        token_ids,
        valid,
        tau_cut=0.9,
        temperature=0.1,
    )
    assert loss.item() > 0
    assert gap.item() > 0
    loss.backward()
    assert confidence.grad is not None
    assert confidence.grad[0, 1].item() < 0
    assert confidence.grad[0, 2].item() > 0


def test_v34_boundary_density_uses_hard_count_with_continuous_gradient() -> None:
    confidence = torch.tensor([[0.0, 0.85, 0.1, 0.1]], requires_grad=True)
    target = torch.tensor([[0.0, 0.95, -0.2, -0.3]])
    token_ids = torch.tensor([[98, 99, 100, 101]])
    valid = torch.ones_like(token_ids, dtype=torch.bool)
    loss, gap = boundary_threshold_density_loss(
        confidence,
        target,
        token_ids,
        valid,
        tau_cut=0.9,
        temperature=0.1,
    )
    assert loss.item() > 0
    assert gap.item() > 0
    loss.backward()
    assert confidence.grad is not None
    assert confidence.grad[0, 1].item() < 0


def test_v34_boundary_positive_margin_pushes_only_rate_positive_positions() -> None:
    confidence = torch.tensor([[0.0, 0.4, 0.4]], requires_grad=True)
    target = torch.tensor([[0.0, 0.95, -0.5]])
    token_ids = torch.tensor([[98, 99, 100]])
    valid = torch.ones_like(token_ids, dtype=torch.bool)
    loss, shortfall = boundary_threshold_positive_margin_loss(
        confidence,
        target,
        token_ids,
        valid,
        tau_cut=0.9,
        margin=0.02,
        temperature=0.1,
    )
    assert loss.item() > 0
    assert shortfall.item() > 0
    loss.backward()
    assert confidence.grad is not None
    assert confidence.grad[0, 1].item() < 0
    assert confidence.grad[0, 2].item() == 0


def test_v34_boundary_rate_dual_pushes_up_only_when_below_minimum_ratio() -> None:
    confidence = torch.tensor([[0.0, 0.4, 0.2]], requires_grad=True)
    target = torch.tensor([[0.0, 0.95, -0.5]])
    token_ids = torch.tensor([[98, 99, 100]])
    valid = torch.ones_like(token_ids, dtype=torch.bool)
    loss, constraint, predicted, target_density = boundary_rate_minimum_ratio_loss(
        confidence,
        target,
        token_ids,
        valid,
        tau_cut=0.9,
        temperature=0.1,
        minimum_ratio=0.8,
        dual_value=torch.tensor(0.5),
        augmented_weight=1.0,
    )
    assert constraint.item() > 0
    assert predicted.item() == 0
    assert target_density.item() > 0
    loss.backward()
    assert confidence.grad is not None
    assert confidence.grad[0, 1].item() < 0
