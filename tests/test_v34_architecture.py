from __future__ import annotations

import inspect
from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from flued.data import text_to_byte_ids
from flued.v34.model import FLUEDV34Probe, FLUEDV34ProbeConfig, SpanDecoder
from flued.v34.rate_emit import MarginalCodingRateSelector, ReadoutEmitController
from tools.train.v3_4.train_v34_pos_ar_probe import apply_boundary_curriculum


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
