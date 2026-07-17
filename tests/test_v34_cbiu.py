from __future__ import annotations

import pytest
import torch

from tools.analysis.v3_4.probe_v34_cbiu import (
    normalize_risks,
    robust_risk,
    score_to_probability,
)
from tools.train.v3_4.cbiu import (
    CBIUState,
    cbiu_keep_utility,
    normalize_cbiu_risks,
    robust_cbiu_risk,
    update_cbiu_dual,
)


def test_cbiu_anchor_normalization_and_worst_risk() -> None:
    rich = {
        "reconstruction_bpb": 0.2,
        "completion_bpb": 1.0,
        "preservation_bpb": 0.1,
    }
    null = {
        "reconstruction_bpb": 1.2,
        "completion_bpb": 3.0,
        "preservation_bpb": 0.5,
    }
    row = {
        "reconstruction_bpb": 0.7,
        "completion_bpb": 1.5,
        "preservation_bpb": 0.3,
    }

    normalized, invalid = normalize_risks(row, rich, null)

    assert invalid == []
    assert normalized["reconstruction_bpb"] == pytest.approx(0.5)
    assert normalized["completion_bpb"] == pytest.approx(0.25)
    assert normalized["preservation_bpb"] == pytest.approx(0.5)
    assert robust_risk(normalized) == pytest.approx(0.5)


def test_cbiu_rejects_nonseparating_anchor() -> None:
    rich = {key: 1.0 for key in ("reconstruction_bpb", "completion_bpb", "preservation_bpb")}
    null = dict(rich)
    normalized, invalid = normalize_risks(dict(rich), rich, null)

    assert set(invalid) == set(rich)
    assert robust_risk(normalized) is None


@pytest.mark.parametrize(
    ("score", "probability"),
    [(-1.0, 0.0), (0.0, 0.5), (0.5, 0.75), (0.75, 0.875), (0.9, 0.95), (1.0, 1.0)],
)
def test_signed_confidence_has_probability_semantics(score: float, probability: float) -> None:
    assert score_to_probability(score) == pytest.approx(probability)


def _state() -> CBIUState:
    return CBIUState(
        protocol_version="CBIU_V1_ONLINE_20260717",
        anchor_rich=torch.tensor([1.0, 2.0, 3.0]),
        anchor_null=torch.tensor([3.0, 6.0, 7.0]),
        anchor_file="anchor.json",
        anchor_checkpoint="checkpoint.pt",
        compute_budget=0.5,
        compute_dual=torch.tensor(0.0),
    )


def test_online_cbiu_tensor_normalization_and_worst_risk() -> None:
    state = _state()
    risks = torch.tensor([[2.0, 3.0, 5.0]])
    normalized = normalize_cbiu_risks(risks, state.anchor_rich, state.anchor_null)

    assert normalized.squeeze(0).tolist() == pytest.approx([0.5, 0.25, 0.5])
    assert robust_cbiu_risk(normalized).item() == pytest.approx(0.5)


def test_online_cbiu_keep_utility_is_positive_when_off_is_worse() -> None:
    state = _state()
    on = torch.tensor([[1.5, 3.0, 4.0]])
    off = torch.tensor([[2.5, 5.0, 6.0]])
    utility = cbiu_keep_utility(
        on,
        off,
        torch.tensor([0.4]),
        torch.tensor([0.3]),
        state,
        augmented_weight=0.0,
    )

    assert utility["quality_utility"].item() > 0.0
    assert utility["net_utility"].item() > 0.0


def test_online_cbiu_dual_uses_hard_budget_violation() -> None:
    state = _state()
    constraint = update_cbiu_dual(
        state,
        torch.tensor([0.75, 0.75]),
        dual_lr=0.2,
        dual_max=10.0,
    )

    assert constraint.item() == pytest.approx(0.5)
    assert state.compute_dual.item() == pytest.approx(0.1)
    assert state.update_count == 1


def test_online_cbiu_state_round_trip_rejects_anchor_change() -> None:
    state = _state()
    state.compute_dual.fill_(0.7)
    payload = state.state_dict()
    restored = _state()
    restored.load_state_dict(payload, torch.device("cpu"))

    assert restored.compute_dual.item() == pytest.approx(0.7)
    payload["anchor_null"] = payload["anchor_null"] + 1.0
    with pytest.raises(RuntimeError, match="anchor mismatch"):
        restored.load_state_dict(payload, torch.device("cpu"))
