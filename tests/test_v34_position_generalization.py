from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flued.data import PAD_ID
from tools.eval.v3_4.eval_v34_position_generalization import (
    PREFIX_CASES,
    _args_for_length,
    _parse_lengths,
    _set_scores,
    _utf8_counts,
)


def test_parse_lengths_preserves_order_and_removes_duplicates() -> None:
    assert _parse_lengths("512, 2048,512,4096") == [512, 2048, 4096]
    with pytest.raises(ValueError):
        _parse_lengths("512,0")


def test_set_scores_reports_exact_f1_and_jaccard() -> None:
    scores = _set_scores({4, 8, 12}, {4, 9, 12})
    assert scores["intersection_count"] == 2
    assert scores["f1"] == pytest.approx(2 / 3)
    assert scores["jaccard"] == pytest.approx(1 / 2)
    assert _set_scores(set(), set())["f1"] == 1.0


def test_utf8_counts_actual_chunk_boundary_violations() -> None:
    # UTF-8 bytes for "中" are e4 b8 ad; position 1 is a continuation and a
    # deliberately injected actual chunk start. Position 0 is excluded as a
    # structural sequence start.
    clean = torch.tensor([[0xE4 + 1, 0xB8 + 1, 0xAD + 1, PAD_ID]])
    out = SimpleNamespace(
        policy=SimpleNamespace(hard_cut=torch.tensor([[True, False, False, False]])),
        chunks=SimpleNamespace(
            chunk_ids=torch.tensor([[0, 1, 1, -1]]),
            offsets=torch.tensor([[0, 0, 1, -1]]),
        ),
    )
    counts = _utf8_counts(clean, out)
    assert counts["utf8_continuation_bytes"] == 2
    assert counts["utf8_policy_violation_count"] == 0
    assert counts["utf8_chunk_boundary_violation_count"] == 1


def test_long_lengths_default_to_configurable_batch_one() -> None:
    base = SimpleNamespace(batch_size=8, data_path="", data_manifest="")
    cli = SimpleNamespace(
        max_eval_batches=3,
        eval_seed=7,
        device="cpu",
        deterministic=True,
        data_path="",
        data_manifest="",
        batch_size=0,
        long_batch_size=1,
        long_length_threshold=512,
    )
    assert _args_for_length(base, cli, 512).batch_size == 8
    assert _args_for_length(base, cli, 2048).batch_size == 1


def test_prefix_cases_cover_required_scenarios() -> None:
    assert {case["id"] for case in PREFIX_CASES} == {
        "english",
        "chinese",
        "code",
        "entity_repetition",
    }
