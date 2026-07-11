from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import text_to_byte_ids  # noqa: E402
from flued.v34.model import FLUEDV34Probe, FLUEDV34ProbeConfig  # noqa: E402
from tools.analysis.v3_4.render_v34_boundary_roi import render  # noqa: E402
from tools.eval.v3_4.eval_v34_boundary_roi import (  # noqa: E402
    _byte_char_labels,
    inspect_case,
)


CONFIG_PATH = REPO_ROOT / "configs/v3_4/v34_boundary_roi_cases.json"


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
            use_ar=False,
            use_emit_controller=True,
            max_chunks=8,
            max_span=8,
            noise_scale=0.0,
        )
    ).eval()


def test_utf8_continuations_never_cut() -> None:
    model = _tiny_model()
    ids = torch.tensor([text_to_byte_ids("中文 boundary")])
    out = model(ids)
    raw = ids - 1
    continuation = raw.ge(0x80) & raw.le(0xBF)
    assert continuation.any()
    assert not out.policy.hard_cut[continuation].any()
    assert out.policy.force_continue[continuation].all()


def test_truncated_labels_follow_original_byte_offsets() -> None:
    text = "中文"
    labels = _byte_char_labels(text)
    assert len(labels) == len(text.encode("utf-8"))
    assert labels[:2] == ["中", "↳"]

    result = inspect_case(
        _tiny_model(),
        {"id": "utf8_cut", "category": "标点/换行/UTF-8", "pair_id": "utf8_cut", "variant": "base", "text": text},
        seq_len=2,
        device=torch.device("cpu"),
    )
    assert result["truncated_to_seq_len"] is True
    assert result["byte_length"] == 2
    assert len(result["bytes"]) == 2
    assert [byte["char"] for byte in result["bytes"]] == ["中", "↳"]


def test_case_config_schema_and_pair_variants() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert len(cases) >= 64
    required = {"id", "category", "pair_id", "variant", "text", "tags", "audit_targets"}
    assert all(required <= set(case) for case in cases)
    assert {case["variant"] for case in cases} >= {"base", "perturbed"}
    pairs: dict[str, set[str]] = {}
    categories: dict[str, set[str]] = {}
    for case in cases:
        pairs.setdefault(case["pair_id"], set()).add(case["variant"])
        categories.setdefault(case["category"], set()).add(case["pair_id"])
    assert len(pairs) >= 32
    assert all(variants >= {"base", "perturbed"} for variants in pairs.values())
    assert len(categories) >= 8
    assert all(len(pair_ids) >= 2 for pair_ids in categories.values())


def test_json_schema_and_html_legends() -> None:
    result = inspect_case(
        _tiny_model(),
        {"id": "schema", "category": "代码", "pair_id": "schema_pair", "variant": "base", "text": "x = 1\n"},
        seq_len=32,
        device=torch.device("cpu"),
    )
    assert {"category", "pair_id", "variant", "bytes", "chunks", "summary", "budget"} <= set(result)
    byte_required = {
        "index", "raw_byte", "hex", "char", "signed_confidence",
        "model_hard_boundary", "hard_chunk_boundary", "logic_transition",
        "utf8_continuation", "forced_max_span_boundary", "chunk_id", "chunk_offset",
    }
    assert all(byte_required <= set(byte) for byte in result["bytes"])
    html = render({"device": "cpu", "checkpoint": {}, "cases": [result]})
    for marker in ("模型 hard boundary", "soft transition", "UTF-8 continuation", "强制 max-span boundary"):
        assert marker in html
