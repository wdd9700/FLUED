"""Guard against configuration drift between three sources of truth.

The training script's argparse defaults, ``configs/canonical_v35.json`` and
``docs/CURRENT_STATE.md`` must agree on the curated key set below. If a future
experiment changes the recommended defaults, all three must be updated in the
same commit; otherwise this test fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "configs" / "canonical_v35.json"
CURRENT_STATE = REPO_ROOT / "docs" / "CURRENT_STATE.md"

CURATED_KEYS = (
    "readout_vectors",
    "use_structured_lookup",
    "use_memory",
    "memory_use_position",
    "memory_position_mode",
    "boundary_mode",
    "boundary_coding_rate_mode",
    "boundary_curriculum_mode",
    "use_prompt_alibi",
    "prompt_position_scale",
    "max_chunks",
    "boundary_bridge_gradient_scale",
    "completion_mask_granularity",
    "decoder_mode",
    "use_emit_controller",
    "emit_forward_mode",
    "emit_threshold",
    "bytes_per_chunk_budget",
    "mask_prob",
    "eval_mask_seed",
)

CURRENT_STATE_MARKERS = (
    ("plain", "byte lookup"),
    ("独立 decoder", "decoder"),
    ("uniform", "boundary"),
    ("关闭", "emit / memory"),
    ("RoPE", "位置"),
)


def _cli_defaults() -> dict:
    from tools.train.v3_4 import train_v34_pos_ar_probe

    argv = sys.argv
    try:
        sys.argv = ["train_v34_pos_ar_probe.py"]
        args = train_v34_pos_ar_probe.parse_args()
    finally:
        sys.argv = argv
    return vars(args)


def test_argparse_defaults_match_canonical() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    defaults = _cli_defaults()
    mismatches = {
        key: (defaults.get(key), canonical.get(key))
        for key in CURATED_KEYS
        if defaults.get(key) != canonical.get(key)
    }
    assert not mismatches, f"CLI defaults drifted from canonical_v35.json: {mismatches}"


def test_current_state_mentions_canonical_version_and_key_defaults() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    text = CURRENT_STATE.read_text(encoding="utf-8")
    assert canonical["canonical_version"] in text, (
        "docs/CURRENT_STATE.md does not reference the canonical_version of configs/canonical_v35.json"
    )
    for marker, label in CURRENT_STATE_MARKERS:
        assert marker in text, f"docs/CURRENT_STATE.md missing {label} marker {marker!r}"


def test_canonical_config_has_required_metadata() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    for key in ("canonical_version", "evidence_base", "supersedes", "usage_note"):
        assert canonical.get(key), f"canonical_v35.json missing metadata field {key}"
