"""Paired teacher-drift comparison: DeepSeek pilot vs K2.5 anchors (T1).

Pairs records by ``retry_of`` (pilot) == ``index`` (anchor file) and reports:
* pass rate of the pilot run (kept / attempted);
* segment-granularity distributions (n_segments, seg_bytes_median) both sides;
* boundary agreement: for each paired text, symmetric boundary match rates —
  exact and within tolerance bytes (default ±3B, one CJK char) — averaged over
  texts, plus the fraction of texts whose boundary SETS match exactly.

Usage:
  py -3.14 -X utf8 tools/analysis/v3_6/compare_teacher_drift.py \
      --pilot outputs/.../s0_teacher_labels.jsonl \
      --anchor L:/FLUED_archive/s0p_k25_v4_sft_20260805/teacher_labels_merged.jsonl \
      --attempted 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _match_rates(k_bounds: list[int], d_bounds: list[int], tol: int) -> tuple[float, float]:
    """Symmetric match: fraction of k25 boundaries with a deepseek boundary
    within tol, and vice versa (greedy one-to-one not needed at this scale)."""
    if not k_bounds or not d_bounds:
        return (1.0, 1.0) if k_bounds == d_bounds else (0.0, 0.0)
    k_hit = sum(1 for b in k_bounds if any(abs(b - d) <= tol for d in d_bounds)) / len(k_bounds)
    d_hit = sum(1 for b in d_bounds if any(abs(b - k) <= tol for k in k_bounds)) / len(d_bounds)
    return k_hit, d_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--attempted", type=int, required=True)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--out", default="")
    cli = ap.parse_args()

    pilot = _load(cli.pilot)
    anchor = {r["index"]: r for r in _load(cli.anchor)}
    pairs = [(r, anchor[r["retry_of"]]) for r in pilot if r.get("retry_of") in anchor]

    seg_med_k = [a["seg_bytes_median"] for _, a in pairs]
    seg_med_d = [p["seg_bytes_median"] for p, _ in pairs]
    nseg_k = [a["n_segments"] for _, a in pairs]
    nseg_d = [p["n_segments"] for p, _ in pairs]

    exact_text = 0
    k_exact_rates, d_exact_rates, k_tol_rates, d_tol_rates = [], [], [], []
    for p, a in pairs:
        kb, db = a["boundaries_bytes"], p["boundaries_bytes"]
        if kb == db:
            exact_text += 1
        ke, de = _match_rates(kb, db, 0)
        kt, dt = _match_rates(kb, db, cli.tol)
        k_exact_rates.append(ke)
        d_exact_rates.append(de)
        k_tol_rates.append(kt)
        d_tol_rates.append(dt)

    n = max(len(pairs), 1)
    result = {
        "attempted": cli.attempted,
        "pilot_kept": len(pilot),
        "pilot_pass_rate": len(pilot) / max(cli.attempted, 1),
        "paired": len(pairs),
        "seg_bytes_median_mean": {"k25": sum(seg_med_k) / n, "pilot": sum(seg_med_d) / n},
        "n_segments_mean": {"k25": sum(nseg_k) / n, "pilot": sum(nseg_d) / n},
        "boundary_exact_match_text_frac": exact_text / n,
        "boundary_recall_k25_side": {"exact": sum(k_exact_rates) / n, f"tol_{cli.tol}b": sum(k_tol_rates) / n},
        "boundary_recall_pilot_side": {"exact": sum(d_exact_rates) / n, f"tol_{cli.tol}b": sum(d_tol_rates) / n},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if cli.out:
        Path(cli.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
