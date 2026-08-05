"""Merge S0 teacher labels: K2.5 anchor + DeepSeek v4-pro increment (S0'' prep).

- Tags every record with a ``source`` field (``k25`` / ``deepseek_v4pro``).
- Machine prefilter (QA spot-check recommendation, 2026-08-05): reject any
  record containing a segment with >21 CJK chars or >9 whitespace-separated
  words (the "must split" hard caps from S05_TEACHER_RULES_CN.md R1);
  segments mixing scripts are checked against both caps.
- Reports per-source counts, rejection counts, and segment-length stats so the
  granularity distribution can be compared against the K2.5 anchor.

Usage:
  py -3.14 -X utf8 tools/analysis/v3_6/merge_s0_teacher_labels.py \
      --anchor L:/FLUED_archive/s0p_k25_v4_sft_20260805/teacher_labels_merged.jsonl \
      --increment outputs/s05_pro_teacher_v4_8k_20260805/s0_teacher_labels.jsonl \
      --out-dir outputs/s05_teacher_merged_v4_20260805
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def _is_cjk(ch: str) -> bool:
    return '一' <= ch <= '鿿'


def _segments(text: str, boundaries: list[int]) -> list[str]:
    raw = text.encode('utf-8')
    out, prev = [], 0
    for b in boundaries + [len(raw)]:
        out.append(raw[prev:b].decode('utf-8', errors='replace'))
        prev = b
    return out


def _violates_hard_cap(seg: str) -> bool:
    cjk = sum(1 for c in seg if _is_cjk(c))
    words = len(seg.split())
    return cjk > 21 or words > 9


def _stats(records: list[dict]) -> dict:
    meds = sorted(r['seg_bytes_median'] for r in records)
    nsegs = sorted(r['n_segments'] for r in records)
    n = max(len(records), 1)
    return {
        'count': len(records),
        'seg_bytes_median_mean': sum(meds) / n,
        'seg_bytes_median_p50': meds[len(meds) // 2] if meds else 0,
        'n_segments_mean': sum(nsegs) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', required=True, help='K2.5 labels jsonl (source=k25)')
    ap.add_argument('--increment', required=True, help='DeepSeek labels jsonl (source=deepseek_v4pro)')
    ap.add_argument('--out-dir', required=True)
    cli = ap.parse_args()

    anchor = _load(cli.anchor)
    increment = _load(cli.increment)
    for r in anchor:
        r['source'] = 'k25'
    for r in increment:
        r['source'] = 'deepseek_v4pro'

    kept_anchor, kept_increment, rejected = [], [], []
    for r in anchor + increment:
        segs = _segments(r['text'], r['boundaries_bytes'])
        if any(_violates_hard_cap(s) for s in segs):
            rejected.append(r)
        elif r['source'] == 'k25':
            kept_anchor.append(r)
        else:
            kept_increment.append(r)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = kept_anchor + kept_increment
    with (out_dir / 'teacher_labels_merged.jsonl').open('w', encoding='utf-8') as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (out_dir / 'teacher_labels_rejected.jsonl').open('w', encoding='utf-8') as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    report = {
        'anchor_in': len(anchor),
        'increment_in': len(increment),
        'kept': {'k25': len(kept_anchor), 'deepseek_v4pro': len(kept_increment), 'total': len(merged)},
        'rejected_hard_cap': len(rejected),
        'rejected_by_source': {
            'k25': sum(1 for r in rejected if r['source'] == 'k25'),
            'deepseek_v4pro': sum(1 for r in rejected if r['source'] == 'deepseek_v4pro'),
        },
        'granularity': {
            'k25_kept': _stats(kept_anchor),
            'deepseek_kept': _stats(kept_increment),
            'merged': _stats(merged),
        },
    }
    (out_dir / 'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
