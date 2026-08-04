"""Quantitative analysis of the 16 hand-annotated S0 samples.

Parses s0_annotation_samples.txt, splits each sample on '|', and reports:
- per-sample and global segment length stats (chars + UTF-8 bytes),
  split Chinese-dominant vs English-dominant samples
- segment-final / segment-initial punctuation inventory
- whitespace / newline attachment (trailing vs leading)
"""

import re
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "s0_annotation_samples.txt"


def parse_samples(text: str):
    parts = re.split(r"### 样本(\d+)（语料 [\d]+% 处）\n", text)
    # parts[0] is header; then alternating (num, body)
    samples = []
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        body = parts[i + 1]
        lines = [l for l in body.strip("\n").split("\n") if l.strip()]
        samples.append((num, "\n".join(lines)))
    return samples


def is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def lang_of(text: str) -> str:
    cjk = sum(1 for c in text if is_cjk(c))
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return "zh" if cjk >= lat else "en"


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def stats(vals):
    return {
        "n": len(vals),
        "min": min(vals),
        "p10": round(pct(vals, 0.10), 1),
        "p25": round(pct(vals, 0.25), 1),
        "median": round(median(vals), 1),
        "p75": round(pct(vals, 0.75), 1),
        "p90": round(pct(vals, 0.90), 1),
        "max": max(vals),
        "mean": round(sum(vals) / len(vals), 1),
    }


def segments_of(annotated: str):
    """Split keeping the '|' as a terminator: seg text is what precedes each '|'.
    Returns list of segments (non-empty after split may still contain '\n')."""
    segs = annotated.split("|")
    return segs  # last element is text after final '|' (usually '')


PUNCT = "，。、；：？！“”‘’（）《》〈〉【】…—·-,.!?;:'\"()[]<>/%"


def main():
    text = SRC.read_text(encoding="utf-8")
    samples = parse_samples(text)
    print(f"parsed {len(samples)} samples")

    per_sample = []
    all_zh_chars, all_zh_bytes = [], []
    all_en_chars, all_en_bytes = [], []

    end_punct = {}   # punct -> count at segment end
    start_punct = {} # punct -> count at segment start
    trail_ws = {"space": 0, "newline": 0, "none": 0}
    lead_ws = {"space": 0, "newline": 0, "none": 0}
    empty_segs = 0

    for num, annotated in samples:
        raw_segs = segments_of(annotated)
        # drop trailing empty after final '|'
        if raw_segs and raw_segs[-1] == "":
            raw_segs = raw_segs[:-1]
        segs = [s for s in raw_segs if s != ""]
        empty_segs += len(raw_segs) - len(segs)

        lang = lang_of(annotated.replace("|", ""))
        chars = [len(s) for s in segs]
        byts = [len(s.encode("utf-8")) for s in segs]
        per_sample.append((num, lang, len(segs), stats(chars), stats(byts)))

        if lang == "zh":
            all_zh_chars += chars
            all_zh_bytes += byts
        else:
            all_en_chars += chars
            all_en_bytes += byts

        for s in segs:
            # final punctuation
            core_end = s.rstrip(" \n")
            core_start = s.lstrip(" \n")
            if core_end and core_end[-1] in PUNCT:
                end_punct[core_end[-1]] = end_punct.get(core_end[-1], 0) + 1
            if core_start and core_start[0] in PUNCT:
                start_punct[core_start[0]] = start_punct.get(core_start[0], 0) + 1
            # whitespace attachment
            if s.endswith("\n"):
                trail_ws["newline"] += 1
            elif s.endswith(" "):
                trail_ws["space"] += 1
            else:
                trail_ws["none"] += 1
            if s.startswith("\n"):
                lead_ws["newline"] += 1
            elif s.startswith(" "):
                lead_ws["space"] += 1
            else:
                lead_ws["none"] += 1

    print("\n=== per-sample stats ===")
    for num, lang, n, cs, bs in per_sample:
        print(f"样本{num:02d} [{lang}] n={n:3d} chars(med={cs['median']},p10={cs['p10']},p90={cs['p90']},min={cs['min']},max={cs['max']}) "
              f"bytes(med={bs['median']},p10={bs['p10']},p90={bs['p90']},min={bs['min']},max={bs['max']})")

    print("\n=== ZH aggregate (samples: %s) ===" % [n for n, l, *_ in per_sample if l == "zh"])
    print("chars:", stats(all_zh_chars))
    print("bytes:", stats(all_zh_bytes))
    print("\n=== EN aggregate (samples: %s) ===" % [n for n, l, *_ in per_sample if l == "en"])
    print("chars:", stats(all_en_chars))
    print("bytes:", stats(all_en_bytes))

    print("\n=== punctuation at segment END (top) ===")
    for k, v in sorted(end_punct.items(), key=lambda x: -x[1]):
        print(f"  {k!r}: {v}")
    print("\n=== punctuation at segment START ===")
    for k, v in sorted(start_punct.items(), key=lambda x: -x[1]):
        print(f"  {k!r}: {v}")

    print("\n=== whitespace ===")
    print("trailing:", trail_ws)
    print("leading:", lead_ws)
    print("empty segments dropped:", empty_segs)

    # how many segments end with a sentence-final punct vs comma-like
    # and distribution of segments that are pure punctuation/short
    print("\n=== short segments (chars<=3) examples ===")
    shown = 0
    for num, annotated in samples:
        raw_segs = segments_of(annotated)
        if raw_segs and raw_segs[-1] == "":
            raw_segs = raw_segs[:-1]
        for s in raw_segs:
            if s and len(s) <= 3 and shown < 60:
                print(f"  样本{num:02d}: {s!r}")
                shown += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
