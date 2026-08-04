"""Second-pass detail analysis on s0_annotation_samples.txt."""

import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "s0_annotation_samples.txt"


def parse_samples(text: str):
    parts = re.split(r"### 样本(\d+)（语料 [\d]+% 处）\n", text)
    return [(int(parts[i]), parts[i + 1].strip("\n")) for i in range(1, len(parts), 2)]


def segs_of(annotated: str):
    raw = annotated.split("|")
    if raw and raw[-1].strip() == "":
        raw = raw[:-1]
    return [s for s in raw if s != ""]


def is_cjk(ch):
    return "一" <= ch <= "鿿"


samples = parse_samples(SRC.read_text(encoding="utf-8"))

# --- EN word counts per segment ---
print("=== EN words per segment (samples 07/12/13/14) ===")
words = []
for num, body in samples:
    if num not in (7, 12, 13, 14):
        continue
    for s in segs_of(body):
        w = len(s.strip().split())
        words.append(w)
ws = sorted(words)
n = len(ws)
print(f"n={n} min={ws[0]} p10={ws[int((n-1)*.1)]} p25={ws[int((n-1)*.25)]} med={median(ws)} "
      f"p75={ws[int((n-1)*.75)]} p90={ws[int((n-1)*.9)]} max={ws[-1]} mean={sum(ws)/n:.1f}")

# --- long ZH segments (>13 chars): characterize ---
print("\n=== ZH segments with len>13 chars ===")
for num, body in samples:
    if num in (7, 12, 13, 14):
        continue
    for s in segs_of(body):
        if len(s) > 13:
            print(f"  样本{num:02d} ({len(s)}字): {s!r}")

# --- long EN segments (>37 chars) ---
print("\n=== EN segments with len>37 chars ===")
for num, body in samples:
    if num not in (7, 12, 13, 14):
        continue
    for s in segs_of(body):
        if len(s) > 37:
            print(f"  样本{num:02d} ({len(s)}ch): {s!r}")

# --- segments ending with connective chars (should be none) ---
print("\n=== segments ENDING with 和/与/及/或/并/而/以及 (bad pattern check) ===")
bad = 0
for num, body in samples:
    for s in segs_of(body):
        core = s.rstrip(" \n，。、；：？！”）")
        if core and core[-1] in "和与及或并而":
            print(f"  样本{num:02d}: {s!r}")
            bad += 1
print(f"total: {bad}")

# --- segments starting with closing punct (bad) ---
print("\n=== segments STARTING with ，。、；：？！”）】》 (bad pattern check) ===")
bad = 0
for num, body in samples:
    for s in segs_of(body):
        core = s.lstrip(" \n")
        if core and core[0] in "，。、；：？！”）】》,.!?;:\")]>":
            print(f"  样本{num:02d}: {s!r}")
            bad += 1
print(f"total: {bad}")

# --- quote-spanning segments: opening/closing quote inside one segment ---
print("\n=== segments containing BOTH “ and ” (short quotes kept whole) ===")
for num, body in samples:
    for s in segs_of(body):
        if "“" in s and "”" in s:
            print(f"  样本{num:02d}: {s!r}")

print("\n=== segments with opening quote but no close (long quoted speech split) ===")
cnt = 0
for num, body in samples:
    for s in segs_of(body):
        if ("“" in s) != ("”" in s):
            print(f"  样本{num:02d}: {s!r}")
            cnt += 1
            if cnt > 14:
                break
    if cnt > 14:
        break

# --- cut-before-的 pattern ---
print("\n=== segments ending with 的 (attributive chain cut) ===")
cnt = 0
for num, body in samples:
    for s in segs_of(body):
        if s.endswith("的") and cnt < 30:
            print(f"  样本{num:02d}: {s!r}")
            cnt += 1

# --- newline segments ---
print("\n=== segments containing \\n (newline attachment) ===")
for num, body in samples:
    for s in segs_of(body):
        if "\n" in s:
            print(f"  样本{num:02d}: {s!r}")

# --- number-entity segments ---
print("\n=== segments that are mostly digits/units ===")
for num, body in samples:
    for s in segs_of(body):
        core = s.strip(" \n")
        if core and sum(c.isdigit() for c in core) >= 2:
            print(f"  样本{num:02d}: {s!r}")
