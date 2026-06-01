"""
analyze_type_bp.py — Per-type boundary probability trend analysis (step 25000+).
"""
import re, math
from pathlib import Path

# ── Parse all type_bp data ───────────────────────────────────────────────────
def parse_type_bp(text: str, min_step: int = 25000):
    pat = re.compile(
        r'step=\s*(\d+)\s+\[type_bp\]\s+'
        r'utf8=([\w.]+)\s+ascii=([\w.]+)\s+'
        r'cjk=([\w.]+)\s+op=([\w.]+)\s+digit=([\w.]+)'
    )
    data = []
    for m in pat.finditer(text):
        step = int(m.group(1))
        if step < min_step: continue
        vals = {'step': step}
        for i, k in enumerate(('utf8', 'ascii', 'cjk', 'op', 'digit'), start=2):
            v = m.group(i)
            if v and v not in ('N/A', 'nan', ''):
                vals[k] = float(v)
        if len(vals) > 1:
            data.append(vals)
    return data

# Load all sources
all_data = []
for log_file in ['checkpoints/e1_merged_full.log', 'checkpoints/e1_run2.log']:
    p = Path(log_file)
    if p.exists():
        all_data.extend(parse_type_bp(p.read_text(encoding='utf-8', errors='ignore')))

# Also try the captured terminal output
import glob, os
caps = Path.home() / 'AppData/Roaming/Code/User/workspaceStorage'
for f in sorted(caps.rglob('content.txt'), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
    txt = f.read_text(encoding='utf-8', errors='ignore')
    if 'step=41600' in txt or 'step=41500' in txt:
        all_data.extend(parse_type_bp(txt))
        break

# Dedup by step
seen = set()
unique = []
for d in sorted(all_data, key=lambda x: x['step']):
    if d['step'] not in seen:
        seen.add(d['step'])
        unique.append(d)

print(f"Total type_bp points (step >= 25000): {len(unique)}")
if unique:
    print(f"Step range: {unique[0]['step']} → {unique[-1]['step']}")
print()

# ── Segment analysis ─────────────────────────────────────────────────────────
segments = []
seg_start = 0
for i in range(1, len(unique)):
    if unique[i]['step'] - unique[i-1]['step'] > 500:
        segments.append((seg_start, i))
        seg_start = i
segments.append((seg_start, len(unique)))

for start, end in segments:
    seg = unique[start:end]
    s0, s1 = seg[0]['step'], seg[-1]['step']
    print(f"Segment {s0}-{s1} (n={len(seg)}, span={s1-s0} steps):")
    for k in ('utf8', 'ascii', 'cjk', 'op', 'digit'):
        vals = [d[k] for d in seg if k in d]
        if len(vals) < 3:
            continue
        mean_v = sum(vals) / len(vals)
        # Slope per 1000 steps
        xs = list(range(len(vals)))
        n = len(vals)
        sx, sy = sum(xs), sum(vals)
        sxy = sum(x * y for x, y in zip(xs, vals))
        sx2 = sum(x * x for x in xs)
        denom = n * sx2 - sx * sx
        slope = (n * sxy - sx * sy) / denom if abs(denom) > 1e-10 else 0.0
        per_1k = slope * len(vals) / max(1, s1 - s0) * 1000
        print(f"  {k:>6s}: mean={mean_v:.4f}  slope/1k={per_1k:+.6f}  "
              f"range=[{min(vals):.4f}, {max(vals):.4f}]")
    print()

# ── Cross-gap comparison ─────────────────────────────────────────────────────
print("=" * 70)
print("Cross-gap analysis: comparing segment means")
print("=" * 70)
for k in ('op', 'ascii', 'digit'):
    print(f"\n{k}:")
    for start, end in segments:
        seg = unique[start:end]
        vals = [d[k] for d in seg if k in d]
        if len(vals) < 3:
            continue
        mean_v = sum(vals) / len(vals)
        s0, s1 = seg[0]['step'], seg[-1]['step']
        # Gap from previous segment
        if start > 0:
            prev_seg = unique[segments[segments.index((start,end))-1][0]:segments[segments.index((start,end))-1][1]]
            prev_vals = [d[k] for d in prev_seg if k in d]
            if prev_vals:
                prev_mean = sum(prev_vals) / len(prev_vals)
                gap = mean_v - prev_mean
                print(f"  {prev_seg[-1]['step']}→{s0}: mean {prev_mean:.4f} → {mean_v:.4f}  (gap Δ={gap:+.4f})")
        else:
            print(f"  First segment {s0}-{s1}: mean={mean_v:.4f}")

# ── Recent slow trends ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Recent segment (after step 39000) detailed trend")
print("=" * 70)
recent = [d for d in unique if d['step'] >= 39000]
if len(recent) >= 5:
    for k in ('op', 'ascii', 'digit'):
        vals = [(d['step'], d[k]) for d in recent if k in d]
        if len(vals) < 3:
            continue
        # Linear regression against actual step
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        n = len(vals)
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        den = sum((xs[i]-mx)**2 for i in range(n))
        slope_per_step = num/den if abs(den) > 1e-15 else 0
        slope_per_1k = slope_per_step * 1000
        # R²
        y_pred = [mx*slope_per_step + (my - mx*slope_per_step) + slope_per_step*x for x in xs]
        ss_res = sum((ys[i]-y_pred[i])**2 for i in range(n)) if y_pred else 0
        ss_tot = sum((y-my)**2 for y in ys)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        print(f"  {k:>6s}: mean={my:.4f}  slope/1k={slope_per_1k:+.6f}  R2={r2:.4f}  "
              f"range=[{min(ys):.4f}, {max(ys):.4f}]")
        # First 5 vs last 5
        if len(vals) >= 10:
            first5 = sum(v[1] for v in vals[:5]) / 5
            last5 = sum(v[1] for v in vals[-5:]) / 5
            print(f"         first5_mean={first5:.4f}  last5_mean={last5:.4f}  Δ={last5-first5:+.4f}")
