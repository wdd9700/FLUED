"""
plot_type_bp.py — Per-type boundary probability standalone chart (gap-aware).

Usage: python plot_type_bp.py checkpoints/e1b_full.log
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Colors ──────────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
GRID    = "#313244"
TEXT    = "#cdd6f4"
UTF8    = "#f38ba8"
ASCII   = "#a6e3a1"
CJK     = "#89dceb"
OP      = "#cba6f7"
DIGIT   = "#f9e2af"


def safe_float(s: str | None) -> float | None:
    if s is None or s in ("N/A", "nan"): return None
    try: return float(s)
    except ValueError: return None


def parse_type_bp(text: str) -> dict:
    pat = re.compile(
        r"step=\s*(\d+)\s+\[type_bp\]\s+"
        r"utf8=([\w.]+)\s+ascii=([\w.]+)\s+"
        r"cjk=([\w.]+)\s+op=([\w.]+)\s+digit=([\w.]+)"
    )
    series = {"step": [], "utf8": [], "ascii": [], "cjk": [], "op": [], "digit": []}
    for m in pat.finditer(text):
        series["step"].append(int(m.group(1)))
        for i, k in enumerate(("utf8", "ascii", "cjk", "op", "digit"), start=2):
            v = safe_float(m.group(i))
            series[k].append(v)
    return series


def split_cuts(steps: list[int], threshold: int = 500) -> list[int]:
    cuts = []
    for i in range(1, len(steps)):
        if steps[i] - steps[i - 1] > threshold:
            cuts.append(i)
    return cuts


# ── Load ────────────────────────────────────────────────────────────────────
log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/e1b_full.log")
text = log_path.read_text(encoding="utf-8", errors="ignore")
s = parse_type_bp(text)
n = len(s["step"])
print(f"Parsed {n} type_bp points ({s['step'][0]}→{s['step'][-1]})")

cuts = split_cuts(s["step"])
seg_starts = [0] + cuts
seg_ends = cuts + [n]

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT, labelsize=10)
ax.grid(True, color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)

curves = [
    ("utf8_cont",  UTF8,  s["utf8"]),
    ("ascii",      ASCII, s["ascii"]),
    ("cjk",        CJK,   s["cjk"]),
    ("op",         OP,    s["op"]),
    ("digit",      DIGIT, s["digit"]),
]

for label, color, vals in curves:
    xy = [(st, v) for st, v in zip(s["step"], vals) if v is not None]
    if not xy:
        continue
    # Plot with gap breaks
    for a, b in zip(seg_starts, seg_ends):
        if b > a:
            chunk_x = [st for st in s["step"][a:b]]
            chunk_y = [v for v in vals[a:b] if v is not None]
            if len(chunk_x) == len(chunk_y) and chunk_x:
                ax.plot(chunk_x, chunk_y, color=color, linewidth=1.5, label=label if a == 0 else "")

# Reference lines
ax.axhline(0.10, color=TEXT, linewidth=0.5, linestyle=":", alpha=0.2)
ax.axhline(0.30, color=TEXT, linewidth=0.5, linestyle=":", alpha=0.2)
ax.axhline(0.50, color=TEXT, linewidth=0.5, linestyle=":", alpha=0.2)
ax.axhline(0.70, color=TEXT, linewidth=0.5, linestyle=":", alpha=0.2)

ax.set_xlabel("Step", color=TEXT, fontsize=11)
ax.set_ylabel("bp_mean per type", color=TEXT, fontsize=11)
ax.set_ylim(0.0, 0.88)

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          loc="upper left", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=10, ncol=5)

# Gap shading
if cuts:
    gap_str = f"gap: {s['step'][cuts[0]-1]}→{s['step'][cuts[0]]}  ({s['step'][cuts[0]]-s['step'][cuts[0]-1]} steps)"
    print(f"  ⚠ {gap_str}")
    for c in cuts:
        ax.axvspan(s["step"][c-1], s["step"][c], alpha=0.08, color="#f38ba8", zorder=0)

# Annotations: final values
final_x = s["step"][-1]
for label, color, vals in curves:
    last_v = vals[-1]
    if last_v is None:
        continue
    ax.annotate(f"{label}\n{last_v:.3f}",
                xy=(final_x, last_v), xytext=(15, 0),
                textcoords="offset points", color=color, fontsize=8,
                va="center")

start, end = s["step"][0], s["step"][-1]
gap_note = f"  [{gap_str}]" if cuts else ""
ax.set_title(
    f"FLUED E1-B  Type-Conditional Boundary Probabilities  (step {start} → {end}){gap_note}",
    color=TEXT, fontsize=13, pad=12,
)

plt.tight_layout()
out = Path("checkpoints/type_bp_trends.png")
fig.savefig(out, dpi=150, facecolor=BG)
print(f"Saved → {out}")
plt.show()
