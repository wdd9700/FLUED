"""
plot_compression.py — Standalone compression ratio trends (gap-aware).

Usage: python plot_compression.py checkpoints/e1b_full.log
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Colors ──────────────────────────────────────────────────────────────────
BG    = "#1e1e2e"
GRID  = "#313244"
TEXT  = "#cdd6f4"
HARD  = "#a6e3a1"
SOFT  = "#94e2d5"
MEAN  = "#89b4fa"
H55   = "#f9e2af"
H60   = "#fab387"
H65   = "#f38ba8"
TGT   = "#f38ba8"   # target 0.3 line


def safe_float(s: str | None) -> float | None:
    if s is None or s in ("N/A", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_all(text: str) -> dict:
    pat = re.compile(
        r"step=\s*(\d+)\s+.*?"
        r"soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+"
        r".*?bp_mean=([\d.]+)"
        r".*?h@55=([\d.]+)\s+h@60=([\d.]+)\s+h@65=([\d.]+)"
    )
    series = {
        "step": [], "soft_mn": [], "hard_mn": [],
        "bp_mean": [], "h55": [], "h60": [], "h65": [],
    }
    for m in pat.finditer(text):
        series["step"].append(int(m.group(1)))
        series["soft_mn"].append(float(m.group(2)))
        series["hard_mn"].append(float(m.group(3)))
        series["bp_mean"].append(float(m.group(4)))
        series["h55"].append(float(m.group(5)))
        series["h60"].append(float(m.group(6)))
        series["h65"].append(float(m.group(7)))
    return series


def split_at_gap(steps: list[int], threshold: int = 500):
    cuts = []
    for i in range(1, len(steps)):
        if steps[i] - steps[i - 1] > threshold:
            cuts.append(i)
    return cuts


# ── Load ────────────────────────────────────────────────────────────────────
log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/e1b_full.log")
text = log_path.read_text(encoding="utf-8", errors="ignore")
s = parse_all(text)
print(f"Parsed {len(s['step'])} steps ({s['step'][0]}→{s['step'][-1]})")

cuts = split_at_gap(s["step"])
seg_starts = [0] + cuts
seg_ends = cuts + [len(s["step"])]

# ── Figure: 3-row layout ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"hspace": 0.10})
fig.patch.set_facecolor(BG)

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)


def plot_gapped(ax, x_all, y_all, **kwargs):
    """Plot lines with gaps broken."""
    for a, b in zip(seg_starts, seg_ends):
        if b > a:
            ax.plot(x_all[a:b], y_all[a:b], **kwargs)


# ── Row 1: hard_m/n + soft_m/n + bp_mean + target ──────────────────────────
ax = axes[0]
plot_gapped(ax, s["step"], s["hard_mn"], color=HARD, linewidth=2.0, label="hard_m/n")
plot_gapped(ax, s["step"], s["soft_mn"], color=SOFT, linewidth=1.2, alpha=0.7, label="soft_m/n")
plot_gapped(ax, s["step"], s["bp_mean"], color=MEAN, linewidth=1.2, alpha=0.7, label="bp_mean")
ax.axhline(0.30, color=TGT, linewidth=1.0, linestyle="--", alpha=0.6, label="target 0.30")
ax.set_ylabel("compression ratio", color=TEXT, fontsize=10)
lo = min(min(s["hard_mn"]), min(s["soft_mn"]), min(s["bp_mean"]), 0.28)
hi = max(max(s["hard_mn"]), max(s["soft_mn"]), max(s["bp_mean"]), 0.72)
ax.set_ylim(lo - 0.01, hi + 0.01)
ax.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=9)

# ── Row 2: h@55 / h@60 / h@65 (threshold sweep) ────────────────────────────
ax2 = axes[1]
plot_gapped(ax2, s["step"], s["h55"], color=H55, linewidth=1.5, label="h@55")
plot_gapped(ax2, s["step"], s["h60"], color=H60, linewidth=1.5, label="h@60")
plot_gapped(ax2, s["step"], s["h65"], color=H65, linewidth=1.5, alpha=0.8, label="h@65")
all_h = s["h55"] + s["h60"] + s["h65"]
lo, hi = min(all_h), max(all_h)
pad = (hi - lo) * 0.10
ax2.set_ylim(lo - pad, hi + pad)
ax2.set_ylabel("hard_m/n @ threshold", color=TEXT, fontsize=10)
ax2.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
           labelcolor=TEXT, fontsize=9)

# ── Row 3: h@60−h@55 (spread proxy for bp sharpness) ───────────────────────
ax3 = axes[2]
spread = [h60 - h55 for h60, h55 in zip(s["h60"], s["h55"])]
plot_gapped(ax3, s["step"], spread, color="#cba6f7", linewidth=1.8, label="h@60 − h@55")
ax3.axhline(0, color=TEXT, linewidth=0.6, linestyle=":", alpha=0.3)
lo_s = min(spread) - 0.01
hi_s = max(spread) + 0.01
ax3.set_ylim(lo_s, hi_s)
ax3.set_ylabel("h@60−h@55  (↓ sharper)", color="#cba6f7", fontsize=10)
ax3.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
           labelcolor=TEXT, fontsize=9)

# ── X-axis ──────────────────────────────────────────────────────────────────
axes[-1].set_xlabel("Step", color=TEXT, fontsize=11)

# ── Gap shading ─────────────────────────────────────────────────────────────
if cuts:
    gap_str = f"gap: {s['step'][cuts[0]-1]}→{s['step'][cuts[0]]}"
    print(f"  ⚠ {gap_str}")
    for ax in axes:
        for c in cuts:
            ax.axvspan(s["step"][c-1], s["step"][c], alpha=0.08, color="#f38ba8", zorder=0)

# ── Title ───────────────────────────────────────────────────────────────────
start, end = s["step"][0], s["step"][-1]
gap_note = f"  [{gap_str}]" if cuts else ""
fig.suptitle(
    f"FLUED E1-B  Compression Ratio Trends  (step {start} → {end}){gap_note}",
    color=TEXT, fontsize=13, y=0.995,
)

plt.tight_layout(rect=[0, 0, 1, 0.99])
out = Path("checkpoints/compression_trends.png")
fig.savefig(out, dpi=150, facecolor=BG)
print(f"Saved → {out}")
plt.show()
