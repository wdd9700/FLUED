"""
plot_e1b_full.py — E1-b comprehensive multi-panel visualization (step 10000+).

Usage:
  python plot_e1b_full.py checkpoints/e1b_full.log
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Catppuccin Mocha ────────────────────────────────────────────────────────
BG    = "#1e1e2e"
GRID  = "#313244"
TEXT  = "#cdd6f4"
ACC   = "#89dceb"
LOSS  = "#f38ba8"
COMP  = "#fab387"
BPSTD = "#cba6f7"
MEAN  = "#89b4fa"
HARD  = "#a6e3a1"
SOFT  = "#94e2d5"
H55   = "#f9e2af"
H60   = "#fab387"
H65   = "#f38ba8"
UTF8  = "#f38ba8"
ASCII = "#a6e3a1"
CJK   = "#89dceb"
OP    = "#cba6f7"
DIGIT = "#f9e2af"


def safe_float(s: str | None) -> float | None:
    if s is None or s in ("N/A", "nan"):
        return None
    try: return float(s)
    except ValueError: return None


def parse_all(text: str) -> dict:
    pat_main = re.compile(
        r"step=\s*(\d+)\s+loss=([\-\d.]+)\s+recon=([\d.]+)\s+comp=([\-\d.]+)\s+"
        r"recon_acc=([\d.]+).*?"
        r"soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+units=([\d.]+).*?"
        r"bp_mean=([\d.]+)\s+bp_std=([\d.]+).*?"
        r"h@55=([\d.]+)\s+h@60=([\d.]+)\s+h@65=([\d.]+)"
    )
    pat_type = re.compile(
        r"step=\s*(\d+)\s+\[type_bp\]\s+"
        r"utf8=([\w.]+)\s+ascii=([\w.]+)\s+"
        r"cjk=([\w.]+)\s+op=([\w.]+)\s+digit=([\w.]+)"
    )

    series = {
        "step": [], "loss": [], "recon": [], "comp": [], "recon_acc": [],
        "soft_mn": [], "hard_mn": [], "units": [],
        "bp_mean": [], "bp_std": [],
        "h55": [], "h60": [], "h65": [],
        "utf8": [], "ascii": [], "cjk": [], "op": [], "digit": [],
        "t_step": [],
    }
    for m in pat_main.finditer(text):
        series["step"].append(int(m.group(1)))
        series["loss"].append(float(m.group(2)))
        series["recon"].append(float(m.group(3)))
        series["comp"].append(float(m.group(4)))
        series["recon_acc"].append(float(m.group(5)))
        series["soft_mn"].append(float(m.group(6)))
        series["hard_mn"].append(float(m.group(7)))
        series["units"].append(float(m.group(8)))
        series["bp_mean"].append(float(m.group(9)))
        series["bp_std"].append(float(m.group(10)))
        series["h55"].append(float(m.group(11)))
        series["h60"].append(float(m.group(12)))
        series["h65"].append(float(m.group(13)))
    for m in pat_type.finditer(text):
        series["t_step"].append(int(m.group(1)))
        for i, k in enumerate(("utf8","ascii","cjk","op","digit"), start=2):
            v = safe_float(m.group(i))
            series[k].append(v)
    return series


# ── load ────────────────────────────────────────────────────────────────────
log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/e1b_full.log")
text = log_path.read_text(encoding="utf-8", errors="ignore")
s = parse_all(text)
n = len(s["step"])
print(f"Parsed {n} main steps ({s['step'][0]} → {s['step'][-1]})  "
      f"+ {len(s['t_step'])} type_bp points")

# ── Detect & flag data gaps (> 500 steps) ─────────────────────────────────
GAP_THRESH = 500
gap_ranges: list[tuple[int, int]] = []
for i in range(1, len(s["step"])):
    if s["step"][i] - s["step"][i - 1] > GAP_THRESH:
        gap_ranges.append((s["step"][i - 1], s["step"][i]))
if gap_ranges:
    for g0, g1 in gap_ranges:
        print(f"  ⚠ data gap: step {g0} → {g1} ({g1 - g0} steps missing)")

# ── Helper: split data at gap indices ──────────────────────────────────────
def split_at_gap(steps: list[int], *arrays) -> list[tuple[list, ...]]:
    """Return segments (step_chunk, arr1_chunk, ...) split at gaps."""
    cut_indices = []
    for i in range(1, len(steps)):
        if steps[i] - steps[i - 1] > GAP_THRESH:
            cut_indices.append(i)
    segments = []
    prev = 0
    for ci in cut_indices:
        chunk = [steps[prev:ci]] + [a[prev:ci] for a in arrays]
        segments.append(tuple(chunk))
        prev = ci
    # Last segment
    chunk = [steps[prev:]] + [a[prev:] for a in arrays]
    segments.append(tuple(chunk))
    return segments

# Pre-split all series for gap-aware plotting
_steps = s["step"]
_seg_main = split_at_gap(_steps,
    s["recon_acc"], s["loss"], s["recon"], s["comp"],
    s["hard_mn"], s["soft_mn"], s["bp_mean"], s["bp_std"],
    s["h55"], s["h60"], s["h65"])
# type_bp segments
_tsteps = s.get("t_step", [])
if _tsteps:
    _seg_type = split_at_gap(_tsteps,
        s["utf8"], s["ascii"], s["cjk"], s["op"], s["digit"])
else:
    _seg_type = []

# Plot helper: draw each segment for a given series index
def plot_segments(ax, segs, idx, **kwargs):
    for seg in segs:
        ax.plot(seg[0], seg[idx], **kwargs)

# ── Figure: 6-row compact layout ───────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True,
                         gridspec_kw={"hspace": 0.12})
fig.patch.set_facecolor(BG)

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

# ── Row 1: recon_acc ───────────────────────────────────────────────────────
ax = axes[0]
plot_segments(ax, _seg_main, 1, color=ACC, linewidth=1.2, label="recon_acc")
ax.axhline(0.99, color=TEXT, linewidth=0.6, linestyle="--", alpha=0.5)
ax.axhline(1.00, color=TEXT, linewidth=0.6, linestyle="--", alpha=0.3)
ymin = min(s["recon_acc"]) - 0.0005
ymax = max(s["recon_acc"]) + 0.0005
ax.set_ylim(ymin, ymax)
ax.set_ylabel("recon_acc", color=ACC, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=7)

# ── Row 2: loss / recon / comp ────────────────────────────────────────────
ax = axes[1]
plot_segments(ax, _seg_main, 2, color=LOSS, linewidth=1.2, label="total loss")
plot_segments(ax, _seg_main, 3, color=ACC, linewidth=0.8, alpha=0.6, label="recon")
plot_segments(ax, _seg_main, 4, color=COMP, linewidth=0.8, alpha=0.6, label="comp")
ax.axhline(0, color=TEXT, linewidth=0.6, linestyle=":", alpha=0.3)
ax.set_ylabel("loss", color=LOSS, fontsize=9)
all_l = s["loss"] + s["recon"] + s["comp"]
lo, hi = min(all_l), max(all_l)
pad = (hi - lo) * 0.08
ax.set_ylim(lo - pad, hi + pad)
ax.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=7, ncol=3)

# ── Row 3: m/n + bp_mean ──────────────────────────────────────────────────
ax = axes[2]
plot_segments(ax, _seg_main, 5, color=HARD, linewidth=1.2, label="hard_m/n")
plot_segments(ax, _seg_main, 6, color=SOFT, linewidth=0.8, alpha=0.6, label="soft_m/n")
plot_segments(ax, _seg_main, 7, color=MEAN, linewidth=0.8, alpha=0.6, label="bp_mean")
ax.axhline(0.30, color=TEXT, linewidth=0.6, linestyle="--", alpha=0.3, label="target 0.3")
all_mn = s["hard_mn"] + s["soft_mn"] + s["bp_mean"]
lo, hi = min(all_mn), max(all_mn)
pad = (hi - lo) * 0.08
ax.set_ylim(lo - pad, hi + pad)
ax.set_ylabel("m/n ratio", color=HARD, fontsize=9)
ax.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=7)

# ── Row 4: bp_std ─────────────────────────────────────────────────────────
ax = axes[3]
plot_segments(ax, _seg_main, 8, color=BPSTD, linewidth=1.5, label="bp_std")
lo, hi = min(s["bp_std"]), max(s["bp_std"])
pad = (hi - lo) * 0.12
ax.set_ylim(lo - pad, hi + pad)
ax.set_ylabel("bp_std", color=BPSTD, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=7)

# ── Row 5: h@55 / h@60 / h@65 ─────────────────────────────────────────────
ax = axes[4]
plot_segments(ax, _seg_main, 9, color=H55, linewidth=1.0, label="h@55")
plot_segments(ax, _seg_main, 10, color=H60, linewidth=1.0, label="h@60")
plot_segments(ax, _seg_main, 11, color=H65, linewidth=1.0, alpha=0.7, label="h@65")
all_h = s["h55"] + s["h60"] + s["h65"]
lo, hi = min(all_h), max(all_h)
pad = (hi - lo) * 0.08
ax.set_ylim(lo - pad, hi + pad)
ax.set_ylabel("hard_m/n sweep", color=H55, fontsize=9)
ax.legend(loc="upper right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=7)

# ── Row 6: type_bp ────────────────────────────────────────────────────────
ax = axes[5]
if _seg_type:
    plot_segments(ax, _seg_type, 1, color=UTF8, linewidth=1.0, label="utf8_cont")
    plot_segments(ax, _seg_type, 2, color=ASCII, linewidth=1.0, label="ascii")
    plot_segments(ax, _seg_type, 3, color=CJK, linewidth=1.2, label="cjk")
    plot_segments(ax, _seg_type, 4, color=OP, linewidth=1.0, label="op")
    plot_segments(ax, _seg_type, 5, color=DIGIT, linewidth=1.0, label="digit")
    ax.axhline(0.05, color=TEXT, linewidth=0.5, linestyle=":", alpha=0.25)
    ax.axhline(0.50, color=TEXT, linewidth=0.5, linestyle=":", alpha=0.25)
    ax.set_ylabel("type_bp", color=TEXT, fontsize=9)
    ax.set_ylim(0.0, 0.90)
else:
    ax.text(0.5, 0.5, "no type_bp data", ha="center", va="center",
            color=TEXT, transform=ax.transAxes)
ax.legend(loc="upper left", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=7, ncol=5)

# ── X-axis ──────────────────────────────────────────────────────────────────
axes[-1].set_xlabel("Step", color=TEXT, fontsize=10)
axes[-1].tick_params(colors=TEXT, labelsize=8)

# ── Title ───────────────────────────────────────────────────────────────────
gap_note = ""
if gap_ranges:
    gap_strs = [f"{g0}→{g1}" for g0, g1 in gap_ranges]
    gap_note = f"  [gap: {', '.join(gap_strs)} — terminal buffer flushed]"
    # Add shaded gap regions on all subplots
    for ax in axes:
        for g0, g1 in gap_ranges:
            ax.axvspan(g0, g1, alpha=0.08, color="#f38ba8", zorder=0)

fig.suptitle(
    f"FLUED E1-B v2  Comprehensive Trends  (step {s['step'][0]} → {s['step'][-1]}){gap_note}",
    color=TEXT, fontsize=13, y=0.995,
)

plt.tight_layout(rect=[0, 0, 1, 0.99])
out = Path("checkpoints/e1b_full_trends.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, facecolor=BG)
print(f"Saved → {out}")
plt.show()
