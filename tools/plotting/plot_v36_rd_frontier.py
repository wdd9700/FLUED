"""
plot_v36_rd_frontier.py — Rate–distortion frontier for the v3.6 readout-packet line.

x = rate: total transmitted scalars per 512-byte prompt (log scale).
y = distortion: strict masked-source completion accuracy (higher = less distortion).

All numbers are hardcoded from archived summary.json files (read 2026-08-02):

  v3.6 frontier (S0 dynamic boundary + 4x KDA state, d_pack=1536; rate = readout_queries * d_pack):
    k=1   acc 0.14849522430449724   L:/FLUED_archive/v36_s0_vs_e2e_20260727/arm_a_s0/summary.json
    k=4   acc 0.14625263400375843   L:/FLUED_archive/v36_attribution_matrix_20260731/k4_s0_4x_rerun/summary.json
    k=16  acc 0.15090026520192623   L:/FLUED_archive/v36_attribution_matrix_20260731/k16_s0_4x/summary.json

  Context points (uniform boundary):
    B1 uniform 4x k=1  acc 0.12929600710049272  L:/FLUED_archive/v36_attribution_matrix_20260731/b1_uniform_4x_k1/summary.json
    B0 uniform 1x k=1  acc 0.12720369175076485  L:/FLUED_archive/v36_attribution_matrix_20260731/b0_uniform_1x_k1/summary.json
                                                    (d_pack=384 in this arm's args -> rate = 1*384)
    legacy uniform probe  eval_completion_acc 0.11351355165243149
                          L:/FLUED_archive/v36_learnability_probe_20k_20260725/summary.json
                          (d_pack not logged in that run; plotted at 384 as same 1x KDA-state family)

  HNet-DiT fair-comparison references (hnet_dit_fair_20260802, d_model=512):
    std (byte-direct, no compression)  eval_masked_acc 0.32363029569387436
        L:/FLUED_archive/hnet_dit_fair_20260802/hnet_dit_std/summary.json
        rate = 512 bytes * 512 dim = 262,144 scalars (derived; no explicit scalar count archived)
    bottleneck  eval_masked_acc 0.14179997285827994
        L:/FLUED_archive/hnet_dit_fair_20260802/hnet_dit_bottleneck/summary.json
        rate = eval_chunks_per_sample 190.5078125 * 512 = 97,540 scalars (derived)

  Ceiling anchor (different metric, annotation only, NOT plotted in acc space):
    AR H-Net next-byte BPB 0.6529716201423995
        L:/FLUED_archive/hnet_repro_512_20k_20260801/summary.json

Usage: py -3.14 tools/plotting/plot_v36_rd_frontier.py
Outputs: results/v3.6/rd_frontier_20260802/rd_frontier.png + points.csv
No torch dependency; pure matplotlib (Agg backend).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style (catppuccin-mocha, matching tools/plotting conventions) ────────────
BG   = "#1e1e2e"
GRID = "#313244"
TEXT = "#cdd6f4"
C_V36    = "#89dceb"   # v3.6 frontier
C_HNET   = "#f9e2af"   # HNet-DiT references
C_CTX    = "#a6adc8"   # context points
C_CEIL   = "#f38ba8"   # ceiling anchor

OUT_DIR = Path("results/v3.6/rd_frontier_20260802")

# ── Data points (label, rate_scalars, masked_acc, group, source) ─────────────
POINTS = [
    # v3.6 frontier: S0 dynamic boundary + 4x KDA state, d_pack=1536
    ("v3.6 k=1 (S0+4x)",  1536,  0.14849522430449724, "v36",
     "v36_s0_vs_e2e_20260727/arm_a_s0"),
    ("v3.6 k=4",          6144,  0.14625263400375843, "v36",
     "v36_attribution_matrix_20260731/k4_s0_4x_rerun"),
    ("v3.6 k=16",         24576, 0.15090026520192623, "v36",
     "v36_attribution_matrix_20260731/k16_s0_4x"),
    # Context: uniform-boundary arms
    ("B1 uniform 4x k=1", 1536,  0.12929600710049272, "ctx",
     "v36_attribution_matrix_20260731/b1_uniform_4x_k1"),
    ("B0 uniform 1x k=1", 384,   0.12720369175076485, "ctx",
     "v36_attribution_matrix_20260731/b0_uniform_1x_k1"),
    ("legacy uniform probe", 384, 0.11351355165243149, "ctx",
     "v36_learnability_probe_20k_20260725"),
    # HNet-DiT fair-comparison references
    ("HNet-DiT std (no compr.)", 262144, 0.32363029569387436, "hnet",
     "hnet_dit_fair_20260802/hnet_dit_std"),
    ("HNet-DiT bottleneck",      97540,  0.14179997285827994, "hnet",
     "hnet_dit_fair_20260802/hnet_dit_bottleneck"),
]

HNET_BPB = 0.6529716201423995  # annotation only

# ── CSV ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUT_DIR / "points.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["label", "rate_scalars", "masked_acc", "group",
                "source_archive_dir (under L:/FLUED_archive/)"])
    for label, rate, acc, group, src in POINTS:
        w.writerow([label, rate, f"{acc:.6f}", group, src])
print(f"Wrote {csv_path}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT, labelsize=10)
ax.grid(True, which="both", color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_edgecolor(GRID)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# v3.6 frontier line
v36 = [(r, a) for _, r, a, g, _ in POINTS if g == "v36"]
v36.sort()
ax.plot([r for r, _ in v36], [a for _, a in v36],
        color=C_V36, linewidth=1.8, marker="o", markersize=8,
        label="v3.6 frontier (S0 + 4x KDA, d_pack=1536)", zorder=5)

# HNet-DiT references
ax.scatter([262144], [0.32363029569387436], marker="s", s=90, color=C_HNET,
           label="HNet-DiT reference", zorder=5)
ax.scatter([97540], [0.14179997285827994], marker="D", s=80, color=C_HNET, zorder=5)

# Context points
ctx = [(r, a) for _, r, a, g, _ in POINTS if g == "ctx"]
ax.scatter([r for r, _ in ctx], [a for _, a in ctx], marker="x", s=60,
           color=C_CTX, linewidths=1.5, label="uniform-boundary context", zorder=4)

# Point annotations
offsets = {
    "v3.6 k=1 (S0+4x)": (-70, 8),
    "v3.6 k=4": (-55, -18),
    "v3.6 k=16": (-60, 8),
    "B1 uniform 4x k=1": (-20, -20),
    "B0 uniform 1x k=1": (10, -4),
    "legacy uniform probe": (10, -14),
    "HNet-DiT std (no compr.)": (-210, -18),
    "HNet-DiT bottleneck": (-20, -26),
}
for label, rate, acc, group, _src in POINTS:
    color = {"v36": C_V36, "hnet": C_HNET, "ctx": C_CTX}[group]
    dx, dy = offsets.get(label, (8, 8))
    ax.annotate(f"{label}\n{acc:.3f} @ {rate:,}",
                xy=(rate, acc), xytext=(dx, dy),
                textcoords="offset points", color=color, fontsize=8.5,
                va="center")

# Ceiling anchor: different metric, annotation box only (not in acc space)
ax.text(0.015, 0.975,
        f"Ceiling anchor (diff. metric): AR H-Net next-byte BPB = {HNET_BPB:.3f}\n"
        "(hnet_repro_512_20k_20260801; not comparable in acc space)",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=C_CEIL,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=C_CEIL, alpha=0.9))

ax.set_xlabel("Rate: transmitted scalars per 512-byte prompt (log scale)",
              color=TEXT, fontsize=11)
ax.set_ylabel("Distortion: strict masked-source completion acc (higher = better)",
              color=TEXT, fontsize=11)
ax.set_ylim(0.08, 0.38)
ax.set_title(
    "FLUED v3.6 Rate–Distortion Frontier  (single seed=42, 20k steps, corpus_v3)",
    color=TEXT, fontsize=13, pad=12,
)

ax.legend(loc="center right", facecolor=GRID, edgecolor="#45475a",
          labelcolor=TEXT, fontsize=9)

plt.tight_layout()
out = OUT_DIR / "rd_frontier.png"
fig.savefig(out, dpi=150, facecolor=BG)
print(f"Saved -> {out}")
