"""
plot_e1.py  —  E1 Stage-A 训练曲线可视化

用法:
  python plot_e1.py                    # 使用内嵌日志数据
  python plot_e1.py training.log       # 从日志文件解析
  python -m flued.e1_stage_a ... 2>&1 | tee training.log

v1.2 — 适配 E1-b 日志格式:
  - 主指标行: loss / recon_acc / bp_std / bhead_gnorm / h@55/60/65
  - 类型行:   [type_bp] utf8 / ascii / cjk / op / digit
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 内嵌日志（每次运行后可整块粘贴替换） ──────────────────────────────────────
INLINE_LOG = """\
"""

# ── Catppuccin Mocha 配色 ────────────────────────────────────────────────────
BG       = "#1e1e2e"
GRID     = "#313244"
TEXT     = "#cdd6f4"
ACC      = "#89dceb"   # 青 — recon_acc
LOSS     = "#f38ba8"   # 粉红 — loss
GNORM    = "#a6e3a1"   # 绿 — bhead_gnorm
BPSTD    = "#cba6f7"   # 紫 — bp_std (E1-b key metric)
TARGET   = "#fab387"   # 橙 — target line
H55      = "#f9e2af"   # 黄
H60      = "#fab387"   # 深橙
H65      = "#f38ba8"   # 红/粉
# Per-type bp 颜色
TYPE_UTF8  = "#f38ba8"  # 红
TYPE_ASCII = "#a6e3a1"  # 绿
TYPE_CJK   = "#89dceb"  # 青
TYPE_OP    = "#cba6f7"  # 紫
TYPE_DIGIT = "#f9e2af"  # 黄


# ── 正则解析 ─────────────────────────────────────────────────────────────────
_PAT_MAIN = re.compile(
    r"step=\s*(\d+)\s+loss=([\d.]+)\s+recon=([\d.]+)\s+comp=([\d.]+)\s+"
    r"recon_acc=([\d.]+).*?"
    r"bp_std=([\d.]+)"
    r"(?:.*?bhead_gnorm=([\w.]+))?"
    r"(?:.*?h@55=([\d.]+))?"
    r"(?:.*?h@60=([\d.]+))?"
    r"(?:.*?h@65=([\d.]+))?"
)

_PAT_TYPE = re.compile(
    r"step=\s*(\d+)\s+\[type_bp\]\s+"
    r"utf8=([\w.]+)\s+ascii=([\w.]+)\s+"
    r"cjk=([\w.]+)\s+op=([\w.]+)\s+digit=([\w.]+)"
)


def _safe_float(s: Optional[str]) -> Optional[float]:
    """Parse float, returning None for 'N/A' or 'nan'."""
    if s is None or s in ("N/A", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_main(text: str) -> Dict[str, List]:
    """Parse main metrics lines → dict of series."""
    series: Dict[str, List] = {
        "step": [], "loss": [], "recon_acc": [],
        "bp_std": [], "bhead_gnorm": [],
        "h55": [], "h60": [], "h65": [],
    }
    for m in _PAT_MAIN.finditer(text):
        series["step"].append(int(m.group(1)))
        series["loss"].append(float(m.group(2)))
        series["recon_acc"].append(float(m.group(5)))
        series["bp_std"].append(float(m.group(6)))
        g = m.group(7)
        series["bhead_gnorm"].append(_safe_float(g))
        series["h55"].append(_safe_float(m.group(8)))
        series["h60"].append(_safe_float(m.group(9)))
        series["h65"].append(_safe_float(m.group(10)))
    return series


def parse_type_bp(text: str) -> Dict[str, List]:
    """Parse [type_bp] lines → dict of series."""
    series: Dict[str, List] = {
        "step": [], "utf8": [], "ascii": [], "cjk": [], "op": [], "digit": [],
    }
    for m in _PAT_TYPE.finditer(text):
        series["step"].append(int(m.group(1)))
        for i, key in enumerate(("utf8", "ascii", "cjk", "op", "digit"), start=2):
            v = _safe_float(m.group(i))
            series[key].append(v)
    return series


# ── 数据加载 ─────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    source = log_path.name
else:
    text = INLINE_LOG
    source = "inline log"

main = parse_main(text)
type_bp = parse_type_bp(text)

if not main["step"]:
    print("未找到日志数据，请检查输入。")
    sys.exit(1)

n_points = len(main["step"])
print(f"解析到 {n_points} 个主数据点 (step {main['step'][0]} → {main['step'][-1]})")
print(f"  recon_acc = {main['recon_acc'][-1]:.4f}  loss = {main['loss'][-1]:.4f}"
      f"  bp_std = {main['bp_std'][-1]:.3f}")
if type_bp["step"]:
    n_tp = len(type_bp["step"])
    last = type_bp["step"][-1]
    print(f"解析到 {n_tp} 个 type_bp 数据点")
    u = type_bp["utf8"][-1]
    a = type_bp["ascii"][-1]
    c = type_bp["cjk"][-1]
    o = type_bp["op"][-1]
    d = type_bp["digit"][-1]
    print(f"  (step {last}) utf8={u} ascii={a} cjk={c} op={o} digit={d}")

has_h = any(v is not None for v in main["h55"])
has_type = len(type_bp["step"]) > 0

# ── 图 1: 主指标 + bp_std + h@ 阈值 ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 7))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)

# --- recon_acc (主轴) ---
ax1.plot(main["step"], main["recon_acc"], color=ACC, linewidth=2,
         label="recon_acc", zorder=4)
ax1.axhline(0.99, color=TARGET, linewidth=1.2, linestyle="--",
            label="target 0.99", zorder=1)
ax1.set_xlabel("Step", color=TEXT, fontsize=11)
ax1.set_ylabel("recon_acc / h@m/n", color=ACC, fontsize=11)
ax1.tick_params(colors=TEXT)
ax1.set_ylim(0, 1.05)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# --- h@ 阈值曲线 (同轴, 虚线) ---
if has_h:
    h_curves = [("h@55", H55, main["h55"]),
                ("h@60", H60, main["h60"]),
                ("h@65", H65, main["h65"])]
    for label, color, vals in h_curves:
        xy = [(s, v) for s, v in zip(main["step"], vals) if v is not None]
        if xy:
            xs, ys = zip(*xy)
            ax1.plot(xs, ys, color=color, linewidth=1, linestyle=":",
                     alpha=0.7, label=label, zorder=3)

# --- loss + bp_std + bhead_gnorm (右轴) ---
ax2 = ax1.twinx()
ax2.set_facecolor(BG)
ax2.plot(main["step"], main["loss"], color=LOSS, linewidth=1.5,
         alpha=0.6, label="loss", zorder=2)
ax2.plot(main["step"], main["bp_std"], color=BPSTD, linewidth=1.8,
         label="bp_std", zorder=3)
# bhead_gnorm (过滤 nan)
gn_xy = [(s, g) for s, g in zip(main["step"], main["bhead_gnorm"]) if g == g]
if gn_xy:
    gx, gy = zip(*gn_xy)
    ax2.plot(gx, gy, color=GNORM, linewidth=1, linestyle=":", alpha=0.5,
             label="bhead_gnorm", zorder=1)
ax2.set_ylabel("loss / bp_std / gnorm", color=TEXT, fontsize=10)
ax2.tick_params(colors=TEXT)
ax2.set_ylim(bottom=0)

# --- 网格 ---
ax1.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax1.set_axisbelow(True)

# --- 图例 ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", facecolor="#313244", edgecolor="#45475a",
           labelcolor=TEXT, fontsize=8)

# --- 标题 ---
title_parts = [f"FLUED E1 Stage-A — {source}"]
title_parts.append(f"step={main['step'][-1]}")
title_parts.append(f"acc={main['recon_acc'][-1]:.4f}")
title_parts.append(f"bp_std={main['bp_std'][-1]:.3f}")
ax1.set_title("  ".join(title_parts), color=TEXT, fontsize=12, pad=10)
for spine in list(ax1.spines.values()) + list(ax2.spines.values()):
    spine.set_edgecolor(GRID)

plt.tight_layout()
out1 = Path("checkpoints/recon_acc_curve.png")
out1.parent.mkdir(exist_ok=True)
fig1.savefig(out1, dpi=150, facecolor=fig1.get_facecolor())
print(f"已保存: {out1}")

# ── 图 2: Per-type boundary prob (仅当有 type_bp 数据时) ──────────────────────
if has_type:
    fig2, ax_tp = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor(BG)
    ax_tp.set_facecolor(BG)

    tp_curves = [
        ("utf8_cont", TYPE_UTF8, type_bp["utf8"]),
        ("ascii",     TYPE_ASCII, type_bp["ascii"]),
        ("cjk",       TYPE_CJK, type_bp["cjk"]),
        ("op",        TYPE_OP, type_bp["op"]),
        ("digit",     TYPE_DIGIT, type_bp["digit"]),
    ]
    for label, color, vals in tp_curves:
        xy = [(s, v) for s, v in zip(type_bp["step"], vals) if v is not None]
        if xy:
            xs, ys = zip(*xy)
            ax_tp.plot(xs, ys, color=color, linewidth=1.5,
                       label=label, zorder=3)

    # E1-b target reference lines
    ax_tp.axhline(0.05, color=TYPE_UTF8, linewidth=0.8, linestyle=":",
                  alpha=0.4)
    ax_tp.axhline(0.30, color=TYPE_CJK, linewidth=0.8, linestyle=":",
                  alpha=0.4)
    ax_tp.axhline(0.80, color=TYPE_ASCII, linewidth=0.8, linestyle=":",
                  alpha=0.4)
    ax_tp.axhline(0.90, color=TYPE_OP, linewidth=0.8, linestyle=":",
                  alpha=0.4)

    ax_tp.set_xlabel("Step", color=TEXT, fontsize=11)
    ax_tp.set_ylabel("bp_mean per type", color=TEXT, fontsize=11)
    ax_tp.tick_params(colors=TEXT)
    ax_tp.set_ylim(0, 1.0)
    ax_tp.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax_tp.set_axisbelow(True)
    ax_tp.legend(loc="upper left", facecolor="#313244", edgecolor="#45475a",
                 labelcolor=TEXT, fontsize=9)
    ax_tp.set_title(
        f"FLUED E1-B Type-Conditional Boundary Probs  ({n_tp} points)",
        color=TEXT, fontsize=12, pad=10,
    )
    for spine in ax_tp.spines.values():
        spine.set_edgecolor(GRID)

    plt.tight_layout()
    out2 = Path("checkpoints/type_bp_curve.png")
    fig2.savefig(out2, dpi=150, facecolor=fig2.get_facecolor())
    print(f"已保存: {out2}")

plt.show()
