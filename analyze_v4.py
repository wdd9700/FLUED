"""
V4 Full Training Analysis — comprehensive visualization & trends
Reads all available V4 log data (e1_retrain_v4.log + terminal outputs)
and generates full analysis plots.
"""
import re
import os
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Parse helpers ──
def parse_log(filepath):
    """Parse E1 training log lines into dict of step→metrics."""
    data = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Main metrics line
            m = re.match(
                r".*step=\s*(\d+)\s+loss=([\-\d.]+)\s+recon=([\-\d.]+)\s+comp=([\-\d.]+)"
                r"\s+recon_acc=([\-\d.]+)\s+soft_m/n=([\-\d.]+)\s+hard_m/n=([\-\d.]+)"
                r"\s+units=([\-\d.]+)\s+bp_mean=([\-\d.]+)\s+bp_std=([\-\d.]+)"
                r"\s+bhead_gnorm=([\-\d.]+)\s+lr=([\-\de.]+)",
                line
            )
            if m:
                step = int(m.group(1))
                data["step"].append(step)
                data["loss"].append(float(m.group(2)))
                data["recon"].append(float(m.group(3)))
                data["comp"].append(float(m.group(4)))
                data["recon_acc"].append(float(m.group(5)))
                data["soft_mn"].append(float(m.group(6)))
                data["hard_mn"].append(float(m.group(7)))
                data["units"].append(float(m.group(8)))
                data["bp_mean"].append(float(m.group(9)))
                data["bp_std"].append(float(m.group(10)))
                data["bhead_gnorm"].append(float(m.group(11)))
                data["lr"].append(float(m.group(12)))
                continue

            # Type BP line
            m2 = re.match(
                r".*step=\s*(\d+)\s+\[type_bp\]\s+utf8=([\d.]+)\s+ascii=([\d.]+)"
                r"\s+cjk=([\d.]+)\s+op=([\d.]+)\s+digit=([\d.]+)",
                line
            )
            if m2:
                step = int(m2.group(1))
                data["type_step"].append(step)
                data["utf8"].append(float(m2.group(2)))
                data["ascii"].append(float(m2.group(3)))
                data["cjk"].append(float(m2.group(4)))
                data["op"].append(float(m2.group(5)))
                data["digit"].append(float(m2.group(6)))
    return data


# ── Also parse manual data from terminal logs ──
def parse_manual_data():
    """Supplement with data manually extracted from terminal captures."""
    # This covers the V4 run from step 10000-50000 (terminal output captures)
    manual = defaultdict(list)
    # Data extracted from terminal output files during the conversation
    # Format: step,loss,recon,comp,recon_acc,soft_mn,bp_mean,bp_std,cjk,utf8,ascii,op,digit
    raw = """
    # This will be populated from terminal captures
    """
    return manual


# ── Plotting ──
def plot_all(v4, v1, outdir="analysis_plots"):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                          "axes.titlesize": 11, "axes.labelsize": 10})

    # ── Figure 1: Loss & Accuracy ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    ax = axes[0, 0]
    if v4["step"]:
        ax.plot(v4["step"], v4["loss"], alpha=0.3, color="blue", linewidth=0.5, label="V4 loss (raw)")
        # smoothed
        window = max(1, len(v4["step"]) // 50)
        if len(v4["step"]) > window:
            loss_smooth = np.convolve(v4["loss"], np.ones(window)/window, mode="valid")
            steps_smooth = v4["step"][window//2:window//2+len(loss_smooth)]
            ax.plot(steps_smooth, loss_smooth, color="blue", linewidth=1.5, label=f"V4 (MA{window})")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Recon vs Comp loss
    ax = axes[0, 1]
    if v4["step"]:
        ax.plot(v4["step"], v4["recon"], alpha=0.5, color="red", linewidth=0.8, label="Recon")
        ax.plot(v4["step"], v4["comp"], alpha=0.5, color="green", linewidth=0.8, label="Compression")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Loss Component")
    ax.set_title("Recon vs Compression Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Recon accuracy
    ax = axes[1, 0]
    if v4["step"]:
        ax.plot(v4["step"], v4["recon_acc"], color="purple", linewidth=1, alpha=0.8)
        ax.set_ylim(0.97, 1.001)
    ax.set_ylabel("Reconstruction Accuracy")
    ax.set_xlabel("Step")
    ax.set_title("Reconstruction Accuracy")
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    if v4["step"]:
        ax.plot(v4["step"], v4["lr"], color="orange", linewidth=1)
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_title("LR Schedule")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{outdir}/01_loss_accuracy.png", dpi=200)
    plt.close(fig)

    # ── Figure 2: Compression & Units ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if v4["step"]:
        ax.plot(v4["step"], v4["bp_mean"], color="teal", linewidth=1, label="bp_mean (m/n)")
        ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5, label="target=0.15")
    ax.set_ylabel("Boundary Probability Mean")
    ax.set_title("Global Compression (bp_mean = m/n)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if v4["step"]:
        ax.plot(v4["step"], v4["soft_mn"], color="blue", linewidth=1, alpha=0.7, label="soft m/n")
        ax.plot(v4["step"], v4["hard_mn"], color="red", linewidth=1, alpha=0.7, label="hard m/n")
    ax.set_ylabel("m/n Ratio")
    ax.set_title("Soft vs Hard m/n")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if v4["step"]:
        ax.plot(v4["step"], v4["units"], color="green", linewidth=1)
    ax.set_ylabel("Number of Dynamic Units")
    ax.set_xlabel("Step")
    ax.set_title("Units per 512-byte Sequence")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if v4["step"]:
        ax.plot(v4["step"], v4["bp_std"], color="magenta", linewidth=1)
    ax.set_ylabel("Boundary Prob Std")
    ax.set_xlabel("Step")
    ax.set_title("Polarization (bp_std)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{outdir}/02_compression.png", dpi=200)
    plt.close(fig)

    # ── Figure 3: Type-specific BP trends ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    types = ["utf8", "ascii", "cjk", "op", "digit"]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
    titles = ["UTF-8 Continuation", "ASCII", "CJK", "Operators/Punct", "Digits"]
    
    for i, (t, c, title) in enumerate(zip(types, colors, titles)):
        ax = axes[i // 3, i % 3]
        if v4["type_step"] and t in v4:
            ax.plot(v4["type_step"], v4[t], color=c, linewidth=1, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Boundary Prob")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        # Add reference line
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    # CJK comparison with 初版
    ax = axes[1, 2]
    if v4["type_step"] and "cjk" in v4:
        ax.plot(v4["type_step"], v4["cjk"], color="#FF5722", linewidth=1.5, label="V4 CJK")
    if v1 and v1["type_step"] and "cjk" in v1:
        ax.plot(v1["type_step"], v1["cjk"], color="gray", linewidth=1, alpha=0.6, label="V1 CJK (初版)")
    ax.axhline(y=0.33, color="green", linestyle="--", alpha=0.5, label="1 char/unit (ideal)")
    ax.axhline(y=0.06, color="red", linestyle="--", alpha=0.3, label="0.058 (初版 final)")
    ax.set_title("CJK: V4 vs 初版")
    ax.set_ylabel("Boundary Prob")
    ax.set_xlabel("Step")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{outdir}/03_type_bp.png", dpi=200)
    plt.close(fig)

    # ── Figure 4: Gradient Norm & Phase Analysis ──
    fig, ax = plt.subplots(figsize=(14, 5))
    if v4["step"]:
        ax.plot(v4["step"], v4["bhead_gnorm"], color="darkred", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Boundary Head Gradient Norm")
    ax.set_xlabel("Step")
    ax.set_title("Boundary Head Gradient Norm (Gate Health)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"{outdir}/04_gradnorm.png", dpi=200)
    plt.close(fig)

    # ── Figure 5: Combined Dashboard ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # Loss
    ax = axes[0, 0]
    if v4["step"]:
        w = max(1, len(v4["step"]) // 50)
        if len(v4["step"]) > w:
            s = np.convolve(v4["loss"], np.ones(w)/w, mode="valid")
            ax.plot(v4["step"][w//2:w//2+len(s)], s, color="blue", linewidth=1.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Loss (smoothed)")
    ax.grid(True, alpha=0.3)

    # BP mean
    ax = axes[0, 1]
    if v4["step"]:
        ax.plot(v4["step"], v4["bp_mean"], color="teal", linewidth=1)
    ax.axhline(y=0.15, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("bp_mean")
    ax.grid(True, alpha=0.3)

    # Recon acc
    ax = axes[1, 0]
    if v4["step"]:
        ax.plot(v4["step"], v4["recon_acc"], color="purple", linewidth=0.8, alpha=0.8)
        ax.set_ylim(0.97, 1.001)
    ax.set_ylabel("Recon Accuracy")
    ax.grid(True, alpha=0.3)

    # Type BP
    ax = axes[1, 1]
    for t, c, lbl in [("utf8", "#2196F3", "UTF-8"), ("ascii", "#4CAF50", "ASCII"),
                       ("cjk", "#FF5722", "CJK"), ("op", "#9C27B0", "Op"),
                       ("digit", "#FF9800", "Digit")]:
        if v4["type_step"] and t in v4:
            ax.plot(v4["type_step"], v4[t], color=c, linewidth=0.8, alpha=0.7, label=lbl)
    ax.set_ylabel("Type BP")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    # BP std
    ax = axes[2, 0]
    if v4["step"]:
        ax.plot(v4["step"], v4["bp_std"], color="magenta", linewidth=1)
    ax.set_ylabel("bp_std (Polarization)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Units
    ax = axes[2, 1]
    if v4["step"]:
        ax.plot(v4["step"], v4["units"], color="green", linewidth=1)
    ax.set_ylabel("Units / 512 bytes")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    fig.suptitle("FLUED V4 — Full Training Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{outdir}/05_dashboard.png", dpi=200)
    plt.close(fig)

    print(f"✅ Plots saved to {outdir}/")


# ── Trend Analysis ──
def trend_analysis(v4, v1):
    """Print quantitative trend analysis."""
    print("\n" + "="*70)
    print("FLUED V4 — Quantitative Trend Analysis")
    print("="*70)

    def get_range(data, key, start, end):
        """Get average of key in step range [start, end]."""
        steps = data.get("step", data.get("type_step", []))
        vals = data.get(key, [])
        if not steps or not vals:
            return None
        subset = [v for s, v in zip(steps, vals) if start <= s <= end]
        return np.mean(subset) if subset else None

    # Phase definitions
    phases = [
        ("Warmup (0-2500)", 0, 2500),
        ("Early Comp (2500-5000)", 2500, 5000),
        ("Compression (5000-15000)", 5000, 15000),
        ("Deep Comp (15000-30000)", 15000, 30000),
        ("Convergence (30000-50000)", 30000, 50000),
    ]

    print(f"\n{'Phase':<25} {'bp_mean':>8} {'cjk':>8} {'utf8':>8} {'ascii':>8} {'recon_acc':>10} {'units':>7}")
    print("-"*75)
    for name, start, end in phases:
        bp = get_range(v4, "bp_mean", start, end)
        cj = get_range(v4, "cjk", start, end) if v4.get("type_step") else None
        ut = get_range(v4, "utf8", start, end) if v4.get("type_step") else None
        ac = get_range(v4, "ascii", start, end) if v4.get("type_step") else None
        ra = get_range(v4, "recon_acc", start, end)
        un = get_range(v4, "units", start, end)
        print(f"{name:<25} {bp or 'N/A':>8} {cj or 'N/A':>8} {ut or 'N/A':>8} {ac or 'N/A':>8} {ra or 'N/A':>10} {un or 'N/A':>7}")

    # Final metrics
    print(f"\n{'─'*70}")
    print("FINAL METRICS (step 50000):")
    for key in ["bp_mean", "cjk", "utf8", "ascii", "op", "digit", "recon_acc", "units", "bp_std"]:
        steps = v4.get("step", v4.get("type_step", []))
        vals = v4.get(key, [])
        if steps and vals:
            # Get last 10 values
            last = vals[-10:] if len(vals) >= 10 else vals
            print(f"  {key:<15}: {np.mean(last):.4f}  (last 10 samples)")

    # CJK comparison
    if v1 and v1.get("type_step") and v1.get("cjk"):
        v1_last = v1["cjk"][-10:] if len(v1["cjk"]) >= 10 else v1["cjk"]
        print(f"\n  初版 CJK final: {np.mean(v1_last):.4f}")
        print(f"  V4 CJK final:    {np.mean(v4.get('cjk', [0])[-10:]):.4f}" if v4.get("cjk") else "")


# ── Main ──
if __name__ == "__main__":
    print("Parsing V4 log...")
    v4 = parse_log("checkpoints/e1_retrain_v4.log")
    print(f"  V4 main: {len(v4.get('step',[]))} data points")
    print(f"  V4 types: {len(v4.get('type_step',[]))} data points")

    print("Parsing 初版 (V1) log...")
    v1 = parse_log("checkpoints/e1b_full.log") if os.path.exists("checkpoints/e1b_full.log") else None
    if v1:
        print(f"  V1 main: {len(v1.get('step',[]))} data points")
        print(f"  V1 types: {len(v1.get('type_step',[]))} data points")

    trend_analysis(v4, v1)
    plot_all(v4, v1)
