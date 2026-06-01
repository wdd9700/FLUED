"""
analyze_e1.py — Numerical trend analysis for FLUED E1 training (paper-ready).

Produces:
  1. Windowed averages (250-step windows) for all metrics
  2. Slope analysis (linear regression per window)  
  3. Phase-level summary (early/mid/late training)
  4. CSV export for paper tables

Usage:
  python analyze_e1.py checkpoints/e1_merged_full.log
"""
import re
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── Parse log ────────────────────────────────────────────────────────────────
def safe_float(s: Optional[str]) -> Optional[float]:
    if s is None or s in ("N/A", "nan", ""):
        return None
    try: return float(s)
    except ValueError: return None

def parse_all(text: str) -> Dict[str, List]:
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
    series: Dict[str, List] = {
        "step": [], "loss": [], "recon_loss": [], "comp_loss": [],
        "recon_acc": [], "soft_mn": [], "hard_mn": [], "units": [],
        "bp_mean": [], "bp_std": [],
        "h55": [], "h60": [], "h65": [],
        "t_step": [], "utf8": [], "ascii": [], "cjk": [], "op": [], "digit": [],
    }
    for m in pat_main.finditer(text):
        series["step"].append(int(m.group(1)))
        series["loss"].append(float(m.group(2)))
        series["recon_loss"].append(float(m.group(3)))
        series["comp_loss"].append(float(m.group(4)))
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

# ── Window analysis ──────────────────────────────────────────────────────────
def window_analysis(s: Dict, window_size: int = 250) -> List[Dict]:
    """Sliding window averages + slope for each metric."""
    n = len(s["step"])
    windows = []
    
    # Build step-indexed type_bp lookup
    type_map: Dict[int, Dict[str, float]] = {}
    for i, step in enumerate(s["t_step"]):
        vals = {}
        for k in ("utf8", "ascii", "cjk", "op", "digit"):
            v = s[k][i]
            if v is not None:
                vals[k] = v
        if vals:
            type_map[step] = vals
    
    i = 0
    while i < n:
        # Find window: steps within [start, start+window_size)
        win_start_step = s["step"][i]
        win_end_step = win_start_step + window_size
        j = i
        while j < n and s["step"][j] < win_end_step:
            j += 1
        
        if j == i:
            i += 1
            continue
        
        win_indices = list(range(i, j))
        win = {
            "start_step": s["step"][i],
            "end_step": s["step"][j-1],
            "n_points": len(win_indices),
        }
        
        # All numeric metrics: mean + slope
        metric_keys = [
            "loss", "recon_loss", "comp_loss", "recon_acc",
            "soft_mn", "hard_mn", "units", "bp_mean", "bp_std",
        ]
        for key in metric_keys:
            vals = [s[key][idx] for idx in win_indices]
            if not vals: continue
            avg = sum(vals) / len(vals)
            win[f"{key}_avg"] = avg
            
            # Slope via simple linear regression (x = step index, y = value)
            xs = [k - i for k in range(len(vals))]  # relative index
            n_pts = len(vals)
            if n_pts >= 3:
                sum_x = sum(xs)
                sum_y = sum(vals)
                sum_xy = sum(x * y for x, y in zip(xs, vals))
                sum_x2 = sum(x * x for x in xs)
                denom = n_pts * sum_x2 - sum_x * sum_x
                if abs(denom) > 1e-10:
                    slope = (n_pts * sum_xy - sum_x * sum_y) / denom
                    win[f"{key}_slope"] = slope
                else:
                    win[f"{key}_slope"] = 0.0
            else:
                win[f"{key}_slope"] = 0.0
        
        # Hard m/n sweep thresholds
        for thr_key in ("h55", "h60", "h65"):
            vals = [s[thr_key][idx] for idx in win_indices]
            if vals:
                win[f"{thr_key}_avg"] = sum(vals) / len(vals)
        
        # Type_bp averages (match by step range)
        type_vals: Dict[str, List[float]] = defaultdict(list)
        for idx in win_indices:
            step = s["step"][idx]
            if step in type_map:
                for k, v in type_map[step].items():
                    type_vals[k].append(v)
        for k, vals in type_vals.items():
            if vals:
                win[f"type_{k}_avg"] = sum(vals) / len(vals)
        
        windows.append(win)
        i = j
    
    return windows

# ── Phase summary ────────────────────────────────────────────────────────────
def phase_summary(windows: List[Dict], s: Dict) -> Dict:
    """Divide training into phases: early (0-15k), mid (15k-35k), late (35k+)."""
    phases = {
        "early (6k-15k)": {"wins": [], "metrics": {}},
        "mid (15k-35k)":  {"wins": [], "metrics": {}},
        "late (35k-40.7k)": {"wins": [], "metrics": {}},
    }
    
    for w in windows:
        if w["end_step"] < 15000:
            phases["early (6k-15k)"]["wins"].append(w)
        elif w["start_step"] < 35000:
            phases["mid (15k-35k)"]["wins"].append(w)
        else:
            phases["late (35k-40.7k)"]["wins"].append(w)
    
    result = {}
    for phase_name, data in phases.items():
        wins = data["wins"]
        if not wins:
            result[phase_name] = {"n_windows": 0}
            continue
        
        phase = {"n_windows": len(wins), "step_range": f"{wins[0]['start_step']}-{wins[-1]['end_step']}"}
        for key in ["recon_acc_avg", "loss_avg", "soft_mn_avg", "hard_mn_avg",
                     "bp_mean_avg", "bp_std_avg", "units_avg",
                     "recon_acc_slope", "loss_slope", "soft_mn_slope"]:
            vals = [w.get(key, float('nan')) for w in wins if key in w]
            if vals:
                valid = [v for v in vals if not math.isnan(v)]
                if valid:
                    phase[key] = sum(valid) / len(valid)
        
        result[phase_name] = phase
    
    return result

# ── Generate report ──────────────────────────────────────────────────────────
def print_report(windows: List[Dict], phases: Dict, s: Dict):
    """Print complete numerical analysis to stdout."""
    
    # ── Overall summary ──
    n = len(s["step"])
    print("=" * 90)
    print("FLUED E1 Stage-A — Numerical Trend Analysis (Paper Data)")
    print("=" * 90)
    print(f"  Data: {n} data points, step {s['step'][0]} → {s['step'][-1]}")
    print(f"  Window size: 250 steps, total windows: {len(windows)}")
    print()
    
    # ── Latest point ──
    print("─ Latest Values (step {}) ─".format(s["step"][-1]))
    latest = {
        "recon_acc": s["recon_acc"][-1],
        "loss": s["loss"][-1],
        "recon_loss": s["recon_loss"][-1],
        "comp_loss": s["comp_loss"][-1],
        "soft_m/n": s["soft_mn"][-1],
        "hard_m/n": s["hard_mn"][-1],
        "units": s["units"][-1],
        "bp_mean": s["bp_mean"][-1],
        "bp_std": s["bp_std"][-1],
    }
    for k, v in latest.items():
        print(f"  {k:>14s}: {v:.6f}" if isinstance(v, float) else f"  {k:>14s}: {v}")
    print()

    # ── Phase summary ──
    print("─ Phase-Level Summary ─")
    print(f"{'Phase':<22s} {'Windows':>8s} {'recon_acc':>10s} {'loss':>10s} {'soft_m/n':>10s} {'bp_mean':>10s} {'bp_std':>10s} {'acc_slope':>12s}")
    print("-" * 90)
    for phase_name, data in phases.items():
        if data.get("n_windows", 0) == 0:
            continue
        print(f"{phase_name:<22s} {data['n_windows']:>8d} "
              f"{data.get('recon_acc_avg', float('nan')):>10.6f} "
              f"{data.get('loss_avg', float('nan')):>10.6f} "
              f"{data.get('soft_mn_avg', float('nan')):>10.6f} "
              f"{data.get('bp_mean_avg', float('nan')):>10.6f} "
              f"{data.get('bp_std_avg', float('nan')):>10.6f} "
              f"{data.get('recon_acc_slope', float('nan')):>12.8f}")
    print()

    # ── Key metrics per window (truncated display) ──
    print("─ Windowed Averages (every 5th window shown) ─")
    header = f"{'Step Range':>16s} {'N':>4s} {'recon_acc':>10s} {'loss':>10s} {'soft_m/n':>10s} {'bp_std':>10s} {'bp_mean':>10s} {'acc_slope':>12s} {'type_utf8':>10s} {'type_cjk':>10s}"
    print(header)
    print("-" * len(header))
    
    for i, w in enumerate(windows):
        if i % 5 != 0:
            continue
        step_range = f"{w['start_step']}-{w['end_step']}"
        print(f"{step_range:>16s} {w['n_points']:>4d} "
              f"{w.get('recon_acc_avg', float('nan')):>10.6f} "
              f"{w.get('loss_avg', float('nan')):>10.6f} "
              f"{w.get('soft_mn_avg', float('nan')):>10.6f} "
              f"{w.get('bp_std_avg', float('nan')):>10.6f} "
              f"{w.get('bp_mean_avg', float('nan')):>10.6f} "
              f"{w.get('recon_acc_slope', float('nan')):>12.8f} "
              f"{w.get('type_utf8_avg', float('nan')):>10.4f} "
              f"{w.get('type_cjk_avg', float('nan')):>10.4f}")
    
    print()

    # ── Type BP phase summary ──
    print("─ Type Boundary Probability — Phase Averages ─")
    print(f"{'Phase':<22s} {'utf8_cont':>10s} {'ascii':>10s} {'cjk':>10s} {'op':>10s} {'digit':>10s}")
    print("-" * 72)
    for phase_name, data in phases.items():
        wins = data.get("wins", [])
        if not wins: continue
        type_avgs = {}
        for tk in ("type_utf8_avg", "type_ascii_avg", "type_cjk_avg", "type_op_avg", "type_digit_avg"):
            vals = [w.get(tk) for w in wins if tk in w and w.get(tk) is not None]
            if vals:
                type_avgs[tk] = sum(vals) / len(vals)
        print(f"{phase_name:<22s} "
              f"{type_avgs.get('type_utf8_avg', float('nan')):>10.4f} "
              f"{type_avgs.get('type_ascii_avg', float('nan')):>10.4f} "
              f"{type_avgs.get('type_cjk_avg', float('nan')):>10.4f} "
              f"{type_avgs.get('type_op_avg', float('nan')):>10.4f} "
              f"{type_avgs.get('type_digit_avg', float('nan')):>10.4f}")
    print()

    # ── Trend interpretation ──
    print("─ Trend Interpretation ─")
    if windows:
        first_w = windows[0]
        last_w = windows[-1]
        
        acc_delta = last_w.get("recon_acc_avg", 0) - first_w.get("recon_acc_avg", 0)
        mn_delta = last_w.get("soft_mn_avg", 0) - first_w.get("soft_mn_avg", 0)
        bpstd_delta = last_w.get("bp_std_avg", 0) - first_w.get("bp_std_avg", 0)
        
        print(f"  recon_acc: {first_w.get('recon_acc_avg', 0):.4f} → {last_w.get('recon_acc_avg', 0):.4f}  (Δ = {acc_delta:+.4f})")
        print(f"  soft_m/n:  {first_w.get('soft_mn_avg', 0):.4f} → {last_w.get('soft_mn_avg', 0):.4f}  (Δ = {mn_delta:+.4f})")
        print(f"  bp_std:    {first_w.get('bp_std_avg', 0):.4f} → {last_w.get('bp_std_avg', 0):.4f}  (Δ = {bpstd_delta:+.4f})")
        
        # Convergence check
        late_wins = [w for w in windows if w["start_step"] >= 35000]
        if len(late_wins) >= 3:
            acc_stds = [w.get("recon_acc_avg", 0) for w in late_wins]
            acc_std = (sum((x - sum(acc_stds)/len(acc_stds))**2 for x in acc_stds) / len(acc_stds)) ** 0.5
            print(f"\n  Late-phase recon_acc std: {acc_std:.6f}  (convergence check: {'✓ STABLE' if acc_std < 0.001 else '⚠ still drifting'})")

    print()
    print("=" * 90)

# ── Export ───────────────────────────────────────────────────────────────────
def export_csv(windows: List[Dict], phases: Dict, s: Dict, out_path: Path):
    """Export window data as CSV for paper tables."""
    import csv
    
    # Full window data
    with open(out_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "start_step", "end_step", "n_points",
            "recon_acc_avg", "recon_acc_slope",
            "loss_avg", "loss_slope",
            "recon_loss_avg", "comp_loss_avg",
            "soft_mn_avg", "soft_mn_slope",
            "hard_mn_avg", "bp_mean_avg", "bp_std_avg",
            "units_avg", "h55_avg", "h60_avg", "h65_avg",
            "type_utf8_avg", "type_ascii_avg", "type_cjk_avg",
            "type_op_avg", "type_digit_avg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for w in windows:
            writer.writerow(w)
    
    # Phase summary JSON
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_steps": s["step"][-1],
            "n_data_points": len(s["step"]),
            "window_size": 250,
            "n_windows": len(windows),
            "latest": {
                "step": s["step"][-1],
                "recon_acc": s["recon_acc"][-1],
                "loss": s["loss"][-1],
                "soft_mn": s["soft_mn"][-1],
                "hard_mn": s["hard_mn"][-1],
                "bp_mean": s["bp_mean"][-1],
                "bp_std": s["bp_std"][-1],
                "units": s["units"][-1],
            },
            "phases": phases,
            "windows": windows,
        }, f, indent=2)
    
    print(f"Exported: {out_path.with_suffix('.csv')}")
    print(f"Exported: {out_path.with_suffix('.json')}")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/e1_merged_full.log")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    
    s = parse_all(text)
    print(f"Parsed {len(s['step'])} main data points, {len(s['t_step'])} type_bp points")
    
    windows = window_analysis(s, window_size=250)
    print(f"Window analysis: {len(windows)} windows of 250 steps")
    
    phases = phase_summary(windows, s)
    
    print_report(windows, phases, s)
    
    out_path = Path("checkpoints/e1_numerical_analysis")
    export_csv(windows, phases, s, out_path)
