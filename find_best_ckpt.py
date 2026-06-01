"""Find best E1 checkpoint between step 35000 and latest."""
import re
from pathlib import Path

# Parse all log data >= step 35000
text = Path('checkpoints/e1_merged_full.log').read_text(encoding='utf-8', errors='ignore')

# Also try run2 log
try:
    text += '\n' + Path('checkpoints/e1_run2.log').read_text(encoding='utf-8', errors='ignore')
except: pass

pat = re.compile(
    r'step=\s*(\d+)\s+loss=([\-\d.]+)\s+recon=([\d.]+)\s+comp=([\-\d.]+)\s+'
    r'recon_acc=([\d.]+)\s+soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+units=([\d.]+)\s+'
    r'bp_mean=([\d.]+)\s+bp_std=([\d.]+)\s+bhead_gnorm=([\d.]+)'
)

pts = []
for m in pat.finditer(text):
    s = int(m.group(1))
    if s < 35000: continue
    acc = float(m.group(5))
    if acc < 0.9 or acc > 1.01: continue  # skip FP16 artifacts
    pts.append({
        'step': s,
        'recon_acc': acc,
        'recon_loss': float(m.group(3)),
        'comp_loss': float(m.group(4)),
        'soft_mn': float(m.group(6)),
        'bp_std': float(m.group(10)),
        'bhead_gnorm': float(m.group(11)),
    })

print(f"Valid data points >= 35000: {len(pts)}")
if not pts:
    print("No data found!")
    exit(1)

print(f"Step range: {pts[0]['step']} -> {pts[-1]['step']}")

# Top 10 by recon_acc
by_acc = sorted(pts, key=lambda p: p['recon_acc'], reverse=True)[:10]
print("\nTop 10 by recon_acc:")
print(f"{'Step':>8s}  {'recon_acc':>10s}  {'recon_loss':>10s}  {'comp_loss':>10s}  {'soft_m/n':>10s}  {'bp_std':>10s}")
print("-" * 72)
for p in by_acc:
    print(f"{p['step']:>8d}  {p['recon_acc']:>10.6f}  {p['recon_loss']:>10.6f}  {p['comp_loss']:>10.6f}  {p['soft_mn']:>10.6f}  {p['bp_std']:>10.6f}")

# Top 10 by combined score (recon_acc - |comp_loss| tradeoff)
by_tradeoff = sorted(pts, key=lambda p: p['recon_acc'] + abs(p['comp_loss']) * 10, reverse=True)[:10]
print("\nTop 10 by recon_acc - comp_loss tradeoff:")
print(f"{'Step':>8s}  {'recon_acc':>10s}  {'recon_loss':>10s}  {'comp_loss':>10s}  {'soft_m/n':>10s}  {'bp_std':>10s}")
print("-" * 72)
for p in by_tradeoff:
    print(f"{p['step']:>8d}  {p['recon_acc']:>10.6f}  {p['recon_loss']:>10.6f}  {p['comp_loss']:>10.6f}  {p['soft_mn']:>10.6f}  {p['bp_std']:>10.6f}")

# Map to nearest available checkpoint
ckpts = []
for f in Path('checkpoints').glob('e1_step*.pt'):
    m = re.search(r'step(\d+)', f.name)
    if m:
        s = int(m.group(1))
        if s >= 35000:
            ckpts.append((s, f.name))
ckpts.sort()

print("\nAvailable checkpoints >= 35000:")
for s, name in ckpts:
    print(f"  {name} (step {s})")

print("\nBest checkpoint recommendations:")
# For each top data point, find nearest checkpoint
seen = set()
for p in by_acc[:5]:
    nearest = min(ckpts, key=lambda c: abs(c[0] - p['step']))
    key = nearest[1]
    if key not in seen:
        seen.add(key)
        dist = nearest[0] - p['step']
        print(f"  {nearest[1]}: nearest to best step {p['step']} (acc={p['recon_acc']:.6f}, dist={dist:+d})")

# Also check: which checkpoint has the most data points showing high acc nearby?
print("\nCheckpoint neighborhood quality (avg recon_acc within +/-250 steps):")
for s, name in ckpts:
    nearby = [p for p in pts if abs(p['step'] - s) <= 250]
    if nearby:
        avg = sum(p['recon_acc'] for p in nearby) / len(nearby)
        avg_rl = sum(p['recon_loss'] for p in nearby) / len(nearby)
        avg_comp = sum(p['comp_loss'] for p in nearby) / len(nearby)
        print(f"  {name}: n={len(nearby):>3d}  avg_acc={avg:.6f}  avg_rl={avg_rl:.6f}  avg_comp={avg_comp:.6f}")
    else:
        print(f"  {name}: no nearby data (in gap)")
