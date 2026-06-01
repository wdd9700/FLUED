"""A-class summary: extract final metrics from multi-seed E1 logs.

Produces a paper-ready table:
  target | seed | recon_acc | soft_m/n | bytes/unit | bp_std | utf8 | cjk | ascii | op | digit
"""
import re, statistics, json, sys
from pathlib import Path

SEED_DIRS = {
    'S1 (42)':  'checkpoints/e1_seed42',
    'S2 (123)': 'checkpoints/e1_seed123',
    'S3 (999)': 'checkpoints/e1_seed999',
}

# Allow overriding via CLI: python summary_a.py checkpoints/e1_tc045 S_045
if len(sys.argv) >= 2:
    base_dir = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) >= 3 else 'custom'
    SEED_DIRS = {tag: base_dir}

PAT_MAIN = re.compile(
    r'step=\s*(\d+)\s+loss=([\-\d.]+)\s+recon=([\d.]+)\s+comp=([\-\d.]+)\s+'
    r'recon_acc=([\d.]+)\s+soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+'
    r'units=([\d.]+)\s+bp_mean=([\d.]+)\s+bp_std=([\d.]+)\s+bhead_gnorm=([\d.]+)'
)
PAT_TYPE = re.compile(
    r'step=\s*(\d+)\s+\[type_bp\]\s+utf8=([\d.NA]+)\s+ascii=([\d.NA]+)\s+'
    r'cjk=([\d.NA]+)\s+op=([\d.NA]+)\s+digit=([\d.NA]+)'
)
import re, statistics, json
from pathlib import Path

SEED_DIRS = {
    'S1 (42)':  'checkpoints/e1_seed42',
    'S2 (123)': 'checkpoints/e1_seed123',
    'S3 (999)': 'checkpoints/e1_seed999',
}

PAT_MAIN = re.compile(
    r'step=\s*(\d+)\s+loss=([\-\d.]+)\s+recon=([\d.]+)\s+comp=([\-\d.]+)\s+'
    r'recon_acc=([\d.]+)\s+soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+'
    r'units=([\d.]+)\s+bp_mean=([\d.]+)\s+bp_std=([\d.]+)\s+bhead_gnorm=([\d.]+)'
)
PAT_TYPE = re.compile(
    r'step=\s*(\d+)\s+\[type_bp\]\s+utf8=([\d.NA]+)\s+ascii=([\d.NA]+)\s+'
    r'cjk=([\d.NA]+)\s+op=([\d.NA]+)\s+digit=([\d.NA]+)'
)

def safe_f(s):
    try: return float(s)
    except: return None

def collect_logs(directory):
    """Read all .log files from a directory, return concatenated text."""
    d = Path(directory)
    if not d.exists():
        return ''
    text = ''
    for log in sorted(d.glob('*.log')):
        try:
            text += log.read_text(encoding='utf-8', errors='replace')
        except Exception:
            pass
    return text

results = {}
for label, d in SEED_DIRS.items():
    text = collect_logs(d)
    if not text:
        print(f'{label}: NO LOGS in {d}')
        continue

    main_entries = [(int(m.group(1)), m) for m in PAT_MAIN.finditer(text)]
    type_entries = [(int(m.group(1)), m) for m in PAT_TYPE.finditer(text)]

    if not main_entries:
        print(f'{label}: no main metrics found')
        continue

    max_step, last = max(main_entries, key=lambda x: x[0])
    r = {
        'seed': label,
        'max_step': max_step,
        'loss': safe_f(last.group(2)),
        'recon': safe_f(last.group(3)),
        'comp': safe_f(last.group(4)),
        'recon_acc': safe_f(last.group(5)),
        'soft_mn': safe_f(last.group(6)),
        'hard_mn': safe_f(last.group(7)),
        'units': safe_f(last.group(8)),
        'bp_mean': safe_f(last.group(9)),
        'bp_std': safe_f(last.group(10)),
        'bhead_gnorm': safe_f(last.group(11)),
    }

    if type_entries:
        nearest = min(type_entries, key=lambda x: abs(x[0] - max_step))
        tm = nearest[1]
        r.update({
            'utf8_bp': safe_f(tm.group(2)),
            'ascii_bp': safe_f(tm.group(3)),
            'cjk_bp': safe_f(tm.group(4)),
            'op_bp': safe_f(tm.group(5)),
            'digit_bp': safe_f(tm.group(6)),
        })

    results[label] = r
    print(f"{label}: step={max_step}  soft_m/n={r['soft_mn']:.4f}  "
          f"bp_std={r['bp_std']:.4f}  recon_acc={r['recon_acc']:.4f}  "
          f"cjk_bp={r.get('cjk_bp', 'N/A')}")

# ---- Aggregate ----
print('\n=== Paper-Ready Table (A: Reconstruction Stability) ===')
print(f'{"target":>8}  {"seed":>6}  {"recon_acc":>9}  {"soft_m/n":>8}  {"bytes/unit":>9}  '
      f'{"bp_std":>7}  {"utf8":>6}  {"cjk":>6}  {"ascii":>6}  {"op":>6}  {"digit":>6}')
print('-' * 95)
for label, r in results.items():
    bu = 1.0 / r['soft_mn'] if r.get('soft_mn', 0) > 0 else float('inf')
    print(f'{"0.30":>8}  {label:>6}  {r.get("recon_acc",0):>9.4f}  {r.get("soft_mn",0):>8.4f}  '
          f'{bu:>9.2f}  {r.get("bp_std",0):>7.4f}  '
          f'{r.get("utf8_bp",0):>6.3f}  {r.get("cjk_bp",0):>6.3f}  '
          f'{r.get("ascii_bp",0):>6.3f}  {r.get("op_bp",0):>6.3f}  {r.get("digit_bp",0):>6.3f}')

# Mean row (when multiple seeds)
if len(results) >= 2:
    keys = ['recon_acc', 'soft_mn', 'bp_std', 'utf8_bp', 'cjk_bp', 'ascii_bp', 'op_bp', 'digit_bp']
    avgs = {}
    for k in keys:
        vals = [r[k] for r in results.values() if k in r and r[k] is not None]
        if vals:
            avgs[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) >= 2 else 0)
    mu_bu = 1.0 / avgs['soft_mn'][0] if avgs.get('soft_mn', (0,))[0] > 0 else float('inf')
    print('-' * 95)
    print(f'{"0.30":>8}  {"mean±σ":>6}  {avgs["recon_acc"][0]:>9.4f}  {avgs["soft_mn"][0]:>8.4f}  '
          f'{mu_bu:>9.2f}  {avgs["bp_std"][0]:>7.4f}  '
          f'{avgs["utf8_bp"][0]:>6.3f}  {avgs["cjk_bp"][0]:>6.3f}  '
          f'{avgs["ascii_bp"][0]:>6.3f}  {avgs["op_bp"][0]:>6.3f}  {avgs["digit_bp"][0]:>6.3f}')
    std_bu = 1.0 / max(0.001, avgs['soft_mn'][0] - avgs['soft_mn'][1]) - mu_bu
    print(f'{"":>8}  {"(±σ)":>6}  {avgs["recon_acc"][1]:>9.4f}  {avgs["soft_mn"][1]:>8.4f}  '
          f'{"±"+str(round(std_bu,2)):>9}  {avgs["bp_std"][1]:>7.4f}  '
          f'{avgs["utf8_bp"][1]:>6.3f}  {avgs["cjk_bp"][1]:>6.3f}  '
          f'{avgs["ascii_bp"][1]:>6.3f}  {avgs["op_bp"][1]:>6.3f}  {avgs["digit_bp"][1]:>6.3f}')

print()
print('Key: target=0.30 throughout. bytes/unit = 1/soft_m/n. '
      'Type bp (utf8/cjk/ascii/op/digit) from per-type boundary head output.')

# ---- Export CSV (paper-ready) ----
csv_path = 'paper_table_a.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
    import csv
    writer = csv.writer(fh)
    writer.writerow(['target', 'seed', 'recon_acc', 'soft_m/n', 'bytes/unit',
                     'bp_std', 'utf8_bp', 'cjk_bp', 'ascii_bp', 'op_bp', 'digit_bp'])
    for label, r in results.items():
        bu = 1.0 / r['soft_mn'] if r.get('soft_mn', 0) > 0 else float('inf')
        writer.writerow([
            '0.30', label, f"{r.get('recon_acc',0):.4f}", f"{r.get('soft_mn',0):.4f}",
            f"{bu:.2f}", f"{r.get('bp_std',0):.4f}",
            f"{r.get('utf8_bp',0):.3f}", f"{r.get('cjk_bp',0):.3f}",
            f"{r.get('ascii_bp',0):.3f}", f"{r.get('op_bp',0):.3f}", f"{r.get('digit_bp',0):.3f}",
        ])
    # Mean row
    if len(results) >= 2:
        keys = ['recon_acc', 'soft_mn', 'bp_std', 'utf8_bp', 'cjk_bp', 'ascii_bp', 'op_bp', 'digit_bp']
        means = {}
        for k in keys:
            vals = [r[k] for r in results.values() if k in r and r[k] is not None]
            means[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) >= 2 else 0)
        mu_bu = 1.0 / means['soft_mn'][0] if means['soft_mn'][0] > 0 else float('inf')
        writer.writerow([
            '0.30', 'mean', f"{means['recon_acc'][0]:.4f}", f"{means['soft_mn'][0]:.4f}",
            f"{mu_bu:.2f}", f"{means['bp_std'][0]:.4f}",
            f"{means['utf8_bp'][0]:.3f}", f"{means['cjk_bp'][0]:.3f}",
            f"{means['ascii_bp'][0]:.3f}", f"{means['op_bp'][0]:.3f}", f"{means['digit_bp'][0]:.3f}",
        ])
        writer.writerow([
            '0.30', 'std', f"{means['recon_acc'][1]:.4f}", f"{means['soft_mn'][1]:.4f}",
            f"±{1.0/max(0.001,means['soft_mn'][0]-means['soft_mn'][1])-mu_bu:.2f}",
            f"{means['bp_std'][1]:.4f}",
            f"{means['utf8_bp'][1]:.3f}", f"{means['cjk_bp'][1]:.3f}",
            f"{means['ascii_bp'][1]:.3f}", f"{means['op_bp'][1]:.3f}", f"{means['digit_bp'][1]:.3f}",
        ])
print(f'CSV exported → {csv_path}')

# ---- Export Markdown ----
md_path = 'paper_table_a.md'
with open(md_path, 'w', encoding='utf-8') as fh:
    fh.write('| target | seed | recon_acc | soft_m/n | bytes/unit | bp_std | utf8 | cjk | ascii | op | digit |\n')
    fh.write('|--------|------|-----------|----------|------------|--------|------|-----|-------|----|-------|\n')
    for label, r in results.items():
        bu = 1.0 / r['soft_mn'] if r.get('soft_mn', 0) > 0 else float('inf')
        fh.write(f'| 0.30 | {label} | {r.get("recon_acc",0):.4f} | {r.get("soft_mn",0):.4f} | '
                 f'{bu:.2f} | {r.get("bp_std",0):.4f} | '
                 f'{r.get("utf8_bp",0):.3f} | {r.get("cjk_bp",0):.3f} | '
                 f'{r.get("ascii_bp",0):.3f} | {r.get("op_bp",0):.3f} | {r.get("digit_bp",0):.3f} |\n')
    if len(results) >= 2:
        mu_bu = 1.0 / means['soft_mn'][0] if means['soft_mn'][0] > 0 else float('inf')
        fh.write(f'| 0.30 | **mean** | {means["recon_acc"][0]:.4f} | {means["soft_mn"][0]:.4f} | '
                 f'{mu_bu:.2f} | {means["bp_std"][0]:.4f} | '
                 f'{means["utf8_bp"][0]:.3f} | {means["cjk_bp"][0]:.3f} | '
                 f'{means["ascii_bp"][0]:.3f} | {means["op_bp"][0]:.3f} | {means["digit_bp"][0]:.3f} |\n')
        fh.write(f'| | *±σ* | {means["recon_acc"][1]:.4f} | {means["soft_mn"][1]:.4f} | '
                 f'— | {means["bp_std"][1]:.4f} | '
                 f'{means["utf8_bp"][1]:.3f} | {means["cjk_bp"][1]:.3f} | '
                 f'{means["ascii_bp"][1]:.3f} | {means["op_bp"][1]:.3f} | {means["digit_bp"][1]:.3f} |\n')
print(f'Markdown exported → {md_path}')
