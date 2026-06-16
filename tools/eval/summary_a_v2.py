"""A-class v2 summary: extract final eval metrics from multi-seed E1 logs.

v2 pure denoising (latent_consistency=0) — three seeds.
Parses Python logger output (e1_class300m_16gb.log) for the final E1 eval line.
"""
import re, json, sys
from pathlib import Path

SEED_DIRS = {
    'seed=42':  'checkpoints/e1_v2_seed42/e1_class300m_16gb.log',
    'seed=123': 'checkpoints/e1_v2_seed123/e1_class300m_16gb.log',
    'seed=999': 'checkpoints/e1_v2_seed999/e1_class300m_16gb.log',
}

PAT_EVAL = re.compile(
    r'E1 eval.*reconstruction_accuracy=([\d.]+)\s+m/n=([\d.]+)'
)
PAT_TRAIN = re.compile(
    r'step=\s*(\d+)\s+loss=([\-\d.]+)\s+recon=([\-\d.]+)\s+comp=([\-\d.]+)\s+'
    r'latent=([\-\d.]+)\s+recon_acc=([\d.]+)\s+soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+'
    r'units=([\d.]+)\s+bp_mean=([\d.]+)\s+bp_std=([\d.]+)\s+bhead_gnorm=([\d.]+)'
)
PAT_TYPE = re.compile(
    r'step=\s*(\d+)\s+\[type_bp\]\s+utf8=([\d.NA]+)\s+ascii=([\d.NA]+)\s+'
    r'cjk=([\d.NA]+)\s+op=([\d.NA]+)\s+digit=([\d.NA]+)'
)

def safe_f(s):
    try: return float(s)
    except: return None

def collect(path):
    if not Path(path).exists():
        return None
    return Path(path).read_text(encoding='utf-8', errors='replace')

results = {}
for label, path in SEED_DIRS.items():
    text = collect(path)
    if not text:
        print(f'{label}: NO LOG at {path}')
        continue

    # Eval line (final metrics)
    eval_m = list(PAT_EVAL.finditer(text))
    eval_acc = safe_f(eval_m[-1].group(1)) if eval_m else None
    eval_mn  = safe_f(eval_m[-1].group(2)) if eval_m else None

    # Last training step + type_bp
    main_entries = [(int(m.group(1)), m) for m in PAT_TRAIN.finditer(text)]
    type_entries = [(int(m.group(1)), m) for m in PAT_TYPE.finditer(text)]

    if not main_entries:
        print(f'{label}: no training metrics found')
        continue

    max_step, last = max(main_entries, key=lambda x: x[0])

    r = {
        'seed': label,
        'max_step': max_step,
        'eval_acc': eval_acc,
        'eval_mn': eval_mn,
        'train_acc': safe_f(last.group(6)),
        'soft_mn': safe_f(last.group(7)),
        'hard_mn': safe_f(last.group(8)),
        'units': safe_f(last.group(9)),
        'bp_std': safe_f(last.group(11)),
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

# ---- Print ----
if not results:
    print("No results found.")
    sys.exit(1)

print('\n=== A-class v2: Reconstruction Stability (pure denoising) ===\n')
header = f"{'Seed':>8} | {'Eval Acc':>9} | {'m/n':>6} | {'bp_std':>7} | {'utf8':>6} | {'cjk':>6} | {'ascii':>6} | {'op':>6} | {'digit':>6} | {'units':>6}"
print(header)
print('-' * len(header))
for label, r in results.items():
    print(f"{label:>8} | {r.get('eval_acc',0):>9.4f} | {r.get('eval_mn',0):>6.4f} | "
          f"{r.get('bp_std',0):>7.4f} | {r.get('utf8_bp',0):>6.3f} | {r.get('cjk_bp',0):>6.3f} | "
          f"{r.get('ascii_bp',0):>6.3f} | {r.get('op_bp',0):>6.3f} | {r.get('digit_bp',0):>6.3f} | "
          f"{r.get('units',0):>6.1f}")

# ---- Aggregate stats ----
if len(results) >= 2:
    accs = [r['eval_acc'] for r in results.values() if r.get('eval_acc')]
    mns  = [r['eval_mn']  for r in results.values() if r.get('eval_mn')]
    bpss = [r['bp_std']   for r in results.values() if r.get('bp_std')]
    cjks = [r.get('cjk_bp',0) for r in results.values() if r.get('cjk_bp')]
    print(f"\n--- Aggregate (mean ± std over {len(results)} seeds) ---")
    if accs: print(f"Eval Acc: {sum(accs)/len(accs):.4f} ± {max(accs)-min(accs):.4f}")
    if mns:  print(f"m/n:      {sum(mns)/len(mns):.4f} ± {max(mns)-min(mns):.4f}")
    if bpss: print(f"bp_std:   {sum(bpss)/len(bpss):.4f} ± {max(bpss)-min(bpss):.4f}")
    if cjks: print(f"cjk_bp:   {sum(cjks)/len(cjks):.4f} ± {max(cjks)-min(cjks):.4f}")

# Save JSON
out = {'results': {k: v for k, v in results.items()}}
with open('checkpoints/a_class_v2_summary.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved → checkpoints/a_class_v2_summary.json")
