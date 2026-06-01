"""Compare 初版 (e1b_full.log) vs V4 (e1_retrain_v4.log) — full type BP + metrics."""
import re
from pathlib import Path

def parse_log(fname):
    txt = Path(fname).read_text(encoding='utf-8')
    pat_t = re.compile(
        r'step=\s*(\d+)\s+\[type_bp\]\s+'
        r'utf8=([\w.]+)\s+ascii=([\w.]+)\s+'
        r'cjk=([\w.]+)\s+op=([\w.]+)\s+digit=([\w.]+)'
    )
    pat2 = re.compile(
        r'step=\s*(\d+)\s+loss=([\d.-]+)\s+recon=[\d.-]+\s+comp=[\d.-]+\s+'
        r'recon_acc=([\d.]+)\s+soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+'
        r'units=([\d.]+)\s+bp_mean=[\d.]+\s+bp_std=([\d.]+)\s+bhead_gnorm=([\d.]+)'
    )
    
    t = {'step':[], 'utf8':[], 'ascii':[], 'cjk':[], 'op':[], 'digit':[]}
    for m in pat_t.finditer(txt):
        t['step'].append(int(m.group(1)))
        t['utf8'].append(float(m.group(2)))
        t['ascii'].append(float(m.group(3)))
        t['cjk'].append(float(m.group(4)))
        t['op'].append(float(m.group(5)))
        t['digit'].append(float(m.group(6)))
    
    mets = {'step':[], 'loss':[], 'acc':[], 'bp_std':[],
            'soft_mn':[], 'hard_mn':[], 'units':[], 'gnorm':[]}
    for m in pat2.finditer(txt):
        mets['step'].append(int(m.group(1)))
        mets['loss'].append(float(m.group(2)))
        mets['acc'].append(float(m.group(3)))
        mets['soft_mn'].append(float(m.group(4)))
        mets['hard_mn'].append(float(m.group(5)))
        mets['units'].append(float(m.group(6)))
        mets['bp_std'].append(float(m.group(7)))
        mets['gnorm'].append(float(m.group(8)))
    return mets, t

m1, t1 = parse_log('checkpoints/e1b_full.log')
m4, t4 = parse_log('checkpoints/e1_retrain_v4.log')

# === Type BP comparison ===
print('=' * 70)
print('TYPE BP: 初版 vs V4')
print('=' * 70)
print(f"{'type':<10s}  {'初版末':>10s}  {'V4末':>10s}  {'初版速率':>10s}  {'V4速率':>10s}")
print('-' * 55)
for k in ['cjk', 'utf8', 'ascii', 'op', 'digit']:
    v1 = t1[k][-1]
    v4 = t4[k][-1]
    r1 = (t1[k][-1] - t1[k][0]) / (t1['step'][-1] - t1['step'][0]) * 10000
    r4 = (t4[k][-1] - t4[k][0]) / (t4['step'][-1] - t4['step'][0]) * 10000
    print(f"{k:<10s}  {v1:>10.3f}  {v4:>10.3f}  {r1:>+10.2f}  {r4:>+10.2f}")

# === Main metrics ===
print()
print('=' * 70)
print('MAIN METRICS: 初版 vs V4')
print('=' * 70)
print(f"{'metric':<14s}  {'初版 (step {})'.format(m1['step'][-1]):>16s}  {'V4 (step {})'.format(m4['step'][-1]):>16s}")
print('-' * 50)
for label, v1k, v4k, fmt in [
    ('recon_acc', 'acc', 'acc', '.4f'),
    ('loss', 'loss', 'loss', '.4f'),
    ('bp_std', 'bp_std', 'bp_std', '.3f'),
    ('soft_m/n', 'soft_mn', 'soft_mn', '.3f'),
    ('hard_m/n', 'hard_mn', 'hard_mn', '.3f'),
    ('units', 'units', 'units', '.1f'),
    ('gnorm', 'gnorm', 'gnorm', '.3f'),
]:
    if len(m1[v1k]) > 0 and len(m4[v4k]) > 0:
        v1str = format(m1[v1k][-1], fmt)
        v4str = format(m4[v4k][-1], fmt)
        print(f"{label:<14s}  {v1str:>16s}  {v4str:>16s}")

# === V4 time series ===
print()
print('=' * 70)
print('V4 TYPE BP TIME SERIES')
print('=' * 70)
print(f"{'step':>6s}  {'cjk':>8s}  {'utf8':>8s}  {'ascii':>8s}  {'op':>8s}  {'digit':>8s}  {'bp_std':>8s}")
print('-' * 55)
n = len(t4['step'])
for i in range(0, n, max(1, n // 15)):
    s = t4['step'][i]
    # find matching bp_std
    bps = '-'
    for j, ms in enumerate(m4['step']):
        if ms >= s:
            bps = format(m4['bp_std'][j], '.3f')
            break
    print(f"{s:>6d}  {t4['cjk'][i]:>8.3f}  {t4['utf8'][i]:>8.3f}  {t4['ascii'][i]:>8.3f}  {t4['op'][i]:>8.3f}  {t4['digit'][i]:>8.3f}  {bps:>8s}")
# Last point
s = t4['step'][-1]
bps = format(m4['bp_std'][-1], '.3f') if len(m4['bp_std']) > 0 else '-'
print(f"{s:>6d}  {t4['cjk'][-1]:>8.3f}  {t4['utf8'][-1]:>8.3f}  {t4['ascii'][-1]:>8.3f}  {t4['op'][-1]:>8.3f}  {t4['digit'][-1]:>8.3f}  {bps:>8s}")

# === Key insights ===
print()
print('=' * 70)
print('KEY INSIGHTS')
print('=' * 70)
print(f"cjk:  V4 rate ({((t4['cjk'][-1]-t4['cjk'][0])/(t4['step'][-1]-t4['step'][0])*10000):+.2f}/10k) is "
      f"{abs((t4['cjk'][-1]-t4['cjk'][0])/(t4['step'][-1]-t4['step'][0])/((t1['cjk'][-1]-t1['cjk'][0])/(t1['step'][-1]-t1['step'][0]))):.1f}x "
      f"faster than 初版 ({((t1['cjk'][-1]-t1['cjk'][0])/(t1['step'][-1]-t1['step'][0])*10000):+.2f}/10k)")
print(f"utf8: 初版={t1['utf8'][-1]:.3f}, V4={t4['utf8'][-1]:.3f} — V4 utf8 is LOWER (better)")
print(f"bp_std: 初版={m1['bp_std'][-1]:.3f}, V4={m4['bp_std'][-1]:.3f} — V4 polarizing fast")
print(f"recon_acc: 初版={m1['acc'][-1]:.4f}, V4={m4['acc'][-1]:.4f} — V4 catching up")
