"""Extract V4 training metrics from conversation transcript and build log file."""
import re
import json
import os

TRANSCRIPT = r"c:\Users\74090\AppData\Roaming\Code\User\workspaceStorage\0a66816d77c54ab5fd1a4e104dbe039d\GitHub.copilot-chat\transcripts\5ff44b84-7d2f-49c2-a545-b3c5aca1d30f.jsonl"
OUTPUT = "checkpoints/e1_retrain_v4.log"

with open(TRANSCRIPT, "r", encoding="utf-8") as f:
    content = f.read()

# Extract step metrics
step_pat = r"step=\s*(\d+)\s+loss=([-\d.]+)\s+recon=([-\d.]+)\s+comp=([-\d.]+)\s+recon_acc=([\d.]+)\s+soft_m/n=([\d.]+)\s+hard_m/n=([\d.]+)\s+units=([\d.]+)\s+bp_mean=([\d.]+)\s+bp_std=([\d.]+)"
type_pat = r"step=\s*(\d+)\s+\[type_bp\]\s+utf8=([\d.]+)\s+ascii=([\d.]+)\s+cjk=([\d.]+)\s+op=([\d.]+)\s+digit=([\d.]+)"

step_metrics = re.findall(step_pat, content)
type_metrics = re.findall(type_pat, content)
print(f"Found {len(step_metrics)} step lines, {len(type_metrics)} type_bp lines")

# Merge
steps = {}
for m in step_metrics:
    s = int(m[0])
    steps[s] = dict(step=s, loss=float(m[1]), recon=float(m[2]), comp=float(m[3]),
                    recon_acc=float(m[4]), soft_mn=float(m[5]), hard_mn=float(m[6]),
                    units=float(m[7]), bp_mean=float(m[8]), bp_std=float(m[9]))
for m in type_metrics:
    s = int(m[0])
    if s in steps:
        steps[s].update(utf8=float(m[1]), ascii=float(m[2]), cjk=float(m[3]),
                        op=float(m[4]), digit=float(m[5]))

unique = sorted(steps.values(), key=lambda x: x["step"])
seen = set()
deduped = [s for s in unique if not (s["step"] in seen or seen.add(s["step"]))]
print(f"Unique steps: {len(deduped)}  range: {deduped[0]['step']} - {deduped[-1]['step']}")

# Write log (full format matching plot_e1.py expectations)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as out:
    for s in deduped:
        line = (
            f"step={s['step']:5d}  loss={s.get('loss',0):.4f}"
            f"  recon={s.get('recon',0):.4f}  comp={s.get('comp',0):.4f}"
            f"  recon_acc={s['recon_acc']:.4f}"
            f"  bp_mean={s['bp_mean']:.3f}  bp_std={s['bp_std']:.3f}"
        )
        out.write(line + "\n")
        # Type line
        if "cjk" in s:
            tline = (
                f"step={s['step']:5d}  [type_bp]"
                f"  utf8={s['utf8']:.3f}  ascii={s['ascii']:.3f}"
                f"  cjk={s['cjk']:.3f}  op={s['op']:.3f}  digit={s['digit']:.3f}"
            )
            out.write(tline + "\n")

print(f"Saved {OUTPUT} ({len(deduped)} lines)")
