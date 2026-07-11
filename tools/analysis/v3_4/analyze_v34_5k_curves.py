"""Analyze FLUED v3.4 curves without endpoint-only bias."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


DEFAULT_SNAPSHOTS = (500, 1000, 2500, 5000)
TASK_METRICS = (
    "loss",
    "identity_acc",
    "completion_mask_acc",
    "completion_preserve_acc",
    "completion_masked_loss",
)
COMPUTE_METRICS = (
    "soft_readout_units_per_byte",
    "actual_backbone_units_per_byte",
    "backbone_padded_units_per_byte",
)

GROUPS = {
    "position_ar": ["full", "pos_off_ar_off", "pos_on_ar_off", "pos_off_ar_on"],
    "components": [
        "full",
        "no_memory",
        "no_logic_prior",
        "no_boundary_bridge",
        "no_boundary_prior",
        "no_memory_usage_constraint",
        "codec_only_no_backbone_loss",
        "no_diffusion_noise",
        "plain_byte_lookup",
    ],
    "rate_emit": [
        "full",
        "uniform_boundaries",
        "no_emit_value",
        "no_compute_cost",
        "soft_emit_no_compaction",
        "l2_coding_rate",
    ],
    "long_horizon": [
        "exact_marginal_rate",
        "l2_marginal_rate",
        "uniform_boundaries",
        "uniform_to_l2_curriculum",
    ],
}


def load_rows(path: Path) -> list[dict]:
    by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def nearest(rows: list[dict], completed_step: int) -> dict:
    target_index = completed_step - 1
    return min(rows, key=lambda row: abs(int(row["step"]) - target_index))


def trapezoid_auc(rows: list[dict], metric: str) -> float:
    if len(rows) < 2:
        return float("nan")
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        width = float(right["step"] - left["step"])
        area += width * 0.5 * (float(left[metric]) + float(right[metric]))
    span = max(float(rows[-1]["step"] - rows[0]["step"]), 1.0)
    return area / span


def linear_slope(rows: list[dict], metric: str, tail_points: int = 25) -> float:
    tail = rows[-tail_points:]
    if len(tail) < 2:
        return float("nan")
    xs = [float(row["step"]) for row in tail]
    ys = [float(row[metric]) for row in tail]
    x_mean, y_mean = mean(xs), mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / max(denom, 1.0)


def summarize_run(run: str, rows: list[dict], summary: dict, snapshots: tuple[int, ...]) -> dict:
    result = {"run": run, "logged_points": len(rows), "final_step": int(summary.get("steps", 0))}
    for metric in TASK_METRICS + COMPUTE_METRICS:
        values = [float(row[metric]) for row in rows]
        maximize = metric.endswith("_acc")
        best = max(values) if maximize else min(values)
        best_index = values.index(best)
        result[f"{metric}_auc"] = trapezoid_auc(rows, metric)
        result[f"{metric}_best"] = best
        result[f"{metric}_best_step"] = int(rows[best_index]["step"])
        result[f"{metric}_tail_mean"] = mean(values[-10:])
        result[f"{metric}_tail_slope_per_1k"] = linear_slope(rows, metric) * 1000.0
        for snapshot in snapshots:
            result[f"{metric}_at_{snapshot}"] = float(nearest(rows, snapshot)[metric])
    result["eval_identity_acc"] = float(summary.get("eval_identity_acc", float("nan")))
    result["eval_completion_mask_acc"] = float(summary.get("eval_completion_mask_acc", float("nan")))
    result["eval_actual_backbone_units_per_byte"] = float(
        summary.get("eval_actual_backbone_units_per_byte", float("nan"))
    )
    result["eval_truncated_tokens"] = float(summary.get("eval_truncated_tokens", float("nan")))
    return result


def write_table(rows: list[dict], out_dir: Path, snapshots: tuple[int, ...]) -> None:
    fields = [
        "run",
        "final_step",
        *[f"identity_acc_at_{step}" for step in snapshots],
        "identity_acc_auc",
        "identity_acc_tail_slope_per_1k",
        *[f"completion_mask_acc_at_{step}" for step in snapshots],
        "completion_mask_acc_auc",
        "completion_mask_acc_tail_slope_per_1k",
        f"actual_backbone_units_per_byte_at_{snapshots[-1]}",
        f"backbone_padded_units_per_byte_at_{snapshots[-1]}",
        "eval_identity_acc",
        "eval_completion_mask_acc",
        "eval_actual_backbone_units_per_byte",
        "eval_truncated_tokens",
    ]
    with (out_dir / "curve_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (out_dir / "curve_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_groups(all_rows: dict[str, list[dict]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for group, names in GROUPS.items():
        present = [name for name in names if name in all_rows]
        if not present:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        specs = (
            ("identity_acc", "Identity accuracy"),
            ("completion_mask_acc", "Masked completion accuracy"),
            ("actual_backbone_units_per_byte", "Actual latent / byte"),
            ("loss", "Total loss"),
        )
        for axis, (metric, title) in zip(axes.flat, specs):
            for name in present:
                rows = all_rows[name]
                axis.plot([row["step"] for row in rows], [row[metric] for row in rows], label=name, linewidth=1.5)
            axis.set_title(title)
            axis.set_xlabel("step")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.savefig(out_dir / f"curves_{group}.png", dpi=160)
        plt.close(fig)


def write_interactive_html(all_rows: dict[str, list[dict]], out_dir: Path) -> None:
    max_step = max(int(row["step"]) for rows in all_rows.values() for row in rows)
    payload = {
        "groups": GROUPS,
        "max_step": max_step,
        "runs": {
            name: [
                {
                    "step": row["step"],
                    **{metric: row[metric] for metric in TASK_METRICS + COMPUTE_METRICS},
                }
                for row in rows
            ]
            for name, rows in all_rows.items()
        },
    }
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLUED v3.4 Training Curves</title><style>
body{margin:0;background:#0b0d10;color:#eef1f4;font:14px system-ui,sans-serif;letter-spacing:0}main{max-width:1500px;margin:auto;padding:24px}
h1{font-size:26px;margin:0 0 18px}.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin-bottom:18px}label{display:grid;gap:6px;color:#aab2bd}
select,input{background:#151920;color:#eef1f4;border:1px solid #343b45;padding:8px}button{background:none;color:#cbd2da;border:0;padding:3px 8px;cursor:pointer}
#plot{width:100%;height:680px;border:1px solid #2b3139;background:#101319}.axis{stroke:#59616c;stroke-width:1}.grid{stroke:#252b33;stroke-width:1}.tick{fill:#9da6b1;font-size:12px}
#legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.muted{opacity:.25}.tip{position:fixed;display:none;background:#050607;border:1px solid #59616c;padding:8px;pointer-events:none;white-space:pre}
</style></head><body><main><h1>FLUED v3.4 · 训练曲线</h1><div class="controls">
<label>实验组<select id="group"></select></label><label>指标<select id="metric"></select></label>
<label>平滑点数 <span id="smoothLabel">5</span><input id="smooth" type="range" min="1" max="20" value="5"></label></div>
<svg id="plot" viewBox="0 0 1400 680" preserveAspectRatio="none"></svg><div id="legend"></div><div id="tip" class="tip"></div>
</main><script>const D=__DATA__;const colors=['#42d3a5','#ffcc66','#7aa2ff','#ff7b8b','#c792ea','#82d2ff','#f78c6c','#c3e88d','#89ddff'];
const group=document.querySelector('#group'),metric=document.querySelector('#metric'),smooth=document.querySelector('#smooth'),svg=document.querySelector('#plot'),legend=document.querySelector('#legend'),tip=document.querySelector('#tip');
Object.keys(D.groups).forEach(x=>group.add(new Option(x,x)));['identity_acc','completion_mask_acc','completion_preserve_acc','loss','completion_masked_loss','soft_readout_units_per_byte','actual_backbone_units_per_byte','backbone_padded_units_per_byte'].forEach(x=>metric.add(new Option(x,x)));
const hidden=new Set();function avg(a,n,k){return a.map((r,i)=>{let s=0,c=0;for(let j=Math.max(0,i-n+1);j<=i;j++){s+=+a[j][k];c++}return {step:+r.step,v:s/c}})}
function draw(){const names=D.groups[group.value].filter(n=>D.runs[n]);const k=metric.value,n=+smooth.value,maxStep=D.max_step;document.querySelector('#smoothLabel').textContent=`${n} (${n*20} steps)`;const series=names.map(name=>({name,data:avg(D.runs[name],n,k)}));const vals=series.flatMap(s=>s.data.map(x=>x.v)).filter(Number.isFinite),min=Math.min(...vals),max=Math.max(...vals),pad=(max-min||1)*.08;const y0=min-pad,y1=max+pad,W=1400,H=680,L=90,R=25,T=25,B=55;const X=x=>L+x/maxStep*(W-L-R),Y=y=>T+(y1-y)/(y1-y0)*(H-T-B);let h='';for(let i=0;i<=10;i++){let x=L+i/10*(W-L-R),v=Math.round(i/10*maxStep);h+=`<line class="grid" x1="${x}" y1="${T}" x2="${x}" y2="${H-B}"/><text class="tick" x="${x}" y="${H-24}" text-anchor="middle">${v}</text>`}for(let i=0;i<=8;i++){let y=T+i/8*(H-T-B),v=y1-i/8*(y1-y0);h+=`<line class="grid" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/><text class="tick" x="${L-10}" y="${y+4}" text-anchor="end">${v.toFixed(3)}</text>`}
series.forEach((s,i)=>{const pts=s.data.map(p=>`${X(p.step)},${Y(p.v)}`).join(' ');h+=`<polyline data-name="${s.name}" class="${hidden.has(s.name)?'muted':''}" points="${pts}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="2"/>`});h+=`<rect id="hit" x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/>`;svg.innerHTML=h;legend.innerHTML='';series.forEach((s,i)=>{const b=document.createElement('button');b.textContent=s.name;b.style.borderBottom=`3px solid ${colors[i%colors.length]}`;if(hidden.has(s.name))b.className='muted';b.onclick=()=>{hidden.has(s.name)?hidden.delete(s.name):hidden.add(s.name);draw()};legend.appendChild(b)});document.querySelector('#hit').onmousemove=e=>{const rect=svg.getBoundingClientRect(),sx=(e.clientX-rect.left)/rect.width*W,step=Math.max(0,Math.min(maxStep,(sx-L)/(W-L-R)*maxStep)),lines=[`step ≈ ${Math.round(step)}`];series.filter(s=>!hidden.has(s.name)).forEach(s=>{const p=s.data.reduce((a,b)=>Math.abs(b.step-step)<Math.abs(a.step-step)?b:a);lines.push(`${s.name}: ${p.v.toFixed(4)}`)});tip.textContent=lines.join('\n');tip.style.display='block';tip.style.left=e.clientX+14+'px';tip.style.top=e.clientY+14+'px'};document.querySelector('#hit').onmouseleave=()=>tip.style.display='none'}
[group,metric,smooth].forEach(x=>x.oninput=draw);draw();</script></body></html>'''
    (out_dir / "curves_interactive.html").write_text(
        template.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, loaded = {}, []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        log_path, summary_path = run_dir / "train_log.jsonl", run_dir / "summary.json"
        if not log_path.exists() or not summary_path.exists():
            continue
        rows = load_rows(log_path)
        all_rows[run_dir.name] = rows
        loaded.append((run_dir.name, rows, json.loads(summary_path.read_text(encoding="utf-8"))))
    max_final_step = max((int(summary.get("steps", 0)) for _, _, summary in loaded), default=5000)
    snapshots = DEFAULT_SNAPSHOTS
    if max_final_step > 5000:
        snapshots = tuple(step for step in (500, 1000, 2500, 5000, 10000, 15000, 20000) if step <= max_final_step)
    summaries = [summarize_run(name, rows, summary, snapshots) for name, rows, summary in loaded]
    (out_dir / "curve_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_table(summaries, out_dir, snapshots)
    plot_groups(all_rows, out_dir)
    write_interactive_html(all_rows, out_dir)
    print(f"analyzed {len(summaries)} completed runs -> {out_dir}")


if __name__ == "__main__":
    main()
