"""
e1_paper_pipeline.py — FLUED E1 Paper Data Pipeline

merge → resample → smooth (S-G filter) → plot
No bridging, no fabrication. Raw data preserved for audit.
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

INTERVAL=200
METRICS=['loss','recon_acc','bp_std','soft_mn','bp_mean','cjk','op','digit','ascii','utf8','bhead_gnorm']
BG="#1e1e2e";GRID="#313244";TEXT="#cdd6f4"
PINK="#f38ba8";GREEN="#a6e3a1";PURPLE="#cba6f7";CYAN="#89dceb";ORANGE="#fab387";YELLOW="#f9e2af"

RE_MAIN=re.compile(r"step=\s*(\d+)\s+loss=([\d.-]+)\s+recon=([\d.-]+)\s+comp=([\d.-]+)\s+recon_acc=([\d.]+).*?soft_m/n=([\d.]+).*?bp_mean=([\d.]+)\s+bp_std=([\d.]+)(?:.*?bhead_gnorm=([\d.]+))?")
RE_TYPE=re.compile(r"step=\s*(\d+)\s+\[type_bp\]\s+utf8=([\d.]+)\s+ascii=([\d.]+)\s+cjk=([\d.]+)\s+op=([\d.]+)\s+digit=([\d.]+)")

def merge(log_dir):
    steps={}
    for lf in sorted(Path(log_dir).glob("e1_v5*.log")):
        for line in lf.read_text(encoding="utf-8",errors="ignore").split("\n"):
            m=RE_MAIN.search(line)
            if m:
                s=int(m.group(1)); steps.setdefault(s,{})["step"]=s; d=steps[s]
                d["loss"]=float(m.group(2)); d["recon_loss"]=float(m.group(3))
                d["comp_loss"]=float(m.group(4)); d["recon_acc"]=float(m.group(5))
                d["soft_mn"]=float(m.group(6)); d["bp_mean"]=float(m.group(7)); d["bp_std"]=float(m.group(8))
                gn=m.group(9)
                if gn and gn!="None": d["bhead_gnorm"]=float(gn)
                continue
            m=RE_TYPE.search(line)
            if m:
                s=int(m.group(1)); steps.setdefault(s,{"step":s})
                steps[s].update({"utf8":float(m.group(2)),"ascii":float(m.group(3)),"cjk":float(m.group(4)),"op":float(m.group(5)),"digit":float(m.group(6))})
    df=pd.DataFrame([steps[s] for s in sorted(steps)])
    print(f"Merged: {len(df)} steps")
    return df

def resample(df):
    lo=(int(df['step'].min())//INTERVAL+1)*INTERVAL; hi=(int(df['step'].max())//INTERVAL)*INTERVAL
    xs=np.arange(lo,hi+INTERVAL,INTERVAL,dtype=float); r=pd.DataFrame({'step':xs})
    for c in METRICS:
        if c not in df.columns: continue
        v=df[df[c].notna()][['step',c]].dropna()
        if len(v)<2: continue
        try: r[c]=PchipInterpolator(v['step'].values,v[c].values)(xs)
        except: r[c]=np.interp(xs,v['step'].values,v[c].values)
    print(f"Resampled: {len(r)} @{INTERVAL}-step")
    return r

def smooth(df):
    df=df.copy(); df['is_smoothed']=False
    for lo,hi,w in [(0,5000,15),(5000,27000,11),(27000,50000,9)]:
        mask=(df['step']>=lo)&(df['step']<=hi)
        for c in METRICS:
            if c not in df.columns: continue
            v=mask&df[c].notna()
            if v.sum()<w: continue
            try: df.loc[v,c]=savgol_filter(df.loc[v,c].values,w,2); df.loc[v,'is_smoothed']=True
            except: pass
    print("Smoothed: S-G w=15/11/9")
    return df

def plot(df,out):
    plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"text.color":TEXT,"axes.labelcolor":TEXT,"axes.edgecolor":GRID,"xtick.color":TEXT,"ytick.color":TEXT,"grid.color":GRID,"figure.dpi":120,"savefig.dpi":200,"savefig.facecolor":BG,"savefig.bbox":"tight"})
    xs=df['step']
    # Core
    fig,axes=plt.subplots(2,2,figsize=(18,10))
    fig.suptitle("FLUED E1 — Training Curves",fontsize=14,color=TEXT,fontweight='bold')
    for ax,(col,color,yl) in zip(axes.flat,[('loss',PINK,"Total Loss"),('recon_acc',CYAN,"Reconstruction Accuracy"),('bp_std',PURPLE,"Boundary Prob Std"),('soft_mn',ORANGE,"Compression Ratio m/n")]):
        ax.plot(xs,df[col],color=color,linewidth=1.2); ax.set_ylabel(yl,color=color,fontsize=11); ax.grid(True,alpha=0.2); ax.ticklabel_format(style="plain",axis="x"); ax.tick_params(labelsize=9)
    axes[0,1].axhline(1.0,color=GREEN,linestyle="--",linewidth=0.8,alpha=0.4)
    fig.tight_layout(); fig.savefig(out/"v5_paper_core.png"); print(">>> v5_paper_core.png")
    # Type BP
    fig,ax=plt.subplots(figsize=(16,6))
    fig.suptitle("FLUED E1 — Per-Type Boundary Probabilities",fontsize=14,color=TEXT,fontweight='bold')
    for col,color,label in [('op',PURPLE,"Operator"),('digit',YELLOW,"Digit"),('ascii',GREEN,"ASCII"),('utf8',PINK,"UTF-8 Cont"),('cjk',CYAN,"CJK Lead")]:
        ax.plot(xs,df[col],color=color,linewidth=1.0,label=label)
    ax.axhline(0.15,color=ORANGE,linestyle="--",linewidth=0.8,alpha=0.5)
    ax.set_xlabel("Training Step",fontsize=10); ax.set_ylabel("Boundary Probability",fontsize=11); ax.legend(loc="upper right",ncol=3,fontsize=9); ax.grid(True,alpha=0.2); ax.ticklabel_format(style="plain",axis="x"); ax.tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(out/"v5_paper_typebp.png"); print(">>> v5_paper_typebp.png")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--merge-only",action="store_true"); ap.add_argument("--plot-only",action="store_true"); ap.add_argument("--log-dir",default="checkpoints"); ap.add_argument("--out-dir",default="checkpoints"); args=ap.parse_args()
    log_dir=Path(args.log_dir); out_dir=Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    raw_csv=out_dir/"e1_v5e_merged.csv"
    if not args.plot_only:
        df=merge(log_dir); df.to_csv(raw_csv,index=False); print(f"Raw → {raw_csv}\n")
    else:
        df=pd.read_csv(raw_csv); print(f"Loaded {len(df)} steps\n")
    if args.merge_only: return
    df=resample(df); df=smooth(df)
    df.to_csv(out_dir/"e1_v5_paper.csv",index=False); print(f"Paper→{out_dir/'e1_v5_paper.csv'}")
    recs=[{k:(float(v) if not isinstance(v,(bool,np.bool_)) and not pd.isna(v) else (bool(v) if isinstance(v,(bool,np.bool_)) else None)) for k,v in row.items()} for _,row in df.iterrows()]
    meta={"description":"FLUED E1 paper data","pipeline":"merge→resample(200-step)→smooth(S-G 15/11/9)","smoothing_method":"Savitzky-Golay (standard low-pass filter)","note":"Raw unsmoothed data in e1_v5e_merged.csv for audit","records":recs}
    with open(out_dir/"e1_v5_paper.json","w") as f: json.dump(meta,f,indent=2)
    plot(df,out_dir)
    print("\n=== Verify ===")
    for s in [200,5000,10000,15000,20000,27000,35000,40000,45000,50000]:
        idx=(df['step']-s).abs().idxmin(); r=df.iloc[idx]; print(f"step={r['step']:5.0f} bp_std={r.get('bp_std',0):.3f} cjk={r.get('cjk',0):.3f} op={r.get('op',0):.3f} m/n={r.get('soft_mn',0):.3f} acc={r.get('recon_acc',0):.4f}")
    print("\nDone.")

if __name__=="__main__": main()
