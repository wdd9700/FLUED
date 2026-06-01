"""
fill_gap.py — Extract E1-b metrics from checkpoint files to fill the 9000→24250 gap.

Usage: python fill_gap.py
Output: checkpoints/e1b_ckpt_metrics.txt (step-level metrics)
"""
import torch
from pathlib import Path

from flued.model import FLUEDAutoencoder

CKPT_DIR = Path("checkpoints")
DEVICE = "cuda"

# Steps with missing data
MISSING_STEPS = list(range(10000, 25000, 1000))

# Small eval batch for fast metric extraction
SRC = torch.randint(1, 256, (2, 512), device=DEVICE, dtype=torch.long)
SRC[:, -64:] = 0  # pad tail
PAD = (SRC == 0)


def extract_metrics(ckpt_path: Path, step: int) -> dict | None:
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        return None

    model = FLUEDAutoencoder(
        d_model=1024, nhead=16, dim_feedforward=4096, num_layers=24,
        max_seq_len=512,
        lambda_var=0.5, lambda_entropy=0.05, lambda_utf8=0.02, lambda_type=0.05,
    ).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, metrics = model(SRC, PAD)

    recon_acc = (logits.argmax(-1) == SRC).float().masked_fill(PAD, 0).sum() / (~PAD).sum()

    bp = metrics["boundary_probs"].detach()
    sweep = metrics.get("hard_mn_sweep", {})
    type_vals = {}
    for key, label in (
        ("utf8_cont", "utf8"), ("ascii", "ascii"), ("cjk", "cjk"),
        ("op", "op"), ("digit", "digit"),
    ):
        type_vals[label] = metrics.get(f"{key}_bp_mean", float("nan"))

    return {
        "step": step,
        "recon_acc": recon_acc.item(),
        "hard_mn": float(metrics["hard_m_over_n"]),
        "soft_mn": float(metrics["soft_m_over_n"]),
        "bp_mean": bp.mean().item(),
        "bp_std": bp.std().item(),
        "h55": sweep.get(0.55, 0.0),
        "h60": sweep.get(0.60, 0.0),
        "h65": sweep.get(0.65, 0.0),
        "utf8": type_vals.get("utf8", float("nan")),
        "ascii": type_vals.get("ascii", float("nan")),
        "cjk": type_vals.get("cjk", float("nan")),
        "op": type_vals.get("op", float("nan")),
        "digit": type_vals.get("digit", float("nan")),
        "recon_loss": metrics.get("compression_loss", torch.tensor(0.0)).item(),
    }


def main() -> None:
    results = []
    for step in MISSING_STEPS:
        ckpt = CKPT_DIR / f"e1_step{step:05d}.pt"
        if not ckpt.exists():
            print(f"  skip {step}: no file")
            continue
        m = extract_metrics(ckpt, step)
        if m is None:
            continue
        results.append(m)
        print(f"  step={step}  hard_m/n={m['hard_mn']:.3f}  bp_std={m['bp_std']:.3f}  "
              f"cjk={m['cjk']:.3f}  recon_acc={m['recon_acc']:.4f}")

    # Write in log-compatible format for easy merging
    out = CKPT_DIR / "e1b_ckpt_metrics.txt"
    with open(out, "w") as f:
        for r in results:
            f.write(
                f"step={r['step']:5d}  loss=N/A  recon={r['recon_loss']:.4f}  comp=N/A  "
                f"recon_acc={r['recon_acc']:.4f}  "
                f"soft_m/n={r['soft_mn']:.3f}  hard_m/n={r['hard_mn']:.3f}  units=N/A  "
                f"bp_mean={r['bp_mean']:.3f}  bp_std={r['bp_std']:.3f}  "
                f"bhead_gnorm=N/A  lr=N/A  skip=0  scale=0  "
                f"h@55={r['h55']:.3f}  h@60={r['h60']:.3f}  h@65={r['h65']:.3f}\n"
            )
            f.write(
                f"step={r['step']:5d}  [type_bp]  utf8={r['utf8']:<7}  "
                f"ascii={r['ascii']:<7}  cjk={r['cjk']:<7}  "
                f"op={r['op']:<7}  digit={r['digit']:<7}\n"
            )
    print(f"Saved {len(results)} checkpoints → {out}")


if __name__ == "__main__":
    main()
