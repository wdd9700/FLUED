"""extract checkpoint metrics to fill the 9000-24250 gap."""
import torch, sys
from pathlib import Path
from flued.model import FLUEDAutoencoder

CKPT_DIR = Path("checkpoints")
STEPS = list(range(10000, 25000, 1000))
DEVICE = "cuda"

model = FLUEDAutoencoder(
    d_model=1024, nhead=16, dim_feedforward=4096, num_layers=24,
    max_seq_len=512,
    lambda_var=0.5, lambda_entropy=0.05, lambda_utf8=0.02, lambda_type=0.05,
).to(DEVICE)

src = torch.randint(1, 256, (2, 512), device=DEVICE, dtype=torch.long)
src[:, -64:] = 0
pad = (src == 0)

for step in STEPS:
    ckpt_path = CKPT_DIR / f"e1_step{step:05d}.pt"
    if not ckpt_path.exists():
        continue
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, metrics = model(src, pad)
    bp = metrics["boundary_probs"].detach()
    sweep = metrics.get("hard_mn_sweep", {})
    acc = (logits.argmax(-1) == src).float().masked_fill(pad, 0).sum() / (~pad).sum()
    hmn = float(metrics["hard_m_over_n"])
    smn = float(metrics["soft_m_over_n"])
    bpm = bp.mean().item()
    bps = bp.std().item()
    h55 = sweep.get(0.55, 0.0)
    h60 = sweep.get(0.60, 0.0)
    h65 = sweep.get(0.65, 0.0)
    u = metrics.get("utf8_cont_bp_mean", "nan")
    a = metrics.get("ascii_bp_mean", "nan")
    c = metrics.get("cjk_bp_mean", "nan")
    o = metrics.get("op_bp_mean", "nan")
    d = metrics.get("digit_bp_mean", "nan")
    print(f"CKPT|step={step:5d}  hard_mn={hmn:.4f}  soft_mn={smn:.4f}  "
          f"bp_mean={bpm:.4f}  bp_std={bps:.4f}  recon_acc={acc.item():.4f}  "
          f"h@55={h55:.4f}  h@60={h60:.4f}  h@65={h65:.4f}  "
          f"utf8={u}  ascii={a}  cjk={c}  op={o}  digit={d}")
    sys.stdout.flush()
