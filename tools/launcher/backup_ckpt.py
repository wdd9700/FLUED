"""
backup_ckpt.py — Strip optimizer states, save model-only checkpoints for GitHub Release.

Usage:
  python tools/launcher/backup_ckpt.py --ckpt checkpoints/e1_seed42/e1_latest.pt
  python tools/launcher/backup_ckpt.py --ckpt checkpoints/e1_blt_stage1/bytel\ m_latest.pt
  python tools/launcher/backup_ckpt.py --all-latest  # backs up all *_latest.pt
"""
import argparse
import logging
import os
import sys
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backup")

BACKUP_DIR = Path("checkpoints/backup")


def strip_and_save(ckpt_path: str, output_path: str = None):
    """Load checkpoint, keep only model weights + metadata, save to backup dir."""
    src = Path(ckpt_path)
    if not src.exists():
        log.error("Not found: %s", src)
        return None

    log.info("Loading %s (%.1f MB)...", src.name, src.stat().st_size / 1e6)
    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    # Extract what we keep
    stripped = {}
    if "model" in ckpt:
        stripped["model"] = {k: v.clone() for k, v in ckpt["model"].items()}
    if "config" in ckpt:
        stripped["config"] = ckpt["config"]
    stripped["global_step"] = ckpt.get("global_step", 0)
    stripped["_stripped_from"] = str(src)
    stripped["_stripped_keys"] = sorted(k for k in ckpt.keys() if k not in ("model", "config"))

    # Output path — preserve parent dir to avoid name collisions
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        parent = src.parent.name  # e.g. "e1_seed42"
        stem = src.stem
        output_path = BACKUP_DIR / f"{parent}_{stem}_model_only.pt"
    else:
        output_path = Path(output_path)

    # Count sizes
    model_mb = sum(v.numel() * v.element_size() for v in stripped.get("model", {}).values()) / 1e6
    torch.save(stripped, str(output_path))
    disk_mb = output_path.stat().st_size / 1e6

    log.info("Saved %s: model=%.0fMB, disk=%.0fMB (stripped: %s)",
             output_path.name, model_mb, disk_mb, stripped["_stripped_keys"])
    return str(output_path)


def find_all_latest(checkpoint_root: str = "checkpoints"):
    """Find all *_latest.pt files under checkpoint_root, excluding smoke tests."""
    root = Path(checkpoint_root)
    latests = list(root.rglob("*_latest.pt"))
    # Exclude backup dir and smoke test artifacts (e3_* from smoke runs)
    latests = [p for p in latests
               if "backup" not in str(p)
               and "e3_byte" not in p.name
               and "e3_fixed_patch" not in p.name]
    return sorted(latest for latest in latests)


def main():
    parser = argparse.ArgumentParser(description="Strip optimizer states from checkpoints for backup")
    parser.add_argument("--ckpt", default=None, help="Single checkpoint path")
    parser.add_argument("--all-latest", action="store_true", help="Backup all *_latest.pt under checkpoints/")
    parser.add_argument("--output-dir", default=None, help="Override backup output dir")
    args = parser.parse_args()

    if args.output_dir:
        global BACKUP_DIR
        BACKUP_DIR = Path(args.output_dir)

    if args.all_latest:
        latests = find_all_latest()
        log.info("Found %d latest checkpoints", len(latests))
        for p in latests:
            strip_and_save(str(p))
    elif args.ckpt:
        strip_and_save(args.ckpt)
    else:
        parser.print_help()

    log.info("Backup dir: %s", BACKUP_DIR.absolute())


if __name__ == "__main__":
    main()
