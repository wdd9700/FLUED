"""Low-capacity readability probe for the v3.5 L0 codec latent.

Trains tiny linear classifiers from mean-pooled frozen chunk readouts to chunk
content labels (punct-heavy / digit-heavy / CJK-heavy / space-heavy), plus a
shuffled-chunk control. Combined with the order probe and the independent
decoder's reconstruction, this separates "decoder is strong" from "the latent
itself is readable".
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.v3_3.train_v33 import make_dataloaders  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import build_model  # noqa: E402

PAD_ID = 0


def _chunk_labels(clean: torch.Tensor, chunk_ids: torch.Tensor, chunk_index: int, max_chunks: int):
    labels = {}
    rows = []
    for b in range(clean.size(0)):
        positions = (chunk_ids[b].eq(chunk_index) & clean[b].ne(PAD_ID)).nonzero(as_tuple=False).flatten()
        if positions.numel() < 4:
            continue
        values = (clean[b, positions] - 1).clamp(min=0, max=255)
        punct = ((values >= 0x21) & (values <= 0x2F)).float().mean().item()
        digit = ((values >= 0x30) & (values <= 0x39)).float().mean().item()
        high = (values >= 0x80).float().mean().item()
        space = ((values == 0x20) | (values == 0x0A)).float().mean().item()
        rows.append((b, punct > 0.15, digit > 0.30, high > 0.30, space > 0.20))
    return rows


@torch.no_grad()
def _collect(model, loader, device, batches: int, shuffle: bool):
    features, labels = [], []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        clean = batch[0].to(device)
        out = model(clean)
        z = out.readout_z
        if shuffle:
            perm = torch.randperm(z.size(1), device=z.device)
            z = z[:, perm]
        pooled = z.mean(dim=2)
        for c in range(z.size(1)):
            if not bool(out.chunks.chunk_mask[:, c].any()):
                continue
            rows = _chunk_labels(clean, out.chunks.chunk_ids, c, z.size(1))
            for b, punct, digit, high, space in rows:
                features.append(pooled[b, c])
                labels.append([punct, digit, high, space])
    if not features:
        return None, None
    stacked = torch.stack(features)
    return stacked, torch.tensor(labels, dtype=torch.float32, device=stacked.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(device=cli.device, num_workers=0)
    args = Namespace(**config)
    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    train_loader, eval_loader = make_dataloaders(args)

    x_train, y_train = _collect(model, train_loader, device, cli.batches, False)
    x_test, y_test = _collect(model, eval_loader, device, 8, False)
    x_shuf, y_shuf = _collect(model, eval_loader, device, 8, True)

    probe = nn.Linear(args.d_model, 4).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    for _ in range(cli.steps):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(probe(x_train), y_train)
        loss.backward()
        optimizer.step()

    def _accuracy(x, y):
        with torch.no_grad():
            pred = probe(x).gt(0)
        return float(pred.eq(y.bool()).float().mean().item())

    majority = float(y_test.float().mean(dim=0).gt(0.5).eq(y_test.bool()).float().mean().item())
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "train_chunks": int(x_train.size(0)),
        "test_chunks": int(x_test.size(0)),
        "final_train_loss": float(loss.item()),
        "linear_probe_chunk_label_acc": _accuracy(x_test, y_test),
        "majority_baseline_acc": majority,
        "shuffled_control_acc": _accuracy(x_shuf, y_shuf),
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
