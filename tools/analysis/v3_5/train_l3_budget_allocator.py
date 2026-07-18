"""Train and evaluate the v3.5 L3 offline budget allocator on the L2 dataset.

Two models are compared on a sample-disjoint split:
- class-mean lookup: utility averaged by (slot, punct bucket, entropy bucket,
  position bucket) on the training split;
- a small MLP on chunk content features.

Reports instance- and class-level calibration (Spearman, AUC, ECE, sign),
plus a greedy-budget oracle-regret simulation. Engineering gate (v3.5 5.4):
class Spearman >= 0.40, AUC >= 0.70, ECE <= 0.10.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

FEATURE_KEYS = (
    "byte_len",
    "letter_ratio",
    "digit_ratio",
    "space_ratio",
    "punct_ratio",
    "high_byte_ratio",
    "unigram_entropy",
    "byte_anchor",
)


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _rank(a.double()), _rank(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb / denom.clamp(min=1e-12)).item())


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    greater = (pos[:, None] > neg[None, :]).float().mean()
    equal = (pos[:, None] == neg[None, :]).float().mean()
    return float((greater + 0.5 * equal).item())


def _ece(prob: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = (prob >= lo) & (prob < hi if i < bins - 1 else prob <= hi)
        if sel.any():
            ece += float(sel.float().mean().item()) * abs(
                float(prob[sel].mean().item()) - float(labels[sel].float().mean().item())
            )
    return ece


def _bucket(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _class_key(record: dict) -> tuple:
    f = record["features"]
    return (
        record["slot"],
        _bucket(f["punct_ratio"], [0.05, 0.15, 0.30]),
        _bucket(f["unigram_entropy"], [3.0, 4.0, 5.0, 6.0]),
        _bucket(f["byte_anchor"], [128.0, 256.0, 384.0]),
    )


def _features(record: dict) -> list[float]:
    f = record["features"]
    row = [float(f[k]) for k in FEATURE_KEYS]
    row[0] = row[0] / 16.0
    row[7] = row[7] / 512.0
    row.append(record["slot"] / 15.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    cli = parser.parse_args()

    records = []
    with Path(cli.dataset).open("r", encoding="utf-8") as source:
        for line in source:
            records.append(json.loads(line))
    train = [r for r in records if r["batch_index"] < 24]
    test = [r for r in records if r["batch_index"] >= 24]

    x_train = torch.tensor([_features(r) for r in train], dtype=torch.float32)
    y_train = torch.tensor([r["net_utility"] for r in train], dtype=torch.float32)
    x_test = torch.tensor([_features(r) for r in test], dtype=torch.float32)
    y_test = torch.tensor([r["net_utility"] for r in test], dtype=torch.float32)

    mean, std = x_train.mean(dim=0), x_train.std(dim=0).clamp(min=1e-6)
    y_mean, y_std = y_train.mean(), y_train.std().clamp(min=1e-6)

    class_sums: dict[tuple, list[float]] = defaultdict(list)
    for r in train:
        class_sums[_class_key(r)].append(r["net_utility"])
    class_mean = {key: sum(values) / len(values) for key, values in class_sums.items()}
    global_mean = float(y_train.mean().item())
    pred_class = torch.tensor(
        [class_mean.get(_class_key(r), global_mean) for r in test], dtype=torch.float32
    )

    model = nn.Sequential(
        nn.Linear(x_train.size(1), cli.hidden), nn.SiLU(), nn.Linear(cli.hidden, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    zx, zy = (x_train - mean) / std, (y_train - y_mean) / y_std
    for epoch in range(cli.epochs):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(zx).squeeze(-1), zy)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_mlp = model((x_test - mean) / std).squeeze(-1) * y_std + y_mean

    def _calibration(pred: torch.Tensor) -> dict:
        labels = y_test.gt(0)
        zpred = (pred - pred.mean()) / pred.std().clamp(min=1e-9)
        prob = torch.sigmoid(zpred)
        top_k = max(1, pred.numel() // 4)
        predicted_top = set(pred.topk(top_k).indices.tolist())
        utility_top = set(y_test.topk(top_k).indices.tolist())
        return {
            "spearman": _spearman(pred, y_test),
            "auc": _auc(pred, labels),
            "ece_10bin": _ece(prob, labels),
            "sign_at_mean": float(pred.gt(global_mean).eq(labels).float().mean().item()),
            "top_quartile_overlap": len(predicted_top & utility_top) / top_k,
        }

    per_class_pred: dict[tuple, list[float]] = defaultdict(list)
    per_class_true: dict[tuple, list[float]] = defaultdict(list)
    for i, r in enumerate(test):
        key = _class_key(r)
        per_class_pred[key].append(float(pred_mlp[i].item()))
        per_class_true[key].append(float(y_test[i].item()))
    class_keys = sorted(per_class_true)
    class_pred_mean = torch.tensor(
        [sum(per_class_pred[k]) / len(per_class_pred[k]) for k in class_keys]
    )
    class_true_mean = torch.tensor(
        [sum(per_class_true[k]) / len(per_class_true[k]) for k in class_keys]
    )
    class_labels = class_true_mean.gt(0)
    class_level = {
        "classes": len(class_keys),
        "spearman": _spearman(class_pred_mean, class_true_mean),
        "auc": _auc(class_pred_mean, class_labels),
        "min_class_support": min(len(per_class_true[k]) for k in class_keys),
        "mean_class_support": sum(len(per_class_true[k]) for k in class_keys) / len(class_keys),
    }

    regret_rows = []
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(test):
        grouped[(r["draw"], r["batch_index"], r["sample_in_batch"], r["chunk_index"])].append(i)
    for key, indices in grouped.items():
        if len(indices) < 2:
            continue
        true_u = y_test[indices]
        pred_u = pred_mlp[indices]
        budget = max(1, len(indices) // 4)
        oracle = true_u.topk(budget).values.sum()
        chosen = true_u[pred_u.topk(budget).indices].sum()
        regret_rows.append(float((oracle - chosen).item()))
    deployment = {
        "groups": len(regret_rows),
        "oracle_regret_mean": sum(regret_rows) / max(len(regret_rows), 1),
        "oracle_regret_p90": (
            sorted(regret_rows)[int(0.9 * len(regret_rows))] if regret_rows else None
        ),
    }

    report = {
        "dataset": str(Path(cli.dataset).resolve()),
        "records": len(records),
        "train_records": len(train),
        "test_records": len(test),
        "mlp_final_train_loss": float(loss.item()),
        "class_mean_lookup": _calibration(pred_class),
        "mlp": _calibration(pred_mlp),
        "class_level_mlp": class_level,
        "deployment_simulation": deployment,
        "gate": {
            "class_spearman>=0.40": class_level["spearman"] >= 0.40,
            "class_auc>=0.70": class_level["auc"] >= 0.70,
            "instance_ece<=0.10": _calibration(pred_mlp)["ece_10bin"] <= 0.10,
        },
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
