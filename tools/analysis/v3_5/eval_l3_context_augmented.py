"""Evaluate context-augmented factor models on the v3.5 L2 utility dataset.

Question (2026-07-24, user hypothesis): does cross-chunk context carry utility
signal that local content factors miss? Baseline = additive factor model on
(slot, punct bucket, entropy bucket, anchor bucket); augmented adds bucketed
context features (sim_prev1, max_sim_prev, sim_window_mean, readout_norm).

Same sample-disjoint split and calibration metrics as train_l3_budget_allocator.py.
Additive models are fit as one-hot ridge regression on the train split; bucket
edges for context features come from train-split quantiles only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

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
CONTEXT_KEYS = ("sim_prev1", "max_sim_prev", "sim_window_mean", "readout_norm")


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


def _quantile_edges(values: list[float], n_buckets: int = 4) -> list[float]:
    s = sorted(values)
    edges = []
    for i in range(1, n_buckets):
        edges.append(s[min(len(s) - 1, int(len(s) * i / n_buckets))])
    return edges


def _one_hot_levels(records: list[dict], factor_fns: dict[str, callable]) -> dict[str, int]:
    levels = {}
    for name, fn in factor_fns.items():
        levels[name] = len({fn(r) for r in records})
    return levels


def _design_matrix(
    records: list[dict], factor_fns: dict[str, callable], levels: dict[str, int]
) -> torch.Tensor:
    cols = []
    for name, fn in factor_fns.items():
        idx = torch.tensor([fn(r) for r in records], dtype=torch.long).clamp(
            max=levels[name] - 1
        )
        cols.append(torch.nn.functional.one_hot(idx, num_classes=levels[name]).float())
    intercept = torch.ones(len(records), 1)
    return torch.cat([intercept] + cols, dim=1)


def _ridge_fit(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    gram = x.T @ x + alpha * torch.eye(x.size(1))
    return torch.linalg.solve(gram, x.T @ y)


def _calibration(pred: torch.Tensor, y_test: torch.Tensor, global_mean: float) -> dict:
    labels = y_test.gt(0)
    zpred = (pred - pred.mean()) / pred.std().clamp(min=1e-9)
    prob = torch.sigmoid(zpred)
    top_k = max(1, pred.numel() // 4)
    predicted_top = set(pred.topk(top_k).indices.tolist())
    utility_top = set(y_test.topk(top_k).indices.tolist())
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        sel = (prob >= lo) & (prob < hi if i < 9 else prob <= hi)
        if sel.any():
            ece += float(sel.float().mean().item()) * abs(
                float(prob[sel].mean().item()) - float(labels[sel].float().mean().item())
            )
    return {
        "spearman": _spearman(pred, y_test),
        "auc": _auc(pred, labels),
        "ece_10bin": ece,
        "sign_at_mean": float(pred.gt(global_mean).eq(labels).float().mean().item()),
        "top_quartile_overlap": len(predicted_top & utility_top) / top_k,
    }


def _regret(pred: torch.Tensor, y_test: torch.Tensor, test: list[dict]) -> dict:
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(test):
        grouped[(r["draw"], r["batch_index"], r["sample_in_batch"], r["chunk_index"])].append(i)
    rows = []
    for key, indices in grouped.items():
        if len(indices) < 2:
            continue
        true_u = y_test[indices]
        pred_u = pred[indices]
        budget = max(1, len(indices) // 4)
        oracle = true_u.topk(budget).values.sum()
        chosen = true_u[pred_u.topk(budget).indices].sum()
        rows.append(float((oracle - chosen).item()))
    return {
        "groups": len(rows),
        "oracle_regret_mean": sum(rows) / max(len(rows), 1),
        "oracle_regret_p90": sorted(rows)[int(0.9 * len(rows))] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    records = []
    with Path(cli.dataset).open("r", encoding="utf-8") as source:
        for line in source:
            records.append(json.loads(line))
    ctx = json.loads(Path(cli.context).read_text(encoding="utf-8"))

    joined = 0
    for r in records:
        key = f"{r['batch_index']}_{r['sample_in_batch']}_{r['chunk_index']}"
        row = ctx.get(key)
        if row is not None:
            r["context"] = row
            joined += 1
    records = [r for r in records if "context" in r]

    train = [r for r in records if r["batch_index"] < 24]
    test = [r for r in records if r["batch_index"] >= 24]
    y_train = torch.tensor([r["net_utility"] for r in train], dtype=torch.float32)
    y_test = torch.tensor([r["net_utility"] for r in test], dtype=torch.float32)
    global_mean = float(y_train.mean().item())

    class_sums: dict[tuple, list[float]] = defaultdict(list)
    for r in train:
        class_sums[_class_key(r)].append(r["net_utility"])
    class_mean = {k: sum(v) / len(v) for k, v in class_sums.items()}
    pred_lookup = torch.tensor(
        [class_mean.get(_class_key(r), global_mean) for r in test], dtype=torch.float32
    )

    edges = {
        k: _quantile_edges([r["context"][k] for r in train]) for k in CONTEXT_KEYS
    }

    base_factors = {
        "slot": lambda r: r["slot"],
        "punct": lambda r: _bucket(r["features"]["punct_ratio"], [0.05, 0.15, 0.30]),
        "entropy": lambda r: _bucket(r["features"]["unigram_entropy"], [3.0, 4.0, 5.0, 6.0]),
        "anchor": lambda r: _bucket(r["features"]["byte_anchor"], [128.0, 256.0, 384.0]),
    }
    ctx_factors = {
        name: (lambda k: lambda r: _bucket(r["context"][k], edges[k]))(name)
        for name in CONTEXT_KEYS
    }

    def fit_eval(factors: dict[str, callable]) -> tuple[torch.Tensor, dict, dict]:
        levels = _one_hot_levels(train, factors)
        x_train = _design_matrix(train, factors, levels)
        x_test = _design_matrix(test, factors, levels)
        beta = _ridge_fit(x_train, y_train)
        pred = x_test @ beta
        return pred, _calibration(pred, y_test, global_mean), _regret(pred, y_test, test)

    results = {}
    pred_b, cal_b, reg_b = fit_eval(base_factors)
    results["factor_baseline"] = {"calibration": cal_b, "deployment": reg_b}
    pred_a, cal_a, reg_a = fit_eval({**base_factors, **ctx_factors})
    results["factor_plus_context"] = {"calibration": cal_a, "deployment": reg_a}
    for name, fn in ctx_factors.items():
        _, cal_s, _ = fit_eval({**base_factors, name: fn})
        results[f"factor_plus_{name}"] = {"calibration": cal_s}

    results["class_mean_lookup"] = _calibration(pred_lookup, y_test, global_mean)
    report = {
        "records": len(records),
        "joined_context": joined,
        "train_records": len(train),
        "test_records": len(test),
        "context_bucket_edges": edges,
        "results": results,
        "delta_plus_context_vs_baseline": {
            "spearman": cal_a["spearman"] - cal_b["spearman"],
            "auc": cal_a["auc"] - cal_b["auc"],
            "oracle_regret_mean": reg_a["oracle_regret_mean"] - reg_b["oracle_regret_mean"],
        },
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
