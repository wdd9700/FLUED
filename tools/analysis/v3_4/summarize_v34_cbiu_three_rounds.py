"""Consolidate the three autonomous v3.4 CBIU experiment rounds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUNS = (
    {
        "round": 1,
        "name": "legacy_joint",
        "label": "Legacy joint",
        "summary": "round1_emit_5k/r1_e0_legacy_emit/summary.json",
        "offline": "round1_offline_audit/r1_e0_legacy_emit/cbiu_v0.json",
        "runtime": "runtime/r1_legacy/runtime.json",
    },
    {
        "round": 1,
        "name": "cbiu_quality_joint",
        "label": "CBIU quality joint",
        "summary": "round1_emit_5k/r1_e1_cbiu_quality/summary.json",
        "offline": "round1_offline_audit/r1_e1_cbiu_quality/cbiu_v0.json",
    },
    {
        "round": 1,
        "name": "cbiu_dual_joint",
        "label": "CBIU dual joint",
        "summary": "round1_emit_5k/r1_e2_cbiu_dual/summary.json",
        "offline": "round1_offline_audit/r1_e2_cbiu_dual/cbiu_v0.json",
    },
    {
        "round": 2,
        "name": "legacy_emit_only",
        "label": "Legacy emit-only",
        "summary": "round2_emit_only_3k/r2_e0_legacy_emit_only/summary.json",
        "offline": "round2_offline_audit/r2_e0_legacy_emit_only/cbiu_v0.json",
        "action": "round2_action_calibration/r2_e0_legacy_emit_only/cbiu_action_calibration.json",
    },
    {
        "round": 2,
        "name": "cbiu_quality_emit_only",
        "label": "CBIU quality emit-only",
        "summary": "round2_emit_only_3k/r2_e1_cbiu_quality_emit_only/summary.json",
        "offline": "round2_offline_audit/r2_e1_cbiu_quality_emit_only/cbiu_v0.json",
        "action": "round2_action_calibration/r2_e1_cbiu_quality_emit_only/cbiu_action_calibration.json",
    },
    {
        "round": 2,
        "name": "cbiu_dual_emit_only",
        "label": "CBIU dual emit-only",
        "summary": "round2_emit_only_3k/r2_e2_cbiu_dual_emit_only/summary.json",
        "offline": "round2_offline_audit/r2_e2_cbiu_dual_emit_only/cbiu_v0.json",
        "action": "round2_action_calibration/r2_e2_cbiu_dual_emit_only/cbiu_action_calibration.json",
    },
    {
        "round": 3,
        "name": "cbiu_linear",
        "label": "CBIU linear",
        "summary": "round3_controller_capacity_3k/r3_c0_linear_reset/summary.json",
        "offline": "round3_offline_audit/r3_c0_linear_reset/cbiu_v0.json",
        "action": "round3_action_calibration/r3_c0_linear_reset/cbiu_action_calibration.json",
        "runtime": "runtime/r3_linear/runtime.json",
    },
    {
        "round": 3,
        "name": "cbiu_mlp64",
        "label": "CBIU MLP-64",
        "summary": "round3_controller_capacity_3k/r3_c1_mlp64/summary.json",
        "offline": "round3_offline_audit/r3_c1_mlp64/cbiu_v0.json",
        "action": "round3_action_calibration/r3_c1_mlp64/cbiu_action_calibration.json",
        "runtime": "runtime/r3_mlp64/runtime.json",
    },
    {
        "round": 3,
        "name": "cbiu_mlp64_slot",
        "label": "CBIU MLP-64 + slot",
        "summary": "round3_controller_capacity_3k/r3_c2_mlp64_slot/summary.json",
        "offline": "round3_offline_audit/r3_c2_mlp64_slot/cbiu_v0.json",
        "action": "round3_action_calibration/r3_c2_mlp64_slot/cbiu_action_calibration.json",
        "runtime": "runtime/r3_mlp64_slot/runtime.json",
    },
)


def _load(root: Path, relative: str | None) -> dict[str, Any] | None:
    if relative is None:
        return None
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _record(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    summary = _load(root, spec["summary"])
    offline = _load(root, spec["offline"])
    action = _load(root, spec.get("action"))
    runtime = _load(root, spec.get("runtime"))
    assert summary is not None and offline is not None
    policy = offline["modes"]["policy"]
    result = {
        "round": spec["round"],
        "name": spec["name"],
        "label": spec["label"],
        "steps": summary["steps"],
        "training_scope": summary["args"].get("training_scope", "joint"),
        "trainable_params": summary.get("trainable_params", summary["total_params"]),
        "identity_accuracy": summary["eval_identity_acc"],
        "completion_accuracy": summary["eval_completion_mask_acc"],
        "completion_perplexity": summary["eval_completion_ppl"],
        "preservation_accuracy": summary["eval_completion_preserve_acc"],
        "policy_readouts_per_byte": policy["actual_readouts_per_byte"],
        "reconstruction_bpb": policy["reconstruction_bpb"],
        "completion_bpb": policy["completion_bpb"],
        "preservation_bpb": policy["preservation_bpb"],
        "train_steps_per_second": summary["train_steps_per_sec"],
        "peak_memory_gb": summary["train_peak_memory_gb"],
        "compute_dual": summary.get("eval_cbiu_compute_dual", 0.0),
    }
    if action is not None:
        result.update(
            action_examples=action["examples"],
            spearman=action["spearman_net"],
            auc=action["auc_net_positive"],
            brier=action["brier"],
            ece=action["ece_10bin"],
            sign_accuracy=action["sign_accuracy_at_0_5"],
            top_quartile_overlap=action["top_quartile_overlap"],
        )
    if runtime is not None:
        result["runtime"] = {
            key: runtime[key]
            for key in (
                "policy_readouts_per_byte",
                "max_active_readouts_per_sample",
                "encode_ms",
                "compact_ms",
                "backbone_ms",
                "decode_ms",
                "compact_backbone_decode_ms",
                "end_to_end_ms",
                "bytes_per_second_end_to_end",
            )
            if key in runtime
        }
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "round",
        "name",
        "steps",
        "training_scope",
        "trainable_params",
        "identity_accuracy",
        "completion_accuracy",
        "completion_perplexity",
        "preservation_accuracy",
        "policy_readouts_per_byte",
        "reconstruction_bpb",
        "completion_bpb",
        "preservation_bpb",
        "spearman",
        "auc",
        "brier",
        "ece",
        "sign_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# FLUED v3.4 CBIU 三轮实验机器汇总",
        "",
        "> 由 `summarize_v34_cbiu_three_rounds.py` 从原始 JSON/JSONL 生成。",
        "",
    ]
    for round_index in (1, 2, 3):
        lines.extend(
            [
                f"## 第 {round_index} 轮",
                "",
                "| 实验 | 重建准确率 | 补全准确率 | 补全困惑度 | 保持准确率 | 潜向量/字节 | 重建 BPB | 补全 BPB | 保持 BPB | Spearman | AUC | ECE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in records:
            if record["round"] != round_index:
                continue
            lines.append(
                "| {label} | {identity} | {completion} | {ppl} | {preserve} | "
                "{units} | {rec} | {fill} | {keep} | {spearman} | {auc} | {ece} |".format(
                    label=record["label"],
                    identity=_fmt(record["identity_accuracy"]),
                    completion=_fmt(record["completion_accuracy"]),
                    ppl=_fmt(record["completion_perplexity"], 3),
                    preserve=_fmt(record["preservation_accuracy"]),
                    units=_fmt(record["policy_readouts_per_byte"]),
                    rec=_fmt(record["reconstruction_bpb"], 3),
                    fill=_fmt(record["completion_bpb"], 3),
                    keep=_fmt(record["preservation_bpb"], 3),
                    spearman=_fmt(record.get("spearman"), 3),
                    auc=_fmt(record.get("auc"), 3),
                    ece=_fmt(record.get("ece"), 3),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## 自动判定",
            "",
            "- 第 1 轮：CBIU 在联合训练中减少潜向量，但三项原始 BPB 未改善。",
            "- 第 2 轮：冻结主体后，CBIU 的动作排序优于旧目标，但校准强度仍不足以接管边界。",
            "- 第 3 轮：MLP-64 明显改善动作排序；slot embedding 没有形成稳定的额外优势。",
            "- 当前边界：CBIU 可作为 emit 价值监督候选，尚不能宣称为 boundary/memory 的统一已验证监督。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _smooth(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        output.append(running / min(index + 1, window))
    return output


def _write_plots(root: Path, records: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; curve images were not generated"]

    warnings: list[str] = []
    metrics = (
        ("identity_acc", "Identity accuracy"),
        ("completion_ppl", "Completion perplexity"),
        ("policy_readout_units_per_byte", "Policy readouts / byte"),
        ("cbiu_brier", "CBIU Brier (valid actions)"),
    )
    spec_by_name = {spec["name"]: spec for spec in RUNS}
    for round_index in (1, 2, 3):
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for record in records:
            if record["round"] != round_index:
                continue
            spec = spec_by_name[record["name"]]
            log_path = root / Path(spec["summary"]).with_name("train_log.jsonl")
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            for axis, (key, title) in zip(axes.flat, metrics):
                selected = [
                    row
                    for row in rows
                    if key in row and (key != "cbiu_brier" or row.get("cbiu_valid_actions", 0) > 0)
                ]
                if not selected:
                    continue
                steps = [row["step"] for row in selected]
                values = [float(row[key]) for row in selected]
                window = max(3, len(values) // 30)
                axis.plot(steps, values, alpha=0.13, linewidth=0.7)
                axis.plot(steps, _smooth(values, window), linewidth=1.8, label=record["label"])
                axis.set_title(title)
                axis.set_xlabel("Step")
                axis.grid(alpha=0.2)
        for axis in axes.flat:
            axis.legend(fontsize=8)
        output = root / f"round{round_index}_training_curves.png"
        figure.savefig(output, dpi=170)
        plt.close(figure)
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    args = parser.parse_args()
    root = Path(args.archive_root)
    records = [_record(root, spec) for spec in RUNS]
    payload = {
        "protocol": "CBIU_THREE_ROUND_AUTONOMOUS_VALIDATION_20260717",
        "archive_root": str(root.resolve()),
        "runs": records,
        "decision": {
            "emit": "promising; keep MLP-64 CBIU controller as the next candidate",
            "boundary": "not admitted; action calibration gate failed",
            "memory": "not admitted; component-specific counterfactual training remains unvalidated",
            "unified_objective": "not yet established",
        },
    }
    warnings = _write_plots(root, records)
    if warnings:
        payload["warnings"] = warnings
    (root / "three_round_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(root / "three_round_summary.csv", records)
    _write_markdown(root / "three_round_summary.md", records)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
