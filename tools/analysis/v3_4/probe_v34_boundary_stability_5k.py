"""Compare v3.4 boundary/emit behavior between 2.5K and 5K checkpoints."""

from __future__ import annotations

from argparse import Namespace
import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import PAD_ID, text_to_byte_ids  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import build_model  # noqa: E402


CASES = {
    "english": "The compiler reuses cached latent states when the function name and argument types remain unchanged.",
    "chinese": "语言编码器需要在保留字节细节的同时，把当前语义片段翻译成平滑的潜空间表示。",
    "code": "def update_cache(key: str, value: Tensor) -> None:\n    memory_pool[key] = value.detach()\n",
    "mixed": "Alethic Insight 在 CUDA graph 中追踪 tensor_42，并保持 FLUED-v3.4 的 chunk 对齐。",
}


def tolerant_f1(left: list[int], right: list[int], tolerance: int = 1) -> float:
    if not left and not right:
        return 1.0
    used = set()
    matches = 0
    for value in left:
        candidates = [(abs(value - other), index) for index, other in enumerate(right) if index not in used]
        if candidates:
            distance, index = min(candidates)
            if distance <= tolerance:
                matches += 1
                used.add(index)
    precision = matches / max(len(left), 1)
    recall = matches / max(len(right), 1)
    return 2 * precision * recall / max(precision + recall, 1.0e-9)


@torch.no_grad()
def inspect(model, text: str, seq_len: int, device: torch.device) -> dict:
    ids = text_to_byte_ids(text)[:seq_len]
    token_ids = torch.tensor([ids + [PAD_ID] * (seq_len - len(ids))], device=device)
    out = model(token_ids)
    cuts = torch.where(out.policy.hard_cut[0, : len(ids)])[0].cpu().tolist()
    rates = out.aux["marginal_coding_rate"][0, : len(ids)].float().cpu()
    top_rate = torch.topk(rates, min(8, rates.numel())).indices.tolist() if rates.numel() else []
    emit_per_chunk = out.emit_hard[0, out.chunks.chunk_mask[0]].sum(dim=-1).float().cpu().tolist()
    return {
        "byte_length": len(ids),
        "cuts": cuts,
        "top_rate_positions": top_rate,
        "emit_per_chunk": emit_per_chunk,
        "emit_mean": sum(emit_per_chunk) / max(len(emit_per_chunk), 1),
    }


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(Namespace(**payload["args"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = {}
    for run in ("full", "l2_coding_rate", "uniform_boundaries"):
        checkpoints = {2500: root / run / "step_002500.pt", 5000: root / run / "step_005000.pt"}
        snapshots = {}
        for step, path in checkpoints.items():
            model = load_checkpoint(path, device)
            snapshots[step] = {name: inspect(model, text, 512, device) for name, text in CASES.items()}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        comparisons = {}
        for name in CASES:
            early, late = snapshots[2500][name], snapshots[5000][name]
            comparisons[name] = {
                "boundary_tolerance_f1": tolerant_f1(early["cuts"], late["cuts"], 1),
                "emit_mean_2500": early["emit_mean"],
                "emit_mean_5000": late["emit_mean"],
                "emit_mean_delta": late["emit_mean"] - early["emit_mean"],
            }
        result[run] = {"snapshots": snapshots, "comparisons": comparisons}
    (out_dir / "boundary_stability.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| run | case | boundary F1@1 (2.5K->5K) | emit@2.5K | emit@5K | delta |", "|---|---|---:|---:|---:|---:|"]
    for run, data in result.items():
        for case, row in data["comparisons"].items():
            lines.append(
                f"| {run} | {case} | {row['boundary_tolerance_f1']:.4f} | "
                f"{row['emit_mean_2500']:.3f} | {row['emit_mean_5000']:.3f} | {row['emit_mean_delta']:.3f} |"
            )
    (out_dir / "boundary_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
