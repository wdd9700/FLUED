#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus_v5 import apply_review, LMStudioReviewer, normalize_text, ReviewItem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="lmstudio_probe.json")
    return parser.parse_args()


def make_items() -> list[ReviewItem]:
    templates = [
        [
            "Home Products Pricing Cookie settings Accept all cookies " * 8,
            "A language encoder should preserve exact byte content while exposing a smoother contextual representation. " * 8,
            "Copyright 2026 Privacy Terms Contact us " * 8,
        ],
        [
            "本文讨论字节流到连续潜空间表示的可逆翻译。模型应保留局部语义、顺序和必要位置，同时避免把词义永久冻结在离散词表中。" * 8,
            "实验需要分别记录重建、补全、实际潜向量数量和边界稳定性，不能只凭单一损失判断。" * 8,
        ],
        [
            "def normalize(text):\n    return ' '.join(text.lower().split())\n" * 18,
            "The function above normalizes whitespace before exact hashing. " * 12,
        ],
        [
            "ï¿½ï¿½ broken navigation subscribe login click here " * 20,
            "A coherent paragraph remains useful only when its original characters are intact. " * 10,
        ],
    ]
    items = []
    for index in range(32):
        paragraphs = templates[index % len(templates)]
        items.append(
            ReviewItem(
                doc_id=f"probe-{index:02d}",
                text="\n\n".join(paragraphs),
                source="probe",
                paragraphs=paragraphs,
            )
        )
    return items


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    models = httpx.get(base_url + "/api/v1/models", timeout=30).json()
    selected = next(
        (row for row in models.get("models", []) if row.get("key") == args.model),
        None,
    )
    if selected is None:
        raise RuntimeError(f"model is not available: {args.model}")

    output = Path(args.output).resolve()
    reviewer = LMStudioReviewer(
        base_url=base_url,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        max_reviews=64,
        timeout=180,
        audit_log=output.with_suffix(".audit.jsonl"),
    )
    items = make_items()
    started = time.perf_counter()
    decisions = reviewer.review_many(items)
    elapsed = time.perf_counter() - started

    invalid = []
    for item in items:
        decision = decisions.get(item.doc_id)
        if decision is None or not decision.valid:
            invalid.append({"doc_id": item.doc_id, "reason": "missing_or_invalid"})
            continue
        if any(i < 0 or i >= len(item.paragraphs) for i in decision.drop_paragraphs):
            invalid.append({"doc_id": item.doc_id, "reason": "drop_index_out_of_range"})
        if any(i < 0 or i >= len(item.paragraphs) - 1 for i in decision.split_after):
            invalid.append({"doc_id": item.doc_id, "reason": "split_index_out_of_range"})
        outputs = apply_review(item, decision, min_chars=1)
        source_paragraphs = {normalize_text(paragraph) for paragraph in item.paragraphs}
        for output_text in outputs:
            output_paragraphs = {normalize_text(paragraph) for paragraph in output_text.split("\n\n")}
            if not output_paragraphs.issubset(source_paragraphs):
                invalid.append({"doc_id": item.doc_id, "reason": "output_contains_generated_text"})

    payload = {
        "model": selected,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "documents": len(items),
        "elapsed_seconds": elapsed,
        "documents_per_second": len(items) / elapsed,
        "valid": not invalid and len(decisions) == len(items),
        "invalid": invalid,
        "decisions": [decisions[item.doc_id].__dict__ for item in items if item.doc_id in decisions],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("documents", "elapsed_seconds", "documents_per_second", "valid", "invalid")}, ensure_ascii=False, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
