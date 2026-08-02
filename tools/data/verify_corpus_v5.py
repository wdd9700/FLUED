#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from build_corpus_v5 import legacy_hash_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-db", required=True)
    parser.add_argument("--report", default="")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    db_path = output_dir / "state" / "dedupe_hashes.sqlite3"
    reference_db = Path(args.reference_db)
    report_path = Path(args.report) if args.report else output_dir / "reports" / "verification_report.json"
    started = time.time()

    required = [db_path, reference_db, output_dir / "manifests" / "shard_manifest.csv"]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        payload = {"passed": False, "missing_required": missing_required}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.execute("ATTACH DATABASE ? AS reference", (str(reference_db),))
    seen_total = int(conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
    reference_total = int(conn.execute("SELECT COUNT(*) FROM reference.seen").fetchone()[0])
    total = int(conn.execute("SELECT COUNT(*) FROM v5_records").fetchone()[0])
    overlap = int(
        conn.execute("SELECT COUNT(*) FROM v5_records r JOIN reference.seen s ON r.h=s.h").fetchone()[0]
    )
    distinct_hashes = int(conn.execute("SELECT COUNT(DISTINCT h) FROM v5_records").fetchone()[0])

    checked = 0
    content_hash_mismatches = 0
    delimiter_mismatches = 0
    missing_shards = 0
    layout_mismatches = 0
    by_shard: dict[str, list[tuple[bytes, int, int]]] = {}
    for digest, shard, offset, length in conn.execute(
        "SELECT h,shard,byte_offset,byte_length FROM v5_records ORDER BY shard,byte_offset"
    ):
        by_shard.setdefault(shard, []).append((digest, int(offset), int(length)))
    conn.close()

    for shard_name, records in by_shard.items():
        shard = Path(shard_name)
        if not shard.exists():
            missing_shards += 1
            continue
        with shard.open("rb") as f:
            expected_offset = 0
            for expected_hash, offset, length in records:
                if offset != expected_offset:
                    layout_mismatches += 1
                f.seek(offset)
                payload = f.read(length)
                delimiter = f.read(2)
                try:
                    text = payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    content_hash_mismatches += 1
                    checked += 1
                    continue
                if legacy_hash_text(text) != expected_hash:
                    content_hash_mismatches += 1
                if delimiter != b"\n\n":
                    delimiter_mismatches += 1
                checked += 1
                expected_offset = offset + length + 2
            if expected_offset != shard.stat().st_size:
                layout_mismatches += 1

    manifest_path = output_dir / "manifests" / "shard_manifest.csv"
    manifest_mismatches = 0
    manifest_paths: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            path = Path(row["path"]).resolve()
            manifest_paths.add(path)
            if not path.exists() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                manifest_mismatches += 1
    actual_paths = {path.resolve() for path in (output_dir / "shards").glob("corpus_v5_increment_*.txt")}
    indexed_paths = {Path(path).resolve() for path in by_shard}
    if manifest_paths != actual_paths or indexed_paths != actual_paths:
        manifest_mismatches += 1

    payload = {
        "output_dir": str(output_dir),
        "reference_db": str(reference_db),
        "indexed_records": total,
        "seen_total": seen_total,
        "reference_total": reference_total,
        "seen_increment": seen_total - reference_total,
        "distinct_hashes": distinct_hashes,
        "reference_overlap": overlap,
        "records_checked": checked,
        "content_hash_mismatches": content_hash_mismatches,
        "delimiter_mismatches": delimiter_mismatches,
        "missing_shards": missing_shards,
        "layout_mismatches": layout_mismatches,
        "manifest_mismatches": manifest_mismatches,
        "elapsed_seconds": time.time() - started,
    }
    payload["passed"] = (
        total == distinct_hashes == checked
        and total == seen_total - reference_total
        and overlap == 0
        and content_hash_mismatches == 0
        and delimiter_mismatches == 0
        and missing_shards == 0
        and layout_mismatches == 0
        and manifest_mismatches == 0
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
