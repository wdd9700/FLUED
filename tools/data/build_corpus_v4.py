#!/usr/bin/env python3
"""Build a deduplicated UTF-8 FLUED corpus from local dataset stores.

The script is deliberately conservative:
- E:/.../corpus_v3.txt is treated as the baseline corpus.
- O:/v11 raw sources are not enabled by default because most of them were
  already used to build corpus_v3.txt.
- W/X/Y/Z sources are treated as incremental additions.
- Deduplication is document/line based via SQLite so the run is resumable and
  does not require keeping all hashes in memory.

Output is a single corpus represented as UTF-8 text shards under:
  <out_dir>/shards/corpus_v4_00000.txt ...
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import html
import io
import json
import lzma
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


TEXT_FIELDS = (
    "text",
    "content",
    "contents",
    "article",
    "body",
    "document",
    "prompt",
    "query",
    "question",
    "answer",
    "response",
    "completion",
    "instruction",
    "input",
    "output",
    "caption",
    "title",
)


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WIKI_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>", re.IGNORECASE | re.DOTALL)
WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]{0,800}\}\}")
WIKI_LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")


@dataclass(frozen=True)
class Source:
    path: Path
    kind: str
    category: str
    max_output_gb: Optional[float] = None
    note: str = ""


DEFAULT_SOURCES = [
    # Baseline: keep exactly as the existing FLUED training corpus source.
    Source(
        Path(r"data/corpus.txt"),
        "plain",
        "baseline",
        None,
        "existing FLUED corpus_v3 baseline, keep original file in place",
    ),
    # W/X additions. WET is huge, so budgets prevent accidental disk fill.
    Source(Path(r"W:\Corpus\general\zhwiki-latest-pages-articles.bz2"), "wiki_xml_bz2", "wiki", 8.0),
    Source(Path(r"W:\Corpus\general\enwiki-latest-pages-articles.bz2"), "wiki_xml_bz2", "wiki", 12.0),
    Source(Path(r"W:\Corpus\general\commoncrawl-wet"), "wet_dir", "web", 48.0),
    Source(Path(r"X:\Corpus\general\commoncrawl-wet"), "wet_dir", "web", 48.0),
    # Y/Z additions not clearly included in corpus_v3.
    Source(Path(r"Y:\v11\fineweb-edu-full\fineweb-edu-full.jsonl.gz"), "jsonl_gz", "web_edu", 10.0),
    Source(Path(r"Y:\v11\cosmopedia\cosmopedia.jsonl.gz"), "jsonl_gz", "synthetic_edu", 10.0),
    Source(Path(r"Y:\v11\C4-en\C4-en.jsonl.gz"), "jsonl_gz", "web", 8.0),
    Source(Path(r"Y:\v11\proof-pile\proof-pile.jsonl.gz"), "jsonl_gz", "stem", 8.0),
    Source(Path(r"Y:\v11\starcoder-data\starcoder-data.jsonl.gz"), "jsonl_gz", "code", 16.0),
    Source(Path(r"Z:\v11\multilingual-cc\multilingual-cc.jsonl.gz"), "jsonl_gz", "multilingual_web", 16.0),
    Source(Path(r"Z:\v11\fineweb\fineweb.jsonl.gz"), "jsonl_gz", "web", 8.0),
    Source(Path(r"Z:\v11\finemath\finemath.jsonl.gz"), "jsonl_gz", "stem", 8.0),
    Source(Path(r"Z:\v11\OpenWebMath\OpenWebMath.jsonl.gz"), "jsonl_gz", "stem", 8.0),
    Source(Path(r"Z:\v11\arxiv-summarization\arxiv-summarization.jsonl.gz"), "jsonl_gz", "stem", 8.0),
]


def open_text(path: Path) -> io.TextIOBase:
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    if name.endswith(".xz") or name.endswith(".lzma"):
        return lzma.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore", newline="")


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\ufeff", "")
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def clean_wiki_text(text: str) -> str:
    text = WIKI_REF_RE.sub(" ", text)
    for _ in range(3):
        text = WIKI_TEMPLATE_RE.sub(" ", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub(" ", text)
    return normalize_text(text)


def flatten_json_value(value) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, str):
        if value:
            yield value
        return
    if isinstance(value, (int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            yield from flatten_json_value(item)
        return
    if isinstance(value, dict):
        if "value" in value and isinstance(value["value"], str):
            yield value["value"]
        for key in TEXT_FIELDS:
            if key in value:
                yield from flatten_json_value(value[key])
        return


def extract_json_text(obj) -> str:
    pieces: list[str] = []
    if isinstance(obj, dict):
        for key in TEXT_FIELDS:
            if key in obj:
                pieces.extend(flatten_json_value(obj[key]))
        if not pieces and "messages" in obj:
            pieces.extend(flatten_json_value(obj["messages"]))
        if not pieces and "conversations" in obj:
            pieces.extend(flatten_json_value(obj["conversations"]))
        if not pieces:
            for value in obj.values():
                if isinstance(value, str) and len(value) > 40:
                    pieces.append(value)
    else:
        pieces.extend(flatten_json_value(obj))
    return normalize_text("\n".join(str(x) for x in pieces if x))


def iter_plain(path: Path) -> Iterator[str]:
    with open_text(path) as f:
        for line in f:
            text = normalize_text(line)
            if text:
                yield text


def iter_jsonl(path: Path) -> Iterator[str]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                text = normalize_text(line)
            else:
                text = extract_json_text(obj)
            if text:
                yield text


def iter_wiki_xml(path: Path) -> Iterator[str]:
    in_text = False
    buf: list[str] = []
    with open_text(path) as f:
        for line in f:
            if "<text" in line:
                in_text = True
                line = line.split(">", 1)[1] if ">" in line else ""
            if in_text:
                if "</text>" in line:
                    before = line.split("</text>", 1)[0]
                    buf.append(before)
                    text = clean_wiki_text("".join(buf))
                    if text:
                        yield text
                    buf.clear()
                    in_text = False
                else:
                    buf.append(line)


def iter_wet_file(path: Path) -> Iterator[str]:
    record_lines: list[str] = []
    in_payload = False
    with open_text(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("WARC/"):
                if record_lines:
                    text = normalize_text("\n".join(record_lines))
                    if text:
                        yield text
                record_lines = []
                in_payload = False
                continue
            if not in_payload:
                if line == "":
                    in_payload = True
                continue
            record_lines.append(line)
    if record_lines:
        text = normalize_text("\n".join(record_lines))
        if text:
            yield text


def iter_source_records(source: Source) -> Iterator[str]:
    path = source.path
    if source.kind == "plain":
        yield from iter_plain(path)
    elif source.kind in {"jsonl", "jsonl_gz"}:
        yield from iter_jsonl(path)
    elif source.kind == "wiki_xml_bz2":
        yield from iter_wiki_xml(path)
    elif source.kind == "wet_gz":
        yield from iter_wet_file(path)
    elif source.kind == "wet_dir":
        files = sorted(path.glob("*.warc.wet.gz"))
        for child in files:
            yield from iter_wet_file(child)
    else:
        raise ValueError(f"unsupported source kind: {source.kind}")


def hash_text(text: str) -> bytes:
    normalized_for_hash = SPACE_RE.sub(" ", text.replace("\n", " ")).strip().lower()
    return hashlib.blake2b(normalized_for_hash.encode("utf-8"), digest_size=16).digest()


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute("CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_stats (
            path TEXT PRIMARY KEY,
            category TEXT,
            kind TEXT,
            seen_records INTEGER DEFAULT 0,
            kept_records INTEGER DEFAULT 0,
            input_bytes INTEGER DEFAULT 0,
            output_bytes INTEGER DEFAULT 0,
            duplicate_records INTEGER DEFAULT 0,
            skipped_records INTEGER DEFAULT 0,
            done INTEGER DEFAULT 0,
            updated_at REAL
        )
        """
    )
    return conn


class ShardWriter:
    def __init__(self, shard_dir: Path, shard_gb: float):
        self.shard_dir = shard_dir
        self.shard_limit = int(shard_gb * 1024**3)
        self.index = 0
        self.current_bytes = 0
        self.file: Optional[io.TextIOWrapper] = None
        self.open_next()

    def open_next(self) -> None:
        if self.file:
            self.file.close()
        while True:
            path = self.shard_dir / f"corpus_v4_{self.index:05d}.txt"
            if not path.exists() or path.stat().st_size < self.shard_limit:
                self.file = open(path, "a", encoding="utf-8", newline="\n")
                self.current_bytes = path.stat().st_size if path.exists() else 0
                return
            self.index += 1

    def write(self, text: str) -> int:
        data = text + "\n"
        n = len(data.encode("utf-8"))
        if self.current_bytes > 0 and self.current_bytes + n > self.shard_limit:
            self.index += 1
            self.open_next()
        assert self.file is not None
        self.file.write(data)
        self.current_bytes += n
        return n

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None


def source_key(source: Source) -> str:
    return str(source.path.resolve() if source.path.exists() else source.path)


def already_done(conn: sqlite3.Connection, source: Source) -> bool:
    row = conn.execute("SELECT done FROM source_stats WHERE path = ?", (source_key(source),)).fetchone()
    return bool(row and row[0])


def update_source_stats(
    conn: sqlite3.Connection,
    source: Source,
    *,
    seen: int,
    kept: int,
    input_bytes: int,
    output_bytes: int,
    duplicates: int,
    skipped: int,
    done: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO source_stats
          (path, category, kind, seen_records, kept_records, input_bytes,
           output_bytes, duplicate_records, skipped_records, done, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          category=excluded.category,
          kind=excluded.kind,
          seen_records=excluded.seen_records,
          kept_records=excluded.kept_records,
          input_bytes=excluded.input_bytes,
          output_bytes=excluded.output_bytes,
          duplicate_records=excluded.duplicate_records,
          skipped_records=excluded.skipped_records,
          done=excluded.done,
          updated_at=excluded.updated_at
        """,
        (
            source_key(source),
            source.category,
            source.kind,
            seen,
            kept,
            input_bytes,
            output_bytes,
            duplicates,
            skipped,
            1 if done else 0,
            time.time(),
        ),
    )
    conn.commit()


def write_source_manifest(sources: list[Source], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "exists", "kind", "category", "max_output_gb", "bytes", "note"],
        )
        writer.writeheader()
        for s in sources:
            writer.writerow(
                {
                    "path": str(s.path),
                    "exists": s.path.exists(),
                    "kind": s.kind,
                    "category": s.category,
                    "max_output_gb": "" if s.max_output_gb is None else s.max_output_gb,
                    "bytes": s.path.stat().st_size if s.path.is_file() else "",
                    "note": s.note,
                }
            )


def export_stats(conn: sqlite3.Connection, out_path: Path) -> None:
    rows = conn.execute(
        """
        SELECT path, category, kind, seen_records, kept_records, input_bytes,
               output_bytes, duplicate_records, skipped_records, done, updated_at
        FROM source_stats
        ORDER BY path
        """
    ).fetchall()
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "category",
                "kind",
                "seen_records",
                "kept_records",
                "input_bytes",
                "output_bytes",
                "duplicate_records",
                "skipped_records",
                "done",
                "updated_at",
            ]
        )
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=r"data/corpus_v4")
    p.add_argument("--target-gb", type=float, default=180.0, help="Stop after this much UTF-8 output.")
    p.add_argument("--shard-gb", type=float, default=4.0)
    p.add_argument("--min-chars", type=int, default=32)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--commit-every", type=int, default=20_000)
    p.add_argument(
        "--state-dir",
        default=None,
        help="Directory for SQLite dedupe state. Defaults to <out-dir>/state.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-o-v11-raw", action="store_true", help="Unsafe for v3 dedupe semantics; off by default.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_dir = out_dir / "manifests"
    report_dir = out_dir / "reports"
    shard_dir = out_dir / "shards"
    state_dir = Path(args.state_dir) if args.state_dir else out_dir / "state"
    for d in (manifest_dir, report_dir, shard_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    sources = [s for s in DEFAULT_SOURCES if s.path.exists()]
    write_source_manifest(sources, manifest_dir / "selected_sources.csv")

    if args.dry_run:
        print(f"Selected {len(sources)} existing sources.")
        print(f"Manifest: {manifest_dir / 'selected_sources.csv'}")
        return 0

    db_path = state_dir / "dedupe_hashes.sqlite3"
    conn = init_db(db_path)
    writer = ShardWriter(shard_dir, args.shard_gb)
    target_bytes = int(args.target_gb * 1024**3)
    total_output = sum(p.stat().st_size for p in shard_dir.glob("corpus_v4_*.txt"))
    t0 = time.time()

    print(f"Output dir: {out_dir}")
    print(f"Existing output bytes: {total_output}")
    print(f"Target bytes: {target_bytes}")
    print(f"Sources: {len(sources)}")

    try:
        for source in sources:
            if total_output >= target_bytes:
                break
            if already_done(conn, source):
                print(f"[skip done] {source.path}")
                continue
            print(f"[source] {source.category:16s} {source.kind:12s} {source.path}", flush=True)
            source_limit = None
            if source.max_output_gb is not None:
                source_limit = int(source.max_output_gb * 1024**3)
            seen = kept = input_bytes = output_bytes = duplicates = skipped = 0
            batch_hashes: list[tuple[bytes]] = []
            stopped_by_target = False
            stopped_by_source_limit = False
            for text in iter_source_records(source):
                seen += 1
                raw_len = len(text.encode("utf-8", errors="ignore"))
                input_bytes += raw_len
                if len(text) < args.min_chars or len(text) > args.max_chars:
                    skipped += 1
                    continue
                h = hash_text(text)
                cur = conn.execute("INSERT OR IGNORE INTO seen(h) VALUES (?)", (h,))
                if cur.rowcount == 0:
                    duplicates += 1
                    continue
                n = writer.write(text)
                kept += 1
                output_bytes += n
                total_output += n
                if total_output >= target_bytes:
                    stopped_by_target = True
                    break
                if source_limit is not None and output_bytes >= source_limit:
                    stopped_by_source_limit = True
                    break
                if kept % args.commit_every == 0:
                    conn.commit()
                    writer.file.flush() if writer.file else None
                    elapsed = max(time.time() - t0, 1.0)
                    mbps = total_output / 1024**2 / elapsed
                    print(
                        f"  kept={kept:,} dup={duplicates:,} "
                        f"source_out={output_bytes/1024**3:.2f}GiB "
                        f"total={total_output/1024**3:.2f}GiB "
                        f"{mbps:.1f}MiB/s",
                        flush=True,
                    )
            conn.commit()
            update_source_stats(
                conn,
                source,
                seen=seen,
                kept=kept,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                duplicates=duplicates,
                skipped=skipped,
                done=not stopped_by_target or stopped_by_source_limit,
            )
            export_stats(conn, report_dir / "source_stats.csv")
            print(
                f"[done] seen={seen:,} kept={kept:,} dup={duplicates:,} "
                f"out={output_bytes/1024**3:.2f}GiB total={total_output/1024**3:.2f}GiB",
                flush=True,
            )
    finally:
        writer.close()
        export_stats(conn, report_dir / "source_stats.csv")
        conn.close()

    elapsed = time.time() - t0
    print(f"Finished: {total_output/1024**3:.2f}GiB in {elapsed/3600:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
