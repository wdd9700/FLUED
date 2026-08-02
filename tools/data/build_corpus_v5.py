#!/usr/bin/env python3
"""Build a clean, deduplicated FLUED corpus_v5 increment.

The pipeline is intentionally conservative:

* W/X/Y are read-only inputs.
* N:/FLUED_corpus_v4/state/dedupe_hashes.sqlite3 is copied, never modified.
* Exact hashes use the corpus_v4 normalization contract, so every emitted
  record is checked against both the original E: corpus and existing N: shards.
* LM Studio may only select/drop/split existing paragraphs. It is never allowed
  to rewrite or invent corpus text.
* A pending-batch journal makes shard appends recoverable without duplicates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import httpx
except ImportError:  # pragma: no cover - optional when --no-qwen is used
    httpx = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - validated at startup for parquet runs
    pa = None
    pq = None

STREAM_EXCEPTIONS = (EOFError, gzip.BadGzipFile, UnicodeDecodeError, OSError, ValueError) + (
    (pa.ArrowException,) if pa is not None else ()
)


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
HTML_TAG_RE = re.compile(r"<[^>]{1,500}>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
WORD_RE = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)
MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â€|â€™|â€œ|â€\x9d|锟斤拷|烫烫烫)")
REPLACEMENT = "\ufffd"

BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^cookie (settings|policy|preferences)$",
        r"^(accept|reject) all cookies$",
        r"^privacy policy$",
        r"^terms (of use|and conditions|of service)$",
        r"^all rights reserved\.?$",
        r"^skip to (main )?content$",
        r"^sign (in|up)|^log (in|out)$",
        r"^share (this|on)|^follow us$",
        r"^advertisement$|^sponsored content$",
        r"^javascript (is )?disabled$",
        r"^enable javascript",
    )
]

SPAM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"buy (cheap|online).{0,30}(viagra|cialis|levitra)",
        r"casino bonus|online casino|sports betting",
        r"click here.{0,30}(free|download|prize)",
        r"seo services|payday loan",
    )
]

SECRET_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}",
        r"gh[pousr]_[A-Za-z0-9]{30,}",
    )
]

CODE_EXCLUDED_PARTS = {
    ".git",
    ".svn",
    "node_modules",
    "vendor",
    "vendors",
    "dist",
    "build",
    "target",
    "coverage",
    "__pycache__",
    "site-packages",
    "third_party",
    "third-party",
    "external",
    "generated",
    "minified",
}

CODE_EXCLUDED_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "cargo.lock",
    "poetry.lock",
    "go.sum",
}

CODE_ALLOWED_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".cs",
    ".fs",
    ".fsx",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".ps1",
    ".sql",
    ".r",
    ".rb",
    ".php",
    ".lua",
    ".dart",
    ".ex",
    ".exs",
    ".erl",
    ".hrl",
    ".clj",
    ".cljs",
    ".vue",
    ".svelte",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".rst",
    ".tex",
}

PERMISSIVE_LICENSES = {
    "apache-2.0",
    "apache-2",
    "mit",
    "isc",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-2",
    "bsd-3",
    "unlicense",
    "cc0-1.0",
    "cc0",
    "zlib",
    "mpl-2.0",
}


@dataclass
class SourceSpec:
    name: str
    category: str
    kind: str
    root: str
    pattern: str = ""
    include_dirs: list[str] = field(default_factory=list)
    text_field: str = "text"
    json_formatter: str = "generic"
    exclude_if_true: list[str] = field(default_factory=list)
    priority: int = 100
    sample_rate: float = 1.0
    max_output_gib: float | None = None
    min_chars: int = 200
    max_chars: int = 2_000_000
    language_score_min: float | None = None
    edu_score_min: float | None = None
    code: bool = False
    qwen_review: bool = True
    enabled: bool = True
    license_status: str = "unknown"
    source_url: str = ""
    notes: str = ""


@dataclass
class RawRecord:
    text: str
    metadata: dict[str, Any]


@dataclass
class CleanResult:
    text: str | None
    reason: str = ""
    removed_lines: int = 0
    suspicious: bool = False


@dataclass
class ReviewItem:
    doc_id: str
    text: str
    source: str
    paragraphs: list[str]
    code: bool = False


@dataclass
class ReviewDecision:
    doc_id: str
    keep: bool
    quality: float
    issue: str
    drop_paragraphs: list[int]
    split_after: list[int]
    valid: bool = True


@dataclass
class SourceStats:
    name: str
    category: str
    files_discovered: int = 0
    files_valid: int = 0
    files_invalid: int = 0
    records_seen: int = 0
    records_sampled: int = 0
    records_clean: int = 0
    records_rejected: int = 0
    qwen_sent: int = 0
    qwen_eligible: int = 0
    qwen_kept: int = 0
    qwen_dropped: int = 0
    qwen_fallback: int = 0
    qwen_cap_skipped: int = 0
    chunks_candidate: int = 0
    chunks_duplicate: int = 0
    chunks_written: int = 0
    output_bytes: int = 0
    input_bytes: int = 0
    completed: bool = False
    stopped_by_limit: bool = False
    elapsed_seconds: float = 0.0
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["rejection_reasons"] = dict(self.rejection_reasons)
        data["output_gib"] = self.output_bytes / 1024**3
        return data


def legacy_hash_text(text: str) -> bytes:
    """Match corpus_v4's exact hash contract byte-for-byte."""

    normalized_for_hash = SPACE_RE.sub(" ", text.replace("\n", " ")).strip().lower()
    return hashlib.blake2b(normalized_for_hash.encode("utf-8"), digest_size=16).digest()


def hash_fraction(digest: bytes) -> float:
    return int.from_bytes(digest[:8], "big") / float(2**64)


def reversible_mojibake_fix(text: str) -> str:
    if not MOJIBAKE_RE.search(text):
        return text
    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def score(value: str) -> tuple[int, int]:
        return (len(MOJIBAKE_RE.findall(value)) + value.count(REPLACEMENT) * 4, -len(value))

    return min(candidates, key=score)


def normalize_text(text: str) -> str:
    text = reversible_mojibake_fix(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def normalize_code_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def strip_web_boilerplate(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    kept: list[str] = []
    removed = 0
    counts = Counter(line.casefold().strip() for line in lines if len(line.strip()) >= 12)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if any(pattern.search(stripped) for pattern in BOILERPLATE_PATTERNS):
            removed += 1
            continue
        if len(stripped) >= 80 and counts[stripped.casefold()] >= 4:
            removed += 1
            continue
        kept.append(stripped)
    return normalize_text("\n".join(kept)), removed


def text_quality_reason(text: str, *, min_chars: int, max_chars: int, code: bool) -> str:
    n = len(text)
    replacement_count = text.count(REPLACEMENT)
    if replacement_count > max(3, math.ceil(n * 0.0015)):
        return "encoding_corruption"
    if n < min_chars:
        return "too_short"
    if n > max_chars:
        return "too_long"
    if any(pattern.search(text) for pattern in SPAM_PATTERNS):
        return "spam_pattern"
    if not code:
        words = WORD_RE.findall(text)
        if len(words) < 20:
            return "too_few_words"
        url_count = len(URL_RE.findall(text))
        if url_count > max(8, len(words) // 20):
            return "url_density"
        visible = sum(ch.isalnum() or "\u3400" <= ch <= "\u9fff" for ch in text)
        if visible / max(n, 1) < 0.25:
            return "low_visible_text_ratio"
    return ""


def code_quality_reason(text: str, metadata: dict[str, Any]) -> str:
    path = str(metadata.get("path") or "").replace("\\", "/")
    path_lower = path.casefold()
    parts = {part for part in path_lower.split("/") if part}
    name = Path(path_lower).name
    if parts & CODE_EXCLUDED_PARTS:
        return "code_vendored_or_generated"
    if name in CODE_EXCLUDED_NAMES or name.endswith((".min.js", ".min.css", ".map")):
        return "code_generated_or_lock"
    suffix = Path(name).suffix
    if suffix not in CODE_ALLOWED_EXTENSIONS:
        return "code_extension"
    license_name = str(metadata.get("license") or "").strip().casefold()
    if not license_name:
        return "code_license_missing"
    if license_name not in PERMISSIVE_LICENSES:
        return "code_license"
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return "code_secret"
    lines = text.splitlines()
    if not lines:
        return "code_empty"
    max_line = max(map(len, lines))
    avg_line = sum(map(len, lines)) / len(lines)
    if max_line > 20_000 or (max_line > 4_000 and avg_line > 300):
        return "code_minified"
    nul_or_binary = sum(ord(ch) < 9 or 13 < ord(ch) < 32 for ch in text)
    if nul_or_binary:
        return "code_binary"
    return ""


def clean_record(record: RawRecord, source: SourceSpec) -> CleanResult:
    text = normalize_code_text(record.text) if source.code else normalize_text(record.text)
    removed = 0
    if not source.code and source.category in {"web", "web_edu", "stem", "academic", "dialogue"}:
        text = HTML_TAG_RE.sub(" ", text)
        text, removed = strip_web_boilerplate(text)
    if source.code:
        reason = code_quality_reason(text, record.metadata)
        if reason:
            return CleanResult(None, reason)
    reason = text_quality_reason(
        text,
        min_chars=source.min_chars,
        max_chars=source.max_chars,
        code=source.code,
    )
    if reason:
        return CleanResult(None, reason, removed_lines=removed)
    suspicious = removed > 0 or bool(MOJIBAKE_RE.search(text))
    return CleanResult(text, removed_lines=removed, suspicious=suspicious)


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    return paragraphs


def split_review_units(text: str, *, target_chars: int = 1800) -> list[str]:
    units: list[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= target_chars * 2:
            units.append(paragraph)
            continue
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9\"'])", paragraph)
            if part.strip()
        ]
        if len(sentences) <= 1:
            sentences = [paragraph[start : start + target_chars] for start in range(0, len(paragraph), target_chars)]
        joiner = "" if sum("\u3400" <= ch <= "\u9fff" for ch in paragraph) > len(paragraph) * 0.2 else " "
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) > target_chars:
                units.append(joiner.join(current).strip())
                current, current_len = [], 0
            current.append(sentence)
            current_len += len(sentence) + 1
        if current:
            units.append(joiner.join(current).strip())
    if len(units) > 128:
        regrouped: list[str] = []
        group_size = math.ceil(len(units) / 128)
        for start in range(0, len(units), group_size):
            regrouped.append("\n\n".join(units[start : start + group_size]))
        units = regrouped
    return units


def deterministic_chunks(text: str, *, max_chars: int, min_chars: int, code: bool) -> list[str]:
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []
    units = text.splitlines(keepends=True) if code else [p + "\n\n" for p in split_paragraphs(text)]
    normalizer = normalize_code_text if code else normalize_text
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > max_chars:
            if current:
                value = normalizer("".join(current))
                if len(value) >= min_chars:
                    chunks.append(value)
                current, current_len = [], 0
            for start in range(0, len(unit), max_chars):
                value = normalizer(unit[start : start + max_chars])
                if len(value) >= min_chars:
                    chunks.append(value)
            continue
        if current and current_len + len(unit) > max_chars:
            value = normalizer("".join(current))
            if len(value) >= min_chars:
                chunks.append(value)
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit)
    if current:
        value = normalizer("".join(current))
        if len(value) >= min_chars:
            chunks.append(value)
    return chunks


def needs_qwen_review(text: str, clean: CleanResult, audit_fraction: float, *, code: bool) -> bool:
    if code:
        return False
    if len(text) < 600 or len(text) > 24_000:
        return False
    digest = legacy_hash_text(text)
    sampled_audit = hash_fraction(digest) < audit_fraction
    paragraphs = split_review_units(text)
    ambiguous = clean.suspicious and len(paragraphs) >= 3
    return sampled_audit or ambiguous


def apply_review(item: ReviewItem, decision: ReviewDecision, min_chars: int) -> list[str]:
    normalizer = normalize_code_text if item.code else normalize_text
    original = normalizer(item.text)
    # The local model may remove boilerplate or suggest split points, but it is
    # not trusted to delete a complete training record by itself.
    if not decision.valid or not decision.keep:
        return [original] if len(original) >= min_chars else []
    drop = {idx for idx in decision.drop_paragraphs if 0 <= idx < len(item.paragraphs)}
    split_after = {idx for idx in decision.split_after if 0 <= idx < len(item.paragraphs) - 1}
    segments: list[str] = []
    current: list[str] = []
    for idx, paragraph in enumerate(item.paragraphs):
        if idx not in drop:
            current.append(paragraph)
        if idx in split_after and current:
            value = normalizer("\n\n".join(current))
            if len(value) >= min_chars:
                segments.append(value)
            current = []
    if current:
        value = normalizer("\n\n".join(current))
        if len(value) >= min_chars:
            segments.append(value)
    return segments or ([original] if len(original) >= min_chars else [])


class LMStudioReviewer:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        workers: int,
        batch_size: int,
        max_reviews: int,
        timeout: float,
        audit_log: Path,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required when Qwen review is enabled")
        if workers < 1 or batch_size < 1 or max_reviews < 0 or timeout <= 0:
            raise ValueError("invalid LM Studio reviewer limits")
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/v1/chat/completions"
        self.model = model
        self.workers = workers
        self.batch_size = batch_size
        self.max_reviews = max_reviews
        self.timeout = timeout
        self.audit_log = audit_log
        self.input_log = audit_log.with_name("qwen_inputs.jsonl")
        self.sent = 0
        self.log_lock = threading.Lock()
        self.logged_input_ids: set[str] = set()
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=min(timeout, 30.0)) as client:
            model_rows = client.get(self.base_url + "/api/v1/models").json().get("models", [])
        model_row = next((row for row in model_rows if row.get("key") == model), None)
        if model_row is None or not model_row.get("loaded_instances"):
            raise RuntimeError(f"LM Studio model is not loaded: {model}")
        if self.audit_log.exists():
            seen_doc_ids: set[str] = set()
            with self.audit_log.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        seen_doc_ids.add(str(json.loads(line)["doc_id"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
            self.sent = len(seen_doc_ids)
        if self.input_log.exists():
            with self.input_log.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        self.logged_input_ids.add(str(json.loads(line)["doc_id"]))
                    except (json.JSONDecodeError, KeyError):
                        continue

    @property
    def remaining(self) -> int:
        return max(0, self.max_reviews - self.sent)

    @staticmethod
    def _schema() -> dict[str, Any]:
        item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "doc_id": {"type": "string"},
                "keep": {"type": "boolean"},
                "quality": {"type": "number", "minimum": 0, "maximum": 1},
                "issue": {"type": "string"},
                "drop_paragraphs": {"type": "array", "items": {"type": "integer"}},
                "split_after": {"type": "array", "items": {"type": "integer"}},
            },
            "required": [
                "doc_id",
                "keep",
                "quality",
                "issue",
                "drop_paragraphs",
                "split_after",
            ],
        }
        return {
            "name": "corpus_batch_audit",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"decisions": {"type": "array", "items": item}},
                "required": ["decisions"],
            },
        }

    def _review_group(self, items: list[ReviewItem]) -> list[ReviewDecision]:
        docs = []
        for item in items:
            numbered = "\n".join(f"[{idx}] {p}" for idx, p in enumerate(item.paragraphs))
            docs.append(f"DOC_ID={item.doc_id}\n{numbered}")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 3000,
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": self._schema()},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是本地预训练语料审计器。只能选择、删除或切分现有段落，禁止改写、补全、纠错或生成正文。"
                        "删除网页导航、cookie、广告、乱码、无意义列表；保留连贯正文、代码、公式和必要上下文。"
                        "quality 必须是 0 到 1 的小数。只有全文完全没有可用内容时 keep 才能为 false。"
                        "drop_paragraphs 填要删除的段落编号；split_after 填应在其后切分的段落编号。"
                    ),
                },
                {"role": "user", "content": "\n\n===== NEXT DOCUMENT =====\n\n".join(docs)},
            ],
        }
        started = time.perf_counter()
        error = ""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, json=payload)
                response.raise_for_status()
                body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            raw_decisions = parsed.get("decisions", [])
            if not isinstance(raw_decisions, list):
                raise ValueError("decisions must be a list")
            requested_ids = [item.doc_id for item in items]
            response_ids = [row.get("doc_id") for row in raw_decisions if isinstance(row, dict)]
            if len(raw_decisions) != len(items) or sorted(response_ids) != sorted(requested_ids):
                raise ValueError("response doc_id coverage mismatch")
            if len(response_ids) != len(set(response_ids)):
                raise ValueError("duplicate response doc_id")
            by_id = {row["doc_id"]: row for row in raw_decisions}
            decisions = []
            for item in items:
                row = by_id.get(item.doc_id)
                if row is None:
                    decisions.append(ReviewDecision(item.doc_id, True, 0.0, "missing_decision", [], [], False))
                    continue
                if type(row["keep"]) is not bool or not isinstance(row["quality"], (int, float)):
                    raise ValueError(f"invalid scalar types for {item.doc_id}")
                quality = float(row["quality"])
                if 1.0 < quality <= 100.0:
                    quality /= 100.0
                if not 0.0 <= quality <= 1.0:
                    raise ValueError(f"quality out of range for {item.doc_id}")
                drop = row["drop_paragraphs"]
                splits = row["split_after"]
                if (
                    not isinstance(drop, list)
                    or not isinstance(splits, list)
                    or any(type(v) is not int or not 0 <= v < len(item.paragraphs) for v in drop)
                    or any(type(v) is not int or not 0 <= v < len(item.paragraphs) - 1 for v in splits)
                ):
                    raise ValueError(f"paragraph index out of range for {item.doc_id}")
                decisions.append(ReviewDecision(item.doc_id, row["keep"], quality, str(row["issue"]), drop, splits))
        except Exception as exc:  # conservative fallback keeps deterministic-cleaned text
            error = f"{type(exc).__name__}: {exc}"
            decisions = [ReviewDecision(item.doc_id, True, 0.0, error, [], [], False) for item in items]
        elapsed = time.perf_counter() - started
        try:
            with self.log_lock, self.audit_log.open("a", encoding="utf-8") as f:
                for item, decision in zip(items, decisions):
                    row = {
                        "time": time.time(),
                        "model": self.model,
                        "doc_id": item.doc_id,
                        "source": item.source,
                        "chars": len(item.text),
                        "paragraphs": len(item.paragraphs),
                        "elapsed_group_seconds": elapsed,
                        "error": error,
                        "decision": asdict(decision),
                        "preview": item.text[:240],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[warn] Qwen audit log write failed: {exc}", flush=True)
        return decisions

    def review_many(self, items: list[ReviewItem]) -> dict[str, ReviewDecision]:
        if not items or self.remaining <= 0:
            return {}
        items = items[: self.remaining]
        self.sent += len(items)
        new_inputs = [item for item in items if item.doc_id not in self.logged_input_ids]
        if new_inputs:
            try:
                with self.input_log.open("a", encoding="utf-8") as f:
                    for item in new_inputs:
                        f.write(
                            json.dumps(
                                {
                                    "doc_id": item.doc_id,
                                    "source": item.source,
                                    "code": item.code,
                                    "paragraphs": item.paragraphs,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        self.logged_input_ids.add(item.doc_id)
            except OSError as exc:
                print(f"[warn] Qwen input log write failed: {exc}", flush=True)
        groups: list[list[ReviewItem]] = []
        current: list[ReviewItem] = []
        current_chars = 0
        for item in items:
            if current and (len(current) >= self.batch_size or current_chars + len(item.text) > 64_000):
                groups.append(current)
                current, current_chars = [], 0
            current.append(item)
            current_chars += len(item.text)
        if current:
            groups.append(current)
        decisions: list[ReviewDecision] = []
        item_by_id = {item.doc_id: item for item in items}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._review_group, group): group for group in groups}
            for future, group in futures.items():
                try:
                    decisions.extend(future.result())
                except Exception as exc:
                    decisions.extend(
                        ReviewDecision(item.doc_id, True, 0.0, f"worker_error:{type(exc).__name__}", [], [], False)
                        for item in group
                    )
        retry_items = [
            item_by_id[decision.doc_id]
            for decision in decisions
            if not decision.valid
            and decision.issue.startswith(("ValueError:", "JSONDecodeError:", "KeyError:"))
        ]
        if retry_items:
            retried: dict[str, ReviewDecision] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._review_group, [item]): item for item in retry_items}
                for future, item in futures.items():
                    try:
                        retried[item.doc_id] = future.result()[0]
                    except Exception as exc:
                        retried[item.doc_id] = ReviewDecision(
                            item.doc_id, True, 0.0, f"retry_error:{type(exc).__name__}", [], [], False
                        )
            decisions = [retried.get(decision.doc_id, decision) for decision in decisions]
        return {decision.doc_id: decision for decision in decisions}


class StorageLimitReached(RuntimeError):
    pass


class RunLock:
    def __init__(self, output_dir: Path, *, enabled: bool) -> None:
        self.path = output_dir / "state" / "RUNNING.lock"
        self.enabled = enabled

    def acquire(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            details = self.path.read_text(encoding="utf-8", errors="replace") if self.path.exists() else ""
            raise RuntimeError(f"output directory is locked: {self.path}\n{details}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": time.time()}, f)

    def release(self) -> None:
        if self.enabled and self.path.exists():
            self.path.unlink()

    def remove_if_stale(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot verify stale lock: {self.path}") from exc
        if self._pid_exists(pid):
            raise RuntimeError(f"refusing to remove live build lock for pid={pid}: {self.path}")
        self.path.unlink()

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            # Access denied still means the process exists.
            return ctypes.get_last_error() == 5
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        else:
            return True


class OutputStore:
    def __init__(
        self,
        output_dir: Path,
        reference_db: Path,
        *,
        shard_gib: float,
        dry_run: bool,
        max_output_gib: float,
        min_free_gib: float,
        run_fingerprint: dict[str, str],
    ) -> None:
        self.output_dir = output_dir
        self.shard_dir = output_dir / "shards"
        self.state_dir = output_dir / "state"
        self.manifest_dir = output_dir / "manifests"
        self.report_dir = output_dir / "reports"
        self.log_dir = output_dir / "logs"
        self.dry_run = dry_run
        self.max_output_bytes = int(max_output_gib * 1024**3)
        self.min_free_bytes = int(min_free_gib * 1024**3)
        self.run_fingerprint = run_fingerprint
        for path in (self.shard_dir, self.state_dir, self.manifest_dir, self.report_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.shard_limit = int(shard_gib * 1024**3)
        self.pending_path = self.state_dir / "pending_batch.jsonl"
        self.db_path = self.state_dir / "dedupe_hashes.sqlite3"
        if dry_run:
            self.conn = sqlite3.connect(":memory:")
            ref = sqlite3.connect(f"file:{reference_db.as_posix()}?mode=ro", uri=True)
            ref_count = int(ref.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
            ref.close()
            self.reference_count = ref_count
            self.conn.execute("CREATE TABLE seen (h BLOB PRIMARY KEY)")
            self.reference_lookup = sqlite3.connect(f"file:{reference_db.as_posix()}?mode=ro", uri=True)
        else:
            self._initialize_db(reference_db, run_fingerprint)
            self.conn = sqlite3.connect(str(self.db_path))
            self.reference_lookup = None
            self.reference_count = int(
                self.conn.execute("SELECT value FROM v5_meta WHERE key='reference_count'").fetchone()[0]
            )
            self._verify_run_fingerprint(run_fingerprint)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-262144")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS v5_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS v5_source_stats (
                name TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS v5_records (
                h BLOB PRIMARY KEY,
                source TEXT NOT NULL,
                shard TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()
        try:
            self.shard_index, self.shard_path, self.shard_size = self._find_last_shard()
            if self.pending_path.exists() and not dry_run:
                self._recover_pending()
            if not dry_run:
                self._verify_existing_layout()
            self.total_output_bytes = sum(
                path.stat().st_size for path in self.shard_dir.glob("corpus_v5_increment_*.txt")
            )
        except Exception:
            if self.reference_lookup is not None:
                self.reference_lookup.close()
            self.conn.close()
            raise

    def _initialize_db(self, reference_db: Path, run_fingerprint: dict[str, str]) -> None:
        if self.db_path.exists():
            return
        partial = self.db_path.with_suffix(".sqlite3.partial")
        if partial.exists():
            partial.unlink()
        print(f"[state] copying reference dedupe DB via SQLite backup: {reference_db}", flush=True)
        source = sqlite3.connect(f"file:{reference_db.as_posix()}?mode=ro", uri=True)
        dest = sqlite3.connect(str(partial))
        source.backup(dest, pages=8192)
        reference_count = int(source.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
        dest.execute("CREATE TABLE IF NOT EXISTS v5_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        dest.execute(
            "INSERT OR REPLACE INTO v5_meta(key,value) VALUES('reference_count',?)",
            (str(reference_count),),
        )
        dest.execute(
            "INSERT OR REPLACE INTO v5_meta(key,value) VALUES('reference_db',?)",
            (str(reference_db),),
        )
        for key, value in run_fingerprint.items():
            dest.execute(
                "INSERT OR REPLACE INTO v5_meta(key,value) VALUES(?,?)",
                (f"fingerprint:{key}", value),
            )
        dest.commit()
        source.close()
        dest.close()
        partial.replace(self.db_path)

    def _verify_run_fingerprint(self, expected: dict[str, str]) -> None:
        actual = {
            row[0].split(":", 1)[1]: row[1]
            for row in self.conn.execute("SELECT key,value FROM v5_meta WHERE key LIKE 'fingerprint:%'")
        }
        if actual != expected:
            raise RuntimeError(
                "existing output was created from a different config/reference/input manifest; "
                f"expected={expected}, actual={actual}"
            )

    def _verify_existing_layout(self) -> None:
        total_seen = int(self.conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
        indexed = int(self.conn.execute("SELECT COUNT(*) FROM v5_records").fetchone()[0])
        if total_seen - self.reference_count != indexed:
            raise RuntimeError(
                "state/index count mismatch before resume: "
                f"seen_increment={total_seen - self.reference_count}, indexed={indexed}"
            )
        rows = self.conn.execute(
            "SELECT shard,byte_offset,byte_length FROM v5_records ORDER BY shard,byte_offset"
        )
        expected_by_shard: dict[str, int] = {}
        for shard_name, offset, length in rows:
            offset = int(offset)
            length = int(length)
            expected = expected_by_shard.get(shard_name, 0)
            if offset != expected:
                raise RuntimeError(
                    f"non-contiguous shard index before resume: {shard_name} offset={offset} expected={expected}"
                )
            expected_by_shard[shard_name] = offset + length + 2
        indexed_paths = {Path(name).resolve(): size for name, size in expected_by_shard.items()}
        actual_paths = {path.resolve(): path.stat().st_size for path in self.shard_dir.glob("corpus_v5_increment_*.txt")}
        if indexed_paths != actual_paths:
            raise RuntimeError(
                "shard/index layout mismatch before resume: "
                f"indexed={indexed_paths}, actual={actual_paths}"
            )

    def _check_storage(self, batch_bytes: int) -> None:
        if self.dry_run:
            return
        if self.total_output_bytes + batch_bytes > self.max_output_bytes:
            raise StorageLimitReached(
                f"global output quota reached: {(self.total_output_bytes + batch_bytes)/1024**3:.2f} GiB "
                f"> {self.max_output_bytes/1024**3:.2f} GiB"
            )
        free = shutil.disk_usage(self.output_dir).free
        if free - batch_bytes < self.min_free_bytes:
            raise StorageLimitReached(
                f"free-space reserve reached: {(free - batch_bytes)/1024**3:.2f} GiB "
                f"< {self.min_free_bytes/1024**3:.2f} GiB"
            )

    def _find_last_shard(self) -> tuple[int, Path, int]:
        paths = sorted(self.shard_dir.glob("corpus_v5_increment_*.txt"))
        if not paths:
            path = self.shard_dir / "corpus_v5_increment_00000.txt"
            return 0, path, path.stat().st_size if path.exists() else 0
        path = paths[-1]
        match = re.search(r"(\d+)\.txt$", path.name)
        index = int(match.group(1)) if match else len(paths) - 1
        size = path.stat().st_size
        if size >= self.shard_limit:
            index += 1
            path = self.shard_dir / f"corpus_v5_increment_{index:05d}.txt"
            size = path.stat().st_size if path.exists() else 0
        return index, path, size

    def _rotate_if_needed(self, batch_bytes: int) -> None:
        if self.shard_size and self.shard_size + batch_bytes > self.shard_limit:
            self.shard_index += 1
            self.shard_path = self.shard_dir / f"corpus_v5_increment_{self.shard_index:05d}.txt"
            self.shard_size = self.shard_path.stat().st_size if self.shard_path.exists() else 0

    def _exists(self, digest: bytes) -> bool:
        if self.dry_run and self.reference_lookup is not None:
            if self.reference_lookup.execute("SELECT 1 FROM seen WHERE h=?", (digest,)).fetchone():
                return True
        return self.conn.execute("SELECT 1 FROM seen WHERE h=?", (digest,)).fetchone() is not None

    @staticmethod
    def _batch_existing(conn: sqlite3.Connection, digests: Sequence[bytes]) -> set[bytes]:
        existing: set[bytes] = set()
        for start in range(0, len(digests), 400):
            group = digests[start : start + 400]
            if not group:
                continue
            placeholders = ",".join("?" for _ in group)
            existing.update(row[0] for row in conn.execute(f"SELECT h FROM seen WHERE h IN ({placeholders})", group))
        return existing

    def _write_pending(self, records: list[tuple[str, bytes, str]]) -> None:
        header = {
            "shard": str(self.shard_path),
            "offset": self.shard_size,
            "created_at": time.time(),
            "count": len(records),
        }
        with self.pending_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for text, digest, source in records:
                f.write(
                    json.dumps(
                        {"h": digest.hex(), "source": source, "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            os.fsync(f.fileno())

    def _read_pending(self) -> tuple[dict[str, Any], list[tuple[str, bytes, str]]]:
        with self.pending_path.open("r", encoding="utf-8") as f:
            header = json.loads(f.readline())
            records = []
            for line in f:
                row = json.loads(line)
                records.append((row["text"], bytes.fromhex(row["h"]), row["source"]))
        return header, records

    def _append_records(self, records: list[tuple[str, bytes, str]]) -> int:
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.shard_path.open("ab") as f:
            for text, _, _ in records:
                payload = text.encode("utf-8") + b"\n\n"
                f.write(payload)
                written += len(payload)
            f.flush()
            os.fsync(f.fileno())
        self.shard_size += written
        return written

    def _record_index_rows(
        self,
        records: list[tuple[str, bytes, str]],
        *,
        shard: Path,
        offset: int,
    ) -> list[tuple[bytes, str, str, int, int]]:
        rows = []
        cursor = offset
        for text, digest, source in records:
            length = len(text.encode("utf-8"))
            rows.append((digest, source, str(shard), cursor, length))
            cursor += length + 2
        return rows

    def _recover_pending(self) -> None:
        header, records = self._read_pending()
        shard = Path(header["shard"])
        offset = int(header["offset"])
        resolved = shard.resolve()
        if self.shard_dir.resolve() not in resolved.parents:
            raise RuntimeError(f"unsafe pending shard path: {shard}")
        if shard.exists():
            size = shard.stat().st_size
            if size < offset:
                raise RuntimeError(f"pending shard is shorter than checkpoint: {size} < {offset}")
            if size > offset:
                with shard.open("r+b") as f:
                    f.truncate(offset)
        self.shard_path = shard
        match = re.search(r"(\d+)\.txt$", shard.name)
        self.shard_index = int(match.group(1)) if match else self.shard_index
        self.shard_size = offset
        self.conn.executemany("INSERT OR IGNORE INTO seen(h) VALUES (?)", [(d,) for _, d, _ in records])
        self.conn.executemany(
            "INSERT OR IGNORE INTO v5_records(h,source,shard,byte_offset,byte_length) VALUES(?,?,?,?,?)",
            self._record_index_rows(records, shard=shard, offset=offset),
        )
        self.conn.commit()
        self._append_records(records)
        self.pending_path.unlink()
        print(f"[recovery] replayed {len(records)} pending records", flush=True)

    def add_many(self, records: list[tuple[str, str]]) -> tuple[int, int, int]:
        """Return (written records, duplicate records, bytes)."""

        candidates: dict[bytes, tuple[str, bytes, str]] = {}
        duplicates = 0
        for text, source in records:
            digest = legacy_hash_text(text)
            if digest in candidates:
                duplicates += 1
                continue
            candidates[digest] = (text, digest, source)
        digests = list(candidates)
        existing = self._batch_existing(self.conn, digests)
        if self.dry_run and self.reference_lookup is not None:
            existing.update(self._batch_existing(self.reference_lookup, digests))
        duplicates += len(existing)
        unique = [record for digest, record in candidates.items() if digest not in existing]
        if not unique:
            return 0, duplicates, 0
        batch_bytes = sum(len(text.encode("utf-8")) + 2 for text, _, _ in unique)
        self._check_storage(batch_bytes)
        self._rotate_if_needed(batch_bytes)
        if self.dry_run:
            self.conn.executemany("INSERT INTO seen(h) VALUES (?)", [(d,) for _, d, _ in unique])
            self.conn.commit()
            return len(unique), duplicates, batch_bytes
        self._write_pending(unique)
        self.conn.executemany("INSERT INTO seen(h) VALUES (?)", [(d,) for _, d, _ in unique])
        self.conn.executemany(
            "INSERT INTO v5_records(h,source,shard,byte_offset,byte_length) VALUES(?,?,?,?,?)",
            self._record_index_rows(unique, shard=self.shard_path, offset=self.shard_size),
        )
        self.conn.commit()
        written = self._append_records(unique)
        self.total_output_bytes += written
        self.pending_path.unlink()
        return len(unique), duplicates, written

    def source_done(self, name: str) -> bool:
        row = self.conn.execute("SELECT payload_json FROM v5_source_stats WHERE name=?", (name,)).fetchone()
        if not row:
            return False
        try:
            return bool(json.loads(row[0]).get("completed"))
        except json.JSONDecodeError:
            return False

    def save_stats(self, stats: SourceStats) -> None:
        payload = json.dumps(stats.serializable(), ensure_ascii=False, sort_keys=True)
        self.conn.execute(
            "INSERT OR REPLACE INTO v5_source_stats(name,payload_json,updated_at) VALUES(?,?,?)",
            (stats.name, payload, time.time()),
        )
        self.conn.commit()
        self.export_stats()

    def export_stats(self) -> None:
        rows = [json.loads(row[0]) for row in self.conn.execute("SELECT payload_json FROM v5_source_stats ORDER BY name")]
        path = self.report_dir / "source_stats.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if rows:
            csv_path = self.report_dir / "source_stats.csv"
            fields = [key for key in rows[0] if key != "rejection_reasons"] + ["rejection_reasons"]
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    row = dict(row)
                    row["rejection_reasons"] = json.dumps(row["rejection_reasons"], ensure_ascii=False)
                    writer.writerow(row)

    def close(self) -> None:
        self.export_manifest()
        if self.reference_lookup is not None:
            self.reference_lookup.close()
        self.conn.close()

    def export_manifest(self) -> None:
        rows = []
        for path in sorted(self.shard_dir.glob("corpus_v5_increment_*.txt")):
            rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        manifest = self.manifest_dir / "shard_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
        (self.output_dir / "shards.txt").write_text(
            "".join(row["path"] + "\n" for row in rows), encoding="utf-8"
        )


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_inventory(source: SourceSpec) -> list[dict[str, Any]]:
    rows = []
    for path in discover_files(source):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def source_inventory_fingerprint(source: SourceSpec) -> str:
    return json_sha256(source_inventory(source))


def config_fingerprint(config: dict[str, Any]) -> str:
    return json_sha256(config)


def load_validated_fingerprint(
    output_dir: Path,
    sources: Sequence[SourceSpec],
    config: dict[str, Any],
    reference_db: Path,
) -> dict[str, str]:
    path = output_dir / "manifests" / "validation_manifest.json"
    if not path.exists():
        raise RuntimeError(f"validation manifest is missing; run --validate-only first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != config_fingerprint(config):
        raise RuntimeError("config changed after validation; run --validate-only again")
    current_reference = {
        "path": str(reference_db.resolve()),
        "bytes": reference_db.stat().st_size,
        "mtime_ns": reference_db.stat().st_mtime_ns,
    }
    if payload.get("reference") != current_reference:
        raise RuntimeError("reference dedupe DB changed after validation; run --validate-only again")
    current_reference_sha256 = file_sha256(reference_db)
    if payload.get("reference_sha256") != current_reference_sha256:
        raise RuntimeError("reference dedupe DB content hash changed after validation; run --validate-only again")
    validated = {row["source"]: row for row in payload.get("sources", [])}
    selected_fingerprints: dict[str, str] = {}
    for source in sources:
        row = validated.get(source.name)
        if row is None or row.get("invalid") or row.get("files", 0) <= 0:
            raise RuntimeError(f"source is not fully validated: {source.name}")
        fingerprint = source_inventory_fingerprint(source)
        if row.get("inventory_sha256") != fingerprint:
            raise RuntimeError(f"source inventory changed after validation: {source.name}")
        selected_fingerprints[source.name] = fingerprint
    return {
        "config_sha256": payload["config_sha256"],
        "reference_sha256": payload["reference_sha256"],
        "inputs_sha256": json_sha256(selected_fingerprints),
    }


def discover_files(source: SourceSpec) -> list[Path]:
    root = Path(source.root)
    if source.kind == "jsonl_gz":
        if source.include_dirs:
            files: list[Path] = []
            pattern = source.pattern or "*.jsonl.gz"
            for directory in source.include_dirs:
                candidate = root / directory
                if candidate.is_file():
                    files.append(candidate)
                elif candidate.is_dir():
                    files.extend(candidate.rglob(pattern))
            return sorted(
                path for path in files
                if path.is_file()
                and path.suffix.casefold() != ".incomplete"
                and not any(part.casefold() in {".cache", "cache", "temp_hf_dl"} for part in path.parts)
            )
        if source.pattern:
            return sorted(root.glob(source.pattern))
        return [root] if root.is_file() else sorted(root.glob("*.jsonl.gz"))
    pattern = source.pattern or "**/*.parquet"
    files = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        lowered = str(path).casefold()
        if ".cache" in lowered or path.suffix.casefold() == ".incomplete":
            continue
        files.append(path)
    return sorted(files)


def validate_parquet(path: Path, *, deep: bool = False) -> tuple[bool, str]:
    if pq is None:
        return False, "pyarrow_missing"
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows <= 0:
            return False, "empty_parquet"
        if deep:
            schema_names = set(parquet.schema_arrow.names)
            probe_columns = [name for name in ("text", "content") if name in schema_names]
            if not probe_columns:
                return False, "missing_text_column"
            for name in probe_columns:
                field_type = parquet.schema_arrow.field(name).type
                if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
                    return False, f"non_string_text_column:{name}:{field_type}"
            row_count = 0
            for batch in parquet.iter_batches(batch_size=8192, columns=probe_columns, use_threads=True):
                row_count += batch.num_rows
            if row_count != parquet.metadata.num_rows:
                return False, f"row_count_mismatch:{row_count}!={parquet.metadata.num_rows}"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_jsonl_gzip(path: Path, *, deep: bool = False) -> tuple[bool, str]:
    """Validate gzip CRC and, in deep mode, every non-empty JSONL record."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if deep:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        return False, f"invalid_json_line_{line_no}: {exc}"
        return True, ""
    except (EOFError, gzip.BadGzipFile, UnicodeDecodeError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def iter_parquet_records(path: Path, source: SourceSpec, batch_size: int) -> Iterator[RawRecord]:
    parquet = pq.ParquetFile(path)
    schema_names = set(parquet.schema_arrow.names)
    wanted = {source.text_field, "text", "content", "path", "license", "language_score", "score", "int_score", "token_count", "id", "url"}
    columns = [name for name in wanted if name in schema_names]
    if source.text_field not in columns and "text" not in columns and "content" not in columns:
        raise ValueError(f"no text column in {path}: {sorted(schema_names)}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=True):
        for row in batch.to_pylist():
            text = row.get(source.text_field)
            if text is None:
                text = row.get("text") if row.get("text") is not None else row.get("content")
            if isinstance(text, str) and text:
                yield RawRecord(text, row)


def _format_choices(choices: Any) -> tuple[str, dict[str, str]]:
    if not isinstance(choices, dict):
        return "", {}
    texts = choices.get("text")
    labels = choices.get("label")
    if not isinstance(texts, list) or not isinstance(labels, list):
        return "", {}
    pairs = [(str(label), str(text)) for label, text in zip(labels, texts)]
    return "\n".join(f"{label}. {text}" for label, text in pairs), dict(pairs)


def extract_json_text(obj: Any, text_field: str, formatter: str = "generic") -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    if formatter in {"qasc", "arc"}:
        question = str(obj.get("formatted_question") or obj.get("question") or "").strip()
        choices_text, choice_map = _format_choices(obj.get("choices"))
        answer_key = str(obj.get("answerKey") or "").strip()
        answer = choice_map.get(answer_key, answer_key)
        parts = [f"Question: {question}"]
        if choices_text:
            parts.append(f"Choices:\n{choices_text}")
        if answer:
            parts.append(f"Answer: {answer_key}. {answer}" if answer_key and answer != answer_key else f"Answer: {answer}")
        if formatter == "qasc":
            facts = [str(obj.get(key) or "").strip() for key in ("fact1", "fact2", "combinedfact")]
            facts = [fact for fact in facts if fact]
            if facts:
                parts.append("Supporting facts:\n" + "\n".join(facts))
        return "\n\n".join(part for part in parts if part.strip())
    if formatter == "boolq":
        passage = str(obj.get("passage") or "").strip()
        question = str(obj.get("question") or "").strip()
        answer = obj.get("answer")
        return f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer: {'Yes' if answer is True else 'No' if answer is False else answer}"
    if formatter == "squad":
        answers = obj.get("answers")
        answer_values = answers.get("text", []) if isinstance(answers, dict) else []
        answer_text = " | ".join(dict.fromkeys(str(value) for value in answer_values if value))
        return "\n\n".join(
            part for part in (
                f"Title: {obj.get('title', '')}" if obj.get("title") else "",
                f"Context: {obj.get('context', '')}",
                f"Question: {obj.get('question', '')}",
                f"Answer: {answer_text}",
            ) if part.strip(": ")
        )
    if formatter == "hotpot_qa":
        context = obj.get("context")
        context_parts: list[str] = []
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences = context.get("sentences", [])
            if isinstance(titles, list) and isinstance(sentences, list):
                for title, rows in zip(titles, sentences):
                    body = " ".join(str(row) for row in rows) if isinstance(rows, list) else str(rows)
                    context_parts.append(f"{title}: {body}")
        return "\n\n".join(
            part for part in (
                f"Question: {obj.get('question', '')}",
                "Context:\n" + "\n".join(context_parts) if context_parts else "",
                f"Answer: {obj.get('answer', '')}",
            ) if part.strip(": \n")
        )
    if formatter == "dolly":
        return "\n\n".join(
            part for part in (
                f"Instruction: {obj.get('instruction', '')}",
                f"Context: {obj.get('context', '')}" if obj.get("context") else "",
                f"Response: {obj.get('response', '')}",
            ) if part.strip(": ")
        )
    candidates = [text_field, "text", "content", "article", "document", "body"]
    for key in candidates:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    ordered_fields = [
        "instruction",
        "INSTRUCTION",
        "question",
        "prompt",
        "input",
        "context",
        "response",
        "RESPONSE",
        "answer",
        "output",
        "target",
        "label",
        "summary",
    ]
    parts = []
    for key in ordered_fields:
        value = obj.get(key)
        if isinstance(value, str) and value and value not in parts:
            parts.append(value)
    if parts:
        return "\n\n".join(parts)
    return ""


def iter_jsonl_gz_records(path: Path, source: SourceSpec) -> Iterator[RawRecord]:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            text = extract_json_text(obj, source.text_field, source.json_formatter)
            if text:
                metadata = obj if isinstance(obj, dict) else {"line": line_no}
                yield RawRecord(text, metadata)


def record_allowed_by_metadata(record: RawRecord, source: SourceSpec) -> str:
    for field_name in source.exclude_if_true:
        if record.metadata.get(field_name) is True:
            return f"metadata_excluded:{field_name}"
    if source.language_score_min is not None:
        score = record.metadata.get("language_score")
        if score is None:
            return "language_score_missing"
        try:
            score = float(score)
        except (TypeError, ValueError):
            return "language_score_invalid"
        if score < source.language_score_min:
            return "language_score"
    if source.edu_score_min is not None:
        score = record.metadata.get("int_score", record.metadata.get("score"))
        if score is None:
            return "edu_score_missing"
        try:
            score = float(score)
        except (TypeError, ValueError):
            return "edu_score_invalid"
        if score < source.edu_score_min:
            return "edu_score"
    return ""


def process_source(
    source: SourceSpec,
    store: OutputStore,
    reviewer: LMStudioReviewer | None,
    *,
    parquet_batch_size: int,
    output_batch_size: int,
    max_records: int | None,
    max_chunk_chars: int,
    qwen_audit_fraction: float,
    qwen_max_per_source: int,
) -> SourceStats:
    stats = SourceStats(source.name, source.category)
    started = time.time()
    files = discover_files(source)
    stats.files_discovered = len(files)
    output_buffer: list[tuple[str, str]] = []
    review_buffer: list[tuple[ReviewItem, SourceSpec]] = []
    qwen_group_capacity = reviewer.workers * reviewer.batch_size if reviewer else 0
    source_limit = int(source.max_output_gib * 1024**3) if source.max_output_gib else None
    stopped_by_source_limit = False
    stopped_by_global_limit = False
    review_sequence = 0
    review_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if reviewer else None
    pending_review: tuple[concurrent.futures.Future[dict[str, ReviewDecision]], list[tuple[ReviewItem, SourceSpec]]] | None = None

    def flush_output() -> None:
        nonlocal output_buffer
        if not output_buffer:
            return
        stats.chunks_candidate += len(output_buffer)
        written, duplicate, written_bytes = store.add_many(output_buffer)
        stats.chunks_written += written
        stats.chunks_duplicate += duplicate
        stats.output_bytes += written_bytes
        output_buffer = []

    def queue_chunks(chunks: Sequence[str]) -> None:
        for chunk in chunks:
            output_buffer.append((chunk, source.name))
            if len(output_buffer) >= output_batch_size:
                flush_output()

    def consume_reviews(
        batch: list[tuple[ReviewItem, SourceSpec]],
        decisions: dict[str, ReviewDecision],
    ) -> None:
        for item, spec in batch:
            decision = decisions.get(item.doc_id)
            if decision is None or not decision.valid:
                stats.qwen_fallback += 1
                chunks = deterministic_chunks(item.text, max_chars=max_chunk_chars, min_chars=spec.min_chars, code=spec.code)
            else:
                segments = apply_review(item, decision, spec.min_chars)
                if segments:
                    stats.qwen_kept += 1
                else:
                    stats.qwen_dropped += 1
                chunks = []
                for segment in segments:
                    chunks.extend(deterministic_chunks(segment, max_chars=max_chunk_chars, min_chars=spec.min_chars, code=spec.code))
            queue_chunks(chunks)

    def flush_reviews(*, wait: bool = True) -> None:
        nonlocal review_buffer, pending_review
        if pending_review is not None:
            future, batch = pending_review
            if wait or future.done() or len(review_buffer) >= qwen_group_capacity:
                consume_reviews(batch, future.result())
                pending_review = None
        if not review_buffer:
            return
        if reviewer is None or reviewer.remaining <= 0:
            for item, spec in review_buffer:
                queue_chunks(deterministic_chunks(item.text, max_chars=max_chunk_chars, min_chars=spec.min_chars, code=spec.code))
            review_buffer = []
            return
        if pending_review is not None:
            return
        batch = review_buffer
        review_buffer = []
        items = [item for item, _ in batch]
        stats.qwen_sent += min(len(items), reviewer.remaining)
        assert review_executor is not None
        pending_review = (review_executor.submit(reviewer.review_many, items), batch)
        if wait:
            future, submitted_batch = pending_review
            consume_reviews(submitted_batch, future.result())
            pending_review = None

    for path in files:
        if source.kind == "parquet":
            valid, reason = validate_parquet(path)
            if not valid:
                stats.files_invalid += 1
                stats.rejection_reasons[f"invalid_file:{reason}"] += 1
                continue
            stats.files_valid += 1
            iterator = iter_parquet_records(path, source, parquet_batch_size)
        elif source.kind == "jsonl_gz":
            valid, reason = validate_jsonl_gzip(path, deep=False)
            if not valid:
                stats.files_invalid += 1
                stats.rejection_reasons[f"invalid_file:{reason}"] += 1
                continue
            stats.files_valid += 1
            iterator = iter_jsonl_gz_records(path, source)
        else:
            stats.files_invalid += 1
            stats.rejection_reasons["unsupported_kind"] += 1
            continue
        try:
            for record in iterator:
                stats.records_seen += 1
                stats.input_bytes += len(record.text.encode("utf-8", errors="ignore"))
                if max_records is not None and stats.records_seen > max_records:
                    stats.stopped_by_limit = True
                    stopped_by_global_limit = True
                    break
                metadata_reason = record_allowed_by_metadata(record, source)
                if metadata_reason:
                    stats.records_rejected += 1
                    stats.rejection_reasons[metadata_reason] += 1
                    continue
                clean = clean_record(record, source)
                if clean.text is None:
                    stats.records_rejected += 1
                    stats.rejection_reasons[clean.reason] += 1
                    continue
                digest = legacy_hash_text(clean.text)
                if hash_fraction(digest) >= source.sample_rate:
                    stats.rejection_reasons["deterministic_sample"] += 1
                    continue
                stats.records_sampled += 1
                stats.records_clean += 1
                prechunks = deterministic_chunks(
                    clean.text,
                    max_chars=min(max_chunk_chars, 24_000),
                    min_chars=source.min_chars,
                    code=source.code,
                )
                for chunk in prechunks:
                    requires_review = source.qwen_review and needs_qwen_review(
                        chunk, clean, qwen_audit_fraction, code=source.code
                    )
                    if requires_review:
                        stats.qwen_eligible += 1
                    source_review_slots = qwen_max_per_source - stats.qwen_sent - len(review_buffer)
                    if requires_review and reviewer and reviewer.remaining > 0 and source_review_slots > 0:
                        review_sequence += 1
                        doc_id = f"{legacy_hash_text(chunk).hex()}-{review_sequence:012d}"
                        review_buffer.append((ReviewItem(doc_id, chunk, source.name, split_review_units(chunk), source.code), source))
                        if len(review_buffer) >= qwen_group_capacity:
                            flush_reviews(wait=False)
                    else:
                        if requires_review:
                            stats.qwen_cap_skipped += 1
                        queue_chunks([chunk])
                if source_limit is not None and stats.output_bytes >= source_limit:
                    stats.stopped_by_limit = True
                    stopped_by_source_limit = True
                    break
            if stats.stopped_by_limit:
                break
        except STREAM_EXCEPTIONS as exc:
            stats.files_invalid += 1
            stats.files_valid = max(0, stats.files_valid - 1)
            stats.rejection_reasons[f"stream_error:{type(exc).__name__}"] += 1
            print(f"[warn] {source.name} {path}: {type(exc).__name__}: {exc}", flush=True)
            continue
        finally:
            flush_reviews()
            flush_output()
        print(
            f"[{source.name}] seen={stats.records_seen:,} sampled={stats.records_sampled:,} "
            f"written={stats.chunks_written:,} dup={stats.chunks_duplicate:,} "
            f"out={stats.output_bytes/1024**3:.2f}GiB qwen={stats.qwen_sent:,}",
            flush=True,
        )
    flush_reviews()
    flush_output()
    if review_executor is not None:
        review_executor.shutdown(wait=True)
    stats.completed = stats.files_discovered > 0 and stats.files_invalid == 0 and not stopped_by_global_limit and (
        not stats.stopped_by_limit or stopped_by_source_limit
    )
    stats.elapsed_seconds = time.time() - started
    store.save_stats(stats)
    return stats


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "sources" not in data or not isinstance(data["sources"], list):
        raise ValueError("config must contain a sources list")
    return data


def write_run_metadata(output_dir: Path, config_path: Path, config: dict[str, Any], args: argparse.Namespace) -> None:
    payload = {
        "created_at": time.time(),
        "config_path": str(config_path.resolve()),
        "config": config,
        "args": vars(args),
        "python": sys.version,
        "platform": sys.platform,
        "lm_studio": {
            "base_url": config.get("qwen", {}).get("base_url"),
            "model": config.get("qwen", {}).get("model"),
        },
    }
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests" / "resolved_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance = {
        "sources": [
            {
                "name": row.get("name"),
                "root": row.get("root"),
                "source_url": row.get("source_url", ""),
                "license_status": row.get("license_status", "unknown"),
                "notes": row.get("notes", ""),
            }
            for row in config.get("sources", [])
        ],
        "excluded_sources": config.get("excluded_sources", []),
    }
    (output_dir / "manifests" / "source_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_final_report(output_dir: Path, store: OutputStore, elapsed: float) -> None:
    stats_path = output_dir / "reports" / "source_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else []
    total_bytes = sum(int(row.get("output_bytes", 0)) for row in stats)
    total_records = sum(int(row.get("chunks_written", 0)) for row in stats)
    total_duplicates = sum(int(row.get("chunks_duplicate", 0)) for row in stats)
    local_count = int(store.conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
    current_count = local_count + store.reference_count if store.dry_run else local_count
    report = [
        "# FLUED corpus_v5 增量构建报告",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 基线精确哈希数：{store.reference_count:,}",
        f"- 当前精确哈希数：{current_count:,}",
        f"- 新增唯一条目：{current_count - store.reference_count:,}",
        f"- 实际写入条目：{total_records:,}",
        f"- 拦截精确重复：{total_duplicates:,}",
        f"- 输出体积：{total_bytes / 1024**3:.3f} GiB",
        f"- 总耗时：{elapsed / 3600:.2f} 小时",
        "",
        "## 去重口径",
        "",
        "输出使用与 corpus_v4 完全相同的空白折叠、小写化和 blake2b-128 哈希合同。",
        "状态库由 corpus_v4 的只读数据库通过 SQLite backup 复制，因此覆盖 E 盘 corpus_v3 和 N 盘 corpus_v4 中已纳入的全部合格条目。",
        "本文所称零重复是规范化文本级精确重复；语义近重复另行抽样审计，不作数学上的零重复承诺。",
        "",
        "## 来源统计",
        "",
        "| 来源 | seen | sampled | written | duplicate | output GiB | Qwen eligible/sent/fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        report.append(
            f"| {row['name']} | {int(row['records_seen']):,} | {int(row['records_sampled']):,} | "
            f"{int(row['chunks_written']):,} | {int(row['chunks_duplicate']):,} | "
            f"{float(row['output_gib']):.3f} | {int(row['qwen_eligible']):,}/"
            f"{int(row['qwen_sent']):,}/{int(row['qwen_fallback']):,} |"
        )
    (output_dir / "reports" / "FINAL_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--reference-db", default="")
    parser.add_argument("--only", default="", help="Comma-separated source names")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-unlock", action="store_true", help="Remove a stale RUNNING.lock after verifying no build is active")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(args.output_dir or config["output_dir"])
    reference_db = Path(args.reference_db or config["reference_db"])
    if not reference_db.exists():
        raise FileNotFoundError(reference_db)
    selected = {name.strip() for name in args.only.split(",") if name.strip()}
    all_sources = [SourceSpec(**row) for row in config["sources"]]
    all_sources = [source for source in all_sources if source.enabled]
    sources = [source for source in all_sources if not selected or source.name in selected]
    sources.sort(key=lambda source: (source.priority, source.name))
    if args.validate_only:
        manifest_path = output_dir / "manifests" / "validation_manifest.json"
        cache_path = output_dir / "manifests" / "validation_file_cache.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        reference = {
            "path": str(reference_db.resolve()),
            "bytes": reference_db.stat().st_size,
            "mtime_ns": reference_db.stat().st_mtime_ns,
        }
        existing: dict[str, Any] = {}
        if manifest_path.exists():
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            if candidate.get("config_sha256") == config_fingerprint(config) and candidate.get("reference") == reference:
                existing = candidate
        reference_sha256 = existing.get("reference_sha256") or file_sha256(reference_db)
        prior_rows = {row["source"]: row for row in existing.get("sources", [])}
        validation_cache: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        validation_cache[row["cache_key"]] = row
                    except (json.JSONDecodeError, KeyError):
                        continue
        rows = []
        for source in sources:
            files = discover_files(source)
            valid = invalid = 0
            errors = []
            results: dict[str, tuple[bool, str]] = {}
            pending: list[tuple[Path, str]] = []
            for path in files:
                stat = path.stat()
                cache_key = json_sha256({"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                cached = validation_cache.get(cache_key)
                if cached is not None:
                    results[str(path)] = (bool(cached["ok"]), str(cached.get("reason", "")))
                else:
                    pending.append((path, cache_key))

            def validate_one(path: Path) -> tuple[bool, str]:
                if source.kind == "parquet":
                    return validate_parquet(path, deep=True)
                if source.kind == "jsonl_gz":
                    return validate_jsonl_gzip(path, deep=True)
                return False, "unsupported_kind"

            with concurrent.futures.ThreadPoolExecutor(max_workers=int(config.get("validation_workers", 4))) as pool:
                futures = {pool.submit(validate_one, path): (path, cache_key) for path, cache_key in pending}
                completed = len(files) - len(pending)
                for future in concurrent.futures.as_completed(futures):
                    path, cache_key = futures[future]
                    try:
                        ok, reason = future.result()
                    except Exception as exc:
                        ok, reason = False, f"validator_error:{type(exc).__name__}:{exc}"
                    completed += 1
                    results[str(path)] = (ok, reason)
                    cache_row = {
                        "cache_key": cache_key,
                        "source": source.name,
                        "path": str(path),
                        "ok": ok,
                        "reason": reason,
                        "validated_at": time.time(),
                    }
                    with cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(cache_row, ensure_ascii=False) + "\n")
                        f.flush()
                    print(f"[validate] {source.name} {completed}/{len(files)} {'OK' if ok else 'FAIL'} {path.name}", flush=True)

            for path in files:
                ok, reason = results[str(path)]
                valid += int(ok)
                invalid += int(not ok)
                if not ok:
                    errors.append({"path": str(path), "reason": reason})
            rows.append(
                {
                    "source": source.name,
                    "files": len(files),
                    "valid": valid,
                    "invalid": invalid,
                    "inventory_sha256": source_inventory_fingerprint(source),
                    "errors": errors,
                }
            )
            print(f"{source.name}: files={len(files)} valid={valid} invalid={invalid}")
        for row in rows:
            prior_rows[row["source"]] = row
        payload = {
            "created_at": time.time(),
            "config_sha256": config_fingerprint(config),
            "reference": reference,
            "reference_sha256": reference_sha256,
            "sources": sorted(prior_rows.values(), key=lambda row: row["source"]),
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 1 if any(row["invalid"] or row["files"] <= 0 for row in rows) else 0

    run_fingerprint = (
        {
            "config_sha256": config_fingerprint(config),
            "reference_sha256": "dry-run",
            "inputs_sha256": "dry-run",
        }
        if args.dry_run
        else load_validated_fingerprint(output_dir, all_sources, config, reference_db)
    )
    run_fingerprint["pipeline_sha256"] = file_sha256(Path(__file__))
    lock = RunLock(output_dir, enabled=not args.dry_run)
    if args.force_unlock:
        lock.remove_if_stale()
    lock.acquire()
    store: OutputStore | None = None
    qwen_config = config.get("qwen", {})
    reviewer = None
    exit_code = 0
    try:
        if not args.no_qwen and bool(qwen_config.get("enabled", True)):
            reviewer = LMStudioReviewer(
                base_url=qwen_config.get("base_url", "http://127.0.0.1:1234"),
                model=qwen_config.get("model", "qwen/qwen3.5-9b"),
                workers=int(qwen_config.get("workers", 8)),
                batch_size=int(qwen_config.get("batch_size", 4)),
                max_reviews=int(qwen_config.get("max_reviews", 50_000)),
                timeout=float(qwen_config.get("timeout", 180.0)),
                audit_log=output_dir / "logs" / "qwen_audit.jsonl",
            )
        store = OutputStore(
            output_dir,
            reference_db,
            shard_gib=float(config.get("shard_gib", 4.0)),
            dry_run=args.dry_run,
            max_output_gib=float(config.get("max_output_gib", 300.0)),
            min_free_gib=float(config.get("min_free_gib", 80.0)),
            run_fingerprint=run_fingerprint,
        )
        write_run_metadata(output_dir, config_path, config, args)
        started = time.time()
        for source in sources:
            if store.source_done(source.name) and not args.dry_run:
                print(f"[skip completed] {source.name}")
                continue
            print(
                f"[source] {source.name} kind={source.kind} category={source.category} "
                f"sample={source.sample_rate:.5f}",
                flush=True,
            )
            process_source(
                source,
                store,
                reviewer,
                parquet_batch_size=int(config.get("parquet_batch_size", 1024)),
                output_batch_size=int(config.get("output_batch_size", 1000)),
                max_records=args.max_records_per_source,
                max_chunk_chars=int(config.get("max_chunk_chars", 32_768)),
                qwen_audit_fraction=float(qwen_config.get("audit_fraction", 0.00005)),
                qwen_max_per_source=int(qwen_config.get("max_reviews_per_source", 10_000)),
            )
        build_final_report(output_dir, store, time.time() - started)
    except StorageLimitReached as exc:
        print(f"[safe-stop] {exc}", flush=True)
        exit_code = 2
        if store is not None:
            build_final_report(output_dir, store, time.time() - started)
    finally:
        if store is not None:
            store.close()
        lock.release()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
