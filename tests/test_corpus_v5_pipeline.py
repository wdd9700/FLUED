from pathlib import Path
import gzip
import json
import os
import sqlite3

from build_corpus_v5 import (
    CleanResult,
    RawRecord,
    SourceSpec,
    apply_review,
    clean_record,
    code_quality_reason,
    deterministic_chunks,
    discover_files,
    extract_json_text,
    legacy_hash_text,
    iter_jsonl_gz_records,
    needs_qwen_review,
    normalize_text,
    normalize_code_text,
    OutputStore,
    RunLock,
    ReviewDecision,
    ReviewItem,
    record_allowed_by_metadata,
    split_review_units,
)


def test_legacy_hash_ignores_case_newlines_and_spaces():
    assert legacy_hash_text("Hello\nworld") == legacy_hash_text("  hello   world  ")


def test_normalize_keeps_math_and_uses_nfc():
    value = normalize_text("e\u0301 = x²\r\n\r\n  α + β  ")
    assert value == "é = x²\n\nα + β"


def test_corrupt_text_is_rejected():
    source = SourceSpec("bad", "web", "jsonl_gz", "x")
    result = clean_record(RawRecord("这是" + "�" * 20 + "损坏文本" * 20, {}), source)
    assert result.text is None
    assert result.reason == "encoding_corruption"


def test_code_vendor_and_secret_filters():
    text = "def f():\n    return 1\n" * 20
    assert code_quality_reason(text, {"path": "node_modules/a.py", "license": "mit"}) == "code_vendored_or_generated"
    secret = 'api_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"\n' + text
    assert code_quality_reason(secret, {"path": "src/a.py", "license": "mit"}) == "code_secret"
    assert code_quality_reason(text, {"path": "src/a.py"}) == "code_license_missing"


def test_code_permissive_file_passes():
    text = "def f(value):\n    return value + 1\n" * 20
    assert code_quality_reason(text, {"path": "src/a.py", "license": "apache-2.0"}) == ""


def test_code_normalization_preserves_indentation():
    text = "def f():\r\n    return 1  \r\n"
    assert normalize_code_text(text) == "def f():\n    return 1"


def test_deterministic_chunks_respect_limit():
    text = "\n\n".join((f"Paragraph {i}. " + "word " * 80) for i in range(20))
    chunks = deterministic_chunks(text, max_chars=1200, min_chars=100, code=False)
    assert len(chunks) > 1
    assert all(100 <= len(chunk) <= 1200 for chunk in chunks)


def test_review_units_split_long_chinese_without_inventing_characters():
    text = ("第一句用于测试。第二句仍然保留原字节！" * 300)
    units = split_review_units(text, target_chars=300)
    assert len(units) > 1
    assert "".join(units) == text


def test_review_only_selects_existing_paragraphs():
    paragraphs = ["first paragraph " * 20, "cookie settings " * 20, "last paragraph " * 20]
    item = ReviewItem("a", "\n\n".join(paragraphs), "test", paragraphs)
    decision = ReviewDecision("a", True, 0.9, "drop boilerplate", [1], [0])
    segments = apply_review(item, decision, min_chars=20)
    assert segments == [paragraphs[0].strip(), paragraphs[2].strip()]


def test_qwen_only_reviews_suspicious_or_audited_text():
    text = "\n\n".join(["This is a coherent paragraph with enough words for deterministic processing."] * 20)
    assert not needs_qwen_review(text, CleanResult(text, suspicious=False), 0.0, code=False)
    assert needs_qwen_review(text, CleanResult(text, suspicious=True), 0.0, code=False)


def test_qwen_cannot_delete_complete_record():
    text = "First useful paragraph.\n\nSecond useful paragraph."
    item = ReviewItem("id", text, "test", ["First useful paragraph.", "Second useful paragraph."], False)
    decision = ReviewDecision("id", False, 0.0, "drop", [], [], True)
    assert apply_review(item, decision, 10) == [text]


def test_invalid_jsonl_line_is_not_silently_skipped(tmp_path: Path):
    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"text": "valid"}) + "\n")
        f.write("{broken}\n")
    source = SourceSpec("bad", "stem", "jsonl_gz", str(tmp_path))
    iterator = iter_jsonl_gz_records(path, source)
    assert next(iterator).text == "valid"
    try:
        next(iterator)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid JSONL line was silently skipped")


def test_qa_formatters_preserve_context_choices_and_answers():
    qasc = extract_json_text(
        {
            "question": "What forms clouds?",
            "choices": {"label": ["A", "B"], "text": ["water", "stone"]},
            "answerKey": "A",
            "fact1": "Clouds contain water.",
        },
        "text",
        "qasc",
    )
    assert "A. water" in qasc and "Answer: A. water" in qasc and "Clouds contain water" in qasc
    boolq = extract_json_text(
        {"passage": "A factual passage.", "question": "Is it factual?", "answer": True},
        "text",
        "boolq",
    )
    assert "Passage: A factual passage." in boolq and "Answer: Yes" in boolq


def test_metadata_true_exclusion_is_enforced():
    source = SourceSpec("oasst", "dialogue", "jsonl_gz", "x", exclude_if_true=["deleted"])
    assert record_allowed_by_metadata(RawRecord("text", {"deleted": True}), source) == "metadata_excluded:deleted"
    assert record_allowed_by_metadata(RawRecord("text", {"deleted": False}), source) == ""


def test_run_lock_detects_current_process():
    assert RunLock._pid_exists(os.getpid())


def test_discover_jsonl_include_dirs(tmp_path: Path):
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / f"{name}.jsonl.gz").write_bytes(b"placeholder")
    source = SourceSpec(
        "curated", "stem", "jsonl_gz", str(tmp_path),
        pattern="*.jsonl.gz", include_dirs=["a", "b"],
    )
    assert [path.name for path in discover_files(source)] == ["a.jsonl.gz", "b.jsonl.gz"]


def test_output_store_indexes_written_records(tmp_path: Path):
    reference = tmp_path / "reference.sqlite3"
    conn = sqlite3.connect(reference)
    conn.execute("CREATE TABLE seen (h BLOB PRIMARY KEY)")
    conn.execute("INSERT INTO seen(h) VALUES (?)", (legacy_hash_text("already present"),))
    conn.commit()
    conn.close()
    fingerprint = {"config_sha256": "a", "reference_sha256": "b", "inputs_sha256": "c"}
    store = OutputStore(
        tmp_path / "out", reference, shard_gib=0.01, dry_run=False,
        max_output_gib=1, min_free_gib=0, run_fingerprint=fingerprint,
    )
    written, duplicate, _ = store.add_many(
        [("already present", "test"), ("new record one", "test"), ("new record two", "test")]
    )
    assert (written, duplicate) == (2, 1)
    rows = list(store.conn.execute("SELECT source,byte_offset,byte_length FROM v5_records ORDER BY byte_offset"))
    assert rows == [("test", 0, 14), ("test", 16, 14)]
    store.close()


def test_output_store_refuses_unindexed_shard_tail(tmp_path: Path):
    reference = tmp_path / "reference.sqlite3"
    conn = sqlite3.connect(reference)
    conn.execute("CREATE TABLE seen (h BLOB PRIMARY KEY)")
    conn.commit()
    conn.close()
    fingerprint = {"config_sha256": "a", "reference_sha256": "b", "inputs_sha256": "c"}
    output = tmp_path / "out"
    store = OutputStore(
        output, reference, shard_gib=0.01, dry_run=False,
        max_output_gib=1, min_free_gib=0, run_fingerprint=fingerprint,
    )
    store.add_many([("new record", "test")])
    shard = store.shard_path
    store.close()
    with shard.open("ab") as f:
        f.write(b"unindexed")
    try:
        OutputStore(
            output, reference, shard_gib=0.01, dry_run=False,
            max_output_gib=1, min_free_gib=0, run_fingerprint=fingerprint,
        )
    except RuntimeError as exc:
        assert "layout mismatch" in str(exc)
    else:
        raise AssertionError("resume accepted an unindexed shard tail")
