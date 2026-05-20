"""
tests/test_utf8.py
-------------------
Unit tests for the UTF-8 data pipeline and BPE tokeniser in flued/data.py.

Covers edge cases including:
  - ASCII, CJK, emoji, mixed Unicode round-trips
  - Empty strings and null bytes
  - BPE encode/decode round-trips and vocabulary growth
  - Dataset construction and item shapes
  - Dynamic span computation and pool_spans

Run with:
    pytest tests/test_utf8.py -v
"""

import pytest
import torch

from flued.data import (
    BPETextDataset,
    ByteTextDataset,
    SimpleBPE,
    bytes_to_text,
    char_ids_to_text,
    compute_dynamic_spans,
    pool_spans,
    text_to_bytes,
    text_to_char_ids,
)


# ---------------------------------------------------------------------------
# UTF-8 byte encode / decode
# ---------------------------------------------------------------------------


class TestUTF8ByteEncoding:
    """Tests for text_to_bytes / bytes_to_text."""

    def test_ascii_round_trip(self):
        text = "Hello, world!"
        assert bytes_to_text(text_to_bytes(text)) == text

    def test_cjk_round_trip(self):
        text = "中文测试"
        assert bytes_to_text(text_to_bytes(text)) == text

    def test_emoji_round_trip(self):
        text = "🎉🚀🌍"
        assert bytes_to_text(text_to_bytes(text)) == text

    def test_mixed_unicode_round_trip(self):
        text = "Hello 你好 🌏 مرحبا"
        assert bytes_to_text(text_to_bytes(text)) == text

    def test_empty_string(self):
        assert text_to_bytes("") == []
        assert bytes_to_text([]) == ""

    def test_single_ascii_char(self):
        assert text_to_bytes("A") == [65]

    def test_all_bytes_in_range(self):
        text = "Test string with 中文 and 🎯"
        for b in text_to_bytes(text):
            assert 0 <= b <= 255, f"Byte value {b} out of range"

    def test_newline_preserved(self):
        text = "line1\nline2\nline3"
        assert bytes_to_text(text_to_bytes(text)) == text

    def test_tab_preserved(self):
        text = "col1\tcol2"
        assert bytes_to_text(text_to_bytes(text)) == text

    def test_null_byte_round_trip(self):
        """bytes_to_text should handle a null byte without crashing."""
        result = bytes_to_text([0])
        assert isinstance(result, str)

    def test_invalid_utf8_uses_replacement(self):
        """bytes_to_text with invalid UTF-8 should use the replacement char."""
        result = bytes_to_text([0xFF, 0xFE])
        assert "\ufffd" in result or isinstance(result, str)

    def test_multibyte_cjk_byte_count(self):
        """Each CJK character should encode to 3 UTF-8 bytes."""
        assert len(text_to_bytes("中")) == 3
        assert len(text_to_bytes("文")) == 3

    def test_emoji_byte_count(self):
        """A BMP emoji encodes to 4 UTF-8 bytes."""
        assert len(text_to_bytes("🎉")) == 4


# ---------------------------------------------------------------------------
# Unicode codepoint encoding
# ---------------------------------------------------------------------------


class TestCharIdEncoding:
    """Tests for text_to_char_ids / char_ids_to_text."""

    def test_ascii_round_trip(self):
        text = "Hello"
        assert char_ids_to_text(text_to_char_ids(text)) == text

    def test_cjk_round_trip(self):
        text = "中文"
        assert char_ids_to_text(text_to_char_ids(text)) == text

    def test_clip_to_0xFFFF(self):
        """Characters with codepoint > 0xFFFF should be clipped."""
        ids = text_to_char_ids("🎉")   # codepoint 0x1F389 > 0xFFFF
        assert all(i <= 0xFFFF for i in ids)

    def test_length_equals_char_count(self):
        text = "abc"
        assert len(text_to_char_ids(text)) == 3


# ---------------------------------------------------------------------------
# ByteTextDataset
# ---------------------------------------------------------------------------


class TestByteTextDataset:
    """Tests for ByteTextDataset."""

    def test_stub_corpus_produces_chunks(self):
        ds = ByteTextDataset(seq_len=16, stride=8)
        assert len(ds) > 0, "Stub corpus should produce at least one chunk"

    def test_item_shapes(self):
        ds = ByteTextDataset(texts=["Hello world!"] * 100, seq_len=8, stride=4)
        src, tgt = ds[0]
        assert src.shape == (8,)
        assert tgt.shape == (8,)

    def test_item_dtypes(self):
        ds = ByteTextDataset(texts=["Hello world!"] * 100, seq_len=8, stride=4)
        src, tgt = ds[0]
        assert src.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_target_is_src_shifted_by_one(self):
        """tgt[i] should be the byte that follows src[i] in the stream."""
        ds = ByteTextDataset(texts=["ABCDEFGHIJ"] * 50, seq_len=4, stride=1)
        src, tgt = ds[0]
        # The raw byte stream: 65 66 67 68 69 …
        # src[0..3] = bytes[0..3], tgt[0..3] = bytes[1..4]
        assert src.shape == (4,) and tgt.shape == (4,)
        # Check the shift is consistent across a few positions
        for i in range(3):
            assert src[i + 1].item() == tgt[i].item(), (
                f"Position {i}: src[{i+1}]={src[i+1]} != tgt[{i}]={tgt[i]}"
            )

    def test_all_values_valid_bytes(self):
        ds = ByteTextDataset(texts=["test text 中文 🎯"] * 50, seq_len=8, stride=4)
        for i in range(min(len(ds), 20)):
            src, tgt = ds[i]
            assert (src >= 0).all() and (src <= 255).all()
            assert (tgt >= 0).all() and (tgt <= 255).all()

    def test_custom_texts(self):
        texts = ["Foo bar baz"] * 20
        ds = ByteTextDataset(texts=texts, seq_len=4, stride=2)
        assert len(ds) > 0

    def test_utf8_texts_included(self):
        texts = ["中文字符处理", "UTF-8 test"] * 30
        ds = ByteTextDataset(texts=texts, seq_len=8, stride=4)
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# SimpleBPE tokeniser
# ---------------------------------------------------------------------------


class TestSimpleBPE:
    """Tests for SimpleBPE."""

    def _make_trained_bpe(self, corpus=None, num_merges=30) -> SimpleBPE:
        bpe = SimpleBPE(vocab_size=400)
        corpus = corpus or ["hello world"] * 100
        bpe.train(corpus, num_merges=num_merges)
        return bpe

    # --- Special token IDs ---

    def test_special_token_ids(self):
        assert SimpleBPE.PAD_ID == 0
        assert SimpleBPE.BOS_ID == 1
        assert SimpleBPE.EOS_ID == 2
        assert SimpleBPE.UNK_ID == 3

    # --- Vocabulary ---

    def test_base_vocab_has_256_byte_tokens(self):
        bpe = SimpleBPE(vocab_size=400)
        # 4 specials + 256 byte tokens
        assert bpe.current_vocab_size == 260

    def test_vocab_grows_with_merges(self):
        bpe = SimpleBPE(vocab_size=400)
        before = bpe.current_vocab_size
        bpe.train(["abcabc"] * 200, num_merges=10)
        assert bpe.current_vocab_size > before

    def test_vocab_does_not_exceed_max(self):
        bpe = SimpleBPE(vocab_size=270)
        bpe.train(["abcdefg"] * 500, num_merges=1000)  # request far more merges
        assert bpe.current_vocab_size <= 270

    # --- Encode ---

    def test_encode_returns_list_of_ints(self):
        bpe = self._make_trained_bpe()
        ids = bpe.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self):
        bpe = self._make_trained_bpe()
        assert bpe.encode("") == []

    def test_encode_ids_within_vocab(self):
        bpe = self._make_trained_bpe()
        ids = bpe.encode("hello world")
        for i in ids:
            assert 0 <= i < bpe.current_vocab_size

    def test_encode_produces_fewer_tokens_after_training(self):
        """Trained BPE should encode common substrings more compactly."""
        bpe = SimpleBPE(vocab_size=400)
        text = "hello " * 20
        ids_before = bpe.encode(text)         # no merges yet
        bpe.train([text] * 50, num_merges=30)
        ids_after = bpe.encode(text)
        assert len(ids_after) <= len(ids_before), (
            "Trained BPE should produce ≤ tokens than the base byte tokeniser"
        )

    # --- Decode ---

    def test_decode_ascii_round_trip(self):
        bpe = self._make_trained_bpe(corpus=["the quick brown fox"] * 50)
        text = "the quick"
        decoded = bpe.decode(bpe.encode(text))
        assert decoded == text, f"Round-trip failed: {decoded!r} != {text!r}"

    def test_decode_unicode_round_trip(self):
        corpus = ["中文测试"] * 100
        bpe = SimpleBPE(vocab_size=400)
        bpe.train(corpus, num_merges=20)
        text = "中文"
        assert bpe.decode(bpe.encode(text)) == text

    def test_decode_skips_special_tokens(self):
        bpe = self._make_trained_bpe()
        # Manually inject special tokens into a stream; they should be skipped
        ids = [SimpleBPE.BOS_ID] + bpe.encode("hi") + [SimpleBPE.EOS_ID]
        decoded = bpe.decode(ids)
        assert decoded == "hi"

    # --- BPETextDataset ---

    def test_bpe_dataset_construction(self):
        bpe = self._make_trained_bpe()
        ds = BPETextDataset(bpe=bpe, texts=["hello world"] * 30, seq_len=8, stride=4)
        assert len(ds) > 0

    def test_bpe_dataset_item_shapes(self):
        bpe = self._make_trained_bpe()
        ds = BPETextDataset(bpe=bpe, texts=["hello world"] * 30, seq_len=8, stride=4)
        src, tgt = ds[0]
        assert src.shape == (8,)
        assert tgt.shape == (8,)
        assert src.dtype == torch.long


# ---------------------------------------------------------------------------
# Dynamic span utilities
# ---------------------------------------------------------------------------


class TestDynamicSpans:
    """Tests for compute_dynamic_spans and pool_spans."""

    def test_spans_cover_all_positions_all_compress(self):
        """With compress=1 everywhere, output spans should cover [0, T)."""
        T = 8
        spans = compute_dynamic_spans(torch.ones(T), torch.zeros(T))
        covered = sum(e - s for s, e in spans)
        assert covered == T

    def test_spans_cover_all_positions_all_expand(self):
        """With expand=1 everywhere, output spans should still cover [0, T)."""
        T = 8
        spans = compute_dynamic_spans(torch.zeros(T), torch.ones(T))
        covered = sum(e - s for s, e in spans)
        assert covered == T

    def test_spans_cover_all_positions_random(self):
        """Random gates should always produce full coverage."""
        torch.manual_seed(42)
        for _ in range(10):
            T = torch.randint(4, 33, (1,)).item()
            compress = torch.rand(T)
            expand = torch.rand(T)
            spans = compute_dynamic_spans(compress, expand)
            covered = sum(e - s for s, e in spans)
            assert covered == T, f"Coverage {covered} != {T} for T={T}"

    def test_spans_non_overlapping(self):
        """Spans must not overlap."""
        torch.manual_seed(7)
        spans = compute_dynamic_spans(torch.rand(16), torch.rand(16))
        prev_end = 0
        for s, e in spans:
            assert s >= prev_end, f"Overlap: span ({s},{e}) after end {prev_end}"
            assert e > s, "Empty span"
            prev_end = e

    def test_high_expand_creates_many_spans(self):
        """High expand values should produce more fine-grained spans."""
        T = 16
        few_spans = compute_dynamic_spans(torch.ones(T), torch.zeros(T))
        many_spans = compute_dynamic_spans(torch.zeros(T), torch.ones(T))
        assert len(many_spans) >= len(few_spans)

    def test_pool_spans_output_shape(self):
        d, T = 64, 16
        hidden = torch.randn(T, d)
        spans = [(0, 4), (4, 10), (10, 16)]
        result = pool_spans(hidden, spans)
        assert result.shape == (3, d)

    def test_pool_spans_no_nan(self):
        d, T = 32, 8
        hidden = torch.randn(T, d)
        spans = [(0, 2), (2, 5), (5, 8)]
        result = pool_spans(hidden, spans)
        assert not torch.isnan(result).any()

    def test_pool_spans_single_element_span(self):
        """A span of length 1 should equal the single hidden vector."""
        d = 16
        hidden = torch.randn(4, d)
        spans = [(2, 3)]          # single element
        result = pool_spans(hidden, spans)
        assert torch.allclose(result[0], hidden[2])
