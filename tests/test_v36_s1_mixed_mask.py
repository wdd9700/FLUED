"""Mixed char/BPE-word mask tests (v36.3 T3口径).

Checks the invariants the training loop relies on:
- realized mask rate tracks mask_prob;
- a UTF-8 char is never split (continuation byte masked => whole char masked);
- masking stays inside the valid (non-PAD) region;
- the dedicated generator makes the mask deterministic (eval protocol).
"""

from pathlib import Path

import pytest
import torch

from flued.data import text_to_byte_ids
from tools.train.v3_6.train_v36_s1 import _utf8_char_spans, make_mixed_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_tokenizer_128k_v4" / "tokenizer.json"

TEXT = "静等失败的隐性成本你算漏了。Q2你说无法绕过授权就静等决策失败——这是对的博弈策略，但它把证明自己正确置于事情成功之上。"


def _batch(text: str, seq_len: int, rows: int = 2) -> torch.Tensor:
    ids = text_to_byte_ids(text)[:seq_len]
    batch = torch.zeros(rows, seq_len, dtype=torch.long)
    for r in range(rows):
        batch[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def _assert_char_alignment(clean: torch.Tensor, mask: torch.Tensor) -> None:
    for b in range(clean.shape[0]):
        ids = clean[b]
        valid_len = int(ids.ne(0).sum())
        bs = bytes(i - 1 for i in ids[:valid_len].tolist())
        m = mask[b, :valid_len]
        for i in range(valid_len):
            is_cont = (bs[i] & 0xC0) == 0x80
            if is_cont and m[i]:
                assert m[i - 1], f"row {b}: continuation byte {i} masked without its lead byte"
        assert not mask[b, valid_len:].any(), f"row {b}: mask leaked into padding"


def test_char_spans_cover_and_align():
    bs = TEXT.encode("utf-8")
    spans = _utf8_char_spans(bs)
    assert spans[0][0] == 0 and spans[-1][1] == len(bs)
    for (s0, e0), (s1, _e1) in zip(spans, spans[1:]):
        assert e0 == s1
    for s, e in spans:
        bs[s:e].decode("utf-8")  # must be exactly one whole char


def test_mixed_mask_char_fallback_rate_and_alignment():
    clean = _batch(TEXT, 256)
    gen = torch.Generator().manual_seed(7)
    mask = make_mixed_mask(clean, 0.05, tokenizer=None, generator=gen)
    _assert_char_alignment(clean, mask)
    rate = mask.float().sum().item() / clean.ne(0).sum().item()
    assert 0.03 <= rate <= 0.09, rate


@pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="128k BPE reference tokenizer not present")
def test_mixed_mask_with_bpe_tokenizer():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    clean = _batch(TEXT, 256)
    gen = torch.Generator().manual_seed(11)
    mask = make_mixed_mask(clean, 0.05, char_frac=0.4, tokenizer=tok, generator=gen)
    _assert_char_alignment(clean, mask)
    rate = mask.float().sum().item() / clean.ne(0).sum().item()
    assert 0.03 <= rate <= 0.09, rate
    # word budget must have fired: at least one masked run of >=2 chars
    ids = clean[0]
    valid_len = int(ids.ne(0).sum())
    bs = bytes(i - 1 for i in ids[:valid_len].tolist())
    spans = _utf8_char_spans(bs)
    m = mask[0, :valid_len]
    runs = 0
    for idx, (s, e) in enumerate(spans):
        if not m[s]:
            continue
        run = 1
        nxt = idx + 1
        while nxt < len(spans) and m[spans[nxt][0]]:
            run += 1
            nxt += 1
        if run >= 2:
            runs += 1
    assert runs >= 1


def test_mixed_mask_deterministic_per_generator_seed():
    clean = _batch(TEXT, 256)
    m1 = make_mixed_mask(clean, 0.05, tokenizer=None, generator=torch.Generator().manual_seed(42))
    m2 = make_mixed_mask(clean, 0.05, tokenizer=None, generator=torch.Generator().manual_seed(42))
    assert torch.equal(m1, m2)


def test_mixed_mask_zero_prob_and_empty_rows():
    clean = _batch(TEXT, 256)
    assert not make_mixed_mask(clean, 0.0, tokenizer=None).any()
    empty = torch.zeros(1, 16, dtype=torch.long)
    assert not make_mixed_mask(empty, 0.05, tokenizer=None).any()
