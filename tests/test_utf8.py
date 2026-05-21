from flued.data import BYTE_OFFSET, byte_ids_to_text, text_to_byte_ids


def test_byte_pad_offset_roundtrip() -> None:
    text = "Hello 中文 🎉"
    ids = text_to_byte_ids(text)
    assert all(i >= BYTE_OFFSET for i in ids)
    assert byte_ids_to_text(ids) == text


def test_pad_id_ignored_on_decode() -> None:
    text = "A"
    ids = [0] + text_to_byte_ids(text) + [0]
    assert byte_ids_to_text(ids) == text
