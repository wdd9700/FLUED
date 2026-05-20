import importlib

import pytest
import torch

from flued.data import ByteReconstructionDataset
from flued.e1_stage_a import build_e1_configs
from flued.e2_compare import OptionalDependencyError, build_adapter, build_e2_preset_config
from flued.model import FLUEDAutoencoder


def test_flued_v04_forward_returns_logits_and_metrics_m_over_n() -> None:
    model = FLUEDAutoencoder(
        vocab_size=257,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        max_seq_len=32,
        dropout=0.0,
    )
    src = torch.randint(1, 257, (2, 16))
    logits, metrics = model(src)
    assert logits.shape == (2, 16, 257)
    assert isinstance(metrics, dict)
    assert "m_over_n" in metrics
    assert metrics["m_over_n"].shape[0] == 2


def test_reconstruction_dataset_returns_src_equal_tgt() -> None:
    ds = ByteReconstructionDataset(texts=["abc"] * 8, seq_len=8, stride=4)
    src, tgt = ds[0]
    assert torch.equal(src, tgt)


def test_tokenizer_adapters_graceful_when_optional_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name in {"sentencepiece", "tiktoken"}:
            raise ImportError(name)
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError):
        build_adapter("sentencepiece", vocab_size=128, texts=["hello"], tiktoken_encoding="cl100k_base")

    with pytest.raises(OptionalDependencyError):
        build_adapter("tiktoken", vocab_size=128, texts=["hello"], tiktoken_encoding="cl100k_base")


def test_e1_config_builder_smoke_mode() -> None:
    model_cfg, train_cfg = build_e1_configs("smoke_cpu")
    assert model_cfg.size == "smoke"
    assert train_cfg.device == "cpu"
    assert train_cfg.max_steps > 0


def test_e2_config_builder_smoke_mode() -> None:
    cfg = build_e2_preset_config("smoke")
    assert cfg["d_model"] > 0
    assert cfg["seq_len"] > 0
