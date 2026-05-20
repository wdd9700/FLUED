"""
tests/test_reconstruction.py
------------------------------
Unit and integration tests for FLUED Stage A autoencoder models and the
shared trainer utilities.

Run with:
    pytest tests/test_reconstruction.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from blt_baseline.model import BLTAutoencoder
from bpe_baseline.model import BPETransformerAutoencoder
from flued.config import ModelConfig
from flued.data import STUB_CORPUS, ByteTextDataset, get_dataloader
from flued.model import FLUEDAutoencoder
from flued.train import (
    Trainer,
    TrainConfig,
    build_model,
    compute_reconstruction_accuracy,
    eval_step,
    set_seed,
)


# ---------------------------------------------------------------------------
# Tiny architecture config used by all tests for speed
# ---------------------------------------------------------------------------

_TINY = dict(
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    max_seq_len=32,
    dropout=0.0,  # disable dropout for deterministic tests
)


# ---------------------------------------------------------------------------
# FLUED autoencoder tests
# ---------------------------------------------------------------------------


class TestFLUEDAutoencoder:
    """Tests for flued/model.py — FLUEDAutoencoder."""

    def _make_model(self) -> FLUEDAutoencoder:
        return FLUEDAutoencoder(
            vocab_size=256,
            shallow_layers=1,
            gate_entropy_weight=0.01,
            **_TINY,
        )

    def test_forward_output_shape(self):
        """logits should be [B, T, vocab_size]."""
        model = self._make_model()
        model.eval()
        B, T = 2, 16
        src = torch.randint(1, 256, (B, T))
        logits, aux_loss = model(src)
        assert logits.shape == (B, T, 256), f"Unexpected shape: {logits.shape}"

    def test_aux_loss_is_scalar(self):
        """SGL auxiliary loss should be a scalar tensor."""
        model = self._make_model()
        src = torch.randint(1, 256, (2, 16))
        _, aux_loss = model(src)
        assert aux_loss.ndim == 0, "aux_loss should be 0-dimensional"

    def test_aux_loss_is_nonnegative(self):
        """Gate entropy regularisation term should be ≥ 0."""
        model = self._make_model()
        src = torch.randint(1, 256, (2, 16))
        _, aux_loss = model(src)
        assert aux_loss.item() >= 0.0

    def test_backward_propagates_gradients(self):
        """A full backward pass should produce non-zero gradients."""
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        src = torch.randint(1, 256, (2, 16))
        logits, aux_loss = model(src)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 256), src.view(-1)) + aux_loss
        loss.backward()
        optimizer.step()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No non-zero gradients after backward pass"

    def test_encode_returns_gate_info(self):
        """encode() should return all four SGL gate tensors."""
        model = self._make_model()
        src = torch.randint(1, 256, (1, 16))
        _, gate_info = model.encode(src)
        required_keys = {"gamma_compress", "gamma_expand", "gamma_bridge", "bridge_potential"}
        assert required_keys == set(gate_info.keys()), (
            f"Missing keys: {required_keys - set(gate_info.keys())}"
        )

    def test_gate_values_in_unit_interval(self):
        """γ_compress, γ_expand, γ_bridge must lie in [0, 1]."""
        model = self._make_model()
        src = torch.randint(1, 256, (1, 16))
        _, gate_info = model.encode(src)
        for name in ("gamma_compress", "gamma_expand", "gamma_bridge"):
            g = gate_info[name]
            assert (g >= 0.0).all() and (g <= 1.0).all(), (
                f"Gate '{name}' has values outside [0, 1]"
            )

    def test_tgt_equals_src_by_default(self):
        """When tgt is None the model should reconstruct src."""
        model = self._make_model()
        model.eval()
        src = torch.randint(1, 256, (1, 16))
        logits_none, _ = model(src, tgt=None)
        logits_same, _ = model(src, tgt=src)
        assert torch.allclose(logits_none, logits_same), (
            "Passing tgt=None should be equivalent to tgt=src"
        )

    def test_parameter_count_positive(self):
        """count_parameters() must return a positive integer."""
        model = self._make_model()
        n = model.count_parameters()
        assert n > 0
        # Informational (shown with pytest -v -s)
        print(f"\nFLUED tiny: {n:,} parameters")


# ---------------------------------------------------------------------------
# BPE-Transformer autoencoder tests
# ---------------------------------------------------------------------------


class TestBPETransformerAutoencoder:
    """Tests for bpe_baseline/model.py — BPETransformerAutoencoder."""

    def _make_model(self, vocab_size: int = 512) -> BPETransformerAutoencoder:
        return BPETransformerAutoencoder(vocab_size=vocab_size, **_TINY)

    def test_forward_output_shape(self):
        model = self._make_model()
        model.eval()
        B, T, V = 2, 16, 512
        src = torch.randint(4, V, (B, T))   # skip special token ids 0–3
        logits, aux_loss = model(src)
        assert logits.shape == (B, T, V), f"Unexpected shape: {logits.shape}"

    def test_aux_loss_is_zero(self):
        """BPE baseline should return exactly zero auxiliary loss."""
        model = self._make_model()
        src = torch.randint(4, 512, (2, 16))
        _, aux_loss = model(src)
        assert aux_loss.item() == 0.0

    def test_backward_propagates_gradients(self):
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        src = torch.randint(4, 512, (2, 16))
        logits, _ = model(src)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 512), src.view(-1))
        loss.backward()
        optimizer.step()
        assert any(p.grad is not None for p in model.parameters())

    def test_encode_returns_memory(self):
        """encode() should return a [B, T, d_model] tensor."""
        model = self._make_model()
        src = torch.randint(4, 512, (2, 16))
        memory = model.encode(src)
        assert memory.shape == (2, 16, _TINY["d_model"])

    def test_parameter_count_positive(self):
        model = self._make_model()
        n = model.count_parameters()
        assert n > 0
        print(f"\nBPE tiny: {n:,} parameters")


# ---------------------------------------------------------------------------
# BLT autoencoder tests
# ---------------------------------------------------------------------------


class TestBLTAutoencoder:
    """Tests for blt_baseline/model.py — BLTAutoencoder."""

    def _make_model(self) -> BLTAutoencoder:
        return BLTAutoencoder(
            vocab_size=256,
            local_layers=1,
            patch_size=4,
            **_TINY,
        )

    def test_forward_output_shape(self):
        model = self._make_model()
        model.eval()
        B, T = 2, 16
        src = torch.randint(1, 256, (B, T))
        logits, aux_loss = model(src)
        assert logits.shape == (B, T, 256), f"Unexpected shape: {logits.shape}"

    def test_aux_loss_is_zero(self):
        model = self._make_model()
        src = torch.randint(1, 256, (2, 16))
        _, aux_loss = model(src)
        assert aux_loss.item() == 0.0

    def test_backward_propagates_gradients(self):
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        src = torch.randint(1, 256, (2, 16))
        logits, _ = model(src)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 256), src.view(-1))
        loss.backward()
        optimizer.step()
        assert any(p.grad is not None for p in model.parameters())

    def test_seq_len_not_multiple_of_patch_size(self):
        """BLT should handle T not divisible by patch_size via padding."""
        model = self._make_model()   # patch_size=4
        model.eval()
        B, T = 2, 13               # 13 is not divisible by 4
        src = torch.randint(1, 256, (B, T))
        logits, _ = model(src)
        assert logits.shape == (B, T, 256), (
            "Output should be trimmed to original T even when T % patch_size != 0"
        )

    def test_parameter_count_positive(self):
        model = self._make_model()
        n = model.count_parameters()
        assert n > 0
        print(f"\nBLT tiny: {n:,} parameters")


# ---------------------------------------------------------------------------
# Trainer utilities tests
# ---------------------------------------------------------------------------


class TestTrainerUtils:
    """Tests for flued/train.py — helper functions."""

    def test_compute_reconstruction_accuracy_perfect(self):
        """Perfect logits should yield accuracy == 1.0."""
        B, T, V = 2, 4, 10
        targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        logits = torch.full((B, T, V), -1e9)
        for b in range(B):
            for t in range(T):
                logits[b, t, targets[b, t]] = 1e9
        acc = compute_reconstruction_accuracy(logits, targets)
        assert abs(acc - 1.0) < 1e-6

    def test_compute_reconstruction_accuracy_in_range(self):
        """Accuracy should always lie in [0, 1]."""
        torch.manual_seed(0)
        logits = torch.randn(4, 8, 256)
        targets = torch.randint(1, 256, (4, 8))
        acc = compute_reconstruction_accuracy(logits, targets)
        assert 0.0 <= acc <= 1.0

    def test_compute_reconstruction_accuracy_ignores_padding(self):
        """Positions where target==0 (PAD) should not count."""
        logits = torch.zeros(1, 4, 5)
        # Correct prediction at pos 0 only; pos 1-3 are padding (id=0)
        targets = torch.tensor([[1, 0, 0, 0]])
        logits[0, 0, 1] = 1.0   # correct
        acc = compute_reconstruction_accuracy(logits, targets)
        assert abs(acc - 1.0) < 1e-6, "Only non-padding position should count"

    def test_set_seed_reproducibility(self):
        """Two runs with the same seed should produce identical tensors."""
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_build_model_flued(self):
        cfg = ModelConfig(
            model_type="flued",
            d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, vocab_size=256, shallow_layers=1,
        )
        model = build_model(cfg)
        assert isinstance(model, FLUEDAutoencoder)

    def test_build_model_bpe(self):
        cfg = ModelConfig(
            model_type="bpe",
            d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, bpe_vocab_size=512,
        )
        model = build_model(cfg)
        assert isinstance(model, BPETransformerAutoencoder)

    def test_build_model_blt(self):
        cfg = ModelConfig(
            model_type="blt",
            d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, vocab_size=256, local_layers=1,
        )
        model = build_model(cfg)
        assert isinstance(model, BLTAutoencoder)

    def test_eval_step_returns_metrics(self):
        """eval_step should return a dict with loss and reconstruction_accuracy."""
        set_seed(7)
        model = FLUEDAutoencoder(
            vocab_size=256, d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, shallow_layers=1,
        )
        dataset = ByteTextDataset(texts=STUB_CORPUS * 5, seq_len=32, stride=16)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
        metrics = eval_step(model, loader, torch.device("cpu"), max_batches=2)
        assert "loss" in metrics
        assert "reconstruction_accuracy" in metrics
        assert metrics["loss"] > 0.0
        assert 0.0 <= metrics["reconstruction_accuracy"] <= 1.0

    def test_trainer_runs_short_loop(self):
        """Trainer should complete 10 steps without errors."""
        set_seed(0)
        model = FLUEDAutoencoder(
            vocab_size=256, d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, shallow_layers=1,
        )
        dataset = ByteTextDataset(texts=STUB_CORPUS * 5, seq_len=32, stride=16)
        train_ds, eval_ds = torch.utils.data.random_split(
            dataset, [len(dataset) - 2, 2],
            generator=torch.Generator().manual_seed(0),
        )
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False, drop_last=True)

        train_cfg = TrainConfig(
            seed=0,
            batch_size=4,
            max_steps=10,
            lr=1e-3,
            warmup_steps=2,
            log_interval=5,
            eval_interval=100,   # skip eval during this short run
            save_interval=100,   # skip save
            output_dir="/tmp/flued_test_ckpt",
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_cfg=train_cfg,
            device=torch.device("cpu"),
        )
        trainer.train()
        assert trainer.global_step == 10


# ---------------------------------------------------------------------------
# 300M parameter count verification (informational)
# ---------------------------------------------------------------------------


class TestParameterCounts300M:
    """Verify that the 300M size preset produces ~300M parameters."""

    @pytest.mark.parametrize("model_type", ["flued", "bpe", "blt"])
    def test_300M_preset_parameter_range(self, model_type: str):
        """Each model at the 300M preset should have 250M–400M parameters."""
        from flued.config import ModelConfig

        cfg = ModelConfig(model_type=model_type, size="300M")
        cfg.apply_size()
        model = build_model(cfg)
        n = model.count_parameters()
        low, high = 240_000_000, 360_000_000
        print(f"\n{model_type} 300M: {n:,} parameters")
        assert low < n < high, (
            f"{model_type} 300M has {n:,} parameters, "
            f"expected {low:,}–{high:,} (±20% of 300M)"
        )
