"""
tests/test_reconstruction.py
------------------------------
Unit and integration tests for FLUED v0.4 Stage A autoencoder models and the
shared trainer utilities.

Test scope
----------
  TestFLUEDAutoencoder     -- v0.4 FLUEDAutoencoder interface
      * forward shape / return type
      * P0-3: boundary_head gradient  (critical correctness gate)
      * differentiable compression loss
      * hard-span coverage
  TestBPETransformerAutoencoder -- BPE baseline (unchanged)
  TestBLTAutoencoder       -- BLT baseline (vocab_size=257 PAD-offset)
  TestTrainerUtils         -- build_model / eval_step / Trainer
  TestParameterCounts300M  -- informational +-20% parameter range

Run with:
    pytest tests/test_reconstruction.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from blt_baseline.model import BLTAutoencoder
from bpe_baseline.model import BPETransformerAutoencoder
from flued.config import ModelConfig, TrainConfig
from flued.data import STUB_CORPUS, ByteTextDataset, get_dataloader, safe_train_eval_split
from flued.model import FLUEDAutoencoder
from flued.train import (
    Trainer,
    build_model,
    compute_reconstruction_accuracy,
    eval_step,
    set_seed,
)


# ---------------------------------------------------------------------------
# Tiny architecture config -- shared across model types
#
# num_encoder_layers=2 / num_decoder_layers=2 is the legacy API accepted by
# BPETransformerAutoencoder and BLTAutoencoder.
# FLUEDAutoencoder v0.4 aliases num_encoder_layers -> num_layers when
# num_layers == 4 (the default), so passing num_encoder_layers=2 correctly
# sets num_layers=2 inside FLUEDAutoencoder.
# ---------------------------------------------------------------------------

_TINY = dict(
    d_model=64,
    nhead=4,
    dim_feedforward=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    max_seq_len=32,
    dropout=0.0,   # disable dropout for deterministic tests
)


# ---------------------------------------------------------------------------
# FLUED v0.4 autoencoder tests
# ---------------------------------------------------------------------------


class TestFLUEDAutoencoder:
    """Tests for flued/model.py -- FLUEDAutoencoder v0.4."""

    def _make_model(self) -> FLUEDAutoencoder:
        # vocab_size=257: PAD=0, byte b -> id b+1  (v0.4 PAD-offset encoding)
        return FLUEDAutoencoder(vocab_size=257, **_TINY)

    # ------------------------------------------------------------------
    # Output shape / return type
    # ------------------------------------------------------------------

    def test_forward_output_shape(self):
        """logits should be [B, T, vocab_size=257]."""
        model = self._make_model()
        model.eval()
        B, T = 2, 16
        src = torch.randint(1, 257, (B, T))
        logits, metrics = model(src)
        assert logits.shape == (B, T, 257), f"Unexpected shape: {logits.shape}"

    def test_forward_returns_metrics_dict(self):
        """Second return value must be a dict containing required keys."""
        model = self._make_model()
        src = torch.randint(1, 257, (2, 16))
        logits, metrics = model(src)
        assert isinstance(metrics, dict), (
            f"Expected dict, got {type(metrics)}"
        )
        required = {
            "compression_loss", "spans", "boundary_probs",
            "m_over_n", "soft_m_over_n", "hard_m_over_n",
            "num_units", "z",
        }
        missing = required - set(metrics.keys())
        assert not missing, f"metrics dict is missing keys: {missing}"

    # ------------------------------------------------------------------
    # P0-3 -- boundary_head gradient (critical correctness gate)
    # ------------------------------------------------------------------

    def test_boundary_head_receives_gradient(self):
        """P0-3: boundary_head.weight must accumulate a non-zero gradient.

        Training uses the soft assignment matrix A so that boundary_probs
        feeds back into both the reconstruction loss (via expanded_soft)
        and the compression loss.  If the gradient path is broken, this
        test fails and highlights the regression.
        """
        model = self._make_model()
        model.train()
        src = torch.randint(1, 257, (2, 16))
        logits, metrics = model(src)
        loss = (
            nn.CrossEntropyLoss()(logits.view(-1, 257), src.view(-1))
            + metrics["compression_loss"]
        )
        loss.backward()

        grad = model.boundary_head.weight.grad
        assert grad is not None, (
            "boundary_head.weight.grad is None -- backward did not reach boundary_head"
        )
        assert grad.norm().item() > 0.0, (
            f"boundary_head.weight gradient is zero "
            f"(norm={grad.norm().item():.6f}) -- soft segmentation path is broken"
        )

    # ------------------------------------------------------------------
    # Differentiable compression loss
    # ------------------------------------------------------------------

    def test_compression_loss_is_differentiable(self):
        """compression_loss must have a grad_fn (allows .backward())."""
        model = self._make_model()
        src = torch.randint(1, 257, (2, 16))
        _, metrics = model(src)
        comp_loss = metrics["compression_loss"]
        assert isinstance(comp_loss, torch.Tensor), (
            f"compression_loss must be a Tensor, got {type(comp_loss)}"
        )
        assert comp_loss.grad_fn is not None or comp_loss.requires_grad, (
            "compression_loss has no grad_fn -- it is not differentiable"
        )
        # Must not raise
        comp_loss.backward()

    def test_soft_m_over_n_is_tensor(self):
        """soft_m_over_n must be a Tensor (not a Python float) for grad flow."""
        model = self._make_model()
        src = torch.randint(1, 257, (2, 16))
        _, metrics = model(src)
        smn = metrics["soft_m_over_n"]
        assert isinstance(smn, torch.Tensor), (
            f"soft_m_over_n must be a torch.Tensor, got {type(smn)}"
        )

    def test_compression_loss_nonnegative(self):
        """Compression loss = weight * (soft_m/n - target)^2 >= 0."""
        model = self._make_model()
        src = torch.randint(1, 257, (2, 16))
        _, metrics = model(src)
        assert metrics["compression_loss"].item() >= 0.0

    # ------------------------------------------------------------------
    # Hard segmentation correctness
    # ------------------------------------------------------------------

    def test_hard_spans_cover_full_sequence(self):
        """Hard spans must tile [0, T) exactly -- no gaps, no overlaps."""
        model = self._make_model()
        model.eval()
        T = 16
        src = torch.randint(1, 257, (1, T))
        _, metrics = model(src)
        spans = metrics["spans"][0]   # batch item 0

        assert spans, "spans must be non-empty"
        assert spans[0][0] == 0, (
            f"First span does not start at 0: {spans[0]}"
        )
        assert spans[-1][1] == T, (
            f"Last span does not end at T={T}: {spans[-1]}"
        )
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0], (
                f"Gap between spans[{i}]={spans[i]} and spans[{i+1}]={spans[i + 1]}"
            )

    # ------------------------------------------------------------------
    # encode / decode interface
    # ------------------------------------------------------------------

    def test_encode_returns_expanded_soft_and_metrics(self):
        """encode() should return (expanded_soft [B,T,d], metrics dict)."""
        model = self._make_model()
        src = torch.randint(1, 257, (2, 16))
        expanded_soft, metrics = model.encode(src)
        assert expanded_soft.shape == (2, 16, _TINY["d_model"]), (
            f"Unexpected expanded_soft shape: {expanded_soft.shape}"
        )
        assert isinstance(metrics, dict)
        assert "boundary_probs" in metrics

    def test_tgt_kwarg_accepted(self):
        """Passing tgt= (unused in v0.4) should not raise and must be idempotent."""
        model = self._make_model()
        model.eval()
        src = torch.randint(1, 257, (1, 16))
        logits_none, _ = model(src, tgt=None)
        logits_same, _ = model(src, tgt=src)
        assert torch.allclose(logits_none, logits_same), (
            "Passing tgt=None vs tgt=src should produce identical logits"
        )

    # ------------------------------------------------------------------
    # Gradient propagation
    # ------------------------------------------------------------------

    def test_backward_propagates_gradients(self):
        """Full backward pass must leave non-zero gradients on model parameters."""
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        src = torch.randint(1, 257, (2, 16))
        logits, metrics = model(src)
        loss = (
            nn.CrossEntropyLoss()(logits.view(-1, 257), src.view(-1))
            + metrics["compression_loss"]
        )
        loss.backward()
        optimizer.step()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No non-zero gradients after backward pass"

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def test_parameter_count_positive(self):
        """count_parameters() must return a positive integer."""
        model = self._make_model()
        n = model.count_parameters()
        assert n > 0
        print(f"\nFLUED tiny: {n:,} parameters")


# ---------------------------------------------------------------------------
# BPE-Transformer autoencoder tests  (baseline -- unchanged from v0.3)
# ---------------------------------------------------------------------------


class TestBPETransformerAutoencoder:
    """Tests for bpe_baseline/model.py -- BPETransformerAutoencoder."""

    def _make_model(self, vocab_size: int = 512) -> BPETransformerAutoencoder:
        return BPETransformerAutoencoder(vocab_size=vocab_size, **_TINY)

    def test_forward_output_shape(self):
        model = self._make_model()
        model.eval()
        B, T, V = 2, 16, 512
        src = torch.randint(4, V, (B, T))   # skip special token ids 0-3
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
# BLT autoencoder tests  (updated for vocab_size=257 PAD-offset encoding)
# ---------------------------------------------------------------------------


class TestBLTAutoencoder:
    """Tests for blt_baseline/model.py -- BLTAutoencoder with vocab_size=257."""

    def _make_model(self) -> BLTAutoencoder:
        # vocab_size=257 matches FLUED PAD-offset encoding (PAD=0, byte b -> b+1)
        return BLTAutoencoder(
            vocab_size=257,
            local_layers=1,
            patch_size=4,
            **_TINY,
        )

    def test_forward_output_shape(self):
        model = self._make_model()
        model.eval()
        B, T = 2, 16
        src = torch.randint(1, 257, (B, T))
        logits, aux_loss = model(src)
        assert logits.shape == (B, T, 257), f"Unexpected shape: {logits.shape}"

    def test_aux_loss_is_zero(self):
        model = self._make_model()
        src = torch.randint(1, 257, (2, 16))
        _, aux_loss = model(src)
        assert aux_loss.item() == 0.0

    def test_backward_propagates_gradients(self):
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        src = torch.randint(1, 257, (2, 16))
        logits, _ = model(src)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 257), src.view(-1))
        loss.backward()
        optimizer.step()
        assert any(p.grad is not None for p in model.parameters())

    def test_seq_len_not_multiple_of_patch_size(self):
        """BLT should handle T not divisible by patch_size via padding."""
        model = self._make_model()   # patch_size=4
        model.eval()
        B, T = 2, 13               # 13 is not divisible by 4
        src = torch.randint(1, 257, (B, T))
        logits, _ = model(src)
        assert logits.shape == (B, T, 257), (
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
    """Tests for flued/train.py -- helper functions and Trainer."""

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
        logits = torch.randn(4, 8, 257)
        targets = torch.randint(1, 257, (4, 8))
        acc = compute_reconstruction_accuracy(logits, targets)
        assert 0.0 <= acc <= 1.0

    def test_compute_reconstruction_accuracy_ignores_padding(self):
        """Positions where target==0 (PAD) should not count."""
        logits = torch.zeros(1, 4, 5)
        targets = torch.tensor([[1, 0, 0, 0]])
        logits[0, 0, 1] = 1.0
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
        """build_model should return a FLUEDAutoencoder for model_type=flued."""
        cfg = ModelConfig(
            model_type="flued",
            d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            num_layers=2,
            max_seq_len=32, vocab_size=257,
        )
        model = build_model(cfg)
        assert isinstance(model, FLUEDAutoencoder)

    def test_build_model_bpe(self):
        """build_model should return a BPETransformerAutoencoder for model_type=bpe."""
        cfg = ModelConfig(
            model_type="bpe",
            d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, bpe_vocab_size=512,
        )
        model = build_model(cfg)
        assert isinstance(model, BPETransformerAutoencoder)

    def test_build_model_blt(self):
        """build_model should return a BLTAutoencoder for model_type=blt."""
        cfg = ModelConfig(
            model_type="blt",
            d_model=64, nhead=4, dim_feedforward=128,
            num_encoder_layers=2, num_decoder_layers=2,
            max_seq_len=32, vocab_size=257, local_layers=1,
        )
        model = build_model(cfg)
        assert isinstance(model, BLTAutoencoder)

    def test_eval_step_returns_metrics(self):
        """eval_step should return a dict with loss and reconstruction_accuracy."""
        set_seed(7)
        model = FLUEDAutoencoder(vocab_size=257, **_TINY)
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
        model = FLUEDAutoencoder(vocab_size=257, **_TINY)
        dataset = ByteTextDataset(texts=STUB_CORPUS * 5, seq_len=32, stride=16)
        train_ds, eval_ds = safe_train_eval_split(dataset, eval_fraction=0.2, seed=0)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False, drop_last=True)

        train_cfg = TrainConfig(
            seed=0,
            batch_size=4,
            max_steps=10,
            lr=1e-3,
            warmup_steps=2,
            log_interval=5,
            eval_interval=100,
            save_interval=100,
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
    """Verify that the 300M size preset produces ~300M parameters.

    FLUED v0.4 uses num_layers=24 (24 tied blocks x ~12.6M params each ~ 302M).
    BPE / BLT use the legacy num_encoder_layers=12 + num_decoder_layers=12.
    All three should fall within +-20% of 300M.
    """

    @pytest.mark.parametrize("model_type", ["flued", "bpe", "blt"])
    def test_300M_preset_parameter_range(self, model_type: str):
        """Each model at the 300M preset should have 240M-360M parameters."""
        cfg = ModelConfig(model_type=model_type, size="300M")
        cfg.apply_size()
        model = build_model(cfg)
        n = model.count_parameters()
        low, high = 240_000_000, 360_000_000
        print(f"\n{model_type} 300M: {n:,} parameters")
        assert low < n < high, (
            f"{model_type} 300M has {n:,} parameters, "
            f"expected {low:,}-{high:,} (+-20% of 300M)"
        )
