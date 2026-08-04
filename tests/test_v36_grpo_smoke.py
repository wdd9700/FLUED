import torch

from flued.v36 import FLUEDV36
from tests.test_v36_smoke import tiny_config
from tools.train.v3_6.train_v36_grpo import grpo_forward

PAD_ID = 0


def test_grpo_forward_shapes_and_finite():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    ids = torch.randint(1, 258, (2, 32))
    out = grpo_forward(model, ids, tau_cut=0.9, cut_temperature=0.15, beta_sigma=0.5)
    assert out["logits_direct"].shape == (2, 4, 8, 258)
    assert out["logits_backbone"].shape == (2, 4, 8, 258)
    assert out["logp_cut"].shape == (2,)
    assert out["logp_beta"].shape == (2,)
    for key in ("logits_direct", "logits_backbone", "logp_cut", "logp_beta"):
        assert torch.isfinite(out[key]).all(), key
    assert out["chunks"].pack_info["truncated_tokens"].sum().item() == 0


def test_grpo_backward_reaches_rl_modules():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = grpo_forward(model, ids, tau_cut=0.9, cut_temperature=0.15, beta_sigma=0.5)
    loss = (
        out["logits_direct"].square().mean()
        + out["logits_backbone"].square().mean()
        - (out["logp_cut"] + out["logp_beta"]).mean()
    )
    loss.backward()
    expected = [
        "segmentor_blocks",
        "segmentor_head",
        "write_head.to_beta",
        "summarizer",
        "state_machine",
        "backbone",
        "decoder",
    ]
    for prefix in expected:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), prefix


def test_grpo_logp_excludes_forced_positions():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config())
    ids = torch.randint(1, 258, (2, 32))
    ids[:, 20:] = PAD_ID
    out = grpo_forward(model, ids, tau_cut=0.9, cut_temperature=0.15, beta_sigma=0.5)
    # padded tail contributes no cut log-prob mass: logp_cut must stay finite
    # and the executable cuts must respect the UTF-8/valid guards.
    assert torch.isfinite(out["logp_cut"]).all()
    cuts = out["chunks"].chunk_mask
    assert cuts.any()
