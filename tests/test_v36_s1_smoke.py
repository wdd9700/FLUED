import torch

from flued.v36 import FLUEDV36
from tests.test_v36_smoke import tiny_config
from tools.train.v3_6.train_v36_s1 import s1_forward, step_model

PAD_ID = 0
MASK_ID = 257


class _Args:
    pass


def test_s1_forward_shapes_and_finite():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = s1_forward(model, ids, _Args())
    assert out["logits_direct"].shape == (2, 4, 8, 258)
    assert out["logits_backbone"].shape == (2, 4, 8, 258)
    assert out["content"].shape == (2, 4, 64)
    for key in ("logits_direct", "logits_backbone", "content"):
        assert torch.isfinite(out[key]).all(), key


def test_s1_predict_target_detached():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = s1_forward(model, ids, _Args())
    content = out["content"].float()
    backbone_out = out["backbone_out"].float()
    pair_mask = (out["chunks"].chunk_mask[:, :-1] & out["chunks"].chunk_mask[:, 1:]).float()
    # direction A (used in training): pred side keeps grads, target detached
    se_a = ((backbone_out[:, :-1] - content[:, 1:].detach()).square().mean(dim=-1) * pair_mask).sum()
    se_a.backward()
    b_grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone")]
    assert any(g is not None and g.abs().sum() > 0 for g in b_grads)
    model.zero_grad()
    # direction B (inverted, fresh forward): if pred were detached, backbone must see nothing
    out = s1_forward(model, ids, _Args())
    content = out["content"].float()
    backbone_out = out["backbone_out"].float()
    se_b = ((backbone_out[:, :-1].detach() - content[:, 1:]).square().mean(dim=-1) * pair_mask).sum()
    se_b.backward()
    b_grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone")]
    assert all(g is None or g.abs().sum() == 0 for g in b_grads)
    s_grads = [p.grad for n, p in model.named_parameters() if n.startswith("summarizer")]
    assert any(g is not None and g.abs().sum() > 0 for g in s_grads)


def test_s1_predict_decode_mode_frozen_decoder():
    import types

    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    model.train()
    args = types.SimpleNamespace(
        mask_prob=0.0, mask_span_min=1, mask_span_max=8, mask_mode="byte_span",
        max_chunks=4, max_span=8, task1_loss_weight=0.0, task2_loss_weight=0.0,
        predict_weight=1.0, predict_mode="decode", predict_latent_weight=0.0, amp=False,
    )
    ids = torch.randint(1, 257, (2, 32))
    loss, metrics = step_model(model, (ids,), args, torch.device("cpu"), train=True)
    loss.backward()
    dec_grads = [p.grad for n, p in model.named_parameters() if n.startswith("decoder")]
    assert all(g is None or g.abs().sum() == 0 for g in dec_grads), "decoder must stay frozen"
    # v2.1: predict CE consumes content.detach() -- the encoder/KDA side must
    # see zero gradient from the predict branch (codec owns the representation)
    for prefix in ("summarizer", "write_head", "state_machine", "encoder_blocks", "byte_lookup"):
        enc_grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert all(g is None or g.abs().sum() == 0 for g in enc_grads), f"{prefix} must stay untouched by predict CE"
    b_grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone")]
    assert any(g is not None and g.abs().sum() > 0 for g in b_grads), "backbone must receive predict gradients"
    assert "predict_ce" in metrics and metrics["predict_byte_acc"] >= 0.0


def test_s1_frozen_decoder_logits_detaches_params_only():
    from tools.train.v3_6.train_v36_s1 import _frozen_decoder_logits

    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    cond = torch.randn(2, 4, 64, requires_grad=True)
    token_mask = torch.ones(2, 4, 8, dtype=torch.bool)
    logits = _frozen_decoder_logits(model, cond, token_mask)
    logits.square().mean().backward()
    assert cond.grad is not None and cond.grad.abs().sum() > 0, "cond (backbone side) must receive grads"
    dec_grads = [p.grad for n, p in model.named_parameters() if n.startswith("decoder")]
    assert all(g is None for g in dec_grads), "decoder params must see no grad through the frozen call"
    assert model.byte_lookup.embedding.weight.grad is None, "byte table must see no grad through the frozen call"


def test_s1_backward_reaches_core():
    torch.manual_seed(0)
    model = FLUEDV36(tiny_config(per_chunk_readout=True))
    model.train()
    ids = torch.randint(1, 258, (2, 32))
    out = s1_forward(model, ids, _Args())
    loss = out["logits_direct"].square().mean() + out["logits_backbone"].square().mean()
    loss.backward()
    for prefix in ["summarizer", "write_head", "state_machine", "backbone", "decoder"]:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), prefix
