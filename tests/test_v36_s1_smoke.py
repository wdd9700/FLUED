import torch

from flued.v36 import FLUEDV36
from tests.test_v36_smoke import tiny_config
from tools.train.v3_6.train_v36_s1 import s1_forward

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
