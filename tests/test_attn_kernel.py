"""Fused Triton attention: math against a float64 reference with the same dropout mask, and determinism."""

import copy

import pytest
import torch

from mathocr.train import train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")


def _inputs(b, lq, lk, h=4, d=32):
    # The model's layout: (B, L, H, D) projections viewed as (B, H, L, D); k and v are halves of one tensor.
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, lq, h, d, device="cuda", generator=g).bfloat16().transpose(1, 2).requires_grad_()
    kv = torch.randn(b, lk, 2 * h * d, device="cuda", generator=g).bfloat16()
    k = kv[..., : h * d].view(b, lk, h, d).transpose(1, 2).detach().requires_grad_()
    v = kv[..., h * d:].view(b, lk, h, d).transpose(1, 2).detach().requires_grad_()
    return q, k, v


def _reference(q, k, v, causal, p, keep):
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v)]
    s = leaves[0] @ leaves[1].mT * q.shape[-1] ** -0.5
    if causal:
        s = s.masked_fill(torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    a = s.softmax(-1)
    if p > 0:
        a = a * keep.view_as(a) / (1 - p)
    return a @ leaves[2], leaves


@cuda
@pytest.mark.parametrize("b,lq,lk,causal,p", [(2, 70, 70, True, 0.0), (2, 70, 70, True, 0.2), (2, 45, 300, False, 0.1),
                                              (1, 257, 257, False, 0.0), (1, 130, 640, False, 0.1)])
def test_flash_attention_matches_reference(b, lq, lk, causal, p):
    from mathocr.attn_kernel import dropout_keep_mask, flash_attention

    q, k, v = _inputs(b, lq, lk)
    o = flash_attention(q, k, v, causal, p, seed=77)
    go = torch.randn_like(o)
    o.backward(go)
    keep = dropout_keep_mask(b * q.shape[1], lq, lk, p, 77).view(b, q.shape[1], lq, lk) if p > 0 else None
    ref, leaves = _reference(q, k, v, causal, p, keep)
    ref.backward(go.double())
    for got, want in ((o, ref), *((t.grad, r.grad) for t, r in zip((q, k, v), leaves))):
        assert ((got.double() - want).norm() / want.norm()).item() < 1e-2
    if keep is not None:
        assert abs(keep.float().mean().item() - (1 - p)) < 0.01


@cuda
def test_flash_attention_is_deterministic():
    from mathocr.attn_kernel import flash_attention

    runs = []
    for _ in range(2):
        q, k, v = _inputs(3, 100, 700)
        o = flash_attention(q, k, v, False, 0.1, seed=5)
        o.backward(torch.ones_like(o))
        runs.append((o, q.grad, k.grad, v.grad))
    assert all(torch.equal(x, y) for x, y in zip(*runs))


def _state(run):
    return torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]


@cuda
def test_training_with_fast_paths_is_bitwise_reproducible_and_resumable(tiny_config, tmp_path, monkeypatch):
    """Triton attention + (batched) Muon: rerun and resume give the same weights."""
    import mathocr.attn_kernel as ak
    import mathocr.model as model

    monkeypatch.setattr(model, "ATTN_KERNEL_MIN_KEYS", 0)  # the tiny model's maps are far below the threshold
    calls = []
    real = ak.flash_attention
    monkeypatch.setattr(ak, "flash_attention", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    cfg = copy.deepcopy(tiny_config)
    cfg["model"]["attn_kernel"] = "triton"
    cfg["train"]["optimizer"] = "muon"
    train(copy.deepcopy(cfg), 7, tmp_path / "a", resume=False)
    assert calls, "the Triton kernel was not used"
    train(copy.deepcopy(cfg), 7, tmp_path / "b", resume=False)
    paused = copy.deepcopy(cfg)
    paused["train"]["stop_after_epoch"] = 0
    train(paused, 7, tmp_path / "c", resume=False)
    train(copy.deepcopy(cfg), 7, tmp_path / "c", resume=True)
    a, b, c = (_state(tmp_path / r) for r in "abc")
    for k in a:
        assert torch.equal(a[k], b[k]), f"rerun differs at {k}"
        assert torch.equal(a[k], c[k]), f"resumed run differs at {k}"
