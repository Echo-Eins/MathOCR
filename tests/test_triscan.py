"""TriScanBlock (model.encoder = "triscan"): the directional streams see only their side, the diagonal stream only
its diagonals, the exchange variant starts as the plain three-stream block, and training with it is reproducible."""

import copy

import pytest
import torch

from mathocr.model import DiagonalConv, DirectionalConv, Im2Latex, ModelConfig, TriScanBlock
from mathocr.train import train


def test_directional_convs_see_only_their_side():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 3, 40)
    y = x.clone()
    y[..., 20] += 5.0  # change one column
    for right in (False, True):
        conv = DirectionalConv(4, right=right)
        a, b = conv(x), conv(y)
        changed = (a - b).abs().amax(dim=(0, 1, 2)) > 1e-6
        if right:  # anti-causal: only columns at or left of 20 can see the change
            assert not changed[21:].any() and changed[20]
        else:  # causal: only columns at or right of 20
            assert not changed[:20].any() and changed[20]


def test_diagonal_conv_reaches_only_the_diagonals():
    conv = DiagonalConv(1)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
    x = torch.zeros(1, 1, 9, 21)
    x[0, 0, 4, 10] = 1.0
    y = conv(x)[0, 0]
    hit = (y != 0).nonzero().tolist()
    assert all(abs(abs(c - 10) - 2 * abs(r - 4)) <= 1 for r, c in hit) and [4, 10] in hit and [2, 14] in hit


def test_exchange_block_starts_as_the_plain_block():
    torch.manual_seed(0)
    plain = TriScanBlock(16, 0.0, exchange=False).eval()
    torch.manual_seed(0)
    ex = TriScanBlock(16, 0.0, exchange=True).eval()
    ex.init_extra()
    ex.load_state_dict({**ex.state_dict(), **plain.state_dict()})
    x = torch.randn(2, 16, 6, 30)
    with torch.no_grad():
        assert torch.allclose(ex(x), plain(x), atol=1e-6)


@pytest.mark.parametrize("exchange", [True, False])
def test_triscan_model_forward(exchange):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=30, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32, n_heads=4,
                      enc_layers=2, dec_layers=1, max_len=16, encoder="triscan", triscan_exchange=exchange)
    m = Im2Latex(cfg).eval()
    out = m(torch.rand(2, 1, 32, 96), torch.randint(4, 30, (2, 5)))
    assert out.shape == (2, 5, 30) and torch.isfinite(out).all()


def test_training_with_triscan_is_reproducible(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["model"].update(encoder="triscan", triscan_exchange=True)
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
