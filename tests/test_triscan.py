"""TriScanBlock (model.encoder = "triscan") at full reach: the scan kernel matches its reference and is
reproducible; the row scans see their whole side of the row and nothing of the other; the column convolution
reaches the whole height and tells up from down; the diagonal stream reaches along both diagonals and only there;
the anchors read only the formula's ink; the exchange variant starts as the plain block; dw_ parameters train on
AdamW; training is bitwise reproducible."""

import copy

import pytest
import torch

from mathocr.model import (ColumnConv, DiagonalConv, Im2Latex, ModelConfig, RowScan, TriScanBlock, shear, thirds,
                           unshear)
from mathocr.optim import build_optimizer
from mathocr.scan_kernel import linear_scan, linear_scan_reference
from mathocr.train import train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _init(m):
    for mod in m.modules():
        if hasattr(mod, "init_extra"):
            mod.init_extra()
    return m


@cuda
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", [(3, 7, 70), (5, 100, 64), (2, 33, 130)])
def test_scan_kernel_matches_reference(shape, reverse):
    torch.manual_seed(0)
    a = torch.rand(*shape, device="cuda", requires_grad=True)
    b = torch.randn(*shape, device="cuda", requires_grad=True)
    g = torch.randn(*shape, device="cuda")
    runs = []
    for fn in (linear_scan, linear_scan_reference, linear_scan):
        h = fn(a, b, reverse)
        da, db = torch.autograd.grad((h * g).sum(), (a, b))
        runs.append((h, da, db))
    for x, y in zip(runs[0], runs[1]):
        assert torch.allclose(x, y, atol=1e-5)
    assert all(torch.equal(x, y) for x, y in zip(runs[0], runs[2]))  # bitwise repeatable


@pytest.mark.parametrize("reverse", [False, True])
def test_row_scan_sees_its_whole_side_only(reverse):
    torch.manual_seed(0)
    scan = _init(RowScan(16, 4, reverse=reverse))
    x = torch.randn(1, 2, 60, 16)
    y = x.clone()
    edge = 0 if not reverse else 59
    y[:, :, edge] += 3.0
    mid = x.clone()
    mid[:, :, 30] += 3.0
    with torch.no_grad():
        base = scan(x, x)
        far = (scan(y, y) - base).abs().amax(dim=(0, 1, 3))
        changed = (scan(mid, mid) - base).abs().amax(dim=(0, 1, 3)) > 0
    assert far[59 - edge] > 0  # the far end of the row hears the first cell
    if reverse:  # anti-causal: only columns at or left of 30
        assert not changed[31:].any() and changed[:31].all()
    else:
        assert not changed[:30].any() and changed[30:].all()


def test_column_conv_reaches_the_whole_height_and_tells_up_from_down():
    torch.manual_seed(0)
    conv = _init(ColumnConv(4))
    with torch.no_grad():
        conv.dw_tail.normal_()
        conv.dw_near[..., : conv.near] = 0.0  # no taps from above within the near field
        conv.dw_tail[:, :, 0] = 0.0  # nor beyond it
    x = torch.zeros(1, 40, 9, 4)
    x[0, 20, 4] = 1.0
    with torch.no_grad():
        y = conv(x)[0].abs().sum(-1)  # (40, 9)
    # the source at row 20 is read only from above it (targets at rows <= 20), up to row 0, in columns 3-5
    assert y[21:].sum() == 0 and y[0].sum() > 0 and y[20].sum() > 0
    assert y[:, :3].sum() == 0 and y[:, 6:].sum() == 0
    h = 5  # shorter maps cut the kernel to the map
    assert conv.kernel(h).shape[-1] == 2 * h - 1


def test_shear_roundtrip_and_lines():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 11, 3)
    s = shear(x, 2)
    assert s.shape == (2, 6, 11 + 2 * 5, 3)
    assert torch.equal(unshear(s, 2, 11), x)
    for r in range(6):  # row r moved right by 2r, zeros elsewhere
        assert torch.equal(s[:, r, 2 * r : 2 * r + 11], x[:, r]) and s[:, r, : 2 * r].abs().sum() == 0


def test_diagonal_conv_reaches_along_both_diagonals_only():
    torch.manual_seed(0)
    conv = _init(DiagonalConv(1))
    with torch.no_grad():
        for c in (conv.rising, conv.falling):
            c.dw_near.fill_(1.0)
            c.dw_tail.fill_(1.0)
    x = torch.zeros(1, 21, 61, 1)
    x[0, 10, 30] = 1.0
    with torch.no_grad():
        y = conv(x)[0, :, :, 0]
    hit = (y.abs() > 1e-6).nonzero().tolist()
    # every reached cell lies within one column of a 2-cells-per-row diagonal through the source
    assert all(min(abs((c - 30) - 2 * (r - 10)), abs((c - 30) + 2 * (r - 10))) <= 1 for r, c in hit)
    assert [0, 30 + 20] in hit or [0, 30 + 20 - 1] in hit or [0, 30 + 20 + 1] in hit  # 10 rows up, up-right
    assert any(r == 20 for r, _ in hit) and any(r == 0 for r, _ in hit)


def test_thirds_read_only_the_ink():
    torch.manual_seed(0)
    y = torch.randn(2, 5, 30, 4)
    ink = torch.zeros(2, 5, 30, dtype=torch.bool)
    ink[0, 1:3, 6:24] = True  # a formula in columns 6-23
    a = thirds(y, ink)
    y2 = y.clone()
    y2[0, :, :6] += 10.0  # off the extent
    y2[0, 4] += 10.0  # a blank row within it
    assert torch.allclose(thirds(y2, ink)[0], a[0])
    assert torch.allclose(a[0, :4], y[0, 1:3, 6:12].mean(dim=(0, 1)), atol=1e-6)  # the start: columns 6-11
    assert torch.allclose(a[1], torch.cat([y[1][:, k * 10 : (k + 1) * 10].mean(dim=(0, 1)) for k in range(3)]), atol=1e-6)


def test_exchange_block_starts_as_the_plain_block():
    torch.manual_seed(0)
    plain = _init(TriScanBlock(16, 4, 0.0, "hvd", exchange=False, rounds=2)).eval()
    ex = _init(TriScanBlock(16, 4, 0.0, "hvd", exchange=True, rounds=2)).eval()
    ex.load_state_dict({**ex.state_dict(), **plain.state_dict()})
    x = torch.randn(2, 6, 30, 16)
    ink = torch.rand(2, 6, 30) > 0.5
    with torch.no_grad():
        assert torch.allclose(ex(x, ink), plain(x, ink), atol=1e-6)


@pytest.mark.parametrize("streams,exchange", [("hd", True), ("vd", False), ("hv", False), ("hvd", False), ("hvd", True)])
def test_triscan_model_forward(streams, exchange):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=30, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32, n_heads=4,
                      enc_layers=2, dec_layers=1, max_len=16, encoder="triscan", triscan_streams=streams,
                      triscan_exchange=exchange, enc_stride="8x4")
    m = Im2Latex(cfg).eval()
    out = m(torch.rand(2, 1, 64, 96), torch.randint(4, 30, (2, 5)))
    assert out.shape == (2, 5, 30) and torch.isfinite(out).all()


def test_dw_parameters_train_on_adamw():
    cfg = ModelConfig(vocab_size=30, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32, n_heads=4,
                      enc_layers=2, dec_layers=1, max_len=16, encoder="triscan", triscan_streams="hvd")
    m = Im2Latex(cfg)
    opt = build_optimizer(m, {"lr": 1e-3, "weight_decay": 0.01, "optimizer": "muon"})
    muon = {id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    names = {n for n, p in m.named_parameters() if ".dw_" in n}
    assert names and not any(id(p) in muon for n, p in m.named_parameters() if n in names)
    assert id(m.encoder[0].value.weight) in muon


def test_training_with_triscan_is_reproducible(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["model"].update(encoder="triscan", triscan_streams="hvd", triscan_exchange=True)
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
