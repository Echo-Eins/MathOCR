"""Transformer-layer variants (attention output gates, QK-norm, value residual) and parameter-free multi-token
prediction: the variants leave every shared parameter's initialization unchanged, MTP with zero weights is the plain
loss, and training with all of them is bitwise reproducible.  Decoding equivalences (cached vs full forward, graphed
vs eager) are in test_image_model.py and test_graphs.py."""

import copy

import torch

from mathocr.model import Im2Latex, ModelConfig
from mathocr.tokenizer import Vocab
from mathocr.train import _loss, train

VARIANT = {"attn_gate": "head", "qk_norm": True, "value_residual": True}


def _model(**kw) -> Im2Latex:
    torch.manual_seed(0)
    return Im2Latex(ModelConfig(vocab_size=40, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                                n_heads=4, enc_layers=2, dec_layers=3, max_len=16, **kw))


def test_variants_leave_shared_initialization_unchanged():
    plain = _model().state_dict()
    for variant in (VARIANT, {"attn_gate": "elem"}):
        m = _model(**variant).state_dict()
        assert set(plain) < set(m)
        assert all(torch.equal(plain[k], m[k]) for k in plain)
        assert all(float(m[k].abs().sum()) == 0 for k in m if ".out_gate." in k or k.endswith(".v_mix"))
    assert "decoder.0.v_mix" not in _model(**VARIANT).state_dict()  # layer 0 gives the values, mixes nothing


def test_mtp_with_zero_weights_is_the_plain_loss(tiny_data):
    vocab = Vocab.load(tiny_data / "vocab.json")
    torch.manual_seed(0)
    m = Im2Latex(ModelConfig(vocab_size=len(vocab), cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                             n_heads=4, enc_layers=1, dec_layers=2, max_len=16)).eval()
    tokens = torch.randint(4, len(vocab), (2, 9))
    tokens[:, 0], tokens[0, 6:], tokens[0, 5] = vocab.bos_id, vocab.pad_id, vocab.eos_id
    batch = {"images": torch.rand(2, 1, 32, 64), "tokens": tokens}
    dev = torch.device("cpu")
    with torch.no_grad():
        base, n, _, _ = _loss(m, batch, dev, vocab, 0.1, False)
        zero, _, _, parts0 = _loss(m, batch, dev, vocab, 0.1, False, mtp=[0.0, 0.0])
        more, _, _, parts = _loss(m, batch, dev, vocab, 0.1, False, mtp=[0.5, 0.25])
    assert torch.equal(base, zero) and "mtp_ce" not in parts0
    assert torch.allclose(more, base + parts["mtp_ce"]) and float(parts["mtp_ce"]) > 0


def test_training_with_layer_variants_and_mtp_is_reproducible(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["model"].update(VARIANT)
    cfg["train"].update(optimizer="muon", mtp=[0.5, 0.25], mtp_until=0.5)
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(float(a[k].abs().sum()) > 0 for k in a if ".out_gate." in k)  # the gates learned


def test_polar_express_orthogonalizes_tighter():
    from mathocr.optim import zeropower_via_newtonschulz5

    torch.manual_seed(0)
    g = torch.randn(3, 64, 32) @ torch.diag(torch.logspace(-2, 0, 32))  # ill-conditioned: singular values 0.01..1
    s_q = torch.linalg.svdvals(zeropower_via_newtonschulz5(g).float())
    s_p = torch.linalg.svdvals(zeropower_via_newtonschulz5(g, polar=True).float())
    assert s_p.min() > 0.5 and s_p.max() < 1.5
    assert (s_p - 1).abs().max() < (s_q - 1).abs().max()  # the fixed quintic oscillates around 1 (0.7..1.2)


def test_normuon_step_has_the_muon_rms():
    from mathocr.optim import Muon

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(48, 96))
    w0 = w.detach().clone()
    opt = Muon([{"params": [w], "use_muon": True, "weight_decay": 0.0}], lr=1e-2, neuron_norm=True)
    w.grad = torch.randn(48, 96) * torch.linspace(0.1, 3.0, 48)[:, None]  # rows of very different scale
    opt.step()
    rms = float((w.detach() - w0).pow(2).mean().sqrt())
    assert abs(rms - 0.2 * 1e-2) < 1e-5  # rescaled to RMS 0.2 (x lr), as the plain Muon update
    assert opt.state[w]["neuron_v"].shape == (48, 1)


def test_training_with_polar_normuon_is_reproducible(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["train"].update(optimizer="muon", muon_polar=True, muon_neuron_norm=True)
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


SKIPS = {"dense_memory": True, "fine_memory": True, "cnn_blur": True, "enc_stride": "8x4"}


def test_skips_leave_shared_initialization_unchanged():
    plain = _model(enc_stride="8x4").state_dict()
    m = _model(**SKIPS).state_dict()
    assert set(plain) < set(m)
    assert all(torch.equal(plain[k], m[k]) for k in plain)


def test_dense_memory_starts_as_the_plain_memory():
    plain, dense = _model().eval(), _model(dense_memory=True).eval()
    dense.load_state_dict({**dense.state_dict(), **plain.state_dict()})
    x, tok = torch.rand(2, 1, 32, 64), torch.randint(4, 40, (2, 7))
    with torch.no_grad():
        assert torch.allclose(dense(x, tok), plain(x, tok), atol=1e-6)
        assert dense._levels is None and dense._fine is None  # taken by the decoder, not kept


def test_blur_keeps_every_shape():
    for stride in ("16x8", "8x8", "8x4"):
        a, b = _model(enc_stride=stride).eval(), _model(enc_stride=stride, cnn_blur=True).eval()
        for hw in ((32, 64), (35, 70), (64, 96)):
            x = torch.rand(1, 1, *hw)
            with torch.no_grad():
                assert a.encode(x).shape == b.encode(x).shape


def test_training_with_skips_is_reproducible(tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["model"].update(SKIPS)
    cfg["train"].update(optimizer="muon")
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert float(a["dense_gain"].abs().sum()) > 0  # the skips learned
