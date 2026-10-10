"""Linear-cost global mixers, the count head and Muon: math and determinism."""

import copy

import pytest
import torch

from mathocr.model import AASAxis, Im2Latex, ModelConfig, SortAttention, _Permute, shift_dictionary, shift_map
from mathocr.optim import zeropower_via_newtonschulz5
from mathocr.train import train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("k", [0, 1, -1, 3, -4, 9])
def test_shift_map_is_a_zero_filled_translation(dim, k):
    x = torch.randn(2, 3, 5, 7)
    y = shift_map(x, k, dim)
    n = x.shape[dim]
    for p in range(n):
        src = p - k
        got = y.select(dim, p)
        assert torch.equal(got, x.select(dim, src) if 0 <= src < n else torch.zeros_like(got))


def test_shift_dictionary_stops_below_the_side():
    assert shift_dictionary(1) == [0]
    assert shift_dictionary(5) == [0, 1, -1, 2, -2, 4, -4]


def test_aas_is_translation_equivariant_away_from_the_border():
    torch.manual_seed(0)
    aas = AASAxis(16, 4, -1, max_power=3).eval()  # shifts 0, +-1, +-2, +-4
    with torch.no_grad():
        aas.action_bias.normal_()
    x = torch.randn(1, 16, 3, 40)
    with torch.no_grad():
        a, b = aas(x), aas(shift_map(x, 5, -1))
    # Columns whose whole shift dictionary stays inside the map in both inputs
    # (out-of-map shifts are masked, so the border is not equivariant).
    assert torch.allclose(shift_map(a, 5, -1)[..., 9:36], b[..., 9:36], atol=1e-5)
    assert not torch.allclose(a, b, atol=1e-3)


def test_sort_attention_is_permutation_equivariant():
    torch.manual_seed(0)
    att = SortAttention(16, 4, window=4).eval()
    x = torch.randn(2, 13, 16)
    perm = torch.randperm(13)
    with torch.no_grad():
        assert torch.allclose(att(x)[:, perm], att(x[:, perm]), atol=1e-5)


def test_permute_gradient_matches_gather():
    x = torch.randn(2, 3, 6, 4, dtype=torch.float64, requires_grad=True)
    order = torch.stack([torch.randperm(6) for _ in range(6)]).view(2, 3, 6)
    inverse = torch.argsort(order, dim=-1)
    torch.autograd.gradcheck(lambda t: _Permute.apply(t, order, inverse), (x,))


def test_newton_schulz_orthogonalizes():
    g = torch.randn(64, 32)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(g).float())
    assert s.min() > 0.5 and s.max() < 1.5


@pytest.mark.parametrize("encoder", ["aas", "aas_sort"])
def test_variants_keep_shape_and_decode(encoder):
    torch.manual_seed(0)
    m = Im2Latex(ModelConfig(vocab_size=30, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                             n_heads=4, enc_layers=1, dec_layers=1, max_len=12, encoder=encoder, sort_window=4,
                             count_head=True)).eval()
    x = torch.rand(2, 1, 64, 96)
    logits, aux = m(x, torch.ones(2, 5, dtype=torch.long), return_aux=True)
    assert logits.shape == (2, 5, 30) and aux["log_count"].shape == (2, 30)
    assert m.generate(x, 1, 2).shape[0] == 2


def _state(run):
    return torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]


@cuda
@pytest.mark.parametrize("overrides", [{"model": {"encoder": "aas"}},
                                       {"model": {"encoder": "aas_sort", "sort_window": 8}},
                                       {"model": {"count_head": True}, "train": {"count_loss": 0.5}},
                                       {"train": {"optimizer": "muon"}},
                                       {"train": {"token_dropout": 0.2}},
                                       {"model": {"fovea": True, "fovea_layer": 1, "fovea_levels": 3}},
                                       {"model": {"rings": True, "inject": True, "depth_pos": True}},
                                       {"model": {"ring_bias": True, "coverage": True}},
                                       {"model": {"deform_layers": 2}}])
def test_variant_training_is_bitwise_reproducible(tiny_config, tmp_path, overrides):
    cfg = copy.deepcopy(tiny_config)
    for section, values in overrides.items():
        cfg[section].update(values)
    train(copy.deepcopy(cfg), 7, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 7, tmp_path / "b", resume=False)
    a, b = _state(tmp_path / "a"), _state(tmp_path / "b")
    for k in a:
        assert torch.equal(a[k], b[k]), f"rerun differs at {k}"


def test_fovea_handles_the_smallest_canvas():
    torch.manual_seed(0)
    m = Im2Latex(ModelConfig(vocab_size=30, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                             n_heads=4, enc_layers=1, dec_layers=2, max_len=12, fovea=True, fovea_layer=1)).train()
    for shape in ((32, 32), (32, 768), (192, 64)):  # default 5 levels -> a 7-level pyramid
        x = torch.rand(2, 1, *shape)
        m(x, torch.ones(2, 5, dtype=torch.long)).sum().backward()
        assert m.eval().generate(x, 1, 2).shape[0] == 2
        m.train()


def test_stacking_is_deterministic_and_shape_consistent(tiny_data):
    from functools import partial

    from torch.utils.data import DataLoader

    from mathocr.data.dataset import AugmentConfig, CropStore, ShapeBatchSampler, TrainItems, collate
    from mathocr.image import ImageConfig
    from mathocr.tokenizer import Vocab

    base = Vocab.load(tiny_data / "vocab.json")  # extended at the end: existing ids are unchanged
    vocab = Vocab(base.itos + [t for t in ("\\begin{array}", "\\end{array}", "l", "\\\\") if t not in base])
    store = CropStore(tiny_data / "train.parquet", vocab)
    cfg, aug = ImageConfig(max_h=384, max_w=1536), AugmentConfig(p_stack=0.6)

    def run(workers):
        sampler = ShapeBatchSampler(store, cfg, 400_000, 8, 5, train=True, aug_cfg=aug, max_tokens=64)
        sampler.set_epoch(1)
        loader = DataLoader(TrainItems(store, cfg, aug, 5), batch_sampler=sampler, num_workers=workers,
                            collate_fn=partial(collate, pad_id=0))
        return list(loader), sampler.batches()

    a, keys = run(0)
    b, _ = run(2)
    assert any(len(k) > 3 for batch in keys for k in batch), "no composition drawn"
    for x, y in zip(a, b):
        assert torch.equal(x["images"], y["images"]) and torch.equal(x["tokens"], y["tokens"])
    stacked = next(k for batch in keys for k in batch if len(k) > 3)
    _, target, _ = TrainItems(store, cfg, aug, 5)[stacked]
    words = vocab.decode(target.tolist())
    assert words[:4] == ["\\begin{array}", "{", "l", "}"] and words[-1] == "\\end{array}"
    assert words.count("\\\\") == len(stacked[3]) and len(target) <= 64
