"""Graphed greedy decoding: same strings as the eager loop, deterministic across decoders and runs."""

import pytest
import torch

from mathocr.data.dataset import CropStore
from mathocr.image import ImageConfig
from mathocr.infer import decode_store, predict_images
from mathocr.model import Im2Latex, ModelConfig
from mathocr.tokenizer import Vocab

from conftest import render

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graphs")


def _model(vocab, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=len(vocab), cnn_channels=(16, 16, 32, 32), cnn_blocks=(1, 1, 1, 1), d_model=64,
                      n_heads=4, enc_layers=1, dec_layers=2, max_len=24, **kw)
    return Im2Latex(cfg).cuda().eval()


@cuda
def test_graphed_decoding_matches_eager(tiny_data, monkeypatch):
    from mathocr import determinism

    determinism.configure(torch.device("cuda"))
    vocab = Vocab.load(tiny_data / "vocab.json")
    store = CropStore(tiny_data / "val.parquet", vocab)
    icfg, dev = ImageConfig(), torch.device("cuda")
    images = [render(t, size=s) for t, s in (("x+y=2", 20), ("a2+b2", 26), ("xyz", 18))]
    for seed in (0, 1):  # random weights: outputs are garbage, but every token must still agree
        model = _model(vocab, seed)
        monkeypatch.setenv("MATHOCR_CUDA_GRAPHS", "0")
        eager = decode_store(model, store, icfg, vocab, dev, 200_000, 8)  # several shapes and batch sizes
        eager_single = predict_images(model, images, icfg, vocab, dev)
        monkeypatch.setenv("MATHOCR_CUDA_GRAPHS", "1")
        graphed = decode_store(model, store, icfg, vocab, dev, 200_000, 8)
        graphed_single = predict_images(model, images, icfg, vocab, dev)
        assert graphed == eager
        assert graphed_single == eager_single
        assert graphed == decode_store(model, store, icfg, vocab, dev, 200_000, 8)  # fresh decoder, same result
        from mathocr.graphs import GraphedGreedy

        tiny_cache = GraphedGreedy(model, vocab.bos_id, vocab.eos_id, cache_bytes=1)  # every key evicts the last
        xs = [torch.rand(1, 1, 64, 32 * w, device="cuda") for w in (2, 3, 2, 4, 3)]
        first = [tiny_cache.decode(x).tolist() for x in xs]
        assert len(tiny_cache.cache) == 1
        assert [GraphedGreedy(model, vocab.bos_id, vocab.eos_id).decode(x).tolist() for x in xs] == first


@cuda
def test_graphed_decoding_stops_on_eos_like_eager(tiny_data):
    """A model that always emits EOS: both paths return it at once (EOS after EOS semantics)."""

    vocab = Vocab.load(tiny_data / "vocab.json")
    model = _model(vocab)
    with torch.no_grad():
        model.dec_norm.weight.zero_()
        model.dec_norm.bias.zero_()
        model.dec_norm.bias[0] = 1.0  # logits = embed @ bias-direction; make EOS win
        model.embed.weight.zero_()
        model.embed.weight[vocab.eos_id, 0] = 1.0
    x = torch.rand(3, 1, 64, 96, device="cuda")
    from mathocr.graphs import GraphedGreedy

    eager = model.generate(x, vocab.bos_id, vocab.eos_id)
    g = GraphedGreedy(model, vocab.bos_id, vocab.eos_id).decode(x)
    assert eager[:, 0].tolist() == [vocab.eos_id] * 3 and (g == vocab.eos_id).all()


def test_variants_fall_back_to_eager(tiny_data):
    from mathocr import graphs

    vocab = Vocab.load(tiny_data / "vocab.json")
    torch.manual_seed(0)
    plain = Im2Latex(ModelConfig(vocab_size=len(vocab), cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                                 n_heads=4, enc_layers=1, dec_layers=1, max_len=12))
    fovea = Im2Latex(ModelConfig(vocab_size=len(vocab), cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                                 n_heads=4, enc_layers=1, dec_layers=2, max_len=12, fovea=True, fovea_layer=1))
    assert graphs.supported(plain) and not graphs.supported(fovea)
    assert not graphs.enabled(plain, torch.device("cpu"))


@cuda
@pytest.mark.parametrize("failures", [1, 2])
def test_failed_capture_recovers(tiny_data, failures):
    """A capture that runs out of memory (as cuMemAddressReserve did on Windows) is retried with a fresh pool;
    a second failure decodes eagerly; either way the ids are the eager ones and later captures work."""

    from mathocr.graphs import GraphedGreedy

    vocab = Vocab.load(tiny_data / "vocab.json")
    model = _model(vocab)
    x = torch.rand(2, 1, 64, 96, device="cuda")
    ref = model.generate(x, vocab.bos_id, vocab.eos_id)
    g = GraphedGreedy(model, vocab.bos_id, vocab.eos_id)
    real, left = g._step, [failures]

    def flaky():
        if torch.cuda.is_current_stream_capturing() and left[0] > 0:
            left[0] -= 1
            raise RuntimeError("CUDA driver error: out of memory")
        real()

    g._step = flaky
    out = g.decode(x)
    assert not torch.cuda.is_current_stream_capturing()
    assert torch.equal(out[:, : ref.shape[1]], ref) and g.eager_fallbacks == (failures == 2)
    g._step = real
    assert torch.equal(g.decode(x)[:, : ref.shape[1]], ref) and (2, 64, 96) in g.cache
