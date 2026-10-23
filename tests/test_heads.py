"""Readout heads on top of the tied output layer (model.head): start exactly at the tied head, the incremental
(KV-cache) decoding path agrees with teacher forcing, and the graphed decoder carries the cells head."""

import pytest
import torch

from mathocr.model import Im2Latex, ModelConfig

HEADS = ("cells", "cells_ml", "cells_vis", "cells_det")


def _model(head: str, seed: int = 0) -> Im2Latex:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=40, cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32, n_heads=4,
                      enc_layers=1, dec_layers=2, max_len=16, head=head, head_k=4)
    return Im2Latex(cfg).eval()


def test_default_model_is_unchanged():
    m = _model("tied")
    assert m.head is None and not any(k.startswith("head") for k in m.state_dict())
    assert torch.equal(_model("tied").embed.weight, _model("cells").embed.weight)  # optional parts init last


@pytest.mark.parametrize("head", HEADS)
def test_heads_start_at_the_tied_head(head):
    tied, m = _model("tied"), _model(head)
    m.load_state_dict({**m.state_dict(), **tied.state_dict()})
    x, tok = torch.rand(2, 1, 32, 64), torch.randint(4, 40, (2, 7))
    with torch.no_grad():
        assert torch.allclose(m(x, tok), tied(x, tok), atol=1e-6)


@pytest.mark.parametrize("head", HEADS)
def test_incremental_decoding_matches_teacher_forcing(head):
    m = _model(head)
    with torch.no_grad():  # make the head matter
        m.head.w2.normal_(0, 1.0)
        m.head.lateral.normal_(0, 0.05)
        if m.head_mix is not None:
            m.head_mix.mix.normal_(0, 1.0)
        if m.head_det is not None:
            m.head_det.proj.weight.normal_(0, 1.0)
    x = torch.rand(2, 1, 32, 64)
    ids = m.generate(x, 1, 2)
    inp = torch.cat([torch.ones(2, 1, dtype=torch.long), ids[:, :-1]], dim=1)
    with torch.no_grad():
        greedy_tf = m(x, inp).argmax(-1)
    for b in range(2):  # until the first EOS the cached path must pick what the full pass picks
        n = int((ids[b] == 2).nonzero()[0]) + 1 if (ids[b] == 2).any() else ids.shape[1]
        assert torch.equal(greedy_tf[b, :n], ids[b, :n])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graphs")
def test_graphed_decoding_carries_the_cells_head():
    from mathocr import graphs

    m = _model("cells").cuda()
    with torch.no_grad():
        m.head.w2.normal_(0, 1.0)
        m.head.lateral.normal_(0, 0.05)
    assert graphs.supported(m) and not graphs.supported(_model("cells_ml")) and not graphs.supported(_model("cells_vis"))
    x = torch.rand(3, 1, 32, 96, device="cuda")
    eager = m.generate(x, 1, 2)
    g = graphs.GraphedGreedy(m, 1, 2)
    assert torch.equal(g.decode(x)[:, : eager.shape[1]], eager)
