import numpy as np
import pytest
import torch
from PIL import Image

from mathocr.image import ImageConfig, analyze, layout, output_shape, prepare
from mathocr.model import Im2Latex, ModelConfig

from conftest import render


def test_prepare_invariants():
    cfg = ImageConfig()
    small, big = render("x+y=2", size=16), render("x+y=2", size=40)
    a, b = prepare(small, cfg), prepare(big, cfg)
    for arr in (a, b):
        assert arr.dtype == np.uint8 and arr.shape[0] % 32 == 0 and arr.shape[1] % 32 == 0
        assert arr.shape[0] <= cfg.max_h and arr.shape[1] <= cfg.max_w
    # glyph normalization maps different font sizes to similar canvases
    assert abs(a.shape[1] - b.shape[1]) <= 64
    assert np.array_equal(prepare(small, cfg), prepare(small.copy(), cfg))


def test_output_shape_matches_layout_and_fits():
    cfg = ImageConfig()
    crop = analyze(render("a very long formula " * 8, size=30))
    for jitter in (0.75, 1.0, 1.34):
        arr = layout(crop, cfg, jitter)
        assert arr.shape == output_shape(*crop.pixels.shape, crop.glyph_px, cfg, jitter)
        assert arr.shape[1] <= cfg.max_w


def test_inverted_and_transparent_inputs():
    cfg = ImageConfig()
    im = render("x+y=2", size=24)
    inv = Image.eval(im, lambda v: 255 - v)
    assert np.array_equal(prepare(im, cfg), prepare(inv, cfg))
    rgba = im.convert("L")
    rgba = Image.merge("RGBA", (Image.new("L", im.size, 0),) * 3 + (Image.eval(rgba, lambda v: 255 - v),))
    assert np.abs(prepare(im, cfg).astype(int) - prepare(rgba, cfg).astype(int)).max() <= 2


@pytest.mark.parametrize("variant", [{}, {"encoder": "aas"}, {"encoder": "aas_sort", "sort_window": 4},
                                     {"rings": True, "inject": True}, {"depth_pos": True, "open_id": 1, "close_id": 3},
                                     {"fovea": True, "fovea_layer": 1, "fovea_levels": 3},
                                     {"count_head": True, "count_feed": True},
                                     {"ring_bias": True, "coverage": True},
                                     {"deform_layers": 2},
                                     {"enc_stride": "8x4", "ring_bias": True, "coverage": True},
                                     {"attn_gate": "head", "qk_norm": True, "value_residual": True},
                                     {"attn_gate": "elem", "value_residual": True, "ring_bias": True, "coverage": True}])
def test_cached_greedy_equals_full_forward(variant):
    torch.manual_seed(0)
    m = Im2Latex(ModelConfig(vocab_size=50, cnn_channels=(16, 16, 32, 32), cnn_blocks=(1, 1, 1, 1),
                             d_model=64, n_heads=4, enc_layers=1, dec_layers=2, max_len=24, **variant)).eval()
    if getattr(m, "feedback", None) is not None:  # tables start at zero: make the feedback act
        with torch.no_grad():
            for f in m.feedback:
                f.ring_table.normal_(0, 1.0)
                f.coverage_gamma.fill_(0.5)
    for name, p in m.named_parameters():  # gates and value shares start neutral: make them act
        if ".out_gate." in name or name.endswith(".v_mix") or name.endswith((".q_norm.weight", ".k_norm.weight")):
            with torch.no_grad():
                p.normal_(0, 1.0)
    x = torch.rand(3, 1, 64, 160)
    out = m.generate(x, bos_id=1, eos_id=2)
    if "depth_pos" in variant:  # the untrained model repeats BOS: with BOS as "open" the depth keeps changing
        assert bool((out == 1).any())
    seq = torch.full((3, 1), 1)
    for _ in range(out.shape[1]):
        nxt = m(x, seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
    ref = seq[:, 1:]
    for row in range(3):  # after EOS, generate pads with EOS
        ended = (ref[row] == 2).nonzero()
        stop = int(ended[0]) + 1 if len(ended) else ref.shape[1]
        assert torch.equal(out[row, :stop], ref[row, :stop])
