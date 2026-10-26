"""Rotation (augment.p_rotate): off by default without extra draws (same batches as before), the sampler's shapes
match the loader's images, and a turned sample is the clockwise rotation of its plain layout."""

import numpy as np

from mathocr.data.dataset import AugmentConfig, CropStore, ShapeBatchSampler, TrainItems
from mathocr.image import ImageConfig
from mathocr.tokenizer import Vocab


def _store(data):
    base = Vocab.load(data / "vocab.json")  # + the array tokens the stacking writes
    vocab = Vocab(base.itos + [t for t in ("\\begin{array}", "\\end{array}", "l", "r", "&", "\\\\") if t not in base])
    return CropStore(data / "train.parquet", vocab)


def test_disabled_rotation_keeps_the_batches(tiny_data):
    store, icfg = _store(tiny_data), ImageConfig()
    a = ShapeBatchSampler(store, icfg, 200_000, 8, seed=0, train=True, aug_cfg=AugmentConfig(), max_tokens=64)
    b = ShapeBatchSampler(store, icfg, 200_000, 8, seed=0, train=True, aug_cfg=AugmentConfig(p_rotate=0.0), max_tokens=64)
    assert a.batches() == b.batches()


def test_sampler_shapes_match_turned_images(tiny_data):
    store, icfg = _store(tiny_data), ImageConfig()
    aug = AugmentConfig(p_rotate=0.5, p_stack=0.3, stack_style="l")
    sampler = ShapeBatchSampler(store, icfg, 200_000, 8, seed=0, train=True, aug_cfg=aug, max_tokens=200)
    items, plain = TrainItems(store, icfg, aug, seed=0), TrainItems(store, icfg, None, seed=0)
    turned = 0
    for batch in sampler:
        shapes = {items[key][0].shape for key in batch}
        assert len(shapes) == 1
        for key in batch:
            if key[4]:
                turned += 1
                assert not key[3]  # compositions are never turned
                ref = plain[(key[0], key[1], key[2])][0][0].numpy()
                np.testing.assert_array_equal(np.rot90(ref, k=-1), TrainItems(store, icfg, None, 0)[key][0][0].numpy())
    assert turned > 0
