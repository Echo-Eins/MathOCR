"""Rasterization augmentation (augment.p_raster / p_gamma / p_jpeg): off by default without extra random draws,
shape- and dtype-preserving, deterministic per generator state."""

import numpy as np

from mathocr.data.dataset import AugmentConfig, augment, rasterize


def _image() -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = np.full((64, 160), 255, np.uint8)
    arr[20:44, 10:150] = rng.integers(0, 255, size=(24, 140), dtype=np.uint8)
    return arr


def test_disabled_rasterization_draws_nothing():
    arr, cfg = _image(), AugmentConfig()
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    augment(arr, r1, cfg)
    augment(arr, r2, AugmentConfig(p_raster=0.0, p_gamma=0.0, p_jpeg=0.0))
    assert r1.random() == r2.random()  # same number of draws as before the option existed


def test_rasterization_keeps_shape_and_is_deterministic():
    arr, cfg = _image(), AugmentConfig(p_raster=1.0, p_gamma=1.0, p_jpeg=1.0)
    a = rasterize(arr, np.random.default_rng(3), cfg)
    b = rasterize(arr, np.random.default_rng(3), cfg)
    assert a.shape == arr.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, arr)
