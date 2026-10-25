"""Canonical image preprocessing: any formula image -> model input array.

``prepare`` is the single path used for evaluation and inference, so a given
image always becomes the same array.  It is split into two halves so the
dataset builder can run the expensive half once:

``analyze`` (source-dependent, run once per image)
    1. grayscale on white (alpha composited); light-on-dark is inverted;
    2. contrast stretch: background (median) -> 255, darkest ink -> 0
       (screenshots have grey paper, renders pure white);
    3. crop to the ink bounding box;
    4. measure the glyph height: 75th percentile of the heights of
       connected ink components at least 3 px tall (fraction bars, dots and
       antialiasing specks excluded).  On im2latex, rendered at one font
       size, this estimate varies by CV 0.18 -- the best of the simple
       estimators tried (plain median: 0.22).

``layout`` (cheap, may take a training-time scale jitter)
    5. rescale so the glyph height equals ``target_glyph_px`` -- sources
       differ in DPI by up to 2.5x (this estimate: im2latex ~12 px,
       UniMER CPE ~19 px, SCE ~27 px);
    6. shrink further if the result does not fit ``max_h x max_w``;
    7. pad with a white margin up to a multiple of ``multiple``.

Shapes are multiples of ``multiple``, so batches are built from images of
*identical* shape and the network never sees padding that ``prepare``
would not produce for that image alone.

Arrays are ``uint8`` with 0 = ink, 255 = paper; ``to_tensor`` inverts them to
ink intensity so the CNN's zero padding means empty paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image


@dataclass(frozen=True)
class ImageConfig:
    target_glyph_px: float = 14.0
    max_h: int = 192
    max_w: int = 768
    margin: int = 8
    multiple: int = 32
    min_scale: float = 0.1
    max_scale: float = 4.0


@dataclass(frozen=True)
class Crop:
    pixels: np.ndarray  # uint8 (h, w), 0 = ink, 255 = paper
    glyph_px: float  # median glyph height in source pixels; 0.0 if blank


def to_gray(image: Image.Image) -> np.ndarray:
    """``float32`` grayscale in [0, 255], transparency composited on white."""

    if image.mode in ("RGBA", "LA", "P", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        image = canvas
    return np.asarray(image.convert("L"), dtype=np.float32)


def _stretch(gray: np.ndarray) -> np.ndarray:
    background = float(np.median(gray))
    if background < 128.0:
        gray = 255.0 - gray
        background = 255.0 - background
    darkest = float(gray.min())
    span = background - darkest
    if span < 1.0:  # blank image
        return np.full_like(gray, 255.0)
    return np.clip((gray - darkest) * (255.0 / span), 0.0, 255.0)


def glyph_height(ink: np.ndarray) -> float:
    """75th percentile height of connected ink components >= 3 px tall."""

    labels, count = ndi.label(ink, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return 0.0
    heights = np.array([s[0].stop - s[0].start for s in ndi.find_objects(labels)])
    tall = heights[heights >= 3]
    if not tall.size:
        return float(heights.max())
    return float(np.percentile(tall, 75))


def analyze(image: Image.Image) -> Crop:
    stretched = _stretch(to_gray(image))
    ink = stretched < 128.0
    if not ink.any():
        return Crop(np.full((1, 1), 255, dtype=np.uint8), 0.0)
    ys = np.flatnonzero(ink.any(axis=1))
    xs = np.flatnonzero(ink.any(axis=0))
    window = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    pixels = np.rint(stretched[window]).astype(np.uint8)
    return Crop(pixels, glyph_height(ink[window]))


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _scaled_size(h: int, w: int, glyph_px: float, cfg: ImageConfig, jitter: float) -> tuple[int, int]:
    scale = cfg.target_glyph_px / glyph_px if glyph_px > 0 else 1.0
    scale = min(max(scale * jitter, cfg.min_scale), cfg.max_scale)
    scale = min(scale, (cfg.max_h - 2 * cfg.margin) / h, (cfg.max_w - 2 * cfg.margin) / w)
    return max(1, round(h * scale)), max(1, round(w * scale))


def output_shape(h: int, w: int, glyph_px: float, cfg: ImageConfig, jitter: float = 1.0) -> tuple[int, int]:
    """Shape ``layout`` will produce, computable from crop metadata alone."""

    new_h, new_w = _scaled_size(h, w, glyph_px, cfg, jitter)
    return _round_up(new_h + 2 * cfg.margin, cfg.multiple), _round_up(new_w + 2 * cfg.margin, cfg.multiple)


def layout(crop: Crop, cfg: ImageConfig, jitter: float = 1.0) -> np.ndarray:
    h, w = crop.pixels.shape
    new_h, new_w = _scaled_size(h, w, crop.glyph_px, cfg, jitter)
    resized = Image.fromarray(crop.pixels, mode="L").resize((new_w, new_h), Image.Resampling.BICUBIC)
    out = np.full(output_shape(h, w, crop.glyph_px, cfg, jitter), 255, dtype=np.uint8)
    out[cfg.margin : cfg.margin + new_h, cfg.margin : cfg.margin + new_w] = np.asarray(resized)
    return out


def layout_into(crop: Crop, cfg: ImageConfig, shape: tuple[int, int], jitter: float = 1.0) -> np.ndarray:
    """``layout`` onto a given canvas ``shape``: the same glyph scale, shrunk further if the crop would not fit
    (a twin render of one label in another font lands on its original's canvas, so both batch together)."""

    h, w = crop.pixels.shape
    new_h, new_w = _scaled_size(h, w, crop.glyph_px, cfg, jitter)
    fit = min(1.0, (shape[0] - 2 * cfg.margin) / new_h, (shape[1] - 2 * cfg.margin) / new_w)
    new_h, new_w = max(1, int(new_h * fit)), max(1, int(new_w * fit))
    resized = Image.fromarray(crop.pixels, mode="L").resize((new_w, new_h), Image.Resampling.BICUBIC)
    out = np.full(shape, 255, dtype=np.uint8)
    out[cfg.margin : cfg.margin + new_h, cfg.margin : cfg.margin + new_w] = np.asarray(resized)
    return out


def prepare(image: Image.Image, cfg: ImageConfig) -> np.ndarray:
    """Canonical model input for ``image`` (no augmentation)."""

    return layout(analyze(image), cfg)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(1, H, W) float32 ink intensity in [0, 1]: paper = 0, ink = 1."""

    return torch.from_numpy(255 - arr).to(torch.float32).div_(255.0).unsqueeze(0)
