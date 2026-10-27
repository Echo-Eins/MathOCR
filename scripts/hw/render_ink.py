"""Online ink (MathWriting strokes) -> an offline image of handwriting on paper.

``render(xy, ends, seed)`` is a pure function of its inputs: every random choice comes from ``seed`` (the builders
pass a sha256 of the ink id), so a split's images are rebuilt byte for byte.  Per sample: a glyph size of 26..44 px
(median stroke extent), a pen of 5.5..12 % of it, a small rotation / shear / aspect change and per-stroke offsets
(less tidy writing), then the paper: tone, ink tone, ruled lines (25 %) or a grid (10 %), a lighting gradient (30 %),
noise, blur and JPEG.  Lines and gradients stay lighter than half the ink contrast, so ``image.analyze`` crops to the
writing, not to the page.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MAX_SIDE = 2400
SS = 2  # supersampling of the stroke drawing


def _strokes(xy: np.ndarray, ends: np.ndarray) -> list[np.ndarray]:
    pts = np.asarray(xy, np.float64).reshape(-1, 2)
    out, start = [], 0
    for e in ends:
        if e > start:
            out.append(pts[start:e])
        start = e
    return out


def render(xy, ends, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    strokes = _strokes(np.asarray(xy), np.asarray(ends))
    ext = np.array([max(np.ptp(s[:, 0]), np.ptp(s[:, 1])) for s in strokes])
    size = float(np.median(ext[ext > 0])) if (ext > 0).any() else 1.0
    glyph = rng.uniform(26.0, 44.0)
    ang = np.deg2rad(np.clip(rng.normal(0.0, 2.0), -5.0, 5.0))
    shear = rng.uniform(-0.18, 0.18)
    aspect = np.exp(rng.uniform(-0.12, 0.12))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    a = rot @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([aspect, 1.0 / aspect]) * (glyph / size)
    jitter = 0.02 * glyph
    strokes = [s @ a.T + rng.normal(0.0, jitter, 2) for s in strokes]
    pen = rng.uniform(0.055, 0.12) * glyph
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    margin = pen + 0.5 * glyph
    span = hi - lo + 2 * margin
    shrink = min(1.0, MAX_SIDE / max(span))
    w, h = (int(np.ceil(v * shrink)) + 1 for v in span)

    canvas = Image.new("L", (w * SS, h * SS), 0)
    draw = ImageDraw.Draw(canvas)
    width = max(1, round(pen * shrink * SS))
    r = width / 2.0
    for s in strokes:
        p = (s - lo + margin) * shrink * SS
        pts = [tuple(q) for q in p]
        if len(pts) > 1:
            draw.line(pts, fill=255, width=width, joint="curve")
        for q in (pts[0], pts[-1]):
            draw.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=255)
    mask = np.asarray(canvas.resize((w, h), Image.Resampling.LANCZOS), np.float32) / 255.0

    paper = rng.uniform(205.0, 255.0)
    ink = rng.uniform(0.0, 80.0)
    contrast = paper - ink
    img = np.full((h, w), paper, np.float32)
    u = rng.random()
    if u < 0.35:  # ruled lines (and a grid for the last 10 %)
        step = rng.uniform(1.1, 1.8) * glyph * shrink
        depth = rng.uniform(0.08, 0.22) * contrast
        thick = 1 + int(rng.random() < 0.5)
        y0 = rng.uniform(0, step)
        for y in np.arange(y0, h, step):
            img[int(y):int(y) + thick, :] -= depth
        if u < 0.10:
            x0 = rng.uniform(0, step)
            for x in np.arange(x0, w, step):
                img[:, int(x):int(x) + thick] -= depth
    if rng.random() < 0.3:  # lighting
        gy, gx = np.mgrid[0:h, 0:w]
        t = (gx / max(w - 1, 1)) * np.cos(rng.uniform(0, 2 * np.pi)) + (gy / max(h - 1, 1)) * np.sin(rng.uniform(0, 2 * np.pi))
        img += rng.uniform(-0.15, 0.15) * contrast * (t - t.mean())
    img = img * (1.0 - mask) + ink * mask
    if rng.random() < 0.5:
        img += rng.normal(0.0, rng.uniform(1.0, 6.0), img.shape)
    out = Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8), mode="L")
    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    buf = io.BytesIO()
    if rng.random() < 0.35:
        out.save(buf, format="JPEG", quality=int(rng.integers(40, 93)))
    else:
        out.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()
