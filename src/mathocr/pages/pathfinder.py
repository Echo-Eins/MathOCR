"""Pathfinder-style long-range task for the page stand (tasks/page_ocr_plan.md 5a: information propagation measured
apart from reading; after Linsley et al. 2018 and the Long Range Arena).

An image holds two circular markers and several dashed curves.  ``label`` is 1 when one curve joins the two markers,
0 when each marker ends a different curve.  Both classes draw the same number of curves of the same length (two
main curves + ``n_distractors``), so ink amount says nothing about the label.  The label is exact by construction;
configurations a human could misread are rejected and redrawn:

* a marker may touch no curve but its own (``clearance`` px from every other curve's points and from the other
  marker);
* end points of different curves keep ``end_clearance`` px apart (no visual continuation of one curve into another);
* the two markers are at least ``min_marker_dist`` apart (the decision is long-range, not local).

Curves are polylines of ``n_segments`` steps of ``seg_len`` px; every other step is drawn (dashes), the first and the
last are drawn (``n_segments`` is odd), so a curve's ends are visible.  The heading turns by up to ``max_turn_deg``
per step and reflects off the canvas margin.  Drawn at 2x and downsampled (antialiasing); paper 255, ink 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

SS = 2


@dataclass(frozen=True)
class PathfinderSpec:
    size: int = 512
    seg_len: float = 9.0
    width: float = 2.0
    n_segments: int = 41  # odd: both end steps are dashes
    max_turn_deg: float = 30.0
    n_distractors: int = 6
    marker_radius: float = 5.0
    clearance: float = 10.0
    end_clearance: float = 24.0
    min_marker_dist: float = 0.3  # of the image size
    margin: float = 12.0


def _curve(rng: np.random.Generator, spec: PathfinderSpec) -> np.ndarray | None:
    lo, hi = spec.margin, spec.size - spec.margin
    p = rng.uniform(lo, hi, 2)
    heading = rng.uniform(0, 2 * math.pi)
    pts = [p]
    turn = math.radians(spec.max_turn_deg)
    for _ in range(spec.n_segments):
        heading += rng.uniform(-turn, turn)
        for _attempt in range(4):
            q = pts[-1] + spec.seg_len * np.array([math.cos(heading), math.sin(heading)])
            if lo <= q[0] <= hi and lo <= q[1] <= hi:
                break
            # reflect off the wall that was crossed
            if not lo <= q[0] <= hi:
                heading = math.pi - heading
            if not lo <= q[1] <= hi:
                heading = -heading
        else:
            return None
        pts.append(q)
    return np.array(pts)


def _dist_to_points(p: np.ndarray, pts: np.ndarray) -> float:
    return float(np.sqrt(((pts - p) ** 2).sum(axis=1)).min())


def _dense(curve: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Points every ``step`` px along the *drawn* dashes of a curve (gaps excluded)."""

    out = []
    for k in range(0, len(curve) - 1, 2):
        a, b = curve[k], curve[k + 1]
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.concatenate(out)


def make(seed: int, label: int, spec: PathfinderSpec = PathfinderSpec(), max_tries: int = 500) -> tuple[Image.Image, dict]:
    """One sample of class ``label`` (0/1): (image, meta with the curves, markers and the number of redraws)."""

    rng = np.random.default_rng(seed)
    for tries in range(max_tries):
        curves = []
        while len(curves) < 2 + spec.n_distractors:
            c = _curve(rng, spec)
            if c is not None:
                curves.append(c)
        a, b = curves[0], curves[1]
        markers = [a[0], a[-1]] if label == 1 else [a[0], b[-1]]
        owners = [0, 0] if label == 1 else [0, 1]
        if np.linalg.norm(markers[0] - markers[1]) < spec.min_marker_dist * spec.size:
            continue
        dense = [_dense(c) for c in curves]
        ok = True
        for m, own in zip(markers, owners):
            for k, d in enumerate(dense):
                if k != own and _dist_to_points(m, d) < spec.clearance:
                    ok = False
        ends = [(k, e) for k, c in enumerate(curves) for e in (c[0], c[-1])]
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                if ends[i][0] != ends[j][0] and np.linalg.norm(ends[i][1] - ends[j][1]) < spec.end_clearance:
                    ok = False
        if ok:
            break
    else:
        raise RuntimeError(f"seed {seed}: no clean configuration in {max_tries} tries")
    big = Image.new("L", (spec.size * SS, spec.size * SS), 255)
    d = ImageDraw.Draw(big)
    w = max(1, round(spec.width * SS))
    for c in curves:
        for k in range(0, len(c) - 1, 2):
            d.line([tuple(c[k] * SS), tuple(c[k + 1] * SS)], fill=0, width=w)
    r = spec.marker_radius * SS
    for m in markers:
        d.ellipse([m[0] * SS - r, m[1] * SS - r, m[0] * SS + r, m[1] * SS + r], fill=0)
    img = big.resize((spec.size, spec.size), Image.Resampling.LANCZOS)
    meta = {"label": int(label), "markers": [[round(float(v), 2) for v in m] for m in markers], "owners": owners,
            "curves": [[[round(float(v), 2) for v in p] for p in c] for c in curves], "redraws": tries}
    return img, meta
