"""Shared fixtures: a tiny synthetic processed dataset (no downloads needed)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw, ImageFont

from mathocr.data.build import SCHEMA, _analyze_bytes
from mathocr.tokenizer import Vocab, tokenize

FORMULAS = ["x + y = 2", "a ^ { 2 } + b ^ { 2 }", "\\alpha - \\beta", "f ( x ) = 3 x", "1 2 3 4",
            "k _ { 1 } < k _ { 2 }", "( a + b ) ^ { n }", "x y z"]
_SHOWN = ["x+y=2", "a2+b2", "alpha-beta", "f(x)=3x", "1234", "k1<k2", "(a+b)n", "xyz"]


def render(text: str, size: int = 22, pad: int = 6) -> Image.Image:
    font = ImageFont.load_default(size=size)
    w = int(font.getlength(text)) + 2 * pad
    im = Image.new("RGB", (w, size + 2 * pad), "white")
    ImageDraw.Draw(im).text((pad, pad), text, fill="black", font=font)
    return im


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny_data")
    rng = np.random.default_rng(0)
    for split, n in (("train", 96), ("val", 24)):
        cols = {f.name: [] for f in SCHEMA}
        for i in range(n):
            k = int(rng.integers(len(FORMULAS)))
            buf = io.BytesIO()
            render(_SHOWN[k], size=int(rng.integers(16, 30))).save(buf, format="PNG")
            crop, h, w, g = _analyze_bytes(buf.getvalue())
            for key, v in zip(SCHEMA.names, (f"syn/{split}/{i}", "syn", FORMULAS[k], crop, h, w, g)):
                cols[key].append(v)
        pq.write_table(pa.table(cols, schema=SCHEMA), out / f"{split}.parquet")
    Vocab.build((tokenize(f) for f in FORMULAS), 1).save(out / "vocab.json")
    (out / "manifest.json").write_text(json.dumps({
        "splits": {"train": {"uid_label_sha256": "syn"}, "val": {"uid_label_sha256": "syn"}},
        "vocab": {"sha256": "syn"},
    }))
    return out


@pytest.fixture()
def tiny_config(tiny_data) -> dict:
    from mathocr.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    cfg["data"]["out_dir"] = str(tiny_data)
    cfg["model"].update(cnn_channels=[16, 16, 32, 32], cnn_blocks=[1, 1, 1, 1], d_model=64, n_heads=4,
                        enc_layers=1, dec_layers=2, max_len=32)
    cfg["train"].update(epochs=2, warmup_steps=5, pixel_budget=200_000, max_batch=8, workers=2,
                        log_every=5, val_decode_samples=8, eval_pixel_budget=200_000, eval_max_batch=8)
    return cfg
