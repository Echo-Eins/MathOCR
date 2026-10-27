import io
import json
import os

import numpy as np
from PIL import Image

from mathocr.aggregate import collect, markdown
from mathocr.data.build import _analyze_bytes, formula_key
from mathocr.data.dataset import CropStore
from mathocr.evaluate import cdm_image_id, cdm_subset
from mathocr.image import Crop, ImageConfig, layout, prepare

from conftest import render


def test_stored_crop_reproduces_canonical_preprocessing():
    """Training/eval read crops from parquet; inference prepares the raw image."""

    cfg = ImageConfig()
    for text, size in (("x+y=2", 18), ("alpha-beta", 31)):
        im = render(text, size=size).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        png, h, w, glyph = _analyze_bytes(buf.getvalue())
        crop = Crop(np.asarray(Image.open(io.BytesIO(png))), glyph)
        assert crop.pixels.shape == (h, w)
        assert np.array_equal(layout(crop, cfg), prepare(Image.open(io.BytesIO(buf.getvalue())), cfg))


def test_store_round_trip(tiny_data):
    store = CropStore(tiny_data / "val.parquet")
    assert len(store) == 24 and store.crop(0).pixels.dtype == np.uint8


def test_formula_key_ignores_spacing_and_markup():
    assert formula_key(r"a \: b \\ \end{array}") == formula_key(r"a b \end{array}")
    assert formula_key("x ^ { 2 } .") == formula_key("x^2")


def test_cdm_ids_have_no_dots_and_subset_is_stable():
    uids = [f"im2latex/test/{i:04x}.png" for i in range(50)]
    assert all("." not in cdm_image_id(u) and "/" not in cdm_image_id(u) for u in uids)
    a, b = cdm_subset(uids, 10), cdm_subset(list(uids), 10)
    assert a == b and len(a) == 10 and a == sorted(a)


def test_aggregate_mean_std(tmp_path):
    runs = []
    for seed, em in ((1, 0.5), (2, 0.7)):
        run = tmp_path / f"s{seed}"
        (run / "eval").mkdir(parents=True)
        (run / "eval" / "metrics.json").write_text(json.dumps(
            {"seed": seed, "splits": {"test": {"exact_match": em, "edit_distance": 0.1, "bleu": 0.8}}}))
        runs.append(run)
    agg = collect(runs)
    em = agg["splits"]["test"]["exact_match"]
    assert abs(em["mean"] - 0.6) < 1e-12 and abs(em["std"] - 0.1414213562) < 1e-9
    assert "60.0 ± 14.1" in markdown(agg)


def test_store_pickles_a_large_buffer_through_a_memory_mapped_file(tiny_data, monkeypatch):
    import pickle

    store = CropStore(tiny_data / "val.parquet")
    monkeypatch.setattr(CropStore, "SPILL_BYTES", 0)  # spill even the tiny buffer
    state = store.__getstate__()
    assert state["buffer"] is None and os.path.exists(state["_spill"])
    copy = pickle.loads(pickle.dumps(store))
    assert isinstance(copy.buffer, np.memmap)
    assert all(np.array_equal(copy.crop(i).pixels, store.crop(i).pixels) for i in range(len(store)))
    assert pickle.loads(pickle.dumps(store)).buffer.filename == copy.buffer.filename  # spilled once
