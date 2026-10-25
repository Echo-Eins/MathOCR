"""Cross-font twins (augment.pair_variants / pair_share, train.pair_consistency): a twin lands on its original's
canvas, enabling twins leaves every original sample as it was, the consistency term vanishes for identical twins,
and training with twins is bitwise reproducible."""

import copy
import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from mathocr.data.build import SCHEMA, _analyze_bytes
from mathocr.data.dataset import AugmentConfig, CropStore, ShapeBatchSampler, TrainItems, collate
from mathocr.image import ImageConfig, layout, layout_into
from mathocr.model import Im2Latex, ModelConfig
from mathocr.tokenizer import Vocab
from mathocr.train import _loss, train


@pytest.fixture(scope="module")
def twin_data(tiny_data):
    """train_bold.parquet: every train crop thickened and widened (another 'font' of the same labels)."""
    rows = pq.read_table(tiny_data / "train.parquet").to_pylist()
    for r in rows:
        im = np.asarray(Image.open(io.BytesIO(r["crop"])).convert("L"))
        bold = np.minimum(im, np.roll(im, 1, axis=1))
        wide = Image.fromarray(bold).resize((int(im.shape[1] * 1.2), im.shape[0]))
        buf = io.BytesIO()
        wide.save(buf, format="PNG")
        r["crop"], r["h"], r["w"], r["glyph"] = _analyze_bytes(buf.getvalue())
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tiny_data / "train_bold.parquet")
    return tiny_data


def test_layout_into_keeps_the_canvas():
    rng = np.random.default_rng(0)
    from mathocr.image import Crop

    crop = Crop(rng.integers(0, 255, size=(30, 200), dtype=np.uint8), 12.0)
    cfg = ImageConfig()
    natural = layout(crop, cfg)
    assert np.array_equal(layout_into(crop, cfg, natural.shape), natural)
    small = (natural.shape[0], natural.shape[1] - 64)  # a narrower canvas: shrunk to fit
    assert layout_into(crop, cfg, small).shape == small


def test_twins_leave_the_originals_unchanged(twin_data):
    vocab = Vocab.load(twin_data / "vocab.json")
    store, bold = CropStore(twin_data / "train.parquet", vocab), CropStore(twin_data / "train_bold.parquet", vocab)
    icfg, aug = ImageConfig(), AugmentConfig(pair_share=0.5)
    plain, paired = TrainItems(store, icfg, aug, 0), TrainItems(store, icfg, aug, 0, [bold])
    n_twins = 0
    for i in range(len(store)):
        a, b = plain[(i, 1.0, 2)], paired[(i, 1.0, 2)]
        assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        if len(b) > 3:
            n_twins += 1
            assert b[3].shape == b[0].shape
    assert 0 < n_twins < len(store)


def test_identical_twins_add_no_consistency_loss(twin_data):
    vocab = Vocab.load(twin_data / "vocab.json")
    torch.manual_seed(0)
    m = Im2Latex(ModelConfig(vocab_size=len(vocab), cnn_channels=(8, 8, 16, 16), cnn_blocks=(1, 1, 1, 1), d_model=32,
                             n_heads=4, enc_layers=1, dec_layers=2, max_len=32)).eval()
    store = CropStore(twin_data / "train.parquet", vocab)
    items = TrainItems(store, ImageConfig(), None, 0)
    batch = collate([items[(i, 1.0, 0)] for i in range(2) if store.shape(i, ImageConfig()) == store.shape(0, ImageConfig())],
                    vocab.pad_id)
    batch["twins"], batch["twin_rows"] = batch["images"].clone(), torch.arange(batch["images"].shape[0])
    with torch.no_grad():
        loss, _, _, parts = _loss(m, batch, torch.device("cpu"), vocab, 0.1, False, pair_weight=1.0)
    assert float(parts["pair_js"]) < 1e-6


def test_training_with_twins_is_reproducible(tiny_config, twin_data, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["augment"].update(pair_variants=["train_bold.parquet"], pair_share=0.5)
    cfg["train"].update(pair_consistency=1.0, optimizer="muon")
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    logs = [l for l in (tmp_path / "a" / "log.jsonl").read_text().splitlines() if "pair_js" in l]
    assert logs  # the consistency term was active
