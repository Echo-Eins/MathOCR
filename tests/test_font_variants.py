"""Font variants (augment.p_font / font_variants): per-epoch choice of a row-aligned alternate rendering; the
sampler's batch shapes match the loader's images, the choice is a function of (seed, epoch), stacking works on the
epoch view, and training with variants is reproducible."""

import copy
import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from mathocr.data.build import SCHEMA, _analyze_bytes
from mathocr.data.dataset import AugmentConfig, CropStore, FontVariants, ShapeBatchSampler, TrainItems
from mathocr.image import ImageConfig
from mathocr.tokenizer import Vocab
from mathocr.train import train


@pytest.fixture(scope="module")
def variant_data(tiny_data):
    """train_big.parquet: every train crop upscaled 1.5x and padded (other glyph size and shape, same labels)."""
    rows = pq.read_table(tiny_data / "train.parquet").to_pylist()
    for r in rows:
        im = Image.open(io.BytesIO(r["crop"]))
        big = Image.new("L", (int(im.width * 1.5) + 12, int(im.height * 1.5) + 6), 255)
        big.paste(im.resize((int(im.width * 1.5), int(im.height * 1.5))), (6, 3))
        buf = io.BytesIO()
        big.save(buf, format="PNG")
        r["crop"], r["h"], r["w"], r["glyph"] = _analyze_bytes(buf.getvalue())
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tiny_data / "train_big.parquet")
    return tiny_data


def _variants(data, p_font, seed=0):
    base_vocab = Vocab.load(data / "vocab.json")  # + the array tokens the "rl" stacking writes
    vocab = Vocab(base_vocab.itos + [t for t in ("\\begin{array}", "\\end{array}", "l", "r", "&", "\\\\") if t not in base_vocab])
    base = CropStore(data / "train.parquet", vocab)
    return FontVariants(base, [CropStore(data / "train_big.parquet", vocab)], p_font, seed), vocab


def test_choice_is_a_function_of_seed_and_epoch(variant_data):
    fv, _ = _variants(variant_data, 0.5)
    a, b = fv.choice(0), fv.choice(0)
    assert np.array_equal(a, b) and 0 < a.mean() < 1
    assert not np.array_equal(fv.choice(0), fv.choice(1))
    assert not _variants(variant_data, 0.0)[0].choice(3).any()


@pytest.mark.parametrize("p_stack", [0.0, 0.5])
def test_sampler_shapes_match_loader_images(variant_data, p_stack):
    fv, vocab = _variants(variant_data, 0.5)
    icfg = ImageConfig()
    aug = AugmentConfig(p_stack=p_stack, stack_style="rl", p_font=0.5)
    sampler = ShapeBatchSampler(fv, icfg, 200_000, 8, seed=0, train=True, aug_cfg=aug, max_tokens=64)
    items = TrainItems(fv, icfg, aug, seed=0)
    used_alt = 0
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        choice = fv.choice(epoch)
        for batch in sampler:
            shapes = {tuple(items[k][0].shape[1:]) for k in batch}
            assert len(shapes) == 1  # one exact shape per batch, as for a plain store
            used_alt += int(sum(choice[k[0]] for k in batch))
    assert used_alt > 0


def test_variants_must_be_row_aligned(variant_data):
    vocab = Vocab.load(variant_data / "vocab.json")
    with pytest.raises(ValueError):
        FontVariants(CropStore(variant_data / "train.parquet", vocab), [CropStore(variant_data / "val.parquet", vocab)], 0.5, 0)


def test_training_with_font_variants_is_reproducible(variant_data, tiny_config, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["augment"].update(p_font=0.5, font_variants=["train_big.parquet"])
    train(copy.deepcopy(cfg), 3, tmp_path / "a", resume=False)
    train(copy.deepcopy(cfg), 3, tmp_path / "b", resume=False)
    import torch

    a = torch.load(tmp_path / "a" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
