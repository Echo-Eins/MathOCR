"""Reproducibility contract: same seed -> same weights; same image -> same LaTeX."""

import copy
import json
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mathocr.data.dataset import AugmentConfig, CropStore, ShapeBatchSampler, TrainItems, collate
from mathocr.image import ImageConfig
from mathocr.infer import decode_store, load_checkpoint, predict_images
from mathocr.tokenizer import Vocab
from mathocr.train import train

from conftest import render

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _batches(store, seed, epoch, workers):
    cfg = ImageConfig()
    sampler = ShapeBatchSampler(store, cfg, 200_000, 8, seed, train=True, aug_cfg=AugmentConfig())
    sampler.set_epoch(epoch)
    loader = DataLoader(TrainItems(store, cfg, AugmentConfig(), seed), batch_sampler=sampler,
                        num_workers=workers, collate_fn=partial(collate, pad_id=0))
    return list(loader)


def test_data_pipeline_independent_of_workers(tiny_data):
    store = CropStore(tiny_data / "train.parquet", Vocab.load(tiny_data / "vocab.json"))
    a, b = _batches(store, 3, 1, 0), _batches(store, 3, 1, 3)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x["images"], y["images"]) and torch.equal(x["tokens"], y["tokens"])
    c = _batches(store, 3, 2, 0)  # another epoch reshuffles and re-augments
    assert [x["index"].tolist() for x in a] != [x["index"].tolist() for x in c]


def _state(run: Path) -> dict:
    return torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)["model"]


@cuda
def test_training_is_bitwise_reproducible_and_resumable(tiny_config, tmp_path):
    train(copy.deepcopy(tiny_config), 7, tmp_path / "a", resume=False)
    train(copy.deepcopy(tiny_config), 7, tmp_path / "b", resume=False)
    paused = copy.deepcopy(tiny_config)
    paused["train"]["stop_after_epoch"] = 0
    train(paused, 7, tmp_path / "c", resume=False)
    train(copy.deepcopy(tiny_config), 7, tmp_path / "c", resume=True)
    train(copy.deepcopy(tiny_config), 8, tmp_path / "d", resume=False)

    a, b, c, d = (_state(tmp_path / r) for r in "abcd")
    for k in a:
        assert torch.equal(a[k], b[k]), f"rerun differs at {k}"
        assert torch.equal(a[k], c[k]), f"resumed run differs at {k}"
    assert any(not torch.equal(a[k], d[k]) for k in a), "seed has no effect"
    log = [json.loads(line) for line in (tmp_path / "a" / "log.jsonl").read_text().splitlines()]
    assert sum("val_exact_match" in r for r in log) == 2


@cuda
def test_inference_is_deterministic(tiny_config, tiny_data, tmp_path):
    train(copy.deepcopy(tiny_config), 1, tmp_path / "r", resume=False)
    device = torch.device("cuda")
    model, vocab, icfg, _ = load_checkpoint(tmp_path / "r" / "checkpoints" / "best.pt", device)
    images = [render(t, size=s) for t, s in (("x+y=2", 20), ("a2+b2", 26), ("xyz", 18))]
    first = predict_images(model, images, icfg, vocab, device)
    for _ in range(3):
        assert predict_images(model, images, icfg, vocab, device) == first
    store = CropStore(tiny_data / "val.parquet", vocab)
    batched = decode_store(model, store, icfg, vocab, device, 200_000, 8)
    assert batched == decode_store(model, store, icfg, vocab, device, 200_000, 8)
    assert len(batched) == len(store)
