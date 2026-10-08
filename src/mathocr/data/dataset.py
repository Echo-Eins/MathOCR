"""Training/evaluation data: in-memory crops, same-shape batches, seeded augmentation.

Every batch holds images of one exact shape, so no image is ever padded
beyond what ``image.prepare`` would produce for it alone.  For training the
sampler first draws each sample's scale jitter for the epoch (the jitter
decides the shape), then groups by shape.  All randomness is keyed by
``(seed, epoch, ...)``; see ``mathocr.determinism``.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.ndimage as ndi
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from mathocr.determinism import epoch_rng, sample_rng
from mathocr.image import Crop, ImageConfig, layout, output_shape, to_tensor
from mathocr.tokenizer import Vocab, tokenize


@dataclass(frozen=True)
class AugmentConfig:
    scale_log_range: float = 0.29  # jitter ~ exp(U(-r, r)): 0.75x .. 1.34x
    p_blur: float = 0.25
    blur_sigma: tuple[float, float] = (0.3, 0.9)
    p_stroke: float = 0.25  # thicken or thin strokes by one pixel
    p_contrast: float = 0.3
    p_noise: float = 0.25
    noise_std: tuple[float, float] = (2.0, 12.0)


class CropStore:
    """One split in memory: PNG bytes in a single buffer (fork-friendly)."""

    def __init__(self, path: Path, vocab: Vocab | None = None):
        table = pq.read_table(path)
        crops = table["crop"].to_pylist()
        lengths = np.fromiter((len(c) for c in crops), dtype=np.int64, count=len(crops))
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])
        self.buffer = np.frombuffer(b"".join(crops), dtype=np.uint8)
        self.h = table["h"].to_numpy().astype(np.int64)
        self.w = table["w"].to_numpy().astype(np.int64)
        self.glyph = table["glyph"].to_numpy().astype(np.float64)
        self.uids = table["uid"].to_pylist()
        self.sources = table["source"].to_pylist()
        self.labels = table["label"].to_pylist()
        self.targets = None
        if vocab is not None:
            ids = [vocab.encode(tokenize(label)) for label in self.labels]
            lens = np.fromiter((len(i) for i in ids), dtype=np.int64, count=len(ids))
            self.target_offsets = np.concatenate([[0], np.cumsum(lens)])
            self.targets = np.fromiter((t for seq in ids for t in seq), dtype=np.int64, count=int(lens.sum()))

    def __len__(self) -> int:
        return len(self.uids)

    def crop(self, i: int) -> Crop:
        data = self.buffer[self.offsets[i] : self.offsets[i + 1]].tobytes()
        with Image.open(io.BytesIO(data)) as im:
            pixels = np.asarray(im.convert("L"), dtype=np.uint8)
        return Crop(pixels, float(self.glyph[i]))

    def target(self, i: int) -> np.ndarray:
        assert self.targets is not None
        return self.targets[self.target_offsets[i] : self.target_offsets[i + 1]]

    def shape(self, i: int, cfg: ImageConfig, jitter: float = 1.0) -> tuple[int, int]:
        return output_shape(int(self.h[i]), int(self.w[i]), float(self.glyph[i]), cfg, jitter)


def augment(arr: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Shape-preserving pixel augmentation of a laid-out uint8 image."""

    x = arr.astype(np.float32)
    if rng.random() < cfg.p_stroke:
        x = ndi.grey_erosion(x, size=(2, 2)) if rng.random() < 0.5 else ndi.grey_dilation(x, size=(2, 2))
    if rng.random() < cfg.p_blur:
        x = ndi.gaussian_filter(x, sigma=rng.uniform(*cfg.blur_sigma))
    if rng.random() < cfg.p_contrast:
        ink, paper = rng.uniform(0.0, 80.0), rng.uniform(190.0, 255.0)
        x = ink + x * ((paper - ink) / 255.0)
    if rng.random() < cfg.p_noise:
        x = x + rng.normal(0.0, rng.uniform(*cfg.noise_std), size=x.shape)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


class TrainItems(Dataset):
    """Indexed by ``(sample index, jitter, epoch)`` tuples from ``ShapeBatchSampler``."""

    def __init__(self, store: CropStore, image_cfg: ImageConfig, aug_cfg: AugmentConfig | None, seed: int):
        self.store, self.image_cfg, self.aug_cfg, self.seed = store, image_cfg, aug_cfg, seed

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, key: tuple[int, float, int]) -> tuple[torch.Tensor, np.ndarray, int]:
        i, jitter, epoch = key
        arr = layout(self.store.crop(i), self.image_cfg, jitter)
        if self.aug_cfg is not None:
            arr = augment(arr, sample_rng(self.seed, epoch, i), self.aug_cfg)
        return to_tensor(arr), self.store.target(i), i


def collate(batch: list[tuple[torch.Tensor, np.ndarray, int]], pad_id: int) -> dict[str, torch.Tensor]:
    images = torch.stack([b[0] for b in batch])  # same shape by construction
    longest = max(len(b[1]) for b in batch)
    tokens = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    for row, (_, ids, _) in enumerate(batch):
        tokens[row, : len(ids)] = torch.from_numpy(ids)
    return {"images": images, "tokens": tokens, "index": torch.tensor([b[2] for b in batch])}


class ShapeBatchSampler(Sampler[list[tuple[int, float, int]]]):
    """Batches of one exact image shape under a pixel budget.

    ``batch = min(max_batch, pixel_budget // (H * W))``.  With ``train=True``
    each epoch draws scale jitters, shuffles within shapes and shuffles the
    batch order; with ``train=False`` the order is fixed (sorted by shape,
    then index) and there is no jitter.
    """

    def __init__(self, store: CropStore, image_cfg: ImageConfig, pixel_budget: int, max_batch: int,
                 seed: int, train: bool, aug_cfg: AugmentConfig | None = None,
                 indices: Sequence[int] | None = None):
        self.store, self.image_cfg = store, image_cfg
        self.indices = range(len(store)) if indices is None else sorted(set(indices))
        self.pixel_budget, self.max_batch = pixel_budget, max_batch
        self.seed, self.train, self.aug_cfg = seed, train, aug_cfg
        self.epoch = 0
        self._cache: tuple[int, list] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[tuple[int, float, int]]]:
        n = len(self.store)
        if self.train and self.aug_cfg is not None and self.aug_cfg.scale_log_range > 0:
            r = self.aug_cfg.scale_log_range
            jitters = np.exp(epoch_rng(self.seed, self.epoch, 0).uniform(-r, r, size=n))
        else:
            jitters = np.ones(n)
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in self.indices:
            groups[self.store.shape(i, self.image_cfg, float(jitters[i]))].append(i)
        order_rng = epoch_rng(self.seed, self.epoch, 1)
        batches = []
        for shape in sorted(groups):
            idx = np.array(groups[shape])
            if self.train:
                idx = idx[order_rng.permutation(len(idx))]
            size = max(1, min(self.max_batch, self.pixel_budget // (shape[0] * shape[1])))
            for start in range(0, len(idx), size):
                batches.append([(int(i), float(jitters[i]), self.epoch) for i in idx[start : start + size]])
        if self.train:
            batches = [batches[j] for j in order_rng.permutation(len(batches))]
        return batches

    def batches(self) -> list[list[tuple[int, float, int]]]:
        if self._cache is None or self._cache[0] != self.epoch:
            self._cache = (self.epoch, self._batches())
        return self._cache[1]

    def __iter__(self) -> Iterator[list[tuple[int, float, int]]]:
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())

