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
    # Multi-line composition: with probability p_stack a training sample is 2..stack_max
    # formulas, each brought to the target glyph height, stacked left-aligned with a gap of
    # stack_gap glyph heights; its label is \begin{array} { l } { f1 } \\ { f2 } ... \end{array}
    # (the normalizer's own spelling of multi-line formulas).
    p_stack: float = 0.0
    stack_max: int = 3
    stack_gap: float = 0.6


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
        self.vocab = vocab
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


def _glyph_scale(glyph: float, cfg: ImageConfig) -> float:
    s = cfg.target_glyph_px / glyph if glyph > 0 else 1.0
    return min(max(s, cfg.min_scale), cfg.max_scale)


def _part_size(store: CropStore, j: int, cfg: ImageConfig) -> tuple[int, int]:
    s = _glyph_scale(float(store.glyph[j]), cfg)
    return max(1, round(int(store.h[j]) * s)), max(1, round(int(store.w[j]) * s))


def stacked_size(store: CropStore, idx: Sequence[int], cfg: ImageConfig, aug: AugmentConfig) -> tuple[int, int]:
    """Crop size of the composition of ``idx`` (computable from metadata alone)."""

    parts = [_part_size(store, j, cfg) for j in idx]
    gap = round(aug.stack_gap * cfg.target_glyph_px)
    return sum(h for h, _ in parts) + gap * (len(parts) - 1), max(w for _, w in parts)


def stacked_crop(store: CropStore, idx: Sequence[int], cfg: ImageConfig, aug: AugmentConfig) -> Crop:
    """Formulas ``idx`` at the target glyph height, stacked left-aligned on white paper."""

    h, w = stacked_size(store, idx, cfg, aug)
    canvas = np.full((h, w), 255, dtype=np.uint8)
    gap, y = round(aug.stack_gap * cfg.target_glyph_px), 0
    for j in idx:
        ph, pw = _part_size(store, j, cfg)
        part = Image.fromarray(store.crop(j).pixels, mode="L").resize((pw, ph), Image.Resampling.BICUBIC)
        canvas[y : y + ph, :pw] = np.asarray(part)
        y += ph + gap
    return Crop(canvas, float(cfg.target_glyph_px))


def stacked_target(store: CropStore, idx: Sequence[int]) -> np.ndarray:
    r"""BOS \begin{array} { l } { f1 } \\ { f2 } ... \end{array} EOS as token ids."""

    v = store.vocab
    ids = [v.bos_id, *(v.stoi[t] for t in ("\\begin{array}", "{", "l", "}"))]
    for n, j in enumerate(idx):
        if n:
            ids.append(v.stoi["\\\\"])
        ids += [v.stoi["{"], *store.target(j)[1:-1].tolist(), v.stoi["}"]]
    ids += [v.stoi["\\end{array}"], v.eos_id]
    return np.asarray(ids, dtype=np.int64)


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

    def __getitem__(self, key: tuple) -> tuple[torch.Tensor, np.ndarray, int]:
        i, jitter, epoch = key[:3]
        partners = key[3] if len(key) > 3 else ()
        if partners:
            idx = (i, *partners)
            arr = layout(stacked_crop(self.store, idx, self.image_cfg, self.aug_cfg), self.image_cfg, jitter)
            target = stacked_target(self.store, idx)
        else:
            arr, target = layout(self.store.crop(i), self.image_cfg, jitter), self.store.target(i)
        if self.aug_cfg is not None:
            arr = augment(arr, sample_rng(self.seed, epoch, i), self.aug_cfg)
        return to_tensor(arr), target, i


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
                 indices: Sequence[int] | None = None, max_tokens: int = 256):
        self.store, self.image_cfg, self.max_tokens = store, image_cfg, max_tokens
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
        partners = self._partners(n) if self.train and self.aug_cfg is not None and self.aug_cfg.p_stack > 0 else {}
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in self.indices:
            if i in partners:
                h, w = stacked_size(self.store, (i, *partners[i]), self.image_cfg, self.aug_cfg)
                shape = output_shape(h, w, self.image_cfg.target_glyph_px, self.image_cfg, float(jitters[i]))
            else:
                shape = self.store.shape(i, self.image_cfg, float(jitters[i]))
            groups[shape].append(i)
        order_rng = epoch_rng(self.seed, self.epoch, 1)
        batches = []
        for shape in sorted(groups):
            idx = np.array(groups[shape])
            if self.train:
                idx = idx[order_rng.permutation(len(idx))]
            size = max(1, min(self.max_batch, self.pixel_budget // (shape[0] * shape[1])))
            for start in range(0, len(idx), size):
                batches.append([(int(i), float(jitters[i]), self.epoch, *((partners[int(i)],) if int(i) in partners else ()))
                                for i in idx[start : start + size]])
        if self.train:
            batches = [batches[j] for j in order_rng.permutation(len(batches))]
        return batches

    def _partners(self, n: int) -> dict[int, tuple[int, ...]]:
        """Per sample (epoch-seeded): whether it becomes a multi-line composition and with
        which other training formulas; partners are dropped from the end until the label
        fits ``max_tokens``.  Arrays of size n are drawn regardless, so the draws of one
        sample never depend on another's."""

        aug, rng = self.aug_cfg, epoch_rng(self.seed, self.epoch, 2)
        pool = np.asarray(self.indices)
        draw = rng.random(n)
        count = rng.integers(2, aug.stack_max + 1, size=n)
        pick = pool[rng.integers(0, len(pool), size=(n, max(1, aug.stack_max - 1)))]
        lengths = np.diff(self.store.target_offsets) - 2  # without BOS/EOS
        out = {}
        for i in self.indices:
            if draw[i] >= aug.p_stack:
                continue
            chosen = [int(j) for j in pick[i, : count[i] - 1]]
            while chosen and lengths[i] + sum(lengths[j] for j in chosen) + 3 * len(chosen) + 7 > self.max_tokens:
                chosen.pop()
            if chosen:
                out[i] = tuple(chosen)
        return out

    def batches(self) -> list[list[tuple[int, float, int]]]:
        if self._cache is None or self._cache[0] != self.epoch:
            self._cache = (self.epoch, self._batches())
        return self._cache[1]

    def __iter__(self) -> Iterator[list[tuple[int, float, int]]]:
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())

