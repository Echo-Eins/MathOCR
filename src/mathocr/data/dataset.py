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

from mathocr.determinism import epoch_rng, sample_rng, twin_rng
from mathocr.image import Crop, ImageConfig, layout, layout_into, output_shape, to_tensor
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
    # formulas, each brought to the target glyph height, stacked with a gap of stack_gap glyph
    # heights.  stack_style "l": lines left-aligned, label \begin{array} { l } { f1 } \\ { f2 } ...
    # \end{array}; "rl": the normalizer's spelling of aligned equations, \begin{array} { r l }
    # { lhs } & { { } = rhs } \\ & { { } f2 } ... \end{array} (split at the first top-level
    # relation; a line without one sits in the right column), with the lines aligned at the
    # relation's estimated column -- the layout 93% of multi-line training labels have; "rows":
    # like "rl", but the partners are whole \begin{array} { r l } blocks (genuine aligned
    # equations, 9% of train) whose rows are merged into one array -- long labels with real rows.
    p_stack: float = 0.0
    stack_max: int = 3
    stack_gap: float = 0.6
    stack_style: str = "l"
    # Font variants (FontVariants): parquet files in the data dir, row-aligned with train.parquet, holding the same
    # labels rendered in other math fonts (scripts/fonts/build_font_variants.py); each epoch a sample takes one of
    # them with probability p_font (uniformly), else its original image.
    p_font: float = 0.0
    font_variants: tuple[str, ...] = ()
    # Rasterization (screen captures, low-DPI renders), after the pixel augmentations above and only when enabled
    # (no extra random draws otherwise): with p_raster the laid-out image is resampled down by a factor in
    # raster_scale and back up (each way box / bilinear / bicubic / nearest at random) -- the glyph read at a lower
    # resolution and then enlarged, as ``layout`` enlarges a small screenshot; with p_gamma the intensities follow
    # a power exp(U(-r, r)) (stroke weight); with p_jpeg JPEG at a quality in jpeg_quality.
    p_raster: float = 0.0
    raster_scale: tuple[float, float] = (0.35, 0.8)
    p_gamma: float = 0.0
    gamma_log_range: float = 0.7
    p_jpeg: float = 0.0
    jpeg_quality: tuple[int, int] = (30, 90)
    # Twins (cross-font consistency, train.pair_consistency): with probability pair_share a single-formula sample
    # also brings the same label from one of pair_variants (row-aligned train copies in other fonts, data dir),
    # laid out on the original's canvas and augmented on its own stream (determinism.twin_rng).
    pair_variants: tuple[str, ...] = ()
    pair_share: float = 0.0
    # Rotation (the multi-row probe): with p_rotate a single-formula sample is laid out and turned 90 degrees
    # clockwise (its start on top, the label unchanged), so the encoder's vertical path reads a long sequence;
    # drawn per (seed, epoch) by the sampler, which batches it at the transposed shape.
    p_rotate: float = 0.0


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


class FontVariants:
    """The training store plus row-aligned copies of it rendered in other fonts.  Each epoch every sample takes one
    variant, drawn from ``(seed, epoch)`` for all samples at once: with probability ``p_font`` one of the alternates
    (uniformly), else the original.  ``for_epoch`` returns a CropStore-like view whose crops and sizes are that
    epoch's variants, so the sampler's shapes and the loader's images agree."""

    def __init__(self, base: CropStore, alts: Sequence[CropStore], p_font: float, seed: int):
        for a in alts:
            if a.uids != base.uids or a.labels != base.labels:
                raise ValueError("font variants must be row-aligned with the training store")
        self.base, self.stores, self.p_font, self.seed = base, [base, *alts], p_font, seed
        self._view: tuple[int, _EpochView] | None = None

    def __len__(self) -> int:
        return len(self.base)

    # labels are the same in every variant; sizes and crops are not, so only the view (for_epoch) has those
    @property
    def labels(self) -> list[str]:
        return self.base.labels

    @property
    def vocab(self) -> Vocab | None:
        return self.base.vocab

    @property
    def uids(self) -> list[str]:
        return self.base.uids

    @property
    def target_offsets(self) -> np.ndarray:
        return self.base.target_offsets

    def target(self, i: int) -> np.ndarray:
        return self.base.target(i)

    def choice(self, epoch: int) -> np.ndarray:
        rng = epoch_rng(self.seed, epoch, 3)
        n = len(self.base)
        use = rng.random(n) < self.p_font
        which = rng.integers(1, len(self.stores), size=n)
        return np.where(use, which, 0)

    def for_epoch(self, epoch: int) -> _EpochView:
        if self._view is None or self._view[0] != epoch:
            self._view = (epoch, _EpochView(self, self.choice(epoch)))
        return self._view[1]


class _EpochView:
    """One epoch of ``FontVariants``: per-sample crop, size and glyph from the chosen variant; labels and targets
    from the base store."""

    def __init__(self, variants: FontVariants, choice: np.ndarray):
        self._variants, self.choice = variants, choice
        stores = variants.stores
        self.h = np.choose(choice, [s.h for s in stores])
        self.w = np.choose(choice, [s.w for s in stores])
        self.glyph = np.choose(choice, [s.glyph for s in stores])

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._variants.base, name)

    def __len__(self) -> int:
        return len(self._variants.base)

    def crop(self, i: int) -> Crop:
        return self._variants.stores[int(self.choice[i])].crop(i)

    def target(self, i: int) -> np.ndarray:
        return self._variants.base.target(i)

    def shape(self, i: int, cfg: ImageConfig, jitter: float = 1.0) -> tuple[int, int]:
        return output_shape(int(self.h[i]), int(self.w[i]), float(self.glyph[i]), cfg, jitter)


def _epoch_store(store, epoch: int):
    return store.for_epoch(epoch) if isinstance(store, FontVariants) else store


def _glyph_scale(glyph: float, cfg: ImageConfig) -> float:
    s = cfg.target_glyph_px / glyph if glyph > 0 else 1.0
    return min(max(s, cfg.min_scale), cfg.max_scale)


def _part_size(store: CropStore, j: int, cfg: ImageConfig) -> tuple[int, int]:
    s = _glyph_scale(float(store.glyph[j]), cfg)
    return max(1, round(int(store.h[j]) * s)), max(1, round(int(store.w[j]) * s))


RELATIONS = frozenset(("=", "\\leq", "\\geq", "\\le", "\\ge", "<", ">", "\\equiv", "\\approx", "\\neq", "\\ne", "\\sim",
                       "\\simeq", "\\propto", "\\to", "\\rightarrow", "\\Rightarrow", "\\subset", "\\subseteq", "\\in"))


def relation_cut(tokens: Sequence[str]) -> int | None:
    """Index of the first relation token at brace/environment depth 0 with something before it, else None."""

    depth = 0
    for i, t in enumerate(tokens):
        if depth == 0 and i > 0 and t in RELATIONS:
            return i
        if t == "{" or t.startswith("\\begin{"):
            depth += 1
        elif t == "}" or t.startswith("\\end{"):
            depth -= 1
    return None


RL_HEAD = ("\\begin{array}", "{", "r", "l", "}")


def rl_rows(tokens: Sequence[str]) -> list[tuple[int, int]] | None:
    """Row spans ``[start, end)`` of a whole ``\\begin{array} { r l } ... \\end{array}`` label (rows split at
    the top-level ``\\\\``, an empty trailing row dropped), or None when the label is not such a block."""

    if len(tokens) < 7 or tuple(tokens[:5]) != RL_HEAD or tokens[-1] != "\\end{array}":
        return None
    rows, start, depth = [], 5, 0
    for i in range(5, len(tokens) - 1):
        t = tokens[i]
        if t == "{" or t.startswith("\\begin{"):
            depth += 1
        elif t == "}" or t.startswith("\\end{"):
            depth -= 1
        elif t == "\\\\" and depth == 0:
            rows.append((start, i))
            start = i + 1
    if start < len(tokens) - 1:
        rows.append((start, len(tokens) - 1))
    return [r for r in rows if r[1] > r[0]] or None


def _cut_fraction(tokens: Sequence[str]) -> float | None:
    """Estimated horizontal position (0..1) of the alignment column: the relation of a single formula, or
    the ``&`` of the first row of an ``r l`` block."""

    rows = rl_rows(tokens)
    if rows:
        s, e = rows[0]
        row = tokens[s:e]
        return (row.index("&") if "&" in row else 0) / max(1, len(row))
    cut = relation_cut(tokens)
    return None if cut is None else cut / len(tokens)


def _part_layout(store: CropStore, idx: Sequence[int], cfg: ImageConfig, aug: AugmentConfig) -> list[tuple[int, int, int]]:
    """(height, width, left offset) of every part.  "l": all at 0.  "rl"/"rows": a part's alignment column
    is estimated as its width times the relation's (or the first ``&``'s) token position; parts are
    aligned at the largest such column, and a part without one starts there (the right column)."""

    parts = [_part_size(store, j, cfg) for j in idx]
    if aug.stack_style not in ("rl", "rows"):
        return [(h, w, 0) for h, w in parts]
    cols = []
    for j, (_, w) in zip(idx, parts):
        f = _cut_fraction(store.labels[j].split())
        cols.append(None if f is None else round(w * f))
    x = max((c for c in cols if c is not None), default=0)
    return [(h, w, x - c if c is not None else x) for (h, w), c in zip(parts, cols)]


def stacked_size(store: CropStore, idx: Sequence[int], cfg: ImageConfig, aug: AugmentConfig) -> tuple[int, int]:
    """Crop size of the composition of ``idx`` (computable from metadata alone)."""

    layout = _part_layout(store, idx, cfg, aug)
    gap = round(aug.stack_gap * cfg.target_glyph_px)
    return sum(h for h, _, _ in layout) + gap * (len(layout) - 1), max(x + w for _, w, x in layout)


def stacked_crop(store: CropStore, idx: Sequence[int], cfg: ImageConfig, aug: AugmentConfig) -> Crop:
    """Formulas ``idx`` at the target glyph height, stacked on white paper (``_part_layout`` offsets)."""

    layout = _part_layout(store, idx, cfg, aug)
    h, w = stacked_size(store, idx, cfg, aug)
    canvas = np.full((h, w), 255, dtype=np.uint8)
    gap, y = round(aug.stack_gap * cfg.target_glyph_px), 0
    for j, (ph, pw, x) in zip(idx, layout):
        part = Image.fromarray(store.crop(j).pixels, mode="L").resize((pw, ph), Image.Resampling.BICUBIC)
        canvas[y : y + ph, x : x + pw] = np.asarray(part)
        y += ph + gap
    return Crop(canvas, float(cfg.target_glyph_px))


def stacked_tokens(store: CropStore, idx: Sequence[int], style: str = "l") -> list[int]:
    r"""Token ids of the composition's label, BOS..EOS.

    "l":    \begin{array} { l } { f1 } \\ { f2 } ... \end{array}
    "rl":   \begin{array} { r l } { lhs1 } & { { } = rhs1 } \\ & { { } f2 } ... \end{array}
    "rows": as "rl", but a part that is itself an r l block contributes its own rows.
    """

    v = store.vocab
    ob, cb = v.stoi["{"], v.stoi["}"]
    if style == "l":
        ids = [v.bos_id, *(v.stoi[t] for t in ("\\begin{array}", "{", "l", "}"))]
    else:
        ids = [v.bos_id, *(v.stoi[t] for t in RL_HEAD)]
    for n, j in enumerate(idx):
        if n:
            ids.append(v.stoi["\\\\"])
        body = store.target(j)[1:-1].tolist()
        if style == "l":
            ids += [ob, *body, cb]
            continue
        toks = store.labels[j].split()  # one id per token: the same indices
        rows = rl_rows(toks) if style == "rows" else None
        if rows:
            for r, (s, e) in enumerate(rows):
                if r:
                    ids.append(v.stoi["\\\\"])
                ids += body[s:e]
            continue
        cut = relation_cut(toks)
        amp = v.stoi["&"]
        if cut is None:
            ids += [amp, ob, ob, cb, *body, cb]
        else:
            ids += [ob, *body[:cut], cb, amp, ob, ob, cb, *body[cut:], cb]
    ids += [v.stoi["\\end{array}"], v.eos_id]
    return ids


def stacked_length(store: CropStore, idx: Sequence[int], style: str = "l") -> int:
    """``len(stacked_tokens(...))``; for "l"/"rl" from the label lengths alone (no id lists built)."""

    if style == "rows":
        return len(stacked_tokens(store, idx, style))
    rows = len(idx)
    lengths = [int(store.target_offsets[j + 1] - store.target_offsets[j]) - 2 for j in idx]  # without BOS/EOS
    if style == "l":  # BOS + 4 + sum(len + 2) + (rows - 1) + 1 + EOS
        return 6 + sum(lengths) + 3 * rows
    extra = [5 if relation_cut(store.labels[j].split()) is None else 7 for j in idx]
    return 7 + sum(lengths) + sum(extra) + rows  # BOS + 5 + sum(len + extra) + (rows - 1) + 1 + EOS


def stacked_target(store: CropStore, idx: Sequence[int], style: str = "l") -> np.ndarray:
    return np.asarray(stacked_tokens(store, idx, style), dtype=np.int64)


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
    out = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    if cfg.p_raster > 0 or cfg.p_gamma > 0 or cfg.p_jpeg > 0:
        out = rasterize(out, rng, cfg)
    return out


_RESAMPLE = (Image.Resampling.BOX, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.NEAREST)


def rasterize(arr: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Shape-preserving rasterization augmentation of a uint8 image (``AugmentConfig.p_raster`` and after)."""

    im = Image.fromarray(arr)
    if cfg.p_raster > 0 and rng.random() < cfg.p_raster:
        f = rng.uniform(*cfg.raster_scale)
        small = (max(1, round(im.width * f)), max(1, round(im.height * f)))
        down, up = (_RESAMPLE[int(k)] for k in rng.integers(0, len(_RESAMPLE), size=2))
        im = im.resize(small, down).resize(arr.shape[::-1], up)
    if cfg.p_gamma > 0 and rng.random() < cfg.p_gamma:
        g = float(np.exp(rng.uniform(-cfg.gamma_log_range, cfg.gamma_log_range)))
        im = im.point([round(255.0 * (i / 255.0) ** g) for i in range(256)])
    if cfg.p_jpeg > 0 and rng.random() < cfg.p_jpeg:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return np.asarray(im, dtype=np.uint8)


class TrainItems(Dataset):
    """Indexed by ``(sample index, jitter, epoch)`` tuples from ``ShapeBatchSampler``."""

    def __init__(self, store: CropStore, image_cfg: ImageConfig, aug_cfg: AugmentConfig | None, seed: int,
                 twins: Sequence[CropStore] = ()):
        self.store, self.image_cfg, self.aug_cfg, self.seed = store, image_cfg, aug_cfg, seed
        for t in twins:
            if t.uids != store.uids or t.labels != store.labels:
                raise ValueError("twin variants must be row-aligned with the training store")
        self.twins = list(twins)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, key: tuple) -> tuple:
        i, jitter, epoch = key[:3]
        partners = key[3] if len(key) > 3 else ()
        turned = bool(key[4]) if len(key) > 4 else False
        store = _epoch_store(self.store, epoch)
        if partners:
            idx = (i, *partners)
            arr = layout(stacked_crop(store, idx, self.image_cfg, self.aug_cfg), self.image_cfg, jitter)
            target = stacked_target(store, idx, self.aug_cfg.stack_style)
        else:
            arr, target = layout(store.crop(i), self.image_cfg, jitter), store.target(i)
            if turned:  # clockwise: the formula's start on top
                arr = np.ascontiguousarray(np.rot90(arr, k=-1))
        twin = None
        if self.twins and not partners and not turned and self.aug_cfg is not None:
            rng = twin_rng(self.seed, epoch, i)
            if rng.random() < self.aug_cfg.pair_share:
                crop = self.twins[int(rng.integers(0, len(self.twins)))].crop(i)
                twin = to_tensor(augment(layout_into(crop, self.image_cfg, arr.shape, jitter), rng, self.aug_cfg))
        if self.aug_cfg is not None:
            arr = augment(arr, sample_rng(self.seed, epoch, i), self.aug_cfg)
        return (to_tensor(arr), target, i) if twin is None else (to_tensor(arr), target, i, twin)


def collate(batch: list[tuple], pad_id: int) -> dict[str, torch.Tensor]:
    images = torch.stack([b[0] for b in batch])  # same shape by construction
    longest = max(len(b[1]) for b in batch)
    tokens = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    for row, b in enumerate(batch):
        tokens[row, : len(b[1])] = torch.from_numpy(b[1])
    out = {"images": images, "tokens": tokens, "index": torch.tensor([b[2] for b in batch])}
    rows = [r for r, b in enumerate(batch) if len(b) > 3]
    if rows:  # twin renders of these rows' labels, on the same canvas
        out["twins"] = torch.stack([batch[r][3] for r in rows])
        out["twin_rows"] = torch.tensor(rows)
    return out


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
        rotating = self.train and self.aug_cfg is not None and self.aug_cfg.p_rotate > 0
        turn = (epoch_rng(self.seed, self.epoch, 4).random(n) < self.aug_cfg.p_rotate) if rotating else None
        store = _epoch_store(self.store, self.epoch)
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in self.indices:
            if i in partners:
                h, w = stacked_size(store, (i, *partners[i]), self.image_cfg, self.aug_cfg)
                shape = output_shape(h, w, self.image_cfg.target_glyph_px, self.image_cfg, float(jitters[i]))
            else:
                shape = store.shape(i, self.image_cfg, float(jitters[i]))
                if rotating and turn[i]:
                    shape = shape[::-1]
            groups[shape].append(i)
        order_rng = epoch_rng(self.seed, self.epoch, 1)
        batches = []
        for shape in sorted(groups):
            idx = np.array(groups[shape])
            if self.train:
                idx = idx[order_rng.permutation(len(idx))]
            size = max(1, min(self.max_batch, self.pixel_budget // (shape[0] * shape[1])))
            for start in range(0, len(idx), size):
                if rotating:  # keys (i, jitter, epoch, partners, turned)
                    batches.append([(int(i), float(jitters[i]), self.epoch, partners.get(int(i), ()),
                                     bool(turn[int(i)]) and int(i) not in partners) for i in idx[start : start + size]])
                else:
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
        if aug.stack_style == "rows":  # partners: whole r l blocks (all samples when the store has too few)
            if not hasattr(self, "_rl_pool"):
                self._rl_pool = np.asarray([i for i in self.indices if rl_rows(self.store.labels[i].split())])
            if len(self._rl_pool) >= 2:
                pool = self._rl_pool
        draw = rng.random(n)
        count = rng.integers(2, aug.stack_max + 1, size=n)
        pick = pool[rng.integers(0, len(pool), size=(n, max(1, aug.stack_max - 1)))]
        out = {}
        for i in self.indices:
            if draw[i] >= aug.p_stack:
                continue
            chosen = [int(j) for j in pick[i, : count[i] - 1]]
            while chosen and stacked_length(self.store, (i, *chosen), aug.stack_style) > self.max_tokens:
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

