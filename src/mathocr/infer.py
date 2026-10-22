"""Inference: checkpoint loading, greedy decoding of splits and single images.

``predict_images`` is the canonical "image -> LaTeX" path: each image goes
through ``image.prepare`` and is decoded alone (batch of one, fp32,
deterministic kernels), so the output is a function of the image bytes only.
``decode_store`` batches images of identical shape for evaluation speed;
``tests/test_determinism.py`` measures its agreement with the canonical path.

On CUDA both run the decoder step as a replayed CUDA graph (``mathocr.graphs``:
static cache, single-query attention as matmuls); ``MATHOCR_CUDA_GRAPHS=0``
selects the eager loop, the reference the graphed path is compared against.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from mathocr import graphs
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate
from mathocr.image import ImageConfig, prepare, to_tensor
from mathocr.model import Im2Latex, ModelConfig
from mathocr.tokenizer import Vocab, detokenize


def load_checkpoint(path: Path, device: torch.device) -> tuple[Im2Latex, Vocab, ImageConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ckpt["model_config"]
    model = Im2Latex(ModelConfig(**{**mcfg, "cnn_channels": tuple(mcfg["cnn_channels"]),
                                     "cnn_blocks": tuple(mcfg["cnn_blocks"])}))
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))  # an EMA run deploys its averaged weights
    model.to(device).eval()
    return model, Vocab(ckpt["vocab"]), ImageConfig(**ckpt["image_config"]), ckpt


def trim_runaway(tokens: list[str], max_period: int = 16, min_repeats: int = 3) -> list[str]:
    r"""A decode that never emitted EOS usually ends in a loop (``\! \! \! ...`` on screenshots): drop the
    periodic tail (period <= ``max_period`` tokens, >= ``min_repeats`` copies), keeping one copy.  Applied only
    to decodes without EOS, so every normally terminated output is untouched."""

    n, best = len(tokens), len(tokens)
    for per in range(1, max_period + 1):
        s = n - per
        while s - per >= 0 and tokens[s - per : s] == tokens[s : s + per]:
            s -= per
        if n - s >= min_repeats * per:
            best = min(best, s + per)
    return tokens[:best]


def _to_strings(ids: torch.Tensor, vocab: Vocab) -> list[str]:
    out = []
    for row in ids.tolist():
        tokens = vocab.decode(row)
        out.append(detokenize(tokens if vocab.eos_id in row else trim_runaway(tokens)))
    return out


def greedy(model: Im2Latex, vocab: Vocab, device: torch.device):
    """``images -> ids`` for greedy decoding: the graphed step on CUDA, the eager loop otherwise.
    Call ``.release()`` (if present) when done to free the graph's buffers."""

    if graphs.enabled(model, device):
        return graphs.GraphedGreedy(model, vocab.bos_id, vocab.eos_id)
    return lambda images: model.generate(images, vocab.bos_id, vocab.eos_id)


@torch.no_grad()
def decode_store(model: Im2Latex, store: CropStore, image_cfg: ImageConfig, vocab: Vocab,
                 device: torch.device, pixel_budget: int, max_batch: int,
                 indices: Sequence[int] | None = None) -> list[str]:
    """Greedy predictions for every sample of ``store`` (or ``indices``), in order."""

    model.eval()
    sampler = ShapeBatchSampler(store, image_cfg, pixel_budget, max_batch, seed=0, train=False, indices=indices)
    items = TrainItems(store, image_cfg, aug_cfg=None, seed=0)
    decoder = greedy(model, vocab, device)
    out: dict[int, str] = {}
    try:
        for batch_keys in sampler:
            batch = collate([items[k] for k in batch_keys], vocab.pad_id)
            ids = decoder.decode(batch["images"].to(device)) if hasattr(decoder, "decode") else decoder(batch["images"].to(device))
            for i, s in zip(batch["index"].tolist(), _to_strings(ids, vocab)):
                out[i] = s
    finally:
        if hasattr(decoder, "release"):
            decoder.release()
    order = range(len(store)) if indices is None else indices
    return [out[i] for i in order]


@torch.no_grad()
def predict_images(model: Im2Latex, images: Sequence[Image.Image], image_cfg: ImageConfig, vocab: Vocab,
                   device: torch.device) -> list[str]:
    """Canonical path: one image at a time."""

    model.eval()
    decoder = greedy(model, vocab, device)
    out = []
    try:
        for image in images:
            x = to_tensor(prepare(image, image_cfg)).unsqueeze(0).to(device)
            ids = decoder.decode(x) if hasattr(decoder, "decode") else decoder(x)
            out.extend(_to_strings(ids, vocab))
    finally:
        if hasattr(decoder, "release"):
            decoder.release()
    return out
