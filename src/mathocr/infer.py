"""Inference: checkpoint loading, greedy decoding of splits and single images.

``predict_images`` is the canonical "image -> LaTeX" path: each image goes
through ``image.prepare`` and is decoded alone (batch of one, fp32,
deterministic kernels), so the output is a function of the image bytes only.
``decode_store`` batches images of identical shape for evaluation speed;
``tests/test_determinism.py`` measures its agreement with the canonical path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

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


def _to_strings(ids: torch.Tensor, vocab: Vocab) -> list[str]:
    return [detokenize(vocab.decode(row)) for row in ids.tolist()]


@torch.no_grad()
def decode_store(model: Im2Latex, store: CropStore, image_cfg: ImageConfig, vocab: Vocab,
                 device: torch.device, pixel_budget: int, max_batch: int,
                 indices: Sequence[int] | None = None) -> list[str]:
    """Greedy predictions for every sample of ``store`` (or ``indices``), in order."""

    model.eval()
    sampler = ShapeBatchSampler(store, image_cfg, pixel_budget, max_batch, seed=0, train=False, indices=indices)
    items = TrainItems(store, image_cfg, aug_cfg=None, seed=0)
    out: dict[int, str] = {}
    for batch_keys in sampler:
        batch = collate([items[k] for k in batch_keys], vocab.pad_id)
        ids = model.generate(batch["images"].to(device), vocab.bos_id, vocab.eos_id)
        for i, s in zip(batch["index"].tolist(), _to_strings(ids, vocab)):
            out[i] = s
    order = range(len(store)) if indices is None else indices
    return [out[i] for i in order]


@torch.no_grad()
def predict_images(model: Im2Latex, images: Sequence[Image.Image], image_cfg: ImageConfig, vocab: Vocab,
                   device: torch.device) -> list[str]:
    """Canonical path: one image at a time."""

    model.eval()
    out = []
    for image in images:
        x = to_tensor(prepare(image, image_cfg)).unsqueeze(0).to(device)
        out.extend(_to_strings(model.generate(x, vocab.bos_id, vocab.eos_id), vocab))
    return out
