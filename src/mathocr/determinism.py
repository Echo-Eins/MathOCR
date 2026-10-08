"""Seeds and the deterministic execution contract.

Two separate guarantees, both tested:

* **inference** -- ``predict`` on the same image returns the same LaTeX:
  canonical preprocessing (``image.prepare``), ``eval()`` mode, greedy
  argmax decoding and deterministic kernels;
* **training** -- the same config and seed reproduce the same weights:
  every random draw comes from generators derived from the seed
  (``numpy.random.SeedSequence``), including per-sample augmentation, which
  is keyed by ``(seed, epoch, sample index)`` and therefore independent of
  the number of DataLoader workers and of which worker loads which sample.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure(device: torch.device) -> dict[str, object]:
    """Enable strict PyTorch/CUDA determinism; return the manifest to record."""

    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "cuda" and workspace not in (":4096:8", ":16:8"):
        raise ValueError("CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": workspace,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": False,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def sample_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Augmentation RNG of one sample in one epoch."""

    return np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))


def epoch_rng(seed: int, epoch: int, stream: int) -> np.random.Generator:
    """Epoch-level RNG (batch order, bucketing); ``stream`` separates uses."""

    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 2**31 + stream]))
