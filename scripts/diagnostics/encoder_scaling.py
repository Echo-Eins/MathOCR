"""Cost of the global mixer vs the size of the feature map (fwd + bwd, batch 1, bf16).

    python scripts/diagnostics/encoder_scaling.py [--cap-gib 12]

Builds only the 3-block mixer stack at d = 512 (Transformer encoder, AAS,
AAS-sort) and times one training step on a random map of growing size,
under the same deterministic settings as training.  The process is capped
at ``--cap-gib`` of GPU memory, so an over-large case fails inside PyTorch
instead of pressing on the machine's shared memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mathocr import determinism  # noqa: E402
from mathocr.model import AASBlock, EncoderLayer, SortBlock  # noqa: E402

KINDS = sys.argv[sys.argv.index("--kinds") + 1].split(",") if "--kinds" in sys.argv else ["transformer", "aas", "aas_sort"]
SIZES = [(12, 96), (24, 192), (48, 384), (96, 768)]  # map cells; image = 16x height, 8x width


def stack(kind: str, d: int = 512) -> nn.ModuleList:
    if kind == "transformer":
        return nn.ModuleList(EncoderLayer(d, 8, 4, 0.0) for _ in range(3))
    if kind in ("aas", "aas_kernel"):
        return nn.ModuleList(AASBlock(d, 8, 1.5, kernel=kind == "aas_kernel") for _ in range(3))
    return nn.ModuleList(SortBlock(d, 8, 1.5, 32) for _ in range(3))


def step(kind: str, layers: nn.ModuleList, h: int, w: int) -> None:
    x = torch.randn(1, 512, h, w, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = x.flatten(2).transpose(1, 2) if kind == "transformer" else x
        for layer in layers:
            y = layer(y)
    y.float().square().mean().backward()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gib", type=float, default=12.0)
    ap.add_argument("--kinds", default="transformer,aas,aas_sort")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    determinism.configure(dev)
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap_gib * 2**30 / total), dev)
    for kind in KINDS:
        layers = stack(kind).to(dev)
        for h, w in SIZES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                step(kind, layers, h, w)  # warm-up
                torch.cuda.synchronize()
                t0 = time.time()
                step(kind, layers, h, w)
                torch.cuda.synchronize()
                rec = {"mixer": kind, "map": f"{h}x{w}", "tokens": h * w, "image_px": f"{16 * h}x{8 * w}",
                       "ms": round(1000 * (time.time() - t0), 1),
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            except torch.OutOfMemoryError:
                rec = {"mixer": kind, "map": f"{h}x{w}", "tokens": h * w, "result": f"OOM at {args.cap_gib} GiB cap"}
            layers.zero_grad(set_to_none=True)
            print(json.dumps(rec), flush=True)
        del layers


if __name__ == "__main__":
    main()
