"""Recognize formula images: ``python -m mathocr.predict --checkpoint best.pt a.png b.jpg``.

Prints one JSON line per image.  Uses the canonical path (one image at a
time, fp32, deterministic kernels), so a given image always yields the same
LaTeX; ``--sha256`` adds the hash of the input bytes for audit trails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from mathocr import determinism
from mathocr.infer import load_checkpoint, predict_images


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sha256", action="store_true")
    ap.add_argument("images", nargs="+", type=Path)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(args.checkpoint, device)
    for path in args.images:
        with Image.open(path) as im:
            im.load()
            latex = predict_images(model, [im], icfg, vocab, device)[0]
        rec = {"image": str(path), "latex": latex}
        if args.sha256:
            rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
