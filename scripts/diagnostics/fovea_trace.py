"""Where does the foveal reader look?  Draws every decoding step's fixation on the image.

    python scripts/diagnostics/fovea_trace.py CHECKPOINT DATA.parquet OUT.png [--n 6]

Each panel: the model input with fixations as numbered dots (colour runs
from blue = first token to red = last), the innermost ring as a square, the
predicted and true LaTeX underneath.  Also prints, per sample, the share of
consecutive fixations that move rightwards (reading order) and the zooms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, TrainItems  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.tokenizer import detokenize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(args.checkpoint, device)
    assert model.fovea is not None, "not a foveal model"
    store = CropStore(args.data, vocab)
    items = TrainItems(store, icfg, None, 0)
    panels, scale = [], 2
    for i in range(args.start, args.start + args.n):
        img, _, _ = items[(i, 1.0, 0)]
        trace: list = []
        with torch.no_grad():
            ids = model.generate(img[None].to(device), vocab.bos_id, vocab.eos_id, trace=trace)
        pred = vocab.decode(ids[0].tolist())
        fx = torch.cat(trace, dim=1)[0, : len(pred) + 1].cpu().numpy()  # (steps, 3): x, y, zoom
        arr = (255 - img[0].numpy() * 255).astype(np.uint8)
        canvas = Image.fromarray(arr).convert("RGB").resize((arr.shape[1] * scale, arr.shape[0] * scale))
        draw = ImageDraw.Draw(canvas)
        half = model.fovea.size * model.fovea.base / 2
        for t, (x, y, z) in enumerate(fx):
            c = int(255 * t / max(1, len(fx) - 1))
            col = (c, 40, 255 - c)
            r = half * z * scale
            draw.rectangle([x * scale - r, y * scale - r, x * scale + r, y * scale + r], outline=col)
            draw.ellipse([x * scale - 3, y * scale - 3, x * scale + 3, y * scale + 3], fill=col)
            draw.text((x * scale + 4, y * scale - 12), str(t), fill=col)
        text = Image.new("RGB", (canvas.width, 36), "white")
        td = ImageDraw.Draw(text)
        td.text((2, 2), "pred: " + detokenize(pred)[:160], fill=(180, 0, 0))
        td.text((2, 18), "true: " + store.labels[i][:160], fill=(0, 100, 0))
        panel = Image.new("RGB", (canvas.width, canvas.height + 36), "white")
        panel.paste(canvas, (0, 0))
        panel.paste(text, (0, canvas.height))
        panels.append(panel)
        dx = np.diff(fx[:, 0])
        print(f"{store.uids[i]}: steps={len(fx)} rightward={np.mean(dx > 0):.2f} "
              f"zoom mean={fx[:, 2].mean():.2f} [{fx[:, 2].min():.2f}, {fx[:, 2].max():.2f}] "
              f"exact={detokenize(pred) == store.labels[i]}")
    width = max(p.width for p in panels)
    sheet = Image.new("RGB", (width, sum(p.height + 8 for p in panels)), "white")
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 8
    sheet.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
