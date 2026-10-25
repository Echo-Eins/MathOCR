"""Line decomposition at inference (MinerU2.5's atomic decomposition, for CPE's aligned multi-line formulas).

A formula whose full-image greedy answer is an aligned block (``\\begin{array} { r l }`` or ``{ l }``) and whose
image splits into lines (``split_lines``: cuts at the emptiest rows, ink components kept whole) is read
line by line: every line image is decoded alone (greedy, the stand's decoder) and the lines are joined in the
convention of the training composition (``augment.stack_style=rl``: ``{ lhs } & { { } = rhs }``, a line without a
top-level relation as ``& { { } line }``).  Everything else keeps its full-image answer.  Deterministic.

    PROBE_ROOT=D:/MathOCR/probe_fonts70 python scripts/diagnostics/line_split.py --run win_b1_tri2_hv_s1 \
        --splits ood_cpe val --out docs/probe_results/line_split_b2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import probe  # noqa: E402
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, relation_cut  # noqa: E402
from mathocr.image import Crop, analyze, output_shape  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from PIL import Image  # noqa: E402

BLOCKS = (("\\begin{array}", "{", "r", "l", "}"), ("\\begin{array}", "{", "l", "}"))


def split_lines(crop: Crop, min_line: float = 1.2, low: float = 0.15, ink_level: int = 160) -> list[np.ndarray]:
    """The crop's lines as images.  Scripts of neighbouring lines overlap in height, so a line break is rarely a
    blank band: cuts go at the emptiest row of every stretch where the row ink count is below ``low`` x the
    median inked row, at least ``min_line`` glyph heights apart; every connected ink component then goes whole to
    the line holding its vertical centre (no glyph is cut).  One image (the crop) when there is no cut."""

    ink = crop.pixels < ink_level
    prof = ink.sum(axis=1).astype(np.float64)
    inked = prof[prof > 0]
    if inked.size == 0:
        return [crop.pixels]
    thr, min_gap = low * float(np.median(inked)), min_line * crop.glyph_px
    cuts, r, n = [], 0, len(prof)
    while r < n:
        if prof[r] <= thr:
            s = r
            while r < n and prof[r] <= thr:
                r += 1
            c = s + int(np.argmin(prof[s:r]))
            if 0 < s and r < n and (not cuts or c - cuts[-1] >= min_gap):  # interior stretches only
                cuts.append(c)
        r += 1
    cuts = [c for c in cuts if c >= min_gap and n - c >= min_gap]
    if not cuts:
        return [crop.pixels]
    labels, k = ndi.label(ink, structure=np.ones((3, 3)))
    centres = ndi.center_of_mass(ink, labels, range(1, k + 1))
    line_of = np.searchsorted(np.array(cuts), [cy for cy, _ in centres])  # component -> line index
    out = []
    for j in range(len(cuts) + 1):
        keep = np.isin(labels, np.flatnonzero(line_of == j) + 1)
        if not keep.any():
            continue
        ys, xs = np.nonzero(keep)
        img = np.where(keep, crop.pixels, 255).astype(np.uint8)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        out.append(img)
    return out


class LineStore:
    """CropStore-like: the line images, analyzed afresh (own bounding box and glyph height)."""

    def __init__(self, crops: list[Crop], vocab):
        self.crops, self.vocab = crops, vocab
        self._target = np.array([vocab.bos_id, vocab.eos_id], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.crops)

    def crop(self, i: int) -> Crop:
        return self.crops[i]

    def target(self, i: int) -> np.ndarray:
        return self._target

    def shape(self, i, cfg, jitter=1.0):
        c = self.crops[i]
        return output_shape(c.pixels.shape[0], c.pixels.shape[1], c.glyph_px, cfg, jitter)


def top_rows(tokens: list[str]) -> int:
    """Rows of a block answer: top-level ``\\\\`` + 1."""

    depth, rows = 0, 1
    for t in tokens[1:-1]:
        if t == "{" or t.startswith("\\begin{"):
            depth += 1
        elif t == "}" or t.startswith("\\end{"):
            depth -= 1
        elif t == "\\\\" and depth == 0:
            rows += 1
    return rows


def join_rl(lines: list[list[str]]) -> list[str]:
    out = list(BLOCKS[0])
    for n, toks in enumerate(lines):
        if n:
            out.append("\\\\")
        cut = relation_cut(toks)
        if cut is None:
            out += ["&", "{", "{", "}", *toks, "}"]
        else:
            out += ["{", *toks[:cut], "}", "&", "{", "{", "}", *toks[cut:], "}"]
    return out + ["\\end{array}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", default=["ood_cpe", "val"])
    ap.add_argument("--min-line", type=float, default=1.2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    determinism.configure(dev)
    torch.cuda.set_per_process_memory_fraction(0.6, torch.cuda.current_device())
    root = Path(os.environ["PROBE_ROOT"])
    model, vocab, icfg, _ = load_checkpoint(root / "runs" / args.run / "checkpoints" / "best.pt", dev)
    report = {"run": args.run, "min_line": args.min_line}
    for split in args.splits:
        store = CropStore(root / "data20k" / f"{split}.parquet", vocab)
        full = probe.decode(model, store, icfg, vocab, dev)
        cand, line_crops, owner = [], [], []
        for i, toks in enumerate(full):
            t = [x for x in toks if x != "<noeos>"]
            if not any(tuple(t[: len(b)]) == b for b in BLOCKS):
                continue
            crop = store.crop(i)
            lines_img = split_lines(crop, args.min_line)
            if len(lines_img) < 2 or len(lines_img) != top_rows(t):  # the image's lines = the answer's rows
                continue
            cand.append(i)
            for img in lines_img:
                line_crops.append(analyze(Image.fromarray(img)))
                owner.append(i)
        lines = probe.decode(model, LineStore(line_crops, vocab), icfg, vocab, dev) if line_crops else []
        per: dict[int, list[list[str]]] = {}
        for i, toks in zip(owner, lines):
            per.setdefault(i, []).append([x for x in toks if x != "<noeos>"])
        joined = [join_rl(per[i]) if i in per else toks for i, toks in enumerate(full)]
        base = probe.score(full, store.labels)
        new = probe.score(joined, store.labels)
        sub = sorted(per)
        report[split] = {"n": len(store), "decomposed": len(sub), "lines": len(line_crops),
                         "full": {k: base[k] for k in ("em", "edit", "bleu")},
                         "split": {k: new[k] for k in ("em", "edit", "bleu")},
                         "on_decomposed": {"full": probe.score([full[i] for i in sub], [store.labels[i] for i in sub])["edit"] if sub else None,
                                           "split": probe.score([joined[i] for i in sub], [store.labels[i] for i in sub])["edit"] if sub else None}}
        print(split, json.dumps(report[split]), flush=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
