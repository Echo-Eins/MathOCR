"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 2: is the remaining error a matter of *where* the decoder reads or of *what* it reads there?

Labels of the stand's val and OOD sets rendered with one colour per visible token (scripts/fonts/render_boxes.py:
CM / Times / Palatino / sans, token boxes).  Our tokens are aligned to CDM's coloured tokens; every box is mapped
through the model's own preprocessing (ink crop, glyph-height scale, margin) onto the encoder memory grid.  Teacher
forcing, explicit cross-attention in every decoder layer:
  none        -- as trained; also the attention mass the decoder puts inside the true box of the token it predicts;
  oracle b    -- +b on the cross-attention logits of the cells of that box (b = 2, 5, 20: 20 ~ a hard mask),
                 in every layer, or only in the last two;
  wrong box   -- the same +5 on the box of another token of the formula (control: must hurt).
Token accuracy on boxed (visible) tokens and on all tokens; formulas with every token right.
Output: docs/probe_results/stage2_oracle_<name>.json."""

import argparse
import json
import math
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr import determinism  # noqa: E402
from mathocr.image import _scaled_size, _stretch, analyze, layout, to_gray, to_tensor  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.model import stride_hw  # noqa: E402

WORK = Path(r"D:\MathOCR\oracle")
FONTS = ("cm", "times", "palatino", "sans")
dev = torch.device("cuda")
CONDITIONS = (("none", None, None), ("oracle 2", 2.0, "all"), ("oracle 5", 5.0, "all"), ("oracle 20", 20.0, "all"),
              ("oracle 5 last2", 5.0, "last2"), ("wrong box 5", 5.0, "wrong"))


def prepare_with_map(image: Image.Image, icfg):
    """The model input and the map from source pixels to its coordinates (mathocr.image.prepare, unrolled)."""
    stretched = _stretch(to_gray(image))
    ink = stretched < 128.0
    ys, xs = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    crop = analyze(image)
    h, w = crop.pixels.shape
    nh, nw = _scaled_size(h, w, crop.glyph_px, icfg, 1.0)
    arr = layout(crop, icfg)
    sy, sx, m = nh / h, nw / w, icfg.margin
    return arr, lambda b: ((b[0] - xs[0]) * sx + m, (b[1] - ys[0]) * sy + m, (b[2] - xs[0]) * sx + m, (b[3] - ys[0]) * sy + m)


def align_boxes(ours: list[str], cdm_tokens: list[str], boxes: list) -> list:
    """Box (or None) for each of our tokens: equal runs of the two token sequences."""
    out = [None] * len(ours)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ours, cdm_tokens, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if boxes[j1 + k]:
                    out[i1 + k] = boxes[j1 + k]
    return out


@torch.no_grad()
def run_sample(model, vocab, icfg, label, img_path, box_path, rng):
    image = Image.open(img_path).convert("L")
    arr, to_model = prepare_with_map(image, icfg)
    meta = json.loads(box_path.read_text(encoding="utf-8"))
    ours = label.split()
    boxes = align_boxes(ours, meta["tokens"], meta["boxes"])
    ids = torch.tensor([vocab.encode(ours)], device=dev)
    inp, out = ids[:, :-1], ids[:, 1:]
    t_len = inp.shape[1]
    x_img = to_tensor(arr)[None].to(dev)
    sh, sw = stride_hw(model.cfg.enc_stride)
    gh, gw = arr.shape[0] // sh, arr.shape[1] // sw
    n_cells = gh * gw
    cell_masks = torch.zeros(t_len, n_cells, dtype=torch.bool, device=dev)
    boxed = torch.zeros(t_len, dtype=torch.bool, device=dev)
    for t in range(min(t_len, len(ours))):  # position t predicts our token t
        b = boxes[t]
        if b is None:
            continue
        x0, y0, x1, y1 = to_model(b)
        r0, r1 = max(0, int(y0 // sh)), min(gh, int(math.ceil(y1 / sh)))
        c0, c1 = max(0, int(x0 // sw)), min(gw, int(math.ceil(x1 / sw)))
        if r1 <= r0 or c1 <= c0:
            continue
        grid = torch.zeros(gh, gw, dtype=torch.bool, device=dev)
        grid[r0:r1, c0:c1] = True
        cell_masks[t] = grid.flatten()
        boxed[t] = True
    if not boxed.any():
        return None
    # control: each boxed position gets the box of another boxed token of the same formula
    pos = boxed.nonzero().flatten().tolist()
    wrong = cell_masks.clone()
    if len(pos) > 1:
        perm = pos[1:] + pos[:1]
        for a, b in zip(pos, perm):
            wrong[a] = cell_masks[b]
    memory = model.encode(x_img)
    kvs = model._memory_kv(memory)
    res = {}
    n_layers = len(model.decoder)
    for name, beta, where in CONDITIONS:
        x = model._embed(inp, 0)
        mass = []
        for li, (layer, kv) in enumerate(zip(model.decoder, kvs)):
            bias = None
            if beta is not None and (where in ("all", "wrong") or (where == "last2" and li >= n_layers - 2)):
                mask = wrong if where == "wrong" else cell_masks
                bias = (mask.float() * beta)[None, None]  # (1, 1, T, N)
            x, w = layer(x, kv, None, return_cross=True, cross_bias=bias)
            if name == "none":
                mass.append((w[0] * cell_masks.float()).sum(-1))  # (T,) attention inside the true box
        pred = model._logits(x).argmax(-1)[0]
        hit = pred == out[0]
        res[name] = {"boxed": hit[boxed].float().mean().item(), "all": hit.float().mean().item(),
                     "formula": bool(hit.all().item()), "n_boxed": int(boxed.sum())}
        if name == "none":
            m = torch.stack(mass).mean(0)[boxed]  # mean over layers
            res[name]["mass_in_box"] = m.mean().item()
            res[name]["hits_low_mass"] = hit[boxed][m < 0.2].tolist()
            res[name]["hits_high_mass"] = hit[boxed][m >= 0.2].tolist()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    determinism.configure(dev)
    torch.cuda.set_per_process_memory_fraction(0.5, torch.cuda.current_device())
    model, vocab, icfg, _ = load_checkpoint(args.ckpt, dev)
    labels = [json.loads(x) for x in (WORK / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = np.random.default_rng(0)
    agg = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for font in FONTS:
        for k, lab in enumerate(labels):
            img, box = WORK / "png" / font / f"{lab['id']}.png", WORK / "box" / font / f"{lab['id']}.json"
            if not (img.exists() and box.exists()) or len(lab["latex"].split()) > model.cfg.max_len - 2:
                skipped += 1
                continue
            try:
                r = run_sample(model, vocab, icfg, lab["latex"], img, box, rng)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1
                continue
            if r is None:
                skipped += 1
                continue
            agg[(lab["split"], font)]["rows"].append(r)
        print(f"{args.name}: font {font} done", flush=True)
    out = {}
    for (split, font), d in sorted(agg.items()):
        rows = d["rows"]
        row = {"n": len(rows)}
        for name, *_ in CONDITIONS:
            row[name] = {"boxed_acc": float(np.mean([r[name]["boxed"] for r in rows])),
                         "all_acc": float(np.mean([r[name]["all"] for r in rows])),
                         "formula_all_right": float(np.mean([r[name]["formula"] for r in rows]))}
        lo = [h for r in rows for h in r["none"]["hits_low_mass"]]
        hi = [h for r in rows for h in r["none"]["hits_high_mass"]]
        row["mass_in_box"] = float(np.mean([r["none"]["mass_in_box"] for r in rows]))
        row["acc_low_mass"], row["share_low_mass"] = (float(np.mean(lo)) if lo else None), len(lo) / max(1, len(lo) + len(hi))
        row["acc_high_mass"] = float(np.mean(hi)) if hi else None
        out[f"{split} {font}"] = row
        print(f"{split:8} {font:9} n={len(rows):4} | box mass {row['mass_in_box']:.2f}, low-mass tokens "
              f"{row['share_low_mass'] * 100:4.1f}% (acc {row['acc_low_mass'] if row['acc_low_mass'] is not None else float('nan'):.3f}) "
              f"vs high (acc {row['acc_high_mass']:.3f}) | " + " | ".join(
                  f"{name} {row[name]['boxed_acc']:.4f}/{row[name]['formula_all_right']:.3f}" for name, *_ in CONDITIONS), flush=True)
    out["skipped"] = skipped
    (ROOT / "docs" / "probe_results" / f"stage2_oracle_{args.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
