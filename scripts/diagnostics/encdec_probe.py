"""(Paths below are those of the Windows laptop.)  What the encoder extracts and how well the decoder uses it, for one
checkpoint, on the box-rendered labels of D:/MathOCR/oracle (stage2_oracle.py: val + OOD labels in CM / Times /
Palatino / sans, one box per visible token, mapped onto the memory grid).

1. Effective receptive field: gradient of one memory cell (a random direction of its features) w.r.t. the input
   pixels, at three depths -- the CNN's projected features, after the first encoder block (TriScan), the final
   memory; |dy| / |dx| (pixels, from the cell centre) holding 50 / 90 % of the gradient mass, and the mass beyond
   the CNN's own reach (|dx| > 64 or |dy| > 32 px).
2. What the memory knows of each glyph: a linear probe (softmax over token ids) on the mean memory feature of a
   visible token's box cells, at the same three depths; trained on 70 % of the val labels in CM, scored on the other
   30 % in every font (the probe never saw Times / sans / Palatino; the model never saw Palatino).  Font stability:
   cosine between the CM and the other-font feature of the same token instance, against other tokens.
3. Decoder: teacher-forced next-token accuracy (visible / structural tokens) per split and font, and the cascade
   after a first error (accuracy of the positions after the first wrong one, under teacher forcing).

    python scripts/diagnostics/encdec_probe.py --name final_v3_s1 --ckpt C:/Coding/Python/MathOCR_final/runs/final_v3_s1/checkpoints/best.pt
Output: docs/probe_results/encdec_<name>.json."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))
from mathocr import determinism  # noqa: E402
from mathocr.image import to_tensor  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.model import stride_hw  # noqa: E402
from stage2_oracle import align_boxes, prepare_with_map  # noqa: E402

WORK = Path(r"D:\MathOCR\oracle")
FONTS = ("cm", "times", "palatino", "sans")
DEPTHS = ("cnn", "block1", "memory")
dev = torch.device("cuda")


def encode_levels(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The three depths of ``model.encode`` (B, N, d), via hooks (the forward itself is the model's)."""
    out = {}
    hooks = [model.proj.register_forward_hook(lambda m, i, o: out.__setitem__("cnn", o)),
             model.encoder[0].register_forward_hook(
                 lambda m, i, o: out.__setitem__("block1", o.reshape(o.shape[0], -1, o.shape[-1]))),
             model.enc_norm.register_forward_hook(lambda m, i, o: out.__setitem__("memory", o))]
    try:
        model.encode(x)
    finally:
        for h in hooks:
            h.remove()
    return out


def receptive_field(model, images: list[np.ndarray], rng) -> dict:
    sh, sw = stride_hw(model.cfg.enc_stride)
    stats = {d: {"dy": [], "dx": [], "w": [], "far": [], "tot": []} for d in DEPTHS}
    for arr in images:
        x = to_tensor(arr)[None].to(dev).requires_grad_(True)
        gh, gw = arr.shape[0] // sh, arr.shape[1] // sw
        ink = torch.from_numpy(arr < 128).to(dev).float()[None, None]
        cell_ink = F.max_pool2d(ink, (sh, sw))[0, 0].flatten()
        cand = cell_ink.nonzero().flatten().tolist()
        if not cand:
            continue
        for c in rng.choice(cand, size=min(3, len(cand)), replace=False).tolist():
            r0, c0 = divmod(c, gw)
            cy, cx = r0 * sh + sh / 2, c0 * sw + sw / 2
            for depth in DEPTHS:
                feats = encode_levels(model, x)[depth]
                v = torch.from_numpy(rng.standard_normal(feats.shape[-1])).to(dev, feats.dtype)
                (g,) = torch.autograd.grad((feats[0, c] * v).sum(), x)
                m = g[0, 0].abs().double().cpu().numpy()
                ys, xs = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
                dy, dx = np.abs(ys + 0.5 - cy), np.abs(xs + 0.5 - cx)
                s = stats[depth]
                s["dy"].append(dy.ravel()), s["dx"].append(dx.ravel()), s["w"].append(m.ravel() / max(m.sum(), 1e-30))
                s["far"].append(float(m[(dx > 64) | (dy > 32)].sum() / max(m.sum(), 1e-30)))
    out = {}
    for depth, s in stats.items():
        w, dy, dx = np.concatenate(s["w"]), np.concatenate(s["dy"]), np.concatenate(s["dx"])
        q = {}
        for name, d in (("dy", dy), ("dx", dx)):
            o = np.argsort(d)
            cw = np.cumsum(w[o]) / w.sum()
            q[name] = {p: float(d[o][np.searchsorted(cw, p / 100)]) for p in (50, 90)}
        out[depth] = {"dy_px_50": q["dy"][50], "dy_px_90": q["dy"][90], "dx_px_50": q["dx"][50], "dx_px_90": q["dx"][90],
                      "mass_beyond_cnn_reach": float(np.mean(s["far"])), "cells": len(s["far"])}
    return out


@torch.no_grad()
def sample_features(model, vocab, label: str, img: Path, box: Path):
    """(token ids, {depth: (T, d) box-mean features}, teacher-forced hits, visible mask) for one rendered label."""
    image = Image.open(img).convert("L")
    arr, to_model = prepare_with_map(image, model_icfg)
    meta = json.loads(box.read_text(encoding="utf-8"))
    ours = label.split()
    boxes = align_boxes(ours, meta["tokens"], meta["boxes"])
    x = to_tensor(arr)[None].to(dev)
    sh, sw = stride_hw(model.cfg.enc_stride)
    gh, gw = arr.shape[0] // sh, arr.shape[1] // sw
    levels = encode_levels(model, x)
    feats, ids = {d: [] for d in DEPTHS}, []
    for t, b in enumerate(boxes):
        if b is None or ours[t] not in vocab.stoi:
            continue
        x0, y0, x1, y1 = to_model(b)
        r0, r1 = max(0, int(y0 // sh)), min(gh, int(math.ceil(y1 / sh)))
        c0, c1 = max(0, int(x0 // sw)), min(gw, int(math.ceil(x1 / sw)))
        if r1 <= r0 or c1 <= c0:
            continue
        cells = torch.tensor([r * gw + c for r in range(r0, r1) for c in range(c0, c1)], device=dev)
        for d in DEPTHS:
            feats[d].append(levels[d][0, cells].float().mean(0))
        ids.append((t, vocab.stoi[ours[t]]))
    # teacher forcing on the full label
    tok = torch.tensor([vocab.encode(ours)], device=dev)
    logits = model(x, tok[:, :-1])
    hits = (logits.argmax(-1) == tok[:, 1:])[0].cpu().numpy()
    visible = np.zeros(len(hits), dtype=bool)
    for t, b in enumerate(boxes[: len(hits)]):
        visible[t] = b is not None
    return ids, {d: torch.stack(v).cpu() if v else None for d, v in feats.items()}, hits, visible


def main() -> None:
    global model_icfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--rf-images", type=int, default=60)
    args = ap.parse_args()
    determinism.configure(dev)
    torch.cuda.set_per_process_memory_fraction(0.5, torch.cuda.current_device())
    model, vocab, model_icfg, _ = load_checkpoint(args.ckpt, dev)
    model.eval()
    labels = [json.loads(x) for x in (WORK / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = np.random.default_rng(0)
    report = {"name": args.name}

    # 1. receptive field on val CM images
    val = [lab for lab in labels if lab["split"] == "val"]
    imgs = []
    for lab in val[: args.rf_images]:
        p = WORK / "png" / "cm" / f"{lab['id']}.png"
        if p.exists():
            imgs.append(prepare_with_map(Image.open(p).convert("L"), model_icfg)[0])
    torch.set_grad_enabled(True)
    report["receptive_field"] = receptive_field(model, imgs, rng)
    print(json.dumps(report["receptive_field"]), flush=True)

    # 2 + 3. features and teacher forcing
    data = defaultdict(list)  # (split, font) -> samples
    for font in FONTS:
        for lab in labels:
            img, box = WORK / "png" / font / f"{lab['id']}.png", WORK / "box" / font / f"{lab['id']}.json"
            if not (img.exists() and box.exists()) or len(lab["latex"].split()) > model.cfg.max_len - 2:
                continue
            try:
                ids, feats, hits, visible = sample_features(model, vocab, lab["latex"], img, box)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            data[(lab["split"], font)].append({"id": lab["id"], "ids": ids, "feats": feats, "hits": hits, "visible": visible})
        print(f"features: {font} done", flush=True)

    # decoder: teacher forcing and the cascade after the first error
    dec = {}
    for (split, font), rows in sorted(data.items()):
        hv = np.concatenate([r["hits"][r["visible"]] for r in rows])
        hs = np.concatenate([r["hits"][~r["visible"]] for r in rows])
        after = [r["hits"][int(np.argmin(r["hits"])) + 1 :] for r in rows if not r["hits"].all()]
        after = np.concatenate([a for a in after if len(a)]) if after else np.array([1.0])
        dec[f"{split} {font}"] = {"n": len(rows), "tf_acc_visible": float(hv.mean()), "tf_acc_structural": float(hs.mean()),
                                   "tf_formula_all_right": float(np.mean([r["hits"].all() for r in rows])),
                                   "tf_acc_after_first_error": float(after.mean())}
    report["decoder"] = dec
    print(json.dumps(dec, indent=0), flush=True)

    # encoder: glyph-identity probe per depth, trained on CM val (70 % of the labels), scored on the rest
    val_ids = sorted({r["id"] for r in data[("val", "cm")]})
    cut = set(rng.permutation(val_ids)[: int(0.7 * len(val_ids))].tolist())
    probe = {}
    for depth in DEPTHS:
        def xy(rows, train):
            xs, ys = [], []
            for r in rows:
                if (r["id"] in cut) == train and r["feats"][depth] is not None:
                    xs.append(r["feats"][depth])
                    ys += [k for _, k in r["ids"]]
            return (torch.cat(xs), torch.tensor(ys)) if xs else (None, None)
        x_tr, y_tr = xy(data[("val", "cm")], True)
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
        lin = torch.nn.Linear(x_tr.shape[1], len(vocab)).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-4)
        xt, yt = ((x_tr - mu) / sd).to(dev), y_tr.to(dev)
        g = torch.Generator(device="cpu").manual_seed(0)
        for _ in range(60):
            for idx in torch.randperm(len(yt), generator=g).split(1024):
                idx = idx.to(dev)
                opt.zero_grad()
                F.cross_entropy(lin(xt[idx]), yt[idx]).backward()
                opt.step()
        res = {}
        for font in FONTS:
            for split in ("val", "ood_spe", "ood_sce", "ood_cpe"):
                rows = data.get((split, font), [])
                x_te, y_te = xy(rows, False) if split == "val" else xy([{**r, "id": -1} for r in rows], False)
                if x_te is None:
                    continue
                with torch.no_grad():
                    acc = (lin(((x_te - mu) / sd).to(dev)).argmax(-1).cpu() == y_te).float().mean().item()
                res[f"{split} {font}"] = round(acc, 4)
        # font stability: the same token instance in CM and in another font (val, held-out labels)
        cm = {r["id"]: r for r in data[("val", "cm")] if r["id"] not in cut and r["feats"][depth] is not None}
        for font in ("times", "palatino", "sans"):
            same, other = [], []
            for r in data[("val", font)]:
                c = cm.get(r["id"])
                if c is None or r["feats"][depth] is None or [t for t, _ in c["ids"]] != [t for t, _ in r["ids"]]:
                    continue
                a, b = F.normalize(c["feats"][depth], dim=-1), F.normalize(r["feats"][depth], dim=-1)
                sim = a @ b.T
                same.append(sim.diag().mean().item())
                if len(sim) > 1:
                    other.append(((sim.sum() - sim.diag().sum()) / (len(sim) ** 2 - len(sim))).item())
            res[f"cosine same token CM-{font}"] = round(float(np.mean(same)), 4) if same else None
            res[f"cosine other tokens CM-{font}"] = round(float(np.mean(other)), 4) if other else None
        probe[depth] = res
        print(depth, json.dumps(res), flush=True)
    report["glyph_probe"] = probe
    (ROOT / "docs" / "probe_results" / f"encdec_{args.name}.json").write_text(json.dumps(report, indent=1) + "\n",
                                                                           encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
