"""(Paths below are those of the Windows laptop; edit them for another machine.  Results of 2026-10-22:
docs/probe_results/fontswap_final_v2_s1.json.)
Decode the font-swap renders with final_v2_s1 (canonical path, with the runaway trim) and compare against the
original test images of the same labels.  Same label, different font -> the font effect with content held fixed."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"C:\Coding\Python\MathOCR\src")
from mathocr import determinism  # noqa: E402
from mathocr.infer import load_checkpoint, predict_images, trim_runaway  # noqa: E402
from mathocr.metrics import normalized_levenshtein  # noqa: E402

HERE = Path(r"D:\MathOCR\fontswap")  # WORK of make_labels.py
RUN = Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1")
FONTS = ("cm", "times", "palatino", "sans")
dev = torch.device("cuda")
determinism.configure(dev)
model, vocab, icfg, _ = load_checkpoint(RUN / "checkpoints" / "best.pt", dev)
labels = [json.loads(x) for x in (HERE / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
orig = {}
for split in {x["split"] for x in labels}:
    rows = [json.loads(x) for x in (RUN / "eval_untrimmed" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    for x in labels:
        if x["split"] == split:
            p = rows[x["index"]]["pred"].split()
            orig[x["id"]] = trim_runaway(p) if len(p) >= 511 else p  # same trim as the new decodes
ok = {x["id"] for x in labels if all((HERE / "png" / f / f"{x['id']}.png").exists() for f in FONTS)}
preds = {}
for f in FONTS:
    items = [x for x in labels if x["id"] in ok]
    images = [Image.open(HERE / "png" / f / f"{x['id']}.png").convert("L") for x in items]
    for x, s in zip(items, predict_images(model, images, icfg, vocab, dev)):
        preds[(x["id"], f)] = s.split()
    print("decoded", f, len(items), flush=True)
res = defaultdict(lambda: defaultdict(list))
for x in labels:
    if x["id"] not in ok:
        continue
    g = x["latex"].split()
    res[x["split"]]["original"].append((normalized_levenshtein(orig[x["id"]], g), orig[x["id"]] == g))
    for f in FONTS:
        p = preds[(x["id"], f)]
        res[x["split"]][f].append((normalized_levenshtein(p, g), p == g))
out = {}
for split, d in sorted(res.items()):
    n = len(d["original"])
    line = []
    for k in ("original",) + FONTS:
        a = np.array(d[k], dtype=float)
        line.append(f"{k} {a[:, 0].mean():.3f}/{a[:, 1].mean():.3f}")
        out.setdefault(split, {})[k] = {"edit": a[:, 0].mean(), "em": a[:, 1].mean(), "n": n}
    print(f"{split:14} n={n:3} (token edit / EM): " + " | ".join(line))
(HERE / "fontswap_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
json.dump({f"{k[0]}|{k[1]}": " ".join(v) for k, v in preds.items()}, open(HERE / "fontswap_preds.json", "w", encoding="utf-8"))
