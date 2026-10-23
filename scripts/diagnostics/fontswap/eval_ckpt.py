"""(Paths below are those of the Windows laptop; edit them for another machine.)
Font-swap suite for any checkpoint, batched by image shape (scripts/probe.py decode): the labels of WORK
(make_labels.py: CPE 400, SPE 300, SCE 200, test_im2latex 200) as their original test images and as our TeX
renders in CM / Times / Palatino / sans (render_fonts.py).  Token-level edit and EM per split and font, on the
labels that compiled in every font.

    python scripts/diagnostics/fontswap/eval_ckpt.py NAME=path/to/best.pt [NAME=...]
Output: docs/probe_results/fontswap_<NAME>.json."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

os.environ.setdefault("PROBE_ROOT", r"D:\MathOCR\probe")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import probe  # noqa: E402
from mathocr import determinism  # noqa: E402
from mathocr.data.build import SCHEMA, _analyze_bytes  # noqa: E402
from mathocr.data.dataset import CropStore  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.metrics import normalized_levenshtein  # noqa: E402

WORK = Path(r"D:\MathOCR\fontswap")
TEST = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
FONTS = ("cm", "times", "palatino", "sans")


def stores() -> dict[str, Path]:
    """One parquet per variant (original + each font) under WORK/stores, built once."""
    out = WORK / "stores"
    out.mkdir(exist_ok=True)
    labels = [json.loads(x) for x in (WORK / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = [x for x in labels if all((WORK / "png" / f / f"{x['id']}.png").exists() for f in FONTS)]
    paths = {v: out / f"{v}.parquet" for v in ("original", *FONTS)}
    if all(p.exists() for p in paths.values()):
        return paths
    orig = {}
    for split in {x["split"] for x in ok}:
        t = pq.read_table(TEST / f"{split}.parquet").to_pylist()
        for x in ok:
            if x["split"] == split:
                orig[x["id"]] = t[x["index"]]
    pq.write_table(pa.Table.from_pylist([{**orig[x["id"]], "uid": x["id"]} for x in ok], schema=SCHEMA), paths["original"])
    for f in FONTS:
        rows = []
        for x in ok:
            crop, h, w, glyph = _analyze_bytes((WORK / "png" / f / f"{x['id']}.png").read_bytes())
            rows.append({"uid": x["id"], "source": x["split"], "label": x["latex"], "crop": crop, "h": h, "w": w, "glyph": glyph})
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), paths[f])
    return paths


def main() -> None:
    dev = torch.device("cuda")
    determinism.configure(dev)
    paths = stores()
    for arg in sys.argv[1:]:
        name, ckpt = arg.split("=", 1)
        model, vocab, icfg, _ = load_checkpoint(Path(ckpt), dev)
        res = defaultdict(dict)
        for variant, path in paths.items():
            store = CropStore(path, vocab)
            preds = probe.decode(model, store, icfg, vocab, dev)
            by_split = defaultdict(list)
            for uid, gt, p in zip(store.uids, store.labels, preds):
                p = [t for t in p if t != "<noeos>"]
                by_split[uid.split("__")[0]].append((normalized_levenshtein(p, gt.split()), p == gt.split()))
            for split, v in by_split.items():
                a = np.array(v, dtype=float)
                res[split][variant] = {"edit": float(a[:, 0].mean()), "em": float(a[:, 1].mean()), "n": len(a)}
        for split in sorted(res):
            print(f"{name} {split:14} " + " | ".join(f"{v} {r['edit']:.3f}/{r['em']:.3f}" for v, r in res[split].items()), flush=True)
        (ROOT / "docs" / "probe_results" / f"fontswap_{name}.json").write_text(json.dumps(res, indent=1),
                                                                             encoding="utf-8", newline="\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
