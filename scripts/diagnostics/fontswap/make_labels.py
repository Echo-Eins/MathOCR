"""(Paths below are those of the Windows laptop; edit them for another machine.)
Font-swap test, step 1 of 3: pick random labels of CPE / SPE / SCE and a test_im2latex control into
WORK/labels.jsonl.  Step 2: render_fonts.py in the CDM image (docker run --rm --entrypoint python3
-v <WORK>:/work mathocr-cdm /work/render_fonts.py, after copying it to WORK); step 3: decode_fonts.py."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Coding\Python\MathOCR\src")
from mathocr.data.dataset import CropStore  # noqa: E402

DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
OUT = Path(r"D:\MathOCR\fontswap")
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for split, n in (("ood_cpe", 400), ("ood_spe", 300), ("ood_sce", 200), ("test_im2latex", 200)):
    store = CropStore(DATA / f"{split}.parquet")
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(store), size=n, replace=False).tolist()):
        rows.append({"id": f"{split}__{i}", "split": split, "index": i, "latex": store.labels[i]})
(OUT / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
print(len(rows), "labels")
