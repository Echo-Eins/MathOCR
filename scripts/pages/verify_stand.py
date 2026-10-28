"""Verify a built stand page by page: image size / mode / multiple of 32, image-layout-label agreement, no ink
outside word boxes, cell targets build, held-out fonts only in ``test_heldout_font``, every article from its own
Wikipedia split, on one page and in one split.

    python scripts/pages/verify_stand.py D:/MathOCR/pages/stand_v0
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mathocr.pages import stand_fonts  # noqa: E402
from mathocr.pages.synth import label  # noqa: E402
from mathocr.pages.targets import cell_targets  # noqa: E402
from mathocr.pages.text import split_of  # noqa: E402

root = Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
owner: dict[str, str] = {}
problems = Counter()
for split in man["splits"]:
    t = pq.read_table(root / f"{split}.parquet")
    used = Counter()
    for r in t.to_pylist():
        lay = json.loads(r["layout"])
        img = Image.open(io.BytesIO(r["png"]))
        a = np.asarray(img)
        if img.mode != "L" or img.size != (r["w"], r["h"]) or (lay["w"], lay["h"]) != img.size:
            problems["size_or_mode"] += 1
        if r["text"] != label(lay) or r["n_lines"] != len(lay["lines"]):
            problems["label"] += 1
        if r["w"] % 32 or r["h"] % 32:
            problems["not_multiple_of_32"] += 1
        allowed = np.zeros(a.shape, bool)
        for l in lay["lines"]:
            for wd in l["words"]:
                x0, y0, x1, y1 = wd["box"]
                allowed[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = True
        for x0, y0, x1, y1 in lay["rules"]:
            allowed[y0:y1, x0:x1] = True
        if ((a < 255) & ~allowed).any():
            problems["stray_ink"] += 1
        cell_targets(lay, (8, 4))
        body = lay["fonts"]["body"]
        if split == "test_heldout_font" and body not in stand_fonts.HELD_OUT:
            problems["heldout_font_leak"] += 1
        if split != "test_heldout_font" and body in stand_fonts.HELD_OUT:
            problems["heldout_font_in_seen_split"] += 1
        art_split = "test" if split.startswith("test") else split
        for aid in lay["articles"]:
            used[aid] += 1
            if split_of(aid) != art_split:
                problems["article_from_wrong_split"] += 1
            if owner.setdefault(aid, split) != split:
                problems["article_in_two_splits"] += 1
    problems["article_on_two_pages"] += sum(1 for c in used.values() if c > 1)
    print(split, len(t), "pages ok so far")
print("problems:", dict(problems) if any(problems.values()) else "none")
