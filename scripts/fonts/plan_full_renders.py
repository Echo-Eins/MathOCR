"""Font assignment for the full training split (final_v3): every train row gets a font by sha256('fonts:' + uid)
with the shares of the stand's variant b (CM 70 % -- the original image --, Times 5 %, sans 5 %, the other 28 fonts
of render.py 20 %; the Palatino family and Asana are not used).  Writes <work>/labels.jsonl ({"id": row, "latex",
"font"} for the rows to render) and <work>/shares.txt for build_font_probe.py.

    python scripts/fonts/plan_full_renders.py --src data/processed --work D:/MathOCR/fonts_full
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_font_probe import assign  # noqa: E402

HELD_OUT = ("cm", "palatino", "otf_pagella", "otf_asana")


def shares_b() -> list[tuple[str, float]]:
    src = (Path(__file__).resolve().parent / "render.py").read_text(encoding="utf-8")
    fonts = [f for f in re.findall(r'^    "(\w+)": \("(?:pdflatex|lualatex)"', src, re.M)
             if f not in HELD_OUT + ("times", "sans")]
    return [("cm", 0.7), ("times", 0.05), ("sans", 0.05)] + [(f, 0.2 / len(fonts)) for f in fonts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    shares = shares_b()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "shares.txt").write_text(",".join(f"{k}={v!r}" for k, v in shares), encoding="utf-8")
    t = pq.read_table(args.src / "train.parquet", columns=["uid", "label"])
    counts = Counter()
    with open(args.work / "labels.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for i, (uid, label) in enumerate(zip(t["uid"].to_pylist(), t["label"].to_pylist())):
            font = assign(uid, shares)
            counts[font] += 1
            if font != "cm":
                fh.write(json.dumps({"id": i, "latex": label, "font": font}) + "\n")
    print(json.dumps({"rows": t.num_rows, "cm": counts["cm"], "to_render": t.num_rows - counts["cm"],
                      "fonts": len(counts) - 1}))


if __name__ == "__main__":
    main()
