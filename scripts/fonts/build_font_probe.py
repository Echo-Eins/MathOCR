"""Probe stand with font diversity (todo section 12, F1).

The standard stand subsets, except that in train a share of the rows (chosen by sha256(uid)) carry the same label
rendered in another math font (scripts/fonts/render.py: e.g. Times = txfonts, sans = sansmath; 300 dpi; the many-font
stand of 2026-10-25 spreads 30 % over 30 fonts, D:/MathOCR/fonts20k/shares_many.txt).  Labels,
uids and splits are unchanged; val and OOD files are copied byte for byte, so probe metrics stay comparable.
Palatino is not used here: it is the held-out font of the fontswap evaluation.  A render that failed to compile
keeps the original image.

    python scripts/fonts/build_font_probe.py --src D:/MathOCR/probe/data20k --renders D:/MathOCR/fonts20k \
        --out D:/MathOCR/probe_fonts/data20k --shares cm=0.5,times=0.25,sans=0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mathocr.data.build import SCHEMA, _analyze_bytes  # noqa: E402


def assign(uid: str, shares: list[tuple[str, float]]) -> str:
    # salted: the stand itself is the smallest sha256(uid) of the full train, so the plain hash is ~0 for all of it
    u = int(hashlib.sha256(("fonts:" + uid).encode()).hexdigest()[:12], 16) / 16**12
    acc = 0.0
    for font, share in shares:
        acc += share
        if u < acc:
            return font
    return shares[-1][0]


def _analyze(path: str):
    return _analyze_bytes(Path(path).read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shares", default="cm=0.5,times=0.25,sans=0.25")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    shares = [(k, float(v)) for k, v in (s.split("=") for s in args.shares.split(","))]
    assert abs(sum(v for _, v in shares) - 1.0) < 1e-9 and shares[0][0] == "cm"

    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.src.iterdir():
        if f.name not in ("train.parquet", "manifest.json"):
            shutil.copy2(f, args.out / f.name)
    table = pq.read_table(args.src / "train.parquet")
    rows = table.to_pylist()
    fonts = [assign(r["uid"], shares) for r in rows]
    todo = [(i, str(args.renders / "png" / f / f"{i}.png")) for i, f in enumerate(fonts)
            if f != "cm" and (args.renders / "png" / f / f"{i}.png").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for (i, _), (crop, h, w, glyph) in zip(todo, pool.map(_analyze, [p for _, p in todo], chunksize=64)):
            rows[i].update(crop=crop, h=h, w=w, glyph=glyph)
    used = Counter(f if (f == "cm" or (args.renders / "png" / f / f"{i}.png").exists()) else "cm (render failed)"
                   for i, f in enumerate(fonts))
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), args.out / "train.parquet")
    manifest = json.loads((args.src / "manifest.json").read_text(encoding="utf-8"))
    manifest["fonts"] = {"rule": "sha256('fonts:' + uid) bucket per train row; other splits unchanged",
                         "shares": dict(shares), "train_rows": dict(used),
                         "renderer": "scripts/fonts/render.py (FONTS there: engine and package per font; 300 dpi, Ghostscript)",
                         "held_out": "palatino (pxfonts) and otf_pagella (TeX Gyre Pagella Math)"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest["fonts"]["train_rows"]))


if __name__ == "__main__":
    main()
