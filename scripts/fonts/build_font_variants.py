"""Font variants of a train split for ``augment.font_variants`` (mathocr.data.dataset.FontVariants).

Copies a processed data dir and adds, per font, ``train_<font>.parquet``: row-aligned with train.parquet (same uid
and label in every row), the crop rendered by scripts/fonts/render.py in that font; a label that failed to compile
keeps its original image there.  Training then picks one variant per sample and epoch (augment.p_font).

    python scripts/fonts/build_font_variants.py --src D:/MathOCR/probe/data20k --renders D:/MathOCR/fonts20k \
        --out D:/MathOCR/probe_fonts_dyn/data20k --fonts times sans
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mathocr.data.build import SCHEMA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from build_font_probe import _analyze  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fonts", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.src.iterdir():
        if f.is_file() and f.name != "manifest.json":
            shutil.copy2(f, args.out / f.name)
    base = pq.read_table(args.src / "train.parquet").to_pylist()
    counts = {}
    for font in args.fonts:
        rows = [dict(r) for r in base]
        todo = [(i, str(args.renders / "png" / font / f"{i}.png")) for i in range(len(rows))
                if (args.renders / "png" / font / f"{i}.png").exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for (i, _), (crop, h, w, glyph) in zip(todo, pool.map(_analyze, [p for _, p in todo], chunksize=64)):
                rows[i].update(crop=crop, h=h, w=w, glyph=glyph)
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), args.out / f"train_{font}.parquet")
        counts[font] = {"rendered": len(todo), "kept_original": len(rows) - len(todo)}
        print(font, counts[font], flush=True)
    manifest = json.loads((args.src / "manifest.json").read_text(encoding="utf-8"))
    manifest["font_variants"] = {"files": [f"train_{f}.parquet" for f in args.fonts], "rows": counts,
                                 "renderer": "scripts/fonts/render.py (pdflatex, 300 dpi, Ghostscript)"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
