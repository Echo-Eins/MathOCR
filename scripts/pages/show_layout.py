"""Draw a page's layout over its image: line metric boxes (blue), word boxes (red), baselines (green), the cap band
the cell targets use (yellow), rules (magenta) and every line's reading-order index -- a visual check of the stand.

    python scripts/pages/show_layout.py --out <dir> --n 6 [--split val] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr.pages import stand_fonts  # noqa: E402
from mathocr.pages.synth import PageSpec, render_page  # noqa: E402
from mathocr.pages.text import articles  # noqa: E402

WIKI = sorted(Path("D:/MathOCR/pages/raw/wikipedia_en/20231101.en").glob("train-*.parquet"))
CATALOG = Path("D:/MathOCR/pages/font_catalog.json")


def overlay(img: Image.Image, layout: dict) -> Image.Image:
    rgb = img.convert("RGB")
    d = ImageDraw.Draw(rgb, "RGBA")
    for i, l in enumerate(layout["lines"]):
        b = l["baseline"]
        x0, x1 = l["char_x"][0], l["box"][2]
        d.rectangle([x0, b - l["cap"], x1, b], fill=(255, 220, 0, 70))
        d.rectangle(l["box"], outline=(0, 90, 255, 200))
        for wd in l["words"]:
            d.rectangle(wd["box"], outline=(230, 0, 0, 160))
        d.line([x0, b, x1, b], fill=(0, 170, 0, 220))
        d.text((max(0, x0 - 22), b - l["cap"]), str(i), fill=(0, 90, 255, 255))
    for r in layout["rules"]:
        d.rectangle([r[0], r[1] - 1, r[2], r[3]], outline=(220, 0, 220, 255))
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fonts = stand_fonts.check(json.loads(CATALOG.read_text(encoding="utf-8"))["faces"])
    families = {"serif": list(stand_fonts.SERIF), "sans": list(stand_fonts.SANS), "mono": list(stand_fonts.MONO)}
    arts = list(islice(articles(WIKI, args.split), 4 * args.n + 8))
    done, k = 0, 0
    while done < args.n:
        res = render_page(args.seed + k, arts[k:k + 4], fonts, families, PageSpec())
        k += 1
        if res is None:
            continue
        img, layout = res
        img.save(args.out / f"page_{done}.png")
        overlay(img, layout).save(args.out / f"page_{done}_layout.png")
        (args.out / f"page_{done}.json").write_text(json.dumps(layout, ensure_ascii=False), encoding="utf-8")
        print(done, layout["w"], layout["h"], layout["fonts"], len(layout["lines"]), "lines",
              {r: sum(l["role"] == r for l in layout["lines"]) for r in ("header", "title", "heading", "body", "footnote", "footer")},
              "cols", len(layout["columns"]), layout["style"])
        done += 1


if __name__ == "__main__":
    main()
