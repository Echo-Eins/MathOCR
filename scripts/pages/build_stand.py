"""Build the page stand (tasks/page_ocr_plan.md 5a): synthetic printed English pages with exact layout.

    python scripts/pages/build_stand.py --out D:/MathOCR/pages/stand_v0 [--train 20000 --val 1000 --test 2000
        --heldout 1000 --workers 8]

Splits: ``train`` / ``val`` / ``test`` draw articles from their own Wikipedia split (``text.split_of``) and fonts
from ``stand_fonts`` SERIF + SANS + MONO; ``test_heldout_font`` uses test-split articles not used by ``test`` and
only the held-out Palatino designs.  Page i of a split takes article i of that split as its main text and, when the
page has room left, continuation articles from a separate reserve (no article text on two pages of a split).  A page
the layout rejects (too few body lines) is re-drawn with the next attempt's seed; seeds are sha256 of
(stand, split, page, attempt).  Each split is one parquet: uid, png (8-bit gray, drawn at final size), text (the
label), layout (JSON, ``synth.render_page``), w, h, n_lines, body_font; ``manifest.json`` records the sources, the
generator settings, the code commit, per-split hashes and the distributions of fonts, roles, columns and sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr.data.sources import sha256_file  # noqa: E402
from mathocr.pages import stand_fonts  # noqa: E402
from mathocr.pages.fonts import CHARSET  # noqa: E402
from mathocr.pages.synth import ROLES, PageSpec, label, render_page  # noqa: E402
from mathocr.pages.text import articles  # noqa: E402

WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI = sorted(Path("D:/MathOCR/pages/raw/wikipedia_en/20231101.en").glob("train-*.parquet"))
CATALOG = Path("D:/MathOCR/pages/font_catalog.json")
STAND = "stand_v0"
CONTINUATIONS = 3
MAX_ATTEMPTS = 8
SCHEMA = pa.schema([("uid", pa.string()), ("png", pa.binary()), ("text", pa.string()), ("layout", pa.string()),
                    ("w", pa.int32()), ("h", pa.int32()), ("n_lines", pa.int32()), ("body_font", pa.string())])

_FONTS: dict | None = None


def _init() -> None:
    global _FONTS
    _FONTS = stand_fonts.check(json.loads(CATALOG.read_text(encoding="utf-8"))["faces"])


def seed_of(split: str, page: int, attempt: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{STAND}\x00{split}\x00{page}\x00{attempt}".encode()).digest()[:8], "big")


def _page(task: tuple) -> tuple:
    split, i, arts, families = task
    for attempt in range(MAX_ATTEMPTS):
        res = render_page(seed_of(split, i, attempt), arts, _FONTS, families, PageSpec())
        if res is not None:
            img, layout = res
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=6)
            return (f"{STAND}/{split}/{i:06d}", buf.getvalue(), label(layout), json.dumps(layout, ensure_ascii=False),
                    layout["w"], layout["h"], len(layout["lines"]), layout["fonts"]["body"]), attempt
    raise RuntimeError(f"{split} page {i}: no acceptable layout in {MAX_ATTEMPTS} attempts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--heldout", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stand_fonts.check(json.loads(CATALOG.read_text(encoding="utf-8"))["faces"])  # fail before any work
    seen = {"serif": list(stand_fonts.SERIF), "sans": list(stand_fonts.SANS), "mono": list(stand_fonts.MONO)}
    heldout_families = {"serif": list(stand_fonts.HELD_OUT)}
    per_art_split = {"train": args.train, "val": args.val, "test": args.test + args.heldout}
    plan = {}
    for asplit, n in per_art_split.items():
        need = n * (1 + CONTINUATIONS)
        arts = list(islice(articles(WIKI, asplit), need))
        if len(arts) < need:
            raise RuntimeError(f"{asplit}: {len(arts)} articles, {need} needed")
        main_arts, reserve = arts[:n], arts[n:]
        tasks = [(i, [main_arts[i]] + reserve[CONTINUATIONS * i:CONTINUATIONS * (i + 1)]) for i in range(n)]
        if asplit == "test":
            plan["test"] = ([(i, a) for i, a in tasks[:args.test]], seen)
            plan["test_heldout_font"] = ([(i - args.test, a) for i, a in tasks[args.test:]], heldout_families)
        else:
            plan[asplit] = (tasks, seen)
    stats = {}
    with Pool(args.workers, initializer=_init) as pool:
        for split, (tasks, families) in plan.items():
            writer = pq.ParquetWriter(args.out / f"{split}.parquet", SCHEMA)
            digest, retries = hashlib.sha256(), Counter()
            fonts, roles, cols, caps, n_lines = Counter(), Counter(), Counter(), [], []
            batch: list[tuple] = []
            for row, attempt in pool.imap(_page, [(split, i, a, families) for i, a in tasks], chunksize=8):
                retries[attempt] += 1
                digest.update(f"{row[0]}\t{row[2]}\n".encode())
                layout = json.loads(row[3])
                fonts[row[7]] += 1
                cols[len(layout["columns"])] += 1
                caps.append(layout["style"]["cap"])
                n_lines.append(row[6])
                roles.update(l["role"] for l in layout["lines"])
                batch.append(row)
                if len(batch) == 256:
                    writer.write_table(pa.Table.from_pylist([dict(zip(SCHEMA.names, r)) for r in batch], schema=SCHEMA))
                    batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist([dict(zip(SCHEMA.names, r)) for r in batch], schema=SCHEMA))
            writer.close()
            stats[split] = {
                "pages": len(tasks), "uid_text_sha256": digest.hexdigest(), "attempts": dict(sorted(retries.items())),
                "body_fonts": dict(sorted(fonts.items())), "roles": {r: roles[r] for r in ROLES},
                "columns": dict(sorted(cols.items())), "lines_mean": sum(n_lines) / len(n_lines),
                "cap_min_mean_max": [min(caps), sum(caps) / len(caps), max(caps)],
            }
            print(json.dumps({"split": split, **{k: v for k, v in stats[split].items() if k != "body_fonts"},
                              "distinct_body_fonts": len(fonts)}), flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src", "scripts/pages"],
                                capture_output=True, text=True).stdout.strip())
    manifest = {
        "stand": STAND, "code": {"commit": commit, "dirty": dirty},
        "sources": {"wikipedia": {"repo": "wikimedia/wikipedia", "revision": WIKI_REV, "license": "CC BY-SA 4.0",
                                  "files": {p.name: sha256_file(p) for p in WIKI}},
                    "font_catalog_sha256": sha256_file(CATALOG)},
        "charset": CHARSET, "spec": asdict(PageSpec()), "continuations": CONTINUATIONS,
        "fonts": {"serif": stand_fonts.SERIF, "sans": stand_fonts.SANS, "mono": stand_fonts.MONO,
                  "held_out": stand_fonts.HELD_OUT},
        "splits": stats,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                            newline="\n")


if __name__ == "__main__":
    main()
