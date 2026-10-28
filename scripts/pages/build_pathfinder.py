"""Build the Pathfinder sets of the page stand (``mathocr.pages.pathfinder``): ``pf512`` (classic scale, curves of
~370 px) and ``pf1024`` (page scale, curves of ~730 px), train / val / test each.

    python scripts/pages/build_pathfinder.py --out D:/MathOCR/pages/pathfinder_v0 [--train 20000 --val 2000 --test 2000]

Sample i of (level, split) has seed sha256(level, split, i) and label = one bit of that hash (not alternating, so
file order says nothing); one parquet per level and split: uid, png, label, meta (JSON); ``manifest.json`` records
the specs, counts, label balance, redraw statistics and the code commit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr.pages.pathfinder import PathfinderSpec, make  # noqa: E402

LEVELS = {
    "pf512": PathfinderSpec(),
    "pf1024": PathfinderSpec(size=1024, n_segments=81, n_distractors=10),
}
SCHEMA = pa.schema([("uid", pa.string()), ("png", pa.binary()), ("label", pa.int8()), ("meta", pa.string())])


def _hash(level: str, split: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"pathfinder\x00{level}\x00{split}\x00{i}".encode()).digest()[:8], "big")


def _sample(task: tuple) -> dict:
    level, split, i = task
    h = _hash(level, split, i)
    img, meta = make(h, h >> 63, LEVELS[level])
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return {"uid": f"{level}/{split}/{i:06d}", "png": buf.getvalue(), "label": meta["label"], "meta": json.dumps(meta)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    with Pool(args.workers) as pool:
        for level in LEVELS:
            for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
                rows = list(pool.imap(_sample, [(level, split, i) for i in range(n)], chunksize=32))
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), args.out / f"{level}_{split}.parquet",
                               row_group_size=1024)
                redraws = [json.loads(r["meta"])["redraws"] for r in rows]
                digest = hashlib.sha256("".join(f"{r['uid']}\t{r['label']}\n" for r in rows).encode()).hexdigest()
                stats[f"{level}_{split}"] = {"n": n, "positive": sum(r["label"] for r in rows),
                                             "redraws_mean": sum(redraws) / n, "redraws_max": max(redraws),
                                             "uid_label_sha256": digest}
                print(json.dumps({f"{level}_{split}": stats[f"{level}_{split}"]}), flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"levels": {k: asdict(v) for k, v in LEVELS.items()}, "code_commit": commit, "splits": stats}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
