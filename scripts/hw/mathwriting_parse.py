"""MathWriting 2024 (CC BY-NC-SA 4.0) -> one parquet per split with the labels and the raw strokes.

    python scripts/hw/mathwriting_parse.py --tgz D:/MathOCR/raw_mathwriting/mathwriting-2024.tgz --out D:/MathOCR/mathwriting

Streams the archive once (no extraction of the 650k small files).  Columns: ``id``, ``label`` (as written),
``norm`` (``normalizedLabel``, the dataset's training target), ``creation`` (``inkCreationMethod``), ``xy``
(float32 x, y of all points, flattened) and ``ends`` (int32 end offset of every stroke in points).  ``synthetic``
and ``symbols`` are skipped (the owner's pair budget, 2026-10-27).
"""

from __future__ import annotations

import argparse
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NS = "{http://www.w3.org/2003/InkML}"
SPLITS = ("train", "valid", "test")
SCHEMA = pa.schema([("id", pa.string()), ("label", pa.string()), ("norm", pa.string()), ("creation", pa.string()),
                    ("xy", pa.list_(pa.float32())), ("ends", pa.list_(pa.int32()))])


def parse_ink(data: bytes) -> dict:
    root = ET.fromstring(data)
    ann = {a.get("type"): (a.text or "") for a in root.iter(f"{NS}annotation")}
    xy, ends = [], []
    for trace in root.iter(f"{NS}trace"):
        for point in (trace.text or "").strip().split(","):
            v = point.split()
            if len(v) >= 2:
                xy.extend((float(v[0]), float(v[1])))
        ends.append(len(xy) // 2)
    return {"id": ann.get("sampleId", ""), "label": ann.get("label", ""), "norm": ann.get("normalizedLabel", ""),
            "creation": ann.get("inkCreationMethod", ""), "xy": np.asarray(xy, np.float32), "ends": np.asarray(ends, np.int32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {s: {f.name: [] for f in SCHEMA} for s in SPLITS}
    member = re.compile(r"^mathwriting-2024/(train|valid|test)/[0-9a-f]{16}\.inkml$")
    with tarfile.open(args.tgz, "r|gz") as tar:
        for info in tar:
            m = member.match(info.name)
            if not (m and info.isfile()):
                continue
            rec = parse_ink(tar.extractfile(info).read())
            for k in SCHEMA.names:
                rows[m.group(1)][k].append(rec[k])
    for s in SPLITS:
        order = np.argsort(rows[s]["id"])  # archive order is not guaranteed; sort by the ink id
        cols = {k: [v[i] for i in order] for k, v in rows[s].items()}
        pq.write_table(pa.table(cols, schema=SCHEMA), args.out / f"{s}.parquet", row_group_size=4096)
        print(s, len(order), flush=True)


if __name__ == "__main__":
    main()
