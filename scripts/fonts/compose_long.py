"""Long multi-row training formulas for the stand (stage 3, the multi-row probes): rows of the stand's own train
labels -- the rows of its aligned blocks (``\\begin{array} { r l }``) and single formulas split at the relation, in the
normalizer's convention ``{ lhs } & { { } = rhs }`` -- joined into blocks of 6..12 rows and typeset by TeX
(scripts/fonts/render.py, the stand's font shares by sha256 of the new uid; "cm" here is our own TeX render).

    python scripts/fonts/compose_long.py plan  --src D:/MathOCR/probe_fonts_many_b/data20k --work D:/MathOCR/long20k --n 4000
    (docker ... render.py with /work = D:/MathOCR/long20k)
    python scripts/fonts/compose_long.py build --src D:/MathOCR/probe_fonts_many_b/data20k --work D:/MathOCR/long20k \
        --out D:/MathOCR/probe_long_b/data20k

``build`` copies the stand and appends the typeset blocks to its train split (uids ``long/<k>``, source ``long``);
val and OOD stay byte for byte.  Labels are built from train rows only: no test leakage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_font_probe import assign  # noqa: E402
from mathocr.data.build import SCHEMA, _analyze_bytes  # noqa: E402
from mathocr.data.dataset import relation_cut, rl_rows  # noqa: E402
from plan_full_renders import shares_b  # noqa: E402


def row_pool(labels: list[str]) -> list[list[str]]:
    rows = []
    for lab in labels:
        t = lab.split()
        spans = rl_rows(t)
        if spans:
            rows += [t[s:e] for s, e in spans]
        elif len(t) <= 60 and not any(x.startswith("\\begin{") for x in t):
            cut = relation_cut(t)
            rows.append(["&", "{", "{", "}", *t, "}"] if cut is None else
                        ["{", *t[:cut], "}", "&", "{", "{", "}", *t[cut:], "}"])
    return rows


def plan(args) -> None:
    labels = pq.read_table(args.src / "train.parquet", columns=["label"])["label"].to_pylist()
    pool = row_pool(labels)
    rng = np.random.default_rng(args.seed)
    shares = shares_b()
    args.work.mkdir(parents=True, exist_ok=True)
    out, k = [], 0
    while len(out) < args.n:
        n_rows = int(rng.integers(6, 13))
        picks = [pool[i] for i in rng.integers(0, len(pool), size=n_rows)]
        toks = ["\\begin{array}", "{", "r", "l", "}"]
        for j, r in enumerate(picks):
            toks += (["\\\\"] if j else []) + r
        toks.append("\\end{array}")
        if len(toks) > args.max_tokens:
            continue
        uid = f"long/{k:05d}"
        out.append({"id": k, "uid": uid, "latex": " ".join(toks), "font": assign(uid, shares), "rows": n_rows})
        k += 1
    with open(args.work / "labels.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    lens = [len(r["latex"].split()) for r in out]
    print(json.dumps({"pool_rows": len(pool), "labels": len(out), "tokens_median": int(np.median(lens)),
                      "tokens_max": max(lens), "rows_mean": float(np.mean([r["rows"] for r in out]))}))


def build(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.src.iterdir():
        if f.name not in ("train.parquet", "manifest.json"):
            shutil.copy2(f, args.out / f.name)
    table = pq.read_table(args.src / "train.parquet")
    rows = table.to_pylist()
    added = 0
    for x in (json.loads(l) for l in (args.work / "labels.jsonl").read_text(encoding="utf-8").splitlines()):
        png = args.work / "png" / x["font"] / f"{x['id']}.png"
        if not png.exists():
            continue
        crop, h, w, glyph = _analyze_bytes(png.read_bytes())
        rows.append({"uid": x["uid"], "source": "long", "label": x["latex"], "crop": crop, "h": h, "w": w, "glyph": glyph})
        added += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), args.out / "train.parquet")
    manifest = json.loads((args.src / "manifest.json").read_text(encoding="utf-8"))
    manifest["long_blocks"] = {"rule": "scripts/fonts/compose_long.py: 6..12 train rows per block, TeX-typeset",
                               "added_train_rows": added}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"train_rows": len(rows), "added": added}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "build"])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (plan if args.mode == "plan" else build)(args)


if __name__ == "__main__":
    main()
