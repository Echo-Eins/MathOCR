"""Handwriting fine-tune data (2026-10-27): a printed track's data + handwritten pairs up to the owner's budget.

    python scripts/hw/build_hw.py --base D:/MathOCR/data_fonts_b_long --convention ours --out D:/MathOCR/data_hw_v4
    python scripts/hw/build_hw.py --base D:/MathOCR/data_unimer_fb_long --convention unimer --out D:/MathOCR/data_hw_unimer

* ``train`` -- the base train byte for byte + handwritten pairs until ``--total`` (300 000, the owner's cap): all of
  CROHME train (8 834, Neeze/CROHME-full), ``--hme-train`` of HME100K train (FineVision ``hme100k``, the 74 492 photos
  UniMERNet trained on), the rest from MathWriting 2024 ``train`` (human inks, drawn by ``render_ink.render``).
  Handwritten labels are converted to the track's convention (``labels.convert``) and must fit the base vocabulary
  (the fine-tune keeps the model's embeddings).  A handwritten formula may repeat (another writer, another image);
  none shares a formula key with an evaluation split;
* ``val`` -- the base val + ``--hme-val`` HME100K pairs held out of training + ``--mw-val`` MathWriting ``valid``;
* ``test_mathwriting`` -- MathWriting ``test``, the same ids and images in both tracks (labels convertible in both);
* every other split and ``vocab.json`` are copied from the base.

Selection is by sha256 of the uid (salted), so the output is a pure function of the inputs.
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import asdict
import hashlib
import json
import shutil
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import convert, usable  # noqa: E402
from render_ink import render  # noqa: E402
from mathocr.config import load_config  # noqa: E402
from mathocr.data.build import SCHEMA, Rows, _analyze_bytes, _hash_int, _leakage, formula_key  # noqa: E402
from mathocr.data.sources import sha256_file  # noqa: E402
from mathocr.image import ImageConfig, output_shape  # noqa: E402
from mathocr.tokenizer import Vocab  # noqa: E402

HME = "D:/MathOCR/raw_hme100k/hme100k/train-*.parquet"
CROHME = "D:/MathOCR/raw_crohme/data/train-*.parquet"
MW = Path("D:/MathOCR/mathwriting")
# Both printed tracks: a new handwriting val/test formula shares no key with either train (one id list for both).
PRINTED_TRAINS = (Path("D:/MathOCR/data_fonts_b_long/train.parquet"), Path("D:/MathOCR/data_unimer_fb_long/train.parquet"))
EVAL_COPY_SKIP = {"train", "val"}


def _seed(uid: str) -> int:
    return _hash_int(uid, "hw-render")


def _work(item: tuple) -> tuple[bytes, int, int, float]:
    kind, payload = item
    if kind == "bytes":
        return _analyze_bytes(payload)
    uid, xy, ends = payload
    return _analyze_bytes(render(xy, ends, _seed(uid)))


def _read_hme() -> list[tuple[str, str, bytes]]:
    out = []
    for path in sorted(glob.glob(HME)):
        shard = Path(path).stem.split("-")[1]
        t = pq.read_table(path, columns=["images", "texts"])
        for i, (ims, txt) in enumerate(zip(t["images"].to_pylist(), t["texts"].to_pylist())):
            out.append((f"hme100k/{shard}/{i}", txt[0]["assistant"], ims[0]["bytes"]))
    return out


def _read_crohme() -> list[tuple[str, str, bytes]]:
    out = []
    for path in sorted(glob.glob(CROHME)):
        shard = Path(path).stem.split("-")[1]
        t = pq.read_table(path)
        for i, (im, lab) in enumerate(zip(t["image"].to_pylist(), t["label"].to_pylist())):
            out.append((f"crohme/train/{shard}/{i}", lab, im["bytes"]))
    return out


def _read_mw(split: str, rows: list[int] | None = None) -> list[dict]:
    t = pq.read_table(MW / f"{split}.parquet", columns=None if rows is not None else ["id", "norm"])
    return (t.take(pa.array(rows, pa.int64())) if rows is not None else t).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--convention", choices=("ours", "unimer"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total", type=int, default=300_000)
    ap.add_argument("--hme-train", type=int, default=45_000)
    ap.add_argument("--hme-val", type=int, default=1_000)
    ap.add_argument("--mw-val", type=int, default=1_000)
    ap.add_argument("--max-tokens", type=int, default=766)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-config", type=Path, default=None,
                    help="training config whose image section the fit filter uses (default: the base manifest's)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    icfg = ImageConfig(**(load_config(args.train_config)["image"] if args.train_config else base_manifest["image_config"]))
    vocab = Vocab.load(args.base / "vocab.json")
    conv = args.convention
    stats: dict[str, Counter] = {}

    # 1. Evaluation keys of the base (every split but train).
    base_splits = sorted(Path(p).stem for p in glob.glob(str(args.base / "*.parquet")))
    eval_keys: set[str] = set()
    for name in base_splits:
        if name != "train":
            eval_keys.update(formula_key(l) for l in pq.read_table(args.base / f"{name}.parquet", columns=["label"])["label"].to_pylist())

    printed_keys: set[str] = set()
    for path in PRINTED_TRAINS:
        printed_keys.update(formula_key(l) for l in pq.read_table(path, columns=["label"])["label"].to_pylist())

    # 2. MathWriting test and valid: labels convertible in both conventions, formulas in neither printed train.
    def both(split: str, st: Counter) -> list[tuple[dict, str]]:
        recs = _read_mw(split)
        mine = convert([r["norm"] for r in recs], conv, spaced=False)
        other = convert([r["norm"] for r in recs], "unimer" if conv == "ours" else "ours", spaced=False)
        idx = [i for i, (m, o) in enumerate(zip(mine, other)) if m and o]
        st["raw"], st["convertible"] = len(recs), len(idx)
        idx = [i for i in idx if formula_key(mine[i]) not in printed_keys and formula_key(other[i]) not in printed_keys]
        st["kept"] = len(idx)
        return list(zip(_read_mw(split, idx), [mine[i] for i in idx]))

    mw_test = both("test", stats.setdefault("mathwriting_test", Counter()))
    mw_valid = both("valid", stats.setdefault("mathwriting_valid", Counter()))
    mw_valid = sorted(mw_valid, key=lambda p: _hash_int(p[0]["id"], "hw-val"))[: args.mw_val]
    for _, lab in mw_test + mw_valid:
        eval_keys.add(formula_key(lab))

    # 3. HME100K: held-out val, then the train pick.
    hme = _read_hme()
    hme_lab = convert([l for _, l, _ in hme], conv, spaced=True)
    order = sorted(range(len(hme)), key=lambda i: _hash_int(hme[i][0], "hw-hme"))
    hme_val = [i for i in order if usable(hme_lab[i], vocab, args.max_tokens)
               and formula_key(hme_lab[i]) not in printed_keys and formula_key(hme_lab[i]) not in eval_keys][: args.hme_val]
    for i in hme_val:
        eval_keys.add(formula_key(hme_lab[i]))
    held = set(hme_val)
    st = stats["hme100k"] = Counter(raw=len(hme), val=len(hme_val))
    hme_pool = []
    for i in order:
        if i in held:
            continue
        if not usable(hme_lab[i], vocab, args.max_tokens):
            st["dropped_label_or_vocab"] += 1
        elif formula_key(hme_lab[i]) in eval_keys:
            st["dropped_in_eval_sets"] += 1
        else:
            hme_pool.append(i)
    st["candidates"] = len(hme_pool)
    hme_train = sorted(hme_pool[: args.hme_train])
    st["train"] = len(hme_train)

    # 4. CROHME train: all usable pairs.
    cro = _read_crohme()
    cro_lab = convert([l for _, l, _ in cro], conv, spaced=True)
    st = stats["crohme"] = Counter(raw=len(cro))
    cro_train = []
    for i in range(len(cro)):
        if not usable(cro_lab[i], vocab, args.max_tokens):
            st["dropped_label_or_vocab"] += 1
        elif formula_key(cro_lab[i]) in eval_keys:
            st["dropped_in_eval_sets"] += 1
        else:
            cro_train.append(i)
    st["train"] = len(cro_train)

    # 5. MathWriting train up to the budget.
    base_train_rows = pq.read_metadata(args.base / "train.parquet").num_rows
    n_mw = args.total - base_train_rows - len(cro_train) - len(hme_train)
    mw_all = _read_mw("train")
    mw_lab = convert([r["norm"] for r in mw_all], conv, spaced=False)
    st = stats["mathwriting_train"] = Counter(raw=len(mw_all), budget=n_mw)
    mw_pool = []
    for i in sorted(range(len(mw_all)), key=lambda i: _hash_int(mw_all[i]["id"], "hw-mw")):
        if not usable(mw_lab[i], vocab, args.max_tokens):
            st["dropped_label_or_vocab"] += 1
        elif formula_key(mw_lab[i]) in eval_keys:
            st["dropped_in_eval_sets"] += 1
        else:
            mw_pool.append(i)
    st["candidates"] = len(mw_pool)
    mw_train = sorted(mw_pool[: max(n_mw, 0)])
    st["train"] = len(mw_train)
    mw_rows = dict(zip(mw_train, _read_mw("train", mw_train)))

    # 6. Images.
    def mw_item(r: dict) -> tuple:
        return ("ink", (f"mathwriting/{r['id']}", np.asarray(r["xy"], np.float32), np.asarray(r["ends"], np.int32)))

    plan = {
        "train": [(hme[i][0], "hme100k", hme_lab[i], ("bytes", hme[i][2])) for i in hme_train]
        + [(cro[i][0], "crohme", cro_lab[i], ("bytes", cro[i][2])) for i in cro_train]
        + [(f"mathwriting/{mw_all[i]['id']}", "mathwriting", mw_lab[i], mw_item(mw_rows[i])) for i in mw_train],
        "val": [(hme[i][0], "hme100k", hme_lab[i], ("bytes", hme[i][2])) for i in sorted(hme_val)]
        + [(f"mathwriting/{r['id']}", "mathwriting", lab, mw_item(r)) for r, lab in mw_valid],
        "test_mathwriting": [(f"mathwriting/{r['id']}", "mathwriting", lab, mw_item(r)) for r, lab in mw_test],
    }
    rows = {k: Rows() for k in plan}
    with Pool(args.workers) as pool:
        for name, items in plan.items():
            for (uid, src, lab, _), analyzed in zip(items, pool.imap(_work, [it[3] for it in items], chunksize=32)):
                rows[name].add(uid, src, lab, analyzed)
            print(json.dumps({"analyzed": name, "rows": len(rows[name])}), flush=True)

    hw = rows["train"].table()
    min_fit = float(base_manifest["data_config"]["min_fit_ratio"])
    keep = []
    for h, w, g in zip(hw["h"].to_pylist(), hw["w"].to_pylist(), hw["glyph"].to_pylist()):
        desired = icfg.target_glyph_px / g if g > 0 else 1.0
        fit = min((icfg.max_h - 2 * icfg.margin) / h, (icfg.max_w - 2 * icfg.margin) / w)
        keep.append(g > 0 and fit >= min_fit * min(max(desired, icfg.min_scale), icfg.max_scale))
    stats["train_filters"] = Counter(before=hw.num_rows, dropped_unreadable_when_fitted=keep.count(False))
    hw = hw.filter(pa.array(keep))
    stats["train_filters"]["after"] = hw.num_rows

    # 7. Write.
    tables = {"train": pa.concat_tables([pq.read_table(args.base / "train.parquet"), hw]),
              "val": pa.concat_tables([pq.read_table(args.base / "val.parquet"), rows["val"].table()]),
              "test_mathwriting": rows["test_mathwriting"].table()}
    for name in base_splits:
        if name not in EVAL_COPY_SKIP:
            shutil.copy2(args.base / f"{name}.parquet", args.out / f"{name}.parquet")
            tables[name] = pq.read_table(args.out / f"{name}.parquet")
    shutil.copy2(args.base / "vocab.json", args.out / "vocab.json")
    splits = {}
    for name, table in sorted(tables.items()):
        if name in plan:
            pq.write_table(table, args.out / f"{name}.parquet", row_group_size=1024)
        digest = hashlib.sha256()
        for uid, label in zip(table["uid"].to_pylist(), table["label"].to_pylist()):
            digest.update(f"{uid}\t{label}\n".encode())
        shapes = Counter(output_shape(h, w, g, icfg) for h, w, g in
                         zip(table["h"].to_pylist(), table["w"].to_pylist(), table["glyph"].to_pylist()))
        splits[name] = {"rows": table.num_rows, "by_source": dict(sorted(Counter(table["source"].to_pylist()).items())),
                        "uid_label_sha256": digest.hexdigest(), "distinct_shapes": len(shapes)}
    hw_train_keys = {formula_key(l) for l in hw["label"].to_pylist()}
    manifest = {
        **base_manifest,
        "handwriting": {
            "rule": "scripts/hw/build_hw.py (2026-10-27): base train + handwritten pairs to the owner's total",
            "base": str(args.base), "convention": conv, "total": args.total, "fit_filter_image_config": asdict(icfg),
            "sources": {"hme100k": {"repo": "HuggingFaceM4/FineVision", "revision": "3c380a731a3429c1d04693d6ec16d7e683def84c",
                                    "files": {Path(p).name: sha256_file(Path(p)) for p in sorted(glob.glob(HME))}},
                        "crohme": {"repo": "Neeze/CROHME-full", "revision": "6d03ee8f86df1d0d93f3de00df75303d53b75f7d",
                                   "files": {Path(p).name: sha256_file(Path(p)) for p in sorted(glob.glob(CROHME))}},
                        "mathwriting": {"url": "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz",
                                        "sha256": "f1977f120b0efa8abf715ac544b0596b213b099133da99b5e64f946730929269",
                                        "license": "CC BY-NC-SA 4.0"}},
            "filters": {k: dict(v) for k, v in stats.items()},
            "hw_train_keys_in_eval": {name: sum(formula_key(l) in hw_train_keys for l in t["label"].to_pylist())
                                      for name, t in sorted(tables.items()) if name != "train"},
        },
        "splits": splits,
        "leakage_check": _leakage(tables),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                            newline="\n")
    print(json.dumps({"handwriting": manifest["handwriting"]["filters"], "splits": {k: v["rows"] for k, v in splits.items()},
                      "leakage": manifest["leakage_check"]}, indent=1))


if __name__ == "__main__":
    main()
