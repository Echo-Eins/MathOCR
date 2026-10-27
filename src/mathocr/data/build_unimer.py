"""The UniMER track (HANDOFF section 1, 2026-10-27): splits from UniMER-1M in the format of ``mathocr.data.build``.

    python -m mathocr.data.build_unimer --config configs/unimer_v4.yaml

* ``train`` -- ``data.unimer_train_size`` pairs of UniMER-1M (the published labels, the normalizer of UniMER), the
  smallest sha256 of the formula key after the held-out buckets are taken out; at most ``data.train_max_tokens``
  tokens; no key shared with any evaluation split;
* ``val`` / ``test_unimer1m`` -- held-out UniMER-1M pairs (hash buckets ``unimer_val_permille`` /
  ``unimer_test_permille`` of the key) of at most ``data.max_tokens`` tokens: checkpoint selection / the in-source
  test (pairs never seen in training);
* ``ood_{cpe,sce,spe,hwe}`` -- UniMER-Test as published (the comparison with UniMERNet);
* ``test_im2latex``, ``test_oleehyo`` -- copied byte for byte from ``data.ours_dir`` (our own tests: a different
  source and labeling convention for this track).

UniMER-1M is read from the pinned zip (``sources.UNIMER_1M``; ``data.unimer_1m_zip`` may point at a local copy, which
is checked against the pinned sha256).  Line i of ``train.txt`` is ``images/{i:07d}.png``; lines without an image
(HME100K, excluded upstream) are skipped.  The same train-only filters as the main build (unreadable when fitted,
rare tokens) and the same leakage check; ``manifest.json`` records the version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mathocr.config import load_config
from mathocr.data import sources
from mathocr.data.build import SCHEMA, Rows, _analyze_bytes, _analyze_path, _hash_int, _leakage, _nice, _read_unimer, formula_key
from mathocr.image import ImageConfig, output_shape
from mathocr.tokenizer import Vocab, tokenize

_ZIP: zipfile.ZipFile | None = None


def _open_zip(path: str) -> None:
    global _ZIP
    _nice()
    _ZIP = zipfile.ZipFile(path)


def _analyze_member(name: str) -> tuple[bytes, int, int, float]:
    return _analyze_bytes(_ZIP.read(name))


def build_unimer(cfg: dict) -> dict:
    dcfg, icfg = cfg["data"], ImageConfig(**cfg["image"])
    out_dir = Path(dcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(os.path.expanduser(dcfg["cache_dir"]))
    train_max = int(dcfg["train_max_tokens"])
    max_tokens = int(dcfg["max_tokens"])
    stats: dict[str, Counter] = {}
    splits: dict[str, Rows] = {}

    zpath = Path(dcfg["unimer_1m_zip"]) if dcfg.get("unimer_1m_zip") else sources.UNIMER_1M.fetch()
    if sources.sha256_file(zpath) != sources.UNIMER_1M.sha256:
        raise RuntimeError(f"{zpath}: sha256 differs from the pinned UniMER-1M")
    workers = int(dcfg["workers"])

    # 1. Evaluation sets that come from elsewhere: UniMER-Test and our own tests.
    eval_keys: set[str] = set()
    pool = Pool(workers, initializer=_nice)
    try:
        root = sources.unimer_test_dir(cache)
        for subset in sources.UNIMER_SUBSETS:
            pairs = _read_unimer(root, subset)
            rows = splits[f"ood_{subset}"] = Rows()
            for (path, label), analyzed in zip(pairs, pool.imap(_analyze_path, [p for p, _ in pairs], chunksize=32)):
                rows.add(f"unimer/{subset}/{Path(path).name}", f"unimer_{subset}", label, analyzed)
                eval_keys.add(formula_key(label))
    finally:
        pool.close()
        pool.join()
    ours = Path(dcfg["ours_dir"])
    for name in ("test_im2latex", "test_oleehyo"):
        eval_keys.update(formula_key(l) for l in pq.read_table(ours / f"{name}.parquet", columns=["label"])["label"].to_pylist())

    # 2. UniMER-1M: labels, held-out buckets, train sample.
    st = stats["unimer_1m"] = Counter()
    with zipfile.ZipFile(zpath) as z:
        have = {int(n.rsplit("/", 1)[1][:-4]) for n in z.namelist() if n.startswith("UniMER-1M/images/") and n.endswith(".png")}
        lines = z.read("UniMER-1M/train.txt").decode("utf-8").split("\n")
    st["label_lines"], st["images"] = len(lines), len(have)
    held = {"val": [], "test": []}
    cand = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        label = line.strip()
        if i not in have:
            st["no_image"] += 1
            continue
        if not label:
            st["empty_label"] += 1
            continue
        key = formula_key(label)
        if key in seen:
            st["duplicate"] += 1
            continue
        seen.add(key)
        if key in eval_keys:
            st["in_eval_sets"] += 1
            continue
        bucket = _hash_int(key, "unimer-split") % 1000
        n_tokens = len(tokenize(label))
        if bucket < dcfg["unimer_val_permille"] + dcfg["unimer_test_permille"]:
            # held out (as mathocr.data.build: val/test within max_tokens; validation teacher-forces every target)
            split = "val" if bucket < dcfg["unimer_val_permille"] else "test"
            if n_tokens > max_tokens:
                st[f"{split}_too_long"] += 1
            else:
                held[split].append((i, label))
        elif n_tokens > train_max:
            st["train_too_long"] += 1
        else:
            cand.append((i, label))
    st["train_candidates"] = len(cand)
    size = int(dcfg["unimer_train_size"])
    cand = sorted(sorted(cand, key=lambda p: _hash_int(formula_key(p[1]), "unimer-train"))[:size])
    st["selected_train"], st["selected_val"], st["selected_test"] = len(cand), len(held["val"]), len(held["test"])

    pool = Pool(workers, initializer=_open_zip, initargs=(str(zpath),))
    try:
        for name, items in (("train", cand), ("val", sorted(held["val"])), ("test_unimer1m", sorted(held["test"]))):
            rows = splits[name] = Rows()
            members = [f"UniMER-1M/images/{i:07d}.png" for i, _ in items]
            for (i, label), analyzed in zip(items, pool.imap(_analyze_member, members, chunksize=64)):
                rows.add(f"unimer1m/{i:07d}", "unimer1m", label, analyzed)
            print(json.dumps({"analyzed": name, "rows": len(rows)}), flush=True)
    finally:
        pool.close()
        pool.join()

    # 3. Train-only filters (as mathocr.data.build).
    train = splits.pop("train").table()
    st = stats["train_filters"] = Counter(before=train.num_rows)
    min_fit = float(dcfg["min_fit_ratio"])
    keep = []
    for h, w, g in zip(train["h"].to_pylist(), train["w"].to_pylist(), train["glyph"].to_pylist()):
        desired = icfg.target_glyph_px / g if g > 0 else 1.0
        fit = min((icfg.max_h - 2 * icfg.margin) / h, (icfg.max_w - 2 * icfg.margin) / w)
        keep.append(g > 0 and fit >= min_fit * min(max(desired, icfg.min_scale), icfg.max_scale))
    st["dropped_unreadable_when_fitted"] = keep.count(False)
    train = train.filter(pa.array(keep))
    vocab = Vocab.build((tokenize(t) for t in train["label"].to_pylist()), int(dcfg["min_token_freq"]))
    in_vocab = [all(t in vocab for t in tokenize(l)) for l in train["label"].to_pylist()]
    st["dropped_rare_token"] = in_vocab.count(False)
    train = train.filter(pa.array(in_vocab))
    st["after"] = train.num_rows

    tables = {"train": train, **{k: v.table() for k, v in splits.items()}}
    for name in ("test_im2latex", "test_oleehyo"):
        shutil.copy2(ours / f"{name}.parquet", out_dir / f"{name}.parquet")
        tables[name] = pq.read_table(out_dir / f"{name}.parquet")
    manifest_splits = {}
    for name, table in sorted(tables.items()):
        if name not in ("test_im2latex", "test_oleehyo"):
            pq.write_table(table, out_dir / f"{name}.parquet", row_group_size=1024)
        digest = hashlib.sha256()
        for uid, label in zip(table["uid"].to_pylist(), table["label"].to_pylist()):
            digest.update(f"{uid}\t{label}\n".encode())
        shapes = Counter(output_shape(h, w, g, icfg) for h, w, g in
                         zip(table["h"].to_pylist(), table["w"].to_pylist(), table["glyph"].to_pylist()))
        manifest_splits[name] = {"rows": table.num_rows, "by_source": dict(sorted(Counter(table["source"].to_pylist()).items())),
                                 "uid_label_sha256": digest.hexdigest(), "distinct_shapes": len(shapes)}
    vocab.save(out_dir / "vocab.json")
    manifest = {
        "sources": {"unimer_1m": asdict(sources.UNIMER_1M), "unimer_test": asdict(sources.UNIMER_TEST),
                    "ours_tests": str(ours)},
        "data_config": dcfg, "image_config": asdict(icfg),
        "filters": {k: dict(v) for k, v in stats.items()}, "splits": manifest_splits,
        "vocab": {"size": len(vocab), "sha256": sources.sha256_file(out_dir / "vocab.json")},
        "leakage_check": _leakage(tables),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                           newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    manifest = build_unimer(load_config(args.config, args.set))
    print(json.dumps({k: manifest[k] for k in ("filters", "splits", "vocab", "leakage_check")}, indent=2))


if __name__ == "__main__":
    main()
