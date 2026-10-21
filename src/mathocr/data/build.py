"""Build the fixed train/val/test splits from the pinned sources.

    python -m mathocr.data.build --config configs/base.yaml

Output (``data.out_dir``): one parquet per split with the analyzed crop of
every image (``image.analyze``), plus ``manifest.json`` (sources, filters,
counts, a sha256 per split over its ``uid``/``label`` list) and
``vocab.json``.  The build is a pure function of the pinned source bytes and
the config, so the manifest hashes identify the data version.

Splits
------
* ``train`` -- im2latex-100k train + an OleehyO (arXiv) sample;
* ``val`` -- im2latex val + OleehyO val (checkpoint selection only);
* ``test_im2latex``, ``test_oleehyo`` -- in-distribution tests;
* ``ood_{cpe,sce,spe,hwe}`` -- UniMER-Test subsets, a different source.

Leakage: a *formula key* is the label's letters and digits only
(``formula_key``).  No training formula shares a key with any
val/test/ood formula.  Test labels are kept
exactly as published (only empty ones are skipped, like UniMERNet's
``test.py``); filters apply to training data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mathocr.config import load_config
from mathocr.data import sources
from mathocr.image import ImageConfig, analyze, output_shape
from mathocr.normalize import normalize_batch, unwrap_display
from mathocr.tokenizer import Vocab, tokenize

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("source", pa.string()),
    ("label", pa.string()),
    ("crop", pa.binary()),  # PNG, uint8 L, 0 = ink
    ("h", pa.int32()),
    ("w", pa.int32()),
    ("glyph", pa.float32()),
])


def formula_key(label: str) -> str:
    """Leakage/dedup key: only the letters and digits of the label.

    Deliberately coarse.  Measured on a one-shard build, a whitespace-only
    key left 88 im2latex-test formulas (1.3%) with a training twin that
    differs only in spacing commands (``\\:``), a trailing ``\\\\`` before
    ``\\end{array}`` or final punctuation -- the same formula from the same
    paper.  Formulas that differ only in operators collide too, which only
    drops a little more training data; evaluation sets are never filtered.
    """

    return _NON_ALNUM.sub("", label)


_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def _hash_int(text: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\x00{text}".encode()).digest()[:8], "big")


# ------------------------------------------------------------------ image work


def _analyze_bytes(data: bytes) -> tuple[bytes, int, int, float]:
    with Image.open(io.BytesIO(data)) as im:
        crop = analyze(im)
    buf = io.BytesIO()
    Image.fromarray(crop.pixels, mode="L").save(buf, format="PNG", optimize=False, compress_level=6)
    h, w = crop.pixels.shape
    return buf.getvalue(), h, w, crop.glyph_px


def _analyze_path(path: str) -> tuple[bytes, int, int, float]:
    return _analyze_bytes(Path(path).read_bytes())


def _nice() -> None:
    if hasattr(os, "nice"):  # POSIX only (Windows has none)
        os.nice(19)


class Rows:
    """Accumulates analyzed rows of one split, in a deterministic order."""

    def __init__(self) -> None:
        self.cols: dict[str, list] = {f.name: [] for f in SCHEMA}

    def add(self, uid: str, source: str, label: str, analyzed: tuple[bytes, int, int, float]) -> None:
        crop, h, w, glyph = analyzed
        for k, v in zip(SCHEMA.names, (uid, source, label, crop, h, w, glyph)):
            self.cols[k].append(v)

    def __len__(self) -> int:
        return len(self.cols["uid"])

    def table(self) -> pa.Table:
        return pa.table(self.cols, schema=SCHEMA)


def _run(pool: Pool, fn: Callable, items: Iterable) -> Iterator:
    return pool.imap(fn, items, chunksize=32)


# --------------------------------------------------------------------- sources


def _read_im2latex(split: str) -> list[dict]:
    path = sources.IM2LATEX[split].fetch()
    return pq.read_table(path).to_pylist()


def _read_unimer(root: Path, subset: str) -> list[tuple[str, str]]:
    """(image path, label) pairs; line i of ``<subset>.txt`` is ``{i:07d}.png``."""

    lines = (root / f"{subset}.txt").read_text(encoding="utf-8").split("\n")
    pairs = []
    for i, line in enumerate(lines):
        image = root / subset / f"{i:07d}.png"
        if line.strip() and image.exists():
            pairs.append((str(image), line.strip()))
    return pairs


def _oleehyo_formulas(shards: list[int]) -> list[tuple[int, int, str]]:
    out = []
    for s in shards:
        path = sources.OLEEHYO_SHARDS[s].fetch()
        for i, f in enumerate(pq.read_table(path, columns=["latex_formula"]).column("latex_formula").to_pylist()):
            out.append((s, i, f))
    return out


def _oleehyo_images(selected: dict[tuple[int, int], str]) -> Iterator[tuple[tuple[int, int], bytes]]:
    """Image bytes of the selected (shard, row) pairs in (shard, row) order."""

    by_shard: dict[int, set[int]] = {}
    for s, i in selected:
        by_shard.setdefault(s, set()).add(i)
    for s in sorted(by_shard):
        pf = pq.ParquetFile(sources.OLEEHYO_SHARDS[s].fetch())
        start = 0
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            wanted = [i for i in range(start, start + n) if i in by_shard[s]]
            if wanted:
                images = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
                for i in wanted:
                    yield (s, i), images[i - start]["bytes"]
            start += n


# ----------------------------------------------------------------------- build


def build(cfg: dict) -> dict:
    dcfg, icfg = cfg["data"], ImageConfig(**cfg["image"])
    out_dir = Path(dcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(os.path.expanduser(dcfg["cache_dir"]))
    max_tokens = int(dcfg["max_tokens"])
    # Train-only length limit (default: the same).  A larger value lets real long formulas into train;
    # val/test keep ``max_tokens``, so every evaluation split stays byte-identical.
    train_max_tokens = int(dcfg.get("train_max_tokens") or max_tokens)
    if train_max_tokens < max_tokens:
        raise ValueError("data.train_max_tokens must be >= data.max_tokens")
    stats: dict[str, Counter] = {}
    splits: dict[str, Rows] = {}

    pool = Pool(int(dcfg["workers"]), initializer=_nice)
    try:
        # 1. Evaluation sets first: their keys define what train must avoid.
        unimer_root = sources.unimer_test_dir(cache)
        eval_keys: set[str] = set()
        for subset in sources.UNIMER_SUBSETS:
            pairs = _read_unimer(unimer_root, subset)
            rows = splits[f"ood_{subset}"] = Rows()
            for (path, label), analyzed in zip(pairs, _run(pool, _analyze_path, [p for p, _ in pairs])):
                rows.add(f"unimer/{subset}/{Path(path).name}", f"unimer_{subset}", label, analyzed)
                eval_keys.add(formula_key(label))

        im2 = {split: _read_im2latex(split) for split in ("train", "val", "test")}
        for split in ("val", "test"):
            st = stats[f"im2latex_{split}"] = Counter(raw=len(im2[split]))
            kept = [r for r in im2[split] if r["formula"].strip()]
            st["empty_label"] = len(im2[split]) - len(kept)
            im2[split] = kept
            eval_keys.update(formula_key(r["formula"]) for r in kept)
        im2_keys = {formula_key(r["formula"]) for split in im2.values() for r in split}

        # 2. OleehyO: unwrap, normalize once, dedup, split by key hash.
        st = stats["oleehyo"] = Counter()
        raw = _oleehyo_formulas(list(dcfg["oleehyo_shards"]))
        st["raw"] = len(raw)
        unwrapped = [(s, i, unwrap_display(f)) for s, i, f in raw]
        st["dropped_unwrap"] = sum(u is None for *_, u in unwrapped)
        unwrapped = [(s, i, u) for s, i, u in unwrapped if u is not None]
        normalized = normalize_batch([u for *_, u in unwrapped])
        st["dropped_katex"] = sum(n is None for n in normalized)
        seen: set[str] = set()
        ol_split: dict[str, dict[tuple[int, int], str]] = {"train": {}, "val": {}, "test": {}}
        for (s, i, _), label in zip(unwrapped, normalized):
            if label is None:
                continue
            n_tokens = len(tokenize(label))
            if n_tokens > max_tokens:
                # Never into val/test; into train only within train_max_tokens (same bucket rule as below).
                bucket = _hash_int(formula_key(label), "oleehyo-split") % 1000
                if n_tokens > train_max_tokens or bucket < dcfg["oleehyo_val_permille"] + dcfg["oleehyo_test_permille"]:
                    st["dropped_too_long"] += 1
                    continue
            key = formula_key(label)
            if key in seen:
                st["dropped_duplicate"] += 1
                continue
            seen.add(key)
            if key in im2_keys:
                st["dropped_in_im2latex"] += 1
                continue
            if key in eval_keys:
                st["dropped_in_eval_sets"] += 1
                continue
            bucket = _hash_int(key, "oleehyo-split") % 1000
            if bucket < dcfg["oleehyo_val_permille"]:
                ol_split["val"][(s, i)] = label
            elif bucket < dcfg["oleehyo_val_permille"] + dcfg["oleehyo_test_permille"]:
                ol_split["test"][(s, i)] = label
            else:
                ol_split["train"][(s, i)] = label
        for split in ("val", "test"):
            eval_keys.update(formula_key(v) for v in ol_split[split].values())
        st["train_candidates"] = len(ol_split["train"])
        size = int(dcfg["oleehyo_train_size"])
        if len(ol_split["train"]) > size:
            order = sorted(ol_split["train"], key=lambda k: _hash_int(formula_key(ol_split["train"][k]), "oleehyo-train"))
            ol_split["train"] = {k: ol_split["train"][k] for k in sorted(order[:size])}
        for split in ("train", "val", "test"):
            st[f"selected_{split}"] = len(ol_split[split])
        st["selected_train_longer_than_max_tokens"] = sum(len(tokenize(l)) > max_tokens for l in ol_split["train"].values())

        # 3. Analyze images.
        rows_val, rows_train = Rows(), Rows()
        for split, rows in (("val", rows_val), ("test", Rows())):
            items = im2[split]
            for r, analyzed in zip(items, _run(pool, _analyze_bytes, (r["image"]["bytes"] for r in items))):
                rows.add(f"im2latex/{split}/{r['filename']}", "im2latex", r["formula"].strip(), analyzed)
            if split == "test":
                splits["test_im2latex"] = rows

        st = stats["im2latex_train"] = Counter(raw=len(im2["train"]))
        train_items = []
        for r in im2["train"]:
            label = r["formula"].strip()
            if not label:
                st["dropped_empty"] += 1
            elif "[object" in label:  # broken upstream normalization
                st["dropped_object_object"] += 1
            elif len(tokenize(label)) > train_max_tokens:
                st["dropped_too_long"] += 1
            elif formula_key(label) in eval_keys:
                st["dropped_in_eval_sets"] += 1
            else:
                train_items.append((r, label))
        for (r, label), analyzed in zip(train_items, _run(pool, _analyze_bytes, (r["image"]["bytes"] for r, _ in train_items))):
            rows_train.add(f"im2latex/train/{r['filename']}", "im2latex", label, analyzed)

        for split, rows in (("train", rows_train), ("val", rows_val), ("test", Rows())):
            items = list(_oleehyo_images(ol_split[split]))
            for (si, data), analyzed in zip(items, _run(pool, _analyze_bytes, (d for _, d in items))):
                s, i = si
                rows.add(f"oleehyo/{s}/{i}", "oleehyo", ol_split[split][si], analyzed)
            if split == "test":
                splits["test_oleehyo"] = rows
        splits["val"] = rows_val
    finally:
        pool.close()
        pool.join()

    # 4. Train-only filters: unreadable after fitting, rare tokens.
    train = rows_train.table()
    st = stats["train_filters"] = Counter(before=train.num_rows)
    keep = []
    min_fit = float(dcfg["min_fit_ratio"])
    for h, w, g in zip(train["h"].to_pylist(), train["w"].to_pylist(), train["glyph"].to_pylist()):
        desired = icfg.target_glyph_px / g if g > 0 else 1.0
        fit = min((icfg.max_h - 2 * icfg.margin) / h, (icfg.max_w - 2 * icfg.margin) / w)
        ok = g > 0 and fit >= min_fit * min(max(desired, icfg.min_scale), icfg.max_scale)
        keep.append(ok)
    st["dropped_unreadable_when_fitted"] = keep.count(False)
    train = train.filter(pa.array(keep))
    vocab = Vocab.build((tokenize(t) for t in train["label"].to_pylist()), int(dcfg["min_token_freq"]))
    in_vocab = [all(t in vocab for t in tokenize(l)) for l in train["label"].to_pylist()]
    st["dropped_rare_token"] = in_vocab.count(False)
    train = train.filter(pa.array(in_vocab))
    st["after"] = train.num_rows

    tables = {"train": train, **{k: v.table() for k, v in splits.items()}}
    manifest_splits = {}
    for name, table in sorted(tables.items()):
        pq.write_table(table, out_dir / f"{name}.parquet", row_group_size=1024)
        digest = hashlib.sha256()
        for uid, label in zip(table["uid"].to_pylist(), table["label"].to_pylist()):
            digest.update(f"{uid}\t{label}\n".encode())
        shapes = Counter(
            output_shape(h, w, g, icfg) for h, w, g in
            zip(table["h"].to_pylist(), table["w"].to_pylist(), table["glyph"].to_pylist())
        )
        manifest_splits[name] = {
            "rows": table.num_rows,
            "by_source": dict(sorted(Counter(table["source"].to_pylist()).items())),
            "uid_label_sha256": digest.hexdigest(),
            "distinct_shapes": len(shapes),
        }
    vocab.save(out_dir / "vocab.json")
    manifest = {
        "sources": {
            "im2latex": [asdict(f) for f in sources.IM2LATEX.values()],
            "oleehyo": [asdict(sources.OLEEHYO_SHARDS[s]) for s in dcfg["oleehyo_shards"]],
            "unimer_test": asdict(sources.UNIMER_TEST),
        },
        "data_config": dcfg,
        "image_config": asdict(icfg),
        "filters": {k: dict(v) for k, v in stats.items()},
        "splits": manifest_splits,
        "vocab": {"size": len(vocab), "sha256": sources.sha256_file(out_dir / "vocab.json")},
        "leakage_check": _leakage(tables),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                           newline="\n")
    return manifest


def _leakage(tables: dict[str, pa.Table]) -> dict[str, int]:
    """Formula keys shared between train and each evaluation split (must be 0)."""

    train_keys = {formula_key(l) for l in tables["train"]["label"].to_pylist()}
    return {
        name: sum(formula_key(l) in train_keys for l in t["label"].to_pylist())
        for name, t in sorted(tables.items()) if name != "train"
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. data.out_dir=data/processed_long data.train_max_tokens=510")
    args = ap.parse_args(argv)
    manifest = build(load_config(args.config, args.set))
    print(json.dumps({k: manifest[k] for k in ("filters", "splits", "vocab", "leakage_check")}, indent=2))


if __name__ == "__main__":
    main()
