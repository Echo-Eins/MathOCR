"""One-off patch (2026-10-27): drop held-out UniMER-1M pairs longer than ``data.max_tokens`` from built data dirs.

    python scripts/ops/patch_unimer_heldout.py D:/MathOCR/data_unimer D:/MathOCR/data_unimer_fb D:/MathOCR/data_unimer_fb_long

``build_unimer`` first applied the length limit to train only; 5 val targets of up to 1 100 tokens then crashed the
validation of ``unimer_v4_s1`` at the end of its first epoch.  The fixed build (same commit as this script) gives the
same train and val / test minus exactly these pairs, so the patch reproduces it without re-rendering: ``val`` and
``test_unimer1m`` are filtered in place (the old files kept as ``*.parquet.before_patch``) and the manifest's split
entries are recomputed; ``filters.heldout_patch`` records the drop.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr.image import ImageConfig, output_shape  # noqa: E402
from mathocr.tokenizer import tokenize  # noqa: E402

MAX_TOKENS = 766


def main() -> None:
    for d in map(Path, sys.argv[1:]):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        icfg = ImageConfig(**manifest["image_config"])
        dropped = {}
        for name in ("val", "test_unimer1m"):
            path = d / f"{name}.parquet"
            t = pq.read_table(path)
            keep = [len(tokenize(l)) <= MAX_TOKENS for l in t["label"].to_pylist()]
            dropped[name] = [u for u, k in zip(t["uid"].to_pylist(), keep) if not k]
            if not dropped[name]:
                continue
            shutil.copy2(path, d / f"{name}.parquet.before_patch")
            t = t.filter(pa.array(keep))
            pq.write_table(t, path, row_group_size=1024)
            digest = hashlib.sha256()
            for uid, label in zip(t["uid"].to_pylist(), t["label"].to_pylist()):
                digest.update(f"{uid}\t{label}\n".encode())
            shapes = Counter(output_shape(h, w, g, icfg) for h, w, g in
                             zip(t["h"].to_pylist(), t["w"].to_pylist(), t["glyph"].to_pylist()))
            manifest["splits"][name] = {"rows": t.num_rows, "by_source": dict(sorted(Counter(t["source"].to_pylist()).items())),
                                        "uid_label_sha256": digest.hexdigest(), "distinct_shapes": len(shapes)}
        manifest.setdefault("filters", {})["heldout_patch"] = {
            "rule": f"scripts/ops/patch_unimer_heldout.py: val / test_unimer1m pairs over {MAX_TOKENS} tokens dropped",
            "dropped": dropped}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                         newline="\n")
        print(d, {k: len(v) for k, v in dropped.items()}, {k: manifest["splits"][k]["rows"] for k in ("val", "test_unimer1m")})


if __name__ == "__main__":
    main()
