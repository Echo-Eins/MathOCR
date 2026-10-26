"""UniMER-Test predictions of a run scored the way UniMERNet's test.py does (comparable with its published edit
distances): all whitespace removed, then character Levenshtein / max(len) per sample, averaged.

    python scripts/diagnostics/unimer_edit.py runs/final_v3_s1 [runs/final_v2_s1 ...]

Reads ``<run>/eval/ood_{spe,cpe,sce,hwe}.jsonl`` (``python -m mathocr.evaluate``).  Our own ``edit_distance`` keeps
the token spaces of the normalized strings, so it is not the same number."""

import json
import re
import sys
from pathlib import Path

import editdistance


def unimer_edit(pred: str, gt: str) -> float:
    p, g = re.sub(r"\s+", "", pred.replace("<noeos>", "")), re.sub(r"\s+", "", gt)
    n = max(len(p), len(g))
    return editdistance.eval(p, g) / n if n else 0.0


def main() -> None:
    for run in map(Path, sys.argv[1:]):
        for split in ("ood_spe", "ood_cpe", "ood_sce", "ood_hwe"):
            path = run / "eval" / f"{split}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            d = [unimer_edit(r["pred"], r["gt"]) for r in rows]
            print(f"{run.name} {split}: n={len(rows)} unimer-style edit {sum(d) / len(d):.4f}")


if __name__ == "__main__":
    main()
