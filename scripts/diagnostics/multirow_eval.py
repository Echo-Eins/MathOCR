"""(Paths below are those of the Windows laptop.)  A checkpoint on the multi-row formulas of the full CPE split
(references with >= 4 rows: 993 of 5 921), greedy as the stand decodes: per number of reference rows, edit (UniMERNet
protocol), the share of answers with the right number of rows / fewer rows, and the error inside rows when the row
count matches -- tells "the rows are lost" from "the rows are misread" (multiline_errors.py).

    python scripts/diagnostics/multirow_eval.py --name win_b3_final6_s1 --ckpt D:/MathOCR/probe_fonts_many_b/runs/win_b3_final6_s1/checkpoints/best.pt
Output: docs/probe_results/multirow_<name>.json."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))
import probe  # noqa: E402
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from multiline_errors import edit, rows  # noqa: E402

CPE = Path(r"D:\MathOCR\data_fonts_b\ood_cpe.parquet")  # = data/processed/ood_cpe.parquet (copied byte for byte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--min-rows", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device("cuda")
    determinism.configure(dev)
    torch.cuda.set_per_process_memory_fraction(0.6, torch.cuda.current_device())
    model, vocab, icfg, _ = load_checkpoint(args.ckpt, dev)
    store = CropStore(CPE, vocab)
    pick = [i for i, lab in enumerate(store.labels) if (r := rows(lab.split())) and len(r[1]) >= args.min_rows]
    preds = probe.decode(model, store, icfg, vocab, dev, indices=pick)
    agg = defaultdict(lambda: {"n": 0, "edit": [], "same": 0, "fewer": 0, "more": 0, "row_edit": []})
    for i, p in zip(pick, preds):
        g = store.labels[i].split()
        p = [t for t in p if t != "<noeos>"]
        k = min(len(rows(g)[1]), 9)
        a = agg[k]
        a["n"] += 1
        a["edit"].append(edit(" ".join(p), store.labels[i]))
        pr = rows(p)
        n_p = len(pr[1]) if pr else 1
        n_g = len(rows(g)[1])
        a["same"] += n_p == n_g
        a["fewer"] += n_p < n_g
        a["more"] += n_p > n_g
        if pr and n_p == n_g:
            a["row_edit"] += [edit(" ".join(x), " ".join(y)) for x, y in zip(pr[1], rows(g)[1])]
    table = {f"{k}{'+' if k == 9 else ''} rows": {
        "n": a["n"], "edit": round(float(np.mean(a["edit"])), 4), "same_rows": round(a["same"] / a["n"], 3),
        "fewer_rows": round(a["fewer"] / a["n"], 3), "more_rows": round(a["more"] / a["n"], 3),
        "row_edit_when_rows_match": round(float(np.mean(a["row_edit"])), 4) if a["row_edit"] else None}
        for k, a in sorted(agg.items())}
    allr = [e for a in agg.values() for e in a["edit"]]
    out = {"name": args.name, "n": len(pick), "edit": round(float(np.mean(allr)), 4),
           "same_rows": round(sum(a["same"] for a in agg.values()) / len(pick), 3), "by_rows": table}
    (ROOT / "docs" / "probe_results" / f"multirow_{args.name}.json").write_text(json.dumps(out, indent=1) + "\n",
                                                                             encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
