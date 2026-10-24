"""Where a probe's errors sit: by kind of token (vertical constructs -- fractions, scripts, big operators, accents,
arrays -- against braces and plain symbols) and by the height of the formula on the memory grid.

Reference and prediction token sequences are aligned (difflib); a reference token in a replaced or deleted run is
missed, a predicted token in a replaced or inserted run is spurious.  Per class: (missed + spurious) / reference
count.  Per height bucket (memory rows at the probe's preprocessing): exact match and token error rate.
Usage: struct_errors.py --root D:\\MathOCR\\probe_fonts70 --runs win_v2_fonts70_s1 win_b1_triscan_x_s1 ..."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mathocr.data.dataset import CropStore  # noqa: E402
from mathocr.image import ImageConfig, output_shape  # noqa: E402

CLASSES = {
    "fraction": {r"\frac", r"\dfrac", r"\tfrac", r"\cfrac", r"\over", r"\atop", r"\binom", r"\choose", r"\stackrel",
                 r"\overset", r"\underset", r"\substack"},
    "script": {"^", "_"},
    "big op": {r"\sum", r"\prod", r"\int", r"\iint", r"\iiint", r"\oint", r"\lim", r"\bigcup", r"\bigcap", r"\coprod",
               r"\bigoplus", r"\bigotimes", r"\limits", r"\nolimits", r"\max", r"\min", r"\sup", r"\inf"},
    "accent": {r"\hat", r"\bar", r"\tilde", r"\vec", r"\dot", r"\ddot", r"\check", r"\breve", r"\acute", r"\grave",
               r"\widehat", r"\widetilde", r"\overline", r"\underline", r"\overbrace", r"\underbrace",
               r"\overrightarrow", r"\overleftarrow"},
    "array": {r"\\", "&", r"\begin", r"\end", r"\hline", "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix",
              "cases", "aligned", "split", "gathered"},
    "radical": {r"\sqrt"},
    "brace": {"{", "}"},
}
VERTICAL = ("fraction", "script", "big op", "accent", "array", "radical")
BUCKETS = ((0, 8, "<=8"), (9, 12, "9-12"), (13, 16, "13-16"), (17, 24, "17-24"), (25, 999, ">24"))


def kind(tok: str) -> str:
    for name, toks in CLASSES.items():
        if tok in toks:
            return name
    return "symbol"


def compare(gt: list[str], pred: list[str]) -> tuple[Counter, Counter, Counter]:
    ref, miss, extra = Counter(kind(t) for t in gt), Counter(), Counter()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes():
        if tag != "equal":
            miss.update(kind(t) for t in gt[i1:i2])
            extra.update(kind(t) for t in pred[j1:j2])
    return ref, miss, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "ood_cpe", "ood_sce", "ood_spe"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    icfg = ImageConfig(target_glyph_px=18.0, max_h=384, max_w=1536, margin=8, multiple=32)
    rows_of = {}
    for split in args.splits:
        s = CropStore(args.root / "data20k" / f"{split}.parquet")
        for uid, h, w, g in zip(s.uids, s.h, s.w, s.glyph):
            rows_of[uid] = output_shape(int(h), int(w), float(g), icfg)[0] // 8
    out = {}
    for split in args.splits:
        print(f"== {split}")
        names = list(CLASSES) + ["symbol"]
        print(f"{'run':24} {'EM':>6} {'tok err':>7} | " + " ".join(f"{n:>8}" for n in names) + " | EM by memory rows "
              + " ".join(b[2] for b in BUCKETS))
        for run in args.runs:
            recs = [json.loads(x) for x in (args.root / "runs" / run / "preds" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            ref, miss, extra = Counter(), Counter(), Counter()
            bucket = defaultdict(lambda: [0, 0, 0, 0])  # formulas, exact, ref tokens, errors
            vert = [0, 0]
            for r in recs:
                gt, pred = r["gt"].split(), [t for t in r["pred"].split() if t != "<noeos>"]
                a, m, e = compare(gt, pred)
                ref += a
                miss += m
                extra += e
                rows = rows_of.get(r["uid"])
                b = next(lab for lo, hi, lab in BUCKETS if rows is not None and lo <= rows <= hi) if rows is not None else "?"
                st = bucket[b]
                st[0] += 1
                st[1] += gt == pred
                st[2] += len(gt)
                st[3] += sum(m.values()) + sum(e.values())
                if any(a[k] for k in VERTICAL):
                    vert[0] += 1
                    vert[1] += gt == pred
            err = {n: (miss[n] + extra[n]) / ref[n] if ref[n] else None for n in names}
            em = sum(st[1] for st in bucket.values()) / len(recs)
            tok = (sum(miss.values()) + sum(extra.values())) / sum(ref.values())
            by_h = {lab: (bucket[lab][1] / bucket[lab][0], bucket[lab][0]) if bucket[lab][0] else None for *_, lab in BUCKETS}
            out[f"{split} {run}"] = {"em": em, "token_error": tok, "class_error": err, "ref_count": dict(ref),
                                     "em_vertical_formulas": vert[1] / max(1, vert[0]), "n_vertical": vert[0],
                                     "em_by_rows": by_h}
            print(f"{run:24} {em:6.3f} {tok:7.4f} | " + " ".join(f"{err[n]:8.4f}" if err[n] is not None else f"{'-':>8}" for n in names)
                  + " | " + " ".join(f"{v[0]:.3f}({v[1]})" if v else "-" for v in by_h.values()))
        print("   reference tokens: " + ", ".join(f"{n} {ref[n]}" for n in names))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
