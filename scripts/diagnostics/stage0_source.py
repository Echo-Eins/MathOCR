"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0b follow-up: are the systematic brace shifts of final_v2_s1 on test_im2latex a *source convention* the
model does not apply?  For every brace span the reference has and the prediction lacks (notation-only errors), the
3-gram around the inserted brace: its rate in im2latex train vs OleehyO train labels (per 1000 formulas).  And can
the source be told from the image itself (crop statistics: glyph height, ink density, crop size) -- a logistic
classifier on train, accuracy on the two tests.  Output: docs/probe_results/stage0_source.json."""

import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mathocr.data.dataset import CropStore  # noqa: E402
from stage0_style import inv  # noqa: E402

DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
EVAL = Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\eval")


def grams(tokens):
    return Counter(zip(tokens, tokens[1:], tokens[2:]))


def main() -> None:
    tr = CropStore(DATA / "train.parquet")
    by_src = {"im2latex": Counter(), "oleehyo": Counter()}
    n_src = Counter(tr.sources)
    for lab, src in zip(tr.labels, tr.sources):
        by_src[src].update(grams(lab.split()))
    res = {}
    for split in ("test_im2latex", "test_oleehyo"):
        rows = [json.loads(x) for x in (EVAL / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        ratios, missing = [], Counter()
        for r in rows:
            g, p = r["gt"].split(), r["pred"].split()
            if g == p or inv(g) != inv(p):
                continue
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, g, p, autojunk=False).get_opcodes():
                if tag in ("delete", "replace") and i2 > i1 and all(t in "{}" for t in g[i1:i2] + p[j1:j2]) and i2 - i1 > j2 - j1:
                    lo, hi = max(0, i1 - 1), min(len(g), i2 + 1)
                    gram = tuple((["<s>"] + g + ["</s>"])[lo:lo + 3]) if hi - lo < 3 else tuple(g[lo:lo + 3])
                    a = by_src["im2latex"].get(gram, 0) / n_src["im2latex"] * 1000
                    b = by_src["oleehyo"].get(gram, 0) / n_src["oleehyo"] * 1000
                    ratios.append(np.log((a + 0.1) / (b + 0.1)))
                    missing[" ".join(gram)] += 1
        ratios = np.array(ratios)
        res[split] = {"missing_brace_spans": int(len(ratios)),
                      "share_more_frequent_in_im2latex_train": float(np.mean(ratios > np.log(2))),
                      "share_more_frequent_in_oleehyo_train": float(np.mean(ratios < -np.log(2))),
                      "median_log_ratio_im2latex_over_oleehyo": float(np.median(ratios)) if len(ratios) else None,
                      "top": missing.most_common(10)}
        print(f"{split}: {len(ratios)} brace spans the reference has and the prediction lacks; "
              f"3-gram >= 2x more frequent in im2latex train {np.mean(ratios > np.log(2)) * 100:.0f}%, "
              f"in OleehyO train {np.mean(ratios < -np.log(2)) * 100:.0f}%; top {missing.most_common(5)}")

    # source from the image: crop statistics only (what the rendering pipeline leaves in the pixels)
    def feats(store, idx):
        out = []
        for i in idx:
            px = store.crop(i).pixels
            ink = px < 128
            out.append([np.log(store.glyph[i] + 1), np.log(store.h[i]), np.log(store.w[i]), ink.mean(),
                        ((px > 32) & (px < 224)).mean() / max(ink.mean(), 1e-3)])  # anti-aliasing ratio
        return np.array(out)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(tr), 20000, replace=False)
    x, y = feats(tr, idx), np.array([tr.sources[i] == "im2latex" for i in idx], float)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w = np.zeros(x.shape[1] + 1)
    xb = np.c_[(x - mu) / sd, np.ones(len(x))]
    for _ in range(3000):  # logistic regression, plain gradient descent
        pr = 1 / (1 + np.exp(-xb @ w))
        w -= 0.5 * xb.T @ (pr - y) / len(y)
    acc = {}
    for split, src in (("test_im2latex", 1.0), ("test_oleehyo", 0.0)):
        st = CropStore(DATA / f"{split}.parquet")
        j = rng.choice(len(st), 2000, replace=False)
        xt = np.c_[(feats(st, j) - mu) / sd, np.ones(len(j))]
        acc[split] = float(np.mean(((1 / (1 + np.exp(-xt @ w))) > 0.5) == src))
    res["source_from_crop_stats_accuracy"] = acc
    print("source told from crop statistics alone (logistic, 5 features):", {k: round(v, 3) for k, v in acc.items()})
    (ROOT / "docs" / "probe_results" / "stage0_source.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
