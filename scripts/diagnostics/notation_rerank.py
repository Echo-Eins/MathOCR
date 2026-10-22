"""(Paths below are those of the Windows laptop; NB = beam-8 n-best dumps, see HANDOFF section 6, 2026-10-22.)
Result: no gain (val EM 0.544 -> 0.547 at best, ceiling 0.721) -- a 4-gram LM cannot pick the notation.
Notation-only reranking: vision decides *what* (the greedy output's notation-normalized content), a separate
n-gram notation LM decides *how* -- it may only pick among beam hypotheses with the same normalized content as the
greedy output, so the rendered formula (CDM) cannot change; only EM can.  Interpolated absolute-discounting
4-gram LMs from train labels: all sources, and per source (the source is what an image classifier would supply)."""
import contextlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Coding\Python\MathOCR\src")
sys.path.insert(0, str(Path(__file__).parent))
if True:  # notation_pools has no side effects on import
    from notation_pools import inv
from mathocr.data.dataset import CropStore  # noqa: E402

DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
NB = Path(r"D:\MathOCR\nbest_final_v2_s1")
ORDER, D = 4, 0.75


class NGram:
    def __init__(self, seqs):
        self.c = [Counter() for _ in range(ORDER + 1)]  # c[n][(ctx..., w)]
        self.ctx = [Counter() for _ in range(ORDER + 1)]  # c[n][ctx] = sum_w c(ctx, w)
        self.types = [Counter() for _ in range(ORDER + 1)]  # distinct continuations of ctx
        for s in seqs:
            t = ["<s>"] * (ORDER - 1) + s + ["</s>"]
            for i in range(ORDER - 1, len(t)):
                for n in range(1, ORDER + 1):
                    g = tuple(t[i - n + 1:i + 1])
                    if self.c[n][g] == 0:
                        self.types[n][g[:-1]] += 1
                    self.c[n][g] += 1
                    self.ctx[n][g[:-1]] += 1
        self.V = len(self.c[1]) + 1

    def logp(self, s):
        t = ["<s>"] * (ORDER - 1) + s + ["</s>"]
        total = 0.0
        for i in range(ORDER - 1, len(t)):
            p = 1.0 / self.V
            for n in range(1, ORDER + 1):
                g = tuple(t[i - n + 1:i + 1])
                cc = self.ctx[n][g[:-1]]
                if cc:
                    p = max(self.c[n][g] - D, 0) / cc + D * self.types[n][g[:-1]] / cc * p
            total += math.log(p)
        return total


st = CropStore(DATA / "train.parquet")
by_src = defaultdict(list)
for lab, src in zip(st.labels, st.sources):
    by_src[src].append(lab.split())
lms = {"all": NGram([s for v in by_src.values() for s in v])}
lms.update({src: NGram(v) for src, v in by_src.items()})
print("LMs built", flush=True)


def load(split):
    return [json.loads(x) for x in (NB / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def candidates(r):
    greedy = r["greedy"].split()
    key = inv(greedy)
    cands = {}
    for s, h in r["hyps"]:
        h = h.split()
        if inv(h) == key and tuple(h) not in cands:
            cands[tuple(h)] = s
    if tuple(greedy) not in cands:
        cands[tuple(greedy)] = max((s for s, _ in r["hyps"]), default=0.0)
    return cands


def evaluate(rows, lm_of, lams):
    em = defaultdict(list)
    for r in rows:
        cands = candidates(r)
        lm = lm_of(r)
        scored = [(s, lm.logp(list(h)) / max(len(h), 1), list(h) == r["gt"].split()) for h, s in cands.items()]
        for lam in lams:
            em[lam].append(max(scored, key=lambda x: x[0] + lam * x[1])[2])
    return {lam: float(np.mean(v)) for lam, v in em.items()}


LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e3)
for split in ("val", "test_im2latex", "test_oleehyo"):
    if not (NB / f"{split}.jsonl").exists():
        continue
    rows = load(split)
    g_em = np.mean([r["greedy"].split() == r["gt"].split() for r in rows])
    reach = np.mean([tuple(r["gt"].split()) in candidates(r) for r in rows])
    multi = np.mean([len(candidates(r)) > 1 for r in rows])
    print(f"{split}: n={len(rows)} greedy EM {g_em:.3f}; reference among same-content candidates {reach:.3f} "
          f"(ceiling); >1 candidate {multi * 100:.0f}%")
    for name, lm_of in (("LM all", lambda r: lms["all"]), ("LM by source", lambda r: lms[r["source"]])):
        res = evaluate(rows, lm_of, LAMS)
        print(f"   {name:13}: " + "  ".join(f"λ={lam:g} {v:.3f}" for lam, v in res.items()), flush=True)
