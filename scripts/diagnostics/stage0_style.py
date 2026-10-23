"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0b: is notation a formula-level style (author consistent, decoder mixing), i.e. is there a basis for a global
style latent?  final_v2_s1, greedy predictions of the full in-distribution tests and beam-8 n-best of 1000 samples
per split (nbest_dump.py).

1. Author consistency: for each synonym family (\\cal/\\mathcal, \\bf/\\mathbf, \\le/\\leq, ...) and for \\left vs plain
   parentheses, formulas of the reference with >= 2 instances -- share written in one variant, against the share
   expected if each instance were chosen independently with the source's marginal (shuffle baseline).
2. The same for the greedy predictions, and how often the prediction mixes variants where the reference does not.
3. Notation-only errors (same rendering, different string): their differing spans, typed (braces / spacing / synonym
   / sizing / style) with direction; per formula, how many independent decisions are wrong, and whether they are
   one systematic style shift or scattered.
4. Consistency re-ranking: among the beam hypotheses with the same rendering as the greedy output, take the one
   that mixes the fewest families (ties: model score) -- EM change with a paired bootstrap interval.
Also: how much the earlier notation normalizer (notation_pools.inv) over-merged (\\cdots = \\ldots, \\colon = :).
Output: docs/probe_results/stage0_style.json."""

import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from notation_pools import SIZE, SPACES, STYLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL = Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\eval")
NB = Path(r"D:\MathOCR\nbest_final_v2_s1")
B = "\\"
FAMILIES = {name: (B + a, B + b) for name, a, b in (
    ("cal", "cal", "mathcal"), ("bf", "bf", "mathbf"), ("le", "le", "leq"), ("ge", "ge", "geq"), ("ne", "ne", "neq"),
    ("to", "to", "rightarrow"), ("dots", "dots", "ldots"), ("gets", "gets", "leftarrow"))}
FAMILY_OF = {t: (name, k) for name, pair in FAMILIES.items() for k, t in enumerate(pair)}
# render-identical synonyms only (\cdots and \colon render differently from \ldots and ':')
SYN = {B + "cal": B + "mathcal", B + "to": B + "rightarrow", B + "le": B + "leq", B + "ge": B + "geq",
       B + "ne": B + "neq", B + "gets": B + "leftarrow", B + "lbrace": B + "{", B + "rbrace": B + "}", B + "vert": "|",
       B + "lvert": "|", B + "rvert": "|", B + "bf": B + "mathbf", B + "rm": B + "mathrm", B + "it": B + "mathit",
       B + "dots": B + "ldots"}


def inv(tokens):
    out = []
    for t in tokens:
        if t in ("{", "}") or t in SPACES or t in STYLE:
            continue
        m = SIZE.match(t)
        if m:
            t = m.group(3)
            if t in ("", "."):
                continue
        out.append(SYN.get(t, t))
    return out


def inv_old(tokens):
    from notation_pools import inv as old
    return old(tokens)


def choices(tokens):
    """family -> list of variant indices in order of appearance (synonyms; parentheses: 1 = \\left/\\bigl...)."""
    ch = defaultdict(list)
    for i, t in enumerate(tokens):
        if t in FAMILY_OF:
            name, k = FAMILY_OF[t]
            ch[name].append(k)
        elif t == "(":
            ch["paren"].append(1 if i and SIZE.match(tokens[i - 1]) else 0)
        elif SIZE.match(t) and t.endswith("("):
            ch["paren"].append(1)
    return ch


def consistent(v):
    return len(set(v)) == 1


def span_type(g, p):
    toks = g + p
    if not toks:
        return "none"
    if all(t in "{}" for t in toks):
        kind = "braces"
    elif all(t in SPACES for t in toks):
        kind = "spacing"
    elif all(t in STYLE for t in toks):
        kind = "style"
    elif all(SIZE.match(t) or t in "()[]|" for t in toks):
        kind = "sizing"
    elif len(g) == len(p) == 1 and SYN.get(g[0], g[0]) == SYN.get(p[0], p[0]):
        return f"synonym {g[0]}->{p[0]}"
    else:
        kind = "mixed"
    direction = "ref more" if len(g) > len(p) else "pred more" if len(p) > len(g) else "swap"
    return f"{kind} ({direction})"


def boot_ci(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    res = {}
    # ---- 1-3 on the full in-distribution tests (greedy)
    for split in ("test_im2latex", "test_oleehyo"):
        rows = [json.loads(x) for x in (EVAL / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        g_tok = [r["gt"].split() for r in rows]
        p_tok = [r["pred"].split() for r in rows]
        r_split = {}
        # author / model consistency per family, with the independent-choice baseline
        fam = {}
        for name in list(FAMILIES) + ["paren"]:
            gt_vecs = [choices(g)[name] for g in g_tok]
            multi = [i for i, v in enumerate(gt_vecs) if len(v) >= 2]
            if len(multi) < 30:
                continue
            p1 = np.mean([k for i in multi for k in gt_vecs[i]])
            expect = np.mean([p1 ** len(gt_vecs[i]) + (1 - p1) ** len(gt_vecs[i]) for i in multi])
            gt_cons = np.mean([consistent(gt_vecs[i]) for i in multi])
            pr_vecs = [choices(p_tok[i])[name] for i in multi]
            pr_multi = [j for j, v in enumerate(pr_vecs) if len(v) >= 2]
            pr_cons = np.mean([consistent(pr_vecs[j]) for j in pr_multi]) if pr_multi else float("nan")
            mix_when_gt_cons = np.mean([not consistent(pr_vecs[j]) for j in pr_multi if consistent(gt_vecs[multi[j]])]) \
                if pr_multi else float("nan")
            fam[name] = {"formulas": len(multi), "variant1_share": float(p1), "ref_consistent": float(gt_cons),
                         "independent_expectation": float(expect), "pred_consistent": float(pr_cons),
                         "pred_mixes_when_ref_consistent": float(mix_when_gt_cons)}
        r_split["families"] = fam
        # notation-only errors: typed spans and structure per formula
        types, per_formula, systematic = Counter(), Counter(), Counter()
        over = 0
        n_notation = 0
        for g, p in zip(g_tok, p_tok):
            if g == p:
                continue
            if inv_old(g) == inv_old(p) and inv(g) != inv(p):
                over += 1
            if inv(g) != inv(p):
                continue
            n_notation += 1
            spans = [span_type(g[i1:i2], p[j1:j2]) for tag, i1, i2, j1, j2 in
                     SequenceMatcher(None, g, p, autojunk=False).get_opcodes() if tag != "equal"]
            types.update(spans)
            k = len(spans)
            per_formula["1" if k == 1 else "2" if k == 2 else "3-4" if k <= 4 else "5+"] += 1
            if k >= 2:
                systematic["one style shift" if len(set(spans)) == 1 else "scattered"] += 1
        n = len(rows)
        r_split["notation_only_share"] = n_notation / n
        r_split["old_normalizer_overmerged_share"] = over / n
        r_split["decisions_per_notation_error"] = {k: v / n_notation for k, v in per_formula.items()}
        r_split["multi_decision_errors"] = {k: v / max(1, sum(systematic.values())) for k, v in systematic.items()}
        r_split["span_types_top"] = [(t, round(c / sum(types.values()), 3)) for t, c in types.most_common(12)]
        res[split] = r_split
        print(f"== {split}: notation-only errors {n_notation / n * 100:.1f}% of formulas "
              f"(old normalizer over-merged {over / n * 100:.2f}% more)")
        for name, f in fam.items():
            print(f"   {name:6} n={f['formulas']:4} variant-1 share {f['variant1_share']:.2f} | ref consistent "
                  f"{f['ref_consistent']:.3f} vs independent {f['independent_expectation']:.3f} | pred consistent "
                  f"{f['pred_consistent']:.3f}, pred mixes where ref is consistent {f['pred_mixes_when_ref_consistent']:.3f}")
        print("   wrong decisions per notation-only error:", {k: round(v, 3) for k, v in sorted(r_split["decisions_per_notation_error"].items())})
        print("   errors with >= 2 decisions:", {k: round(v, 3) for k, v in r_split["multi_decision_errors"].items()})
        print("   span types:", r_split["span_types_top"][:8])

    # ---- 4. consistency re-ranking on the n-best dumps
    for split in ("val", "test_im2latex", "test_oleehyo"):
        path = NB / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        base, rer, oracle = [], [], []
        changed = 0
        for r in rows:
            greedy = r["greedy"].split()
            key = inv(greedy)
            cands = {tuple(greedy): max((s for s, _ in r["hyps"]), default=0.0)}
            for s, h in r["hyps"]:
                h = tuple(h.split())
                if inv(list(h)) == key and h not in cands:
                    cands[h] = s
            def mixed(h):
                return sum(not consistent(v) for v in choices(list(h)).values() if len(v) >= 2)
            best = min(cands, key=lambda h: (mixed(h), -cands[h]))
            gt = r["gt"].split()
            base.append(greedy == gt)
            rer.append(list(best) == gt)
            oracle.append(any(list(h) == gt for h in cands))
            changed += list(best) != greedy
        d = np.array(rer, float) - np.array(base, float)
        lo, hi = boot_ci(d)
        res.setdefault("rerank", {})[split] = {"n": len(rows), "greedy_em": float(np.mean(base)),
                                               "consistency_rerank_em": float(np.mean(rer)), "delta_ci95": [lo, hi],
                                               "changed": changed, "oracle_same_rendering": float(np.mean(oracle))}
        print(f"== rerank {split}: greedy EM {np.mean(base):.3f} -> most consistent {np.mean(rer):.3f} "
              f"(delta {d.mean() * 100:+.2f} pp, 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]), changed {changed}, "
              f"oracle among same-rendering beams {np.mean(oracle):.3f}")
    (ROOT / "docs" / "probe_results" / "stage0_style.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
