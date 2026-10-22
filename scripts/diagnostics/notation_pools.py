"""(Paths below are those of the Windows laptop; edit them for another machine.)
Where is the remaining gain?  Decompose final_v2_s1 errors per split into notation-only (renders the same)
vs visible errors, with a conservative invisible-notation normalizer (``inv``, also used by notation_rerank.py)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Coding\Python\MathOCR\src")
from mathocr.metrics import normalized_levenshtein  # noqa: E402

BS = "\\"
SPACES = {BS + s for s in (",", ";", ":", "!", "quad", "qquad", "", "enspace", "thinspace", "hspace", "hfill")} | {"~"}
STYLE = {BS + s for s in ("displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits", "nolimits")}
SIZE = re.compile(r"^\\(left|right|big|Big|bigg|Bigg)(l|r|m)?(.*)$")
SYN = {BS + a: (BS + b if b[0].isalpha() or b in "{}" else b) for a, b in (
    ("cal", "mathcal"), ("to", "rightarrow"), ("le", "leq"), ("ge", "geq"), ("ne", "neq"), ("gets", "leftarrow"),
    ("lbrace", "{"), ("rbrace", "}"), ("vert", "|"), ("lvert", "|"), ("rvert", "|"), ("bf", "mathbf"), ("rm", "mathrm"),
    ("it", "mathit"), ("dots", "ldots"), ("cdots", "ldots"), ("colon", ":"))}
SYN[BS + "lbrace"], SYN[BS + "rbrace"] = BS + "{", BS + "}"


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


if __name__ == "__main__":
    E = Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\eval_untrimmed")
    cdm = json.loads(Path(r"C:\Coding\Python\MathOCR\runs\final_v2_s1\eval_untrimmed\cdm.json").read_text(encoding="utf-8"))
    print(f"{'split':14} {'EM':>6} {'EM_inv':>7} {'notation-only':>13} {'visible err':>11} {'tok-edit':>8} {'edit_inv':>8} {'CDM=1':>6}")
    for split in ("test_im2latex", "test_oleehyo", "ood_spe", "ood_sce", "ood_cpe"):
        rows = [json.loads(x) for x in (E / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        em = np.mean([r["pred"].split() == r["gt"].split() for r in rows])
        emi = np.mean([inv(r["pred"].split()) == inv(r["gt"].split()) for r in rows])
        ed = np.mean([normalized_levenshtein(r["pred"].split(), r["gt"].split()) for r in rows])
        edi = np.mean([normalized_levenshtein(inv(r["pred"].split()), inv(r["gt"].split())) for r in rows])
        print(f"{split:14} {em:6.3f} {emi:7.3f} {emi - em:13.3f} {1 - emi:11.3f} {ed:8.3f} {edi:8.3f} {cdm[split]['cdm_exp_rate']:6.3f}")
        cls = Counter()
        for r in rows:
            g, p = r["gt"].split(), r["pred"].split()
            if g == p or inv(g) != inv(p):
                continue
            strip = lambda s, drop: [t for t in s if t not in drop]  # noqa: E731
            if strip(g, {"{", "}"}) == strip(p, {"{", "}"}):
                cls["only {} grouping"] += 1
            elif strip(g, SPACES) == strip(p, SPACES):
                cls["only spacing"] += 1
            elif strip(strip(g, SPACES), {"{", "}"}) == strip(strip(p, SPACES), {"{", "}"}):
                cls["grouping+spacing"] += 1
            else:
                cls["sizes/style/synonyms"] += 1
        print("    notation-only classes (share of split):", {k: round(v / len(rows), 3) for k, v in cls.most_common()})
