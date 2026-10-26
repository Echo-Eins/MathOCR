"""Where multi-line formulas lose (CPE): by the number of rows of the reference -- error, exact match, whether the
answer has as many rows; for answers with the right row count, the error of each row (aligned by index), so a
lost or merged row is told apart from errors inside rows; the alignment column and the array specification.

    python scripts/diagnostics/multiline_errors.py runs/final_v3_s1 --split ood_cpe --out docs/probe_results/multiline_final_v3_s1.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import editdistance


def edit(a: str, b: str) -> float:  # UniMERNet's protocol: no whitespace, characters, / max(len)
    a, b = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    n = max(len(a), len(b))
    return editdistance.eval(a, b) / n if n else 0.0


def rows(tokens: list[str]) -> tuple[str, list[list[str]]] | None:
    """(array specification, top-level rows) of a ``\\begin{array} {spec} ... \\end{array}`` answer, else None."""

    if len(tokens) < 4 or tokens[0] != "\\begin{array}" or tokens[1] != "{":
        return None
    j = tokens.index("}", 2)
    spec, body = " ".join(tokens[2:j]), tokens[j + 1 : -1 if tokens[-1] == "\\end{array}" else len(tokens)]
    out, cur, depth = [], [], 0
    for t in body:
        if t == "{" or t.startswith("\\begin{"):
            depth += 1
        elif t == "}" or t.startswith("\\end{"):
            depth -= 1
        if t == "\\\\" and depth == 0:
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return spec, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", default="ood_cpe")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = [json.loads(l) for l in (args.run / "eval" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by_rows = defaultdict(lambda: {"n": 0, "edit": 0.0, "em": 0, "same_rows": 0, "spec_ok": 0, "pred_is_array": 0,
                                   "row_edit": defaultdict(list), "fewer": 0, "more": 0})
    single = {"n": 0, "edit": 0.0, "em": 0, "pred_is_array": 0}
    for r in data:
        g, p = r["gt"].split(), r["pred"].replace("<noeos>", "").split()
        gr, pr = rows(g), rows(p)
        e = edit(r["pred"].replace("<noeos>", ""), r["gt"])
        if gr is None:
            single["n"] += 1
            single["edit"] += e
            single["em"] += p == g
            single["pred_is_array"] += pr is not None
            continue
        k = min(len(gr[1]), 9)
        b = by_rows[k]
        b["n"] += 1
        b["edit"] += e
        b["em"] += p == g
        if pr is None:
            continue
        b["pred_is_array"] += 1
        b["spec_ok"] += pr[0] == gr[0]
        if len(pr[1]) == len(gr[1]):
            b["same_rows"] += 1
            for i, (a, c) in enumerate(zip(pr[1], gr[1])):
                b["row_edit"][min(i, 5)].append(edit(" ".join(a), " ".join(c)))
        elif len(pr[1]) < len(gr[1]):
            b["fewer"] += 1
        else:
            b["more"] += 1
    report = {"split": args.split, "n": len(data),
              "single_line": {**single, "edit": single["edit"] / max(1, single["n"]), "em": single["em"] / max(1, single["n"])}}
    table = {}
    for k in sorted(by_rows):
        b = by_rows[k]
        n = b["n"]
        table[f"{k}{'+' if k == 9 else ''} rows"] = {
            "n": n, "edit": round(b["edit"] / n, 4), "em": round(b["em"] / n, 4),
            "answer_is_array": round(b["pred_is_array"] / n, 3), "same_row_count": round(b["same_rows"] / n, 3),
            "fewer_rows": round(b["fewer"] / n, 3), "more_rows": round(b["more"] / n, 3),
            "spec_ok": round(b["spec_ok"] / n, 3),
            "row_edit_when_rows_match": {f"row {i + 1}{'+' if i == 5 else ''}": round(sum(v) / len(v), 4)
                                         for i, v in sorted(b["row_edit"].items())}}
    report["by_reference_rows"] = table
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["single_line"]))
    for k, v in table.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
