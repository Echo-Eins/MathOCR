"""UniMER-Test predictions of runs scored exactly as UniMERNet's ``test.py`` (opendatalab/UniMERNet, main):
``normalize_text`` on predictions and references (pix2tex whitespace rules: spaces go except between two letters;
``\\operatorname`` / ``\\mathrm`` / ``\\text`` / ``\\mathbf`` arguments lose theirs), then corpus BLEU as HF
``evaluate.load("bleu")`` computes it (tokenizer 13a, max order 4, no smoothing: the TensorFlow NMT
``compute_bleu``) and the mean ``Levenshtein.normalized_distance`` (distance / max length).

    python scripts/diagnostics/unimer_protocol.py runs/final_v4_s1 [more runs]

Reads ``<run>/eval/<split>.jsonl`` for the UniMER-Test splits (and ``test_mathwriting`` when present).  This
replaces ``unimer_edit.py``'s approximation (all whitespace removed), which is kept for the older numbers.
"""

from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

import editdistance
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

SPLITS = ("ood_spe", "ood_cpe", "ood_sce", "ood_hwe", "test_mathwriting")
_TEXT = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
_LETTER = "[a-zA-Z]"
_NOLETTER = r"[\W_^\d]"
_TOK = Tokenizer13a()


def normalize_text(text: str) -> str:
    names = [x[0].replace(" ", "") for x in re.findall(_TEXT, text)]
    text = re.sub(_TEXT, lambda m: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (_NOLETTER, _NOLETTER), r"\1\2", text)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (_NOLETTER, _LETTER), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (_LETTER, _NOLETTER), r"\1\2", news)
        if news == text:
            break
    return text


def _ngrams(tokens: list[str], n: int) -> collections.Counter:
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(preds: list[str], refs: list[str], max_order: int = 4) -> float:
    """TensorFlow NMT ``compute_bleu`` (what HF ``evaluate``'s bleu wraps), one reference per prediction."""

    matches, possible = [0] * max_order, [0] * max_order
    pred_len = ref_len = 0
    for p, r in zip(preds, refs):
        pt, rt = _TOK(p).split(), _TOK(r).split()
        pred_len += len(pt)
        ref_len += len(rt)
        for n in range(1, max_order + 1):
            pc, rc = _ngrams(pt, n), _ngrams(rt, n)
            matches[n - 1] += sum((pc & rc).values())
            possible[n - 1] += max(len(pt) - n + 1, 0)
    precisions = [m / p if p > 0 else 0.0 for m, p in zip(matches, possible)]
    if min(precisions) <= 0:
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precisions) / max_order)
    ratio = pred_len / ref_len if ref_len else 0.0
    bp = 1.0 if ratio > 1.0 else (math.exp(1 - 1.0 / ratio) if ratio > 0 else 0.0)
    return geo * bp


def edit(preds: list[str], refs: list[str]) -> float:
    d = [editdistance.eval(p, r) / max(len(p), len(r)) if max(len(p), len(r)) else 0.0 for p, r in zip(preds, refs)]
    return sum(d) / len(d)


def score(rows: list[dict]) -> dict:
    preds = [normalize_text(r["pred"].replace("<noeos>", "")) for r in rows]
    refs = [normalize_text(r["gt"]) for r in rows]
    return {"n": len(rows), "bleu": round(bleu(preds, refs), 4), "edit": round(edit(preds, refs), 4)}


def main() -> None:
    for run in map(Path, sys.argv[1:]):
        out = {}
        for split in SPLITS:
            path = run / "eval" / f"{split}.jsonl"
            if path.exists():
                out[split] = score([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])
        print(json.dumps({"run": run.name, **out}))


if __name__ == "__main__":
    main()
