"""Text metrics, computed the way UniMERNet's ``test.py`` computes them.

* ``exact_match`` -- token sequences identical (strict; equivalent LaTeX
  spellings count as errors, which is what CDM exists to correct);
* ``edit_distance`` -- mean normalized character Levenshtein distance
  between ``normalize_text`` strings (lower is better), as UniMERNet's
  ``score_text``;
* ``bleu`` -- corpus BLEU-4 without smoothing over 13a-tokenized
  ``normalize_text`` strings: the HuggingFace ``evaluate`` "bleu" metric
  (tensorflow/nmt ``compute_bleu``), which UniMERNet calls.  Ported here so
  the evaluation has no ``evaluate`` dependency; ``tests/test_metrics.py``
  checks it against sacrebleu.
* ``token_edit_distance`` -- the same normalized distance on token lists,
  insensitive to how many characters a token spells.
"""

from __future__ import annotations

import collections
import math
import re
from collections.abc import Sequence

import editdistance
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

_TEXT_REG = re.compile(r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})")
_LETTER = "[a-zA-Z]"
_NOLETTER = r"[\W_^\d]"
_RE_NN = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (_NOLETTER, _NOLETTER))
_RE_NL = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (_NOLETTER, _LETTER))
_RE_LN = re.compile(r"(%s)\s+?(%s)" % (_LETTER, _NOLETTER))


def normalize_text(text: str) -> str:
    """UniMERNet ``test.py::normalize_text``: drop insignificant spaces."""

    names = [x[0].replace(" ", "") for x in _TEXT_REG.findall(text)]
    text = _TEXT_REG.sub(lambda _m: names.pop(0), text)
    news = text
    while True:
        text = news
        news = _RE_NN.sub(r"\1\2", text)
        news = _RE_NL.sub(r"\1\2", news)
        news = _RE_LN.sub(r"\1\2", news)
        if news == text:
            break
    return text


def normalized_levenshtein(a: Sequence, b: Sequence) -> float:
    """``rapidfuzz.Levenshtein.normalized_distance``: d / max(len)."""

    longest = max(len(a), len(b))
    return editdistance.eval(a, b) / longest if longest else 0.0


_TOKENIZER_13A = Tokenizer13a()


def _ngrams(tokens: Sequence[str], max_order: int) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(tokens) - order + 1):
            counts[tuple(tokens[i : i + order])] += 1
    return counts


def corpus_bleu(predictions: Sequence[str], references: Sequence[str], max_order: int = 4) -> float:
    """tensorflow/nmt ``compute_bleu`` (one reference, no smoothing)."""

    matches = [0] * max_order
    possible = [0] * max_order
    ref_len = pred_len = 0
    for pred, ref in zip(predictions, references, strict=True):
        p = _TOKENIZER_13A(pred).split()
        r = _TOKENIZER_13A(ref).split()
        ref_len += len(r)
        pred_len += len(p)
        overlap = _ngrams(p, max_order) & _ngrams(r, max_order)
        for ngram, count in overlap.items():
            matches[len(ngram) - 1] += count
        for order in range(1, max_order + 1):
            possible[order - 1] += max(0, len(p) - order + 1)
    precisions = [m / p if p > 0 else 0.0 for m, p in zip(matches, possible)]
    if min(precisions) <= 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    ratio = pred_len / ref_len if ref_len else 0.0
    bp = 1.0 if ratio > 1.0 else (math.exp(1 - 1.0 / ratio) if ratio > 0 else 0.0)
    return geo_mean * bp


def text_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """All text metrics for space-separated normalized LaTeX strings."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references differ in length")
    if not references:
        raise ValueError("no samples")
    norm_p = [normalize_text(p) for p in predictions]
    norm_r = [normalize_text(r) for r in references]
    n = len(references)
    return {
        "n": n,
        "exact_match": sum(p.split() == r.split() for p, r in zip(predictions, references)) / n,
        "edit_distance": sum(normalized_levenshtein(p, r) for p, r in zip(norm_p, norm_r)) / n,
        "token_edit_distance": sum(
            normalized_levenshtein(p.split(), r.split()) for p, r in zip(predictions, references)
        ) / n,
        "bleu": corpus_bleu(norm_p, norm_r),
    }
