"""Handwriting labels in the two conventions of the formula task.

* ``ours`` -- the normalizer of the main build (``mathocr.normalize.normalize_batch``, KaTeX / im2markup); its comment
  handling cuts a label at ``%`` (even ``\\%``), so such labels are dropped;
* ``unimer`` -- the published UniMER labels are space-separated tokens; HME100K and CROHME labels already are, a
  MathWriting ``normalizedLabel`` (no spaces) is split by ``lex``.  ``\\begin{matrix}``-type environments have no
  UniMER token (its vocabulary has ``\\begin{array}`` only), so labels with an environment are dropped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from mathocr.normalize import normalize_batch
from mathocr.tokenizer import Vocab, tokenize

_LEX = re.compile(r"\\operatorname\*|\\[A-Za-z]+|\\.|\S")


def lex(latex: str) -> str:
    """``\\frac{dy}{dt}`` -> ``\\frac { d y } { d t }``: commands, escaped characters and single characters."""

    return " ".join(_LEX.findall(latex))


def convert(labels: Sequence[str], convention: str, spaced: bool) -> list[str | None]:
    """Labels of one source in ``convention``; ``spaced`` -- the source is already space-tokenized."""

    if convention == "ours":
        keep = [l if "%" not in l else "" for l in labels]
        out = normalize_batch([k or "x" for k in keep])
        return [o if k else None for k, o in zip(keep, out)]
    if convention == "unimer":
        return [(" ".join(l.split()) if spaced else lex(l)) if "\\begin" not in l else None for l in labels]
    raise ValueError(convention)


def usable(label: str | None, vocab: Vocab, max_tokens: int) -> bool:
    if not label:
        return False
    toks = tokenize(label)
    return 0 < len(toks) <= max_tokens and all(t in vocab for t in toks)
