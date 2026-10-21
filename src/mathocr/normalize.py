"""Raw LaTeX -> im2markup-normalized token string (the label convention).

Python half of UniMERNet's ``cdm/modules/tokenize_latex/tokenize_latex.py``
around the vendored KaTeX normalizer; the node half runs as one streaming
process for a whole batch of formulas (it answers one line per input line,
an empty line when KaTeX cannot parse the formula).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

NORMALIZER_JS = Path(__file__).resolve().parents[2] / "third_party" / "latex_normalizer" / "preprocess_formula.js"

_ENV_TO_ALIGNED = re.compile(r"\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}", re.S)
_SMALLMATRIX = re.compile(r"\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}", re.S)
_OPERATORS = r"\s?".join(
    "|".join(["arccos", "arcsin", "arctan", "arg", "cos", "cosh", "cot", "coth", "csc", "deg", "det", "dim",
              "exp", "gcd", "hom", "inf", "injlim", "ker", "lg", "lim", "liminf", "limsup", "ln", "log", "max",
              "min", "Pr", "projlim", "sec", "sin", "sinh", "sup", "tan", "tanh"])
)
_OPERATORNAME = re.compile(r"\\operatorname {(%s)}" % _OPERATORS)


def _pre(latex: str) -> str:
    latex = _ENV_TO_ALIGNED.sub(r"\\begin{aligned}\2\\end{aligned}", latex)
    latex = _SMALLMATRIX.sub(r"\\begin{matrix}\2\\end{matrix}", latex)
    # The node side reads one formula per line.
    return " ".join(latex.split())


def _post(normalized: str) -> str:
    post = " ".join(normalized.split())
    post = _OPERATORNAME.sub(lambda m: "\\" + m.group(1).replace(" ", ""), post)
    return post.replace(r"\\ \end{array}", r"\end{array}")


def normalize_batch(formulas: Sequence[str], timeout_s: float = 3600.0) -> list[str | None]:
    """Normalize every formula; ``None`` where KaTeX rejects it."""

    if not formulas:
        return []
    # Bytes with bare "\n", not text mode: on Windows text mode sends "\r\n", and node's readline
    # counts a "\r" | "\n" pair split across two pipe reads as two line ends (extra output lines).
    payload = "".join(_pre(f) + "\n" for f in formulas).encode("utf-8")
    proc = subprocess.run(
        ["node", str(NORMALIZER_JS), "normalize"],
        input=payload, capture_output=True, timeout=timeout_s, check=True,
    )
    lines = proc.stdout.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(formulas):
        raise RuntimeError(f"normalizer returned {len(lines)} lines for {len(formulas)} formulas")
    return [(_post(line) or None) if line.strip() else None for line in lines]


# ----------------------------------------------------------------- display envs

_DISPLAY = re.compile(r"^\s*\\begin\{(align|gather|equation|multline)(\*?)\}(.*)\\end\{\1\2\}\s*$", re.S)
_SCAN = re.compile(r"\\begin\{[^}]*\}|\\end\{[^}]*\}|\\\\|\\.|[{}&]|[^\\{}&]+", re.S)
_INVISIBLE = re.compile(r"\\(nonumber|notag)\b|\\label\{[^}]*\}")
_TRAILING_BREAKS = re.compile(r"(\\\\\s*)+$")


def _top_level(inner: str) -> tuple[int, int, bool]:
    """(line breaks, alignment marks, balanced) at nesting depth zero.

    A trailing ``\\\\`` with nothing after it is not a line break.
    """

    depth = breaks = amps = 0
    pending_break = False
    for m in _SCAN.finditer(inner):
        tok = m.group(0)
        if tok.startswith("\\begin{") or tok == "{":
            depth += 1
        elif tok.startswith("\\end{") or tok == "}":
            depth -= 1
            if depth < 0:
                return breaks, amps, False
        if depth != 0:
            continue
        if pending_break and not tok.isspace():
            breaks += 1
            pending_break = False
        if tok == "\\\\":
            pending_break = True
        elif tok == "&":
            amps += 1
    return breaks, amps, depth == 0


def _drop_top_level_amp(inner: str) -> str:
    depth, parts = 0, []
    for m in _SCAN.finditer(inner):
        tok = m.group(0)
        if tok.startswith("\\begin{") or tok == "{":
            depth += 1
        elif tok.startswith("\\end{") or tok == "}":
            depth -= 1
        parts.append(" " if tok == "&" and depth == 0 else tok)
    return "".join(parts).strip()


def unwrap_display(raw: str) -> str | None:
    """arXiv display environment -> what the image shows, or ``None`` to drop.

    A single-line ``align*``/``gather*``/``equation*`` is just its formula: the
    environment is invisible, and the normalizer would otherwise turn it into a
    ``\\begin{array}`` the image does not show; a lone alignment mark on that
    line is dropped for the same reason.  A multi-line ``align*``
    becomes ``aligned`` (same rendering) and ``gather*`` a centered
    ``array``.  Numbered (unstarred) environments render an equation number
    that is not in the label, so they are dropped, as are ``\\tag`` and
    ``multline`` (no KaTeX equivalent).
    """

    if "%" in raw or "\\tag" in raw:
        return None
    m = _DISPLAY.match(raw)
    if not m:
        return raw.strip() or None
    env, star, inner = m.groups()
    if not star or env == "multline":
        return None
    inner = _TRAILING_BREAKS.sub("", _INVISIBLE.sub(" ", inner)).strip()
    breaks, amps, balanced = _top_level(inner)
    if not balanced or not inner:
        return None
    if breaks == 0 and amps == 0:
        return inner
    if breaks == 0 and amps == 1:
        # One line, one alignment point: ``a &= b`` renders as ``a = b``.
        return _drop_top_level_amp(inner)
    if env == "align":
        return f"\\begin{{aligned}} {inner} \\end{{aligned}}"
    if env == "gather" and amps == 0:
        return f"\\begin{{array}}{{c}} {inner} \\end{{array}}"
    return None
