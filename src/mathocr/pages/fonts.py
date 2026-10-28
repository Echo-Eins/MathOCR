"""Font catalog for the page stand: which files are usable text faces, and their metrics.

    python -m mathocr.pages.fonts --root D:/MathOCR/pages/fonts_raw --out D:/MathOCR/pages/font_catalog.json

Every ``.otf`` / ``.ttf`` under ``root`` is read with fontTools: family and style names, weight, italic flag,
fixed pitch, units per em, cap height and x-height (OS/2, else measured from the glyph outlines of ``H`` / ``x``),
ascender / descender, and the characters of ``CHARSET`` it lacks.  A face is *usable* when it covers the whole
charset, is a static font (no ``fvar``) and has cap and x heights; ``faces_by_family`` groups usable faces into the
regular / bold / italic / bold-italic roles the page renderer draws with.  Which families the stand actually uses
is an explicit allowlist (``scripts/pages/stand_fonts.py``), not a heuristic: display, script and symbol faces also
cover Latin.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

# Printable ASCII, the typographic quotes / dashes / ellipsis of edited prose, and the Latin-1 letters of names.
CHARSET = (
    "".join(chr(c) for c in range(0x20, 0x7F))
    + "\u2018\u2019\u201c\u201d\u2013\u2014\u2026"
    + "àáâäãåçèéêëìíîïñòóôöõøùúûüýÿßæœ"
    + "ÀÁÂÄÃÅÇÈÉÊËÌÍÎÏÑÒÓÔÖÕØÙÚÛÜÝÆŒ"
    + "°£€×"
)


def _name(font: TTFont, *ids: int) -> str:
    table = font["name"]
    for i in ids:
        rec = table.getDebugName(i)
        if rec:
            return rec.strip()
    return ""


def _glyph_height(font: TTFont, char: str) -> float | None:
    cmap = font.getBestCmap() or {}
    glyph = cmap.get(ord(char))
    if glyph is None:
        return None
    pen = BoundsPen(font.getGlyphSet())
    font.getGlyphSet()[glyph].draw(pen)
    if pen.bounds is None:
        return None
    return float(pen.bounds[3])  # top of the glyph above the baseline


def describe(path: Path) -> dict:
    font = TTFont(path, lazy=True)
    os2 = font["OS/2"] if "OS/2" in font else None
    head, hhea = font["head"], font["hhea"]
    cmap = font.getBestCmap() or {}
    style = _name(font, 17, 2)
    italic = bool(os2 and (os2.fsSelection & 1)) or bool(head.macStyle & 2) or any(
        w in style.lower() for w in ("italic", "oblique", "slanted"))
    cap = getattr(os2, "sCapHeight", 0) if os2 else 0
    xh = getattr(os2, "sxHeight", 0) if os2 else 0
    cap = cap or _glyph_height(font, "H") or 0
    xh = xh or _glyph_height(font, "x") or 0
    missing = [c for c in CHARSET if ord(c) not in cmap]
    variable = "fvar" in font
    out = {
        "path": str(path),
        "family": _name(font, 16, 1),
        "style": style,
        "full": _name(font, 4),
        "weight": int(os2.usWeightClass) if os2 else 400,
        "width": int(os2.usWidthClass) if os2 else 5,  # 5 = normal; condensed < 5 < expanded
        "italic": italic,
        "fixed_pitch": bool(font["post"].isFixedPitch) if "post" in font else False,
        "upem": int(head.unitsPerEm),
        "cap_height": float(cap),
        "x_height": float(xh),
        "ascender": int(hhea.ascent),
        "descender": int(hhea.descent),
        "variable": variable,
        "missing": "".join(missing),
    }
    out["usable"] = not missing and not variable and cap > 0 and xh > 0
    return out


_STYLE_ROLE = {
    "regular": "regular", "roman": "regular", "book": "regular", "normal": "regular", "text": "regular",
    "upright": "regular", "italic": "italic", "oblique": "italic", "book italic": "italic", "text italic": "italic",
    "bold": "bold", "bold italic": "bolditalic", "bold oblique": "bolditalic", "bolditalic": "bolditalic",
}


def canonical_role(face: dict) -> str | None:
    """The role its style name states outright ("Bold", "Italic", "Regular", ...), else None."""

    return _STYLE_ROLE.get(" ".join(face["style"].lower().replace("-", " ").split()))


def role(face: dict) -> str | None:
    """regular / bold / italic / bolditalic: from the style name when it is a canonical one (OS/2 weights are not
    always standard -- JetBrains Mono Bold says 558), else from the weight; None for light, black and the like."""

    named = canonical_role(face)
    if named:
        return named
    w = face["weight"]
    if 350 <= w <= 500:
        return "italic" if face["italic"] else "regular"
    if 600 <= w <= 800:
        return "bolditalic" if face["italic"] else "bold"
    return None


def faces_by_family(catalog: list[dict]) -> dict[str, dict[str, dict]]:
    """family -> {role: face}.  Per role the best face wins: normal width first (a family name can also cover its
    condensed cut, e.g. ``Junicode Cond``), then a style name that states the role, then the weight nearest 400
    (regular / italic) or 700 (bold), then path order -- deterministic."""

    best: dict[str, dict[str, tuple]] = defaultdict(dict)
    for face in catalog:
        r = role(face)
        if not (face["usable"] and r):
            continue
        key = (face.get("width", 5) != 5, canonical_role(face) != r,
               abs(face["weight"] - (700 if r.startswith("bold") else 400)), face["path"])
        if r not in best[face["family"]] or key < best[face["family"]][r][0]:
            best[face["family"]][r] = (key, face)
    return {fam: {r: kf[1] for r, kf in roles.items()} for fam, roles in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    catalog, errors = [], []
    for path in sorted(p for p in args.root.rglob("*") if p.suffix.lower() in (".otf", ".ttf")):
        try:
            catalog.append(describe(path))
        except Exception as e:  # a broken file is recorded and skipped, never guessed at
            errors.append({"path": str(path), "error": f"{type(e).__name__}: {e}"})
    args.out.write_text(json.dumps({"charset": CHARSET, "faces": catalog, "errors": errors}, indent=1, ensure_ascii=False),
                        encoding="utf-8", newline="\n")
    fams = faces_by_family(catalog)
    full = [f for f, r in fams.items() if {"regular", "bold", "italic"} <= set(r)]
    print(json.dumps({"files": len(catalog), "errors": len(errors), "usable": sum(f["usable"] for f in catalog),
                      "families": len(fams), "families_with_regular_bold_italic": len(full)}))


if __name__ == "__main__":
    main()
