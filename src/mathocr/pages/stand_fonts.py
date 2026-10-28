"""The page stand's font families: one per design (clones of Times, Helvetica, Courier, Computer Modern ... would
otherwise outnumber everything else), grouped by kind.  The Palatino designs are held out (``HELD_OUT``) for the
unseen-font test, as in the formula task.  Noto Serif / Noto Sans are absent on purpose: TeX Live ships them without
the Regular and Bold files (only Thin ... Black), and Medium is not a stand-in for Regular.  Every name is a
``family`` of ``font_catalog.json`` that has regular, bold and italic faces covering ``fonts.CHARSET`` (``check``
raises otherwise).
"""

from __future__ import annotations

from mathocr.pages.fonts import faces_by_family

SERIF = (
    "TeX Gyre Termes", "STIX Two Text", "TeX Gyre Schola", "TeX Gyre Bonum", "Latin Modern Roman", "CMU Concrete",
    "KpRoman", "Libertinus Serif", "EB Garamond", "Garamond Libre", "Crimson Pro", "Cochineal", "Libre Baskerville",
    "BaskervilleF", "Libre Bodoni", "Libre Caslon Text", "Gelasio", "Gentium Book Plus", "Droid Serif",
    "Source Serif Pro", "Spectral", "Bitter", "Merriweather", "IBM Plex Serif", "Roboto Serif", "Inria Serif",
    "XCharter", "Heuristica", "fbb", "Old Standard", "MFB Oldstyle", "Junicode", "Coelacanth", "Alegreya",
    "Antykwa Poltawskiego", "Antykwa Torunska", "Venturis ADF", "Ibarra Real Nova",
)
SANS = (
    "TeX Gyre Heros", "TeX Gyre Heros Cn", "TeX Gyre Adventor", "Latin Modern Sans", "CMU Bright", "KpSans",
    "Libertinus Sans", "Source Sans Pro", "Fira Sans", "IBM Plex Sans", "IBM Plex Sans Condensed",
    "Roboto", "Roboto Condensed", "Carlito", "Libre Franklin", "Merriweather Sans", "Alegreya Sans", "Inria Sans",
    "Atkinson Hyperlegible", "Arsenal", "Chivo", "Archiv0", "Cuprum", "Josefin Sans", "Montserrat", "Nunito",
    "Raleway", "Rosario", "Quattrocento Sans", "Ysabeau", "Iwona", "Kurier", "VenturisSans ADF", "Mint Spirit",
)
MONO = (
    "TeX Gyre Cursor", "CMU Typewriter Text", "KpMono", "UM Typewriter", "IBM Plex Mono", "Source Code Pro",
    "Fira Mono", "Roboto Mono", "JetBrains Mono", "DejaVu Sans Mono", "Go Mono", "Anonymous Pro",
)
HELD_OUT = ("TeX Gyre Pagella", "TeX Gyre PagellaX", "P052", "Domitian")  # Palatino and its clones
ROLES = ("regular", "bold", "italic")


def check(catalog: list[dict]) -> dict[str, dict[str, dict]]:
    """family -> {role: face} for every listed family; raises if one is missing a role or is listed twice."""

    names = SERIF + SANS + MONO + HELD_OUT
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"listed twice: {sorted(dup)}")
    fams = faces_by_family(catalog)
    bad = {n: sorted(set(ROLES) - set(fams.get(n, {}))) for n in names if not set(ROLES) <= set(fams.get(n, {}))}
    if bad:
        raise ValueError(f"families without regular/bold/italic covering the charset: {bad}")
    return {n: fams[n] for n in names}
