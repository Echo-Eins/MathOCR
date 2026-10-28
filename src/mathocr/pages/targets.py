"""Cell-level structure targets of a stand page (tasks/page_ocr_plan.md 5a, test 1: does line knowledge reach
every memory cell?).

``cell_targets(layout, stride)`` maps the layout onto the encoder's memory grid -- cell (i, j) covers pixels
``[i*sy, (i+1)*sy) x [j*sx, (j+1)*sx)`` of the page as drawn (the model sees that image unresampled) -- and returns
per-cell arrays.  A cell is a *text cell* of line k when its centre lies in line k's cap band (baseline - cap ..
baseline, from its first character's start to the end of its last advance); bands of different lines never share a
cell centre (checked: an overlap raises instead of being overwritten).  For text cells:

  line          reading-order index of the line on the page (0 = first line read)
  n_lines       lines on the page
  column        column of the line (-1: spans the text area -- header, title, footnote, footer)
  line_in_col   index of the line among the lines of its column (and role group for column -1)
  n_lines_col   number of lines in that group
  line_chars    characters of the line (spaces included)
  chars_before  characters starting at or left of the cell centre (0 = left of the first character)
  chars_after   line_chars - chars_before
  rel_x         (cell centre - line start) / line width, in [0, 1]
  dist_start    pixels from the line start to the cell centre
  dist_end      pixels from the cell centre to the line end
  cap           cap height of the line in pixels
  role          index into ``synth.ROLES``

Other cells hold -1 (integers) or nan (floats); ``mask`` marks the text cells.
"""

from __future__ import annotations

import numpy as np

from mathocr.pages.synth import ROLES

INT_KEYS = ("line", "n_lines", "column", "line_in_col", "n_lines_col", "line_chars", "chars_before", "chars_after", "role")
FLOAT_KEYS = ("rel_x", "dist_start", "dist_end", "cap")


def cell_targets(layout: dict, stride: tuple[int, int]) -> dict[str, np.ndarray]:
    sy, sx = stride
    h, w = layout["h"], layout["w"]
    if h % sy or w % sx:
        raise ValueError(f"page {w}x{h} is not a whole number of {sy}x{sx} cells")
    gh, gw = h // sy, w // sx
    yc = (np.arange(gh) + 0.5) * sy
    xc = (np.arange(gw) + 0.5) * sx
    out = {k: np.full((gh, gw), -1, np.int32) for k in INT_KEYS}
    out.update({k: np.full((gh, gw), np.nan, np.float32) for k in FLOAT_KEYS})
    lines = layout["lines"]
    groups: dict[tuple, list[int]] = {}
    for k, l in enumerate(lines):
        key = (l["column"],) if l["column"] >= 0 else (-1, l["role"])
        groups.setdefault(key, []).append(k)
    for k, l in enumerate(lines):
        key = (l["column"],) if l["column"] >= 0 else (-1, l["role"])
        start, end = float(l["char_x"][0]), float(l["box"][2])
        rows = np.flatnonzero((yc >= l["baseline"] - l["cap"]) & (yc <= l["baseline"]))
        cols = np.flatnonzero((xc >= start) & (xc < end))
        if not len(rows) or not len(cols):
            continue  # a line thinner than one cell (tiny text at a coarse stride) owns no cell centre
        sub = np.ix_(rows, cols)
        if (out["line"][sub] >= 0).any():
            raise ValueError(f"cap bands of lines {np.unique(out['line'][sub][out['line'][sub] >= 0])} and {k} share cells")
        xs = xc[cols]
        before = np.searchsorted(np.asarray(l["char_x"], np.float64), xs, side="right")
        vals = {
            "line": k, "n_lines": len(lines), "column": l["column"], "line_in_col": groups[key].index(k),
            "n_lines_col": len(groups[key]), "line_chars": len(l["text"]), "role": ROLES.index(l["role"]),
            "cap": l["cap"],
        }
        for name, v in vals.items():
            out[name][sub] = v
        out["chars_before"][sub] = before[None, :]
        out["chars_after"][sub] = (len(l["text"]) - before)[None, :]
        out["rel_x"][sub] = ((xs - start) / max(end - start, 1e-6))[None, :]
        out["dist_start"][sub] = (xs - start)[None, :]
        out["dist_end"][sub] = (end - xs)[None, :]
    out["mask"] = out["line"] >= 0
    return out
