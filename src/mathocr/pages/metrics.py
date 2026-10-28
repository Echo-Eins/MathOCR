"""Reading metrics of a page transcript against its reference (tasks/page_ocr_plan.md section 2).

``page_metrics(pred, ref, ref_y=None)`` on ``\\n``-separated texts:

* ``cer`` / ``wer`` -- edit distance over characters / whitespace-separated words of the whole texts, divided by the
  reference length (the transcript's line breaks count as characters, so a merged or split line costs one edit);
* line assignment: predicted lines are matched one-to-one to reference lines by the Hungarian method on the
  normalized edit distance (NED = edit / max length) of the two lines.  A reference line whose match has
  NED <= ``read_ned`` is *read*; any other reference line is *skipped*.  A predicted line not matched within
  ``read_ned`` is a *repeat* when it is within ``repeat_ned`` of a reference line that another predicted line
  already reads, else *extra* (hallucinated or garbled);
* ``order_tau`` -- Kendall's tau between the order of the read reference lines and the order of the predicted
  lines that read them (1 = the reading order is kept);
* ``line_ned`` -- per reference line, its match's NED (1.0 if skipped); with ``ref_y`` (each reference line's
  vertical position, 0 = page top .. 1 = bottom) also ``ned_by_height``: the mean over ten height bins, which shows
  a reader that loses its place lower on the page.

Matching is not monotone on purpose: a monotone alignment cannot see two lines read in swapped order.
"""

from __future__ import annotations

import editdistance
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau


def ned(a: str, b: str) -> float:
    n = max(len(a), len(b))
    return editdistance.eval(a, b) / n if n else 0.0


def page_metrics(pred: str, ref: str, ref_y: list[float] | None = None, read_ned: float = 0.5,
                 repeat_ned: float = 0.2) -> dict:
    p_lines = [l for l in pred.split("\n") if l.strip()]
    r_lines = [l for l in ref.split("\n") if l.strip()]
    out = {
        "cer": editdistance.eval(pred, ref) / max(len(ref), 1),
        "wer": editdistance.eval(pred.split(), ref.split()) / max(len(ref.split()), 1),
        "ref_lines": len(r_lines), "pred_lines": len(p_lines),
    }
    cost = np.ones((len(p_lines), len(r_lines)))
    for i, p in enumerate(p_lines):
        for j, r in enumerate(r_lines):
            cost[i, j] = ned(p, r)
    rows, cols = linear_sum_assignment(cost) if cost.size else (np.zeros(0, int), np.zeros(0, int))
    match_of_ref = {int(j): int(i) for i, j in zip(rows, cols) if cost[i, j] <= read_ned}
    line_ned = [float(cost[match_of_ref[j], j]) if j in match_of_ref else 1.0 for j in range(len(r_lines))]
    read_by = set(match_of_ref.values())
    repeats = extra = 0
    for i in range(len(p_lines)):
        if i in read_by:
            continue
        near = [j for j in range(len(r_lines)) if cost[i, j] <= repeat_ned and j in match_of_ref]
        if near:
            repeats += 1
        else:
            extra += 1
    out.update({
        "read": len(match_of_ref), "skipped": len(r_lines) - len(match_of_ref), "repeats": repeats, "extra": extra,
        "line_ned": line_ned,
    })
    pairs = sorted(match_of_ref.items())  # (ref index, pred index) in reference order
    if len(pairs) >= 2:
        tau = kendalltau([j for j, _ in pairs], [i for _, i in pairs]).statistic
        out["order_tau"] = float(tau) if np.isfinite(tau) else 1.0
    else:
        out["order_tau"] = 1.0
    if ref_y is not None:
        if len(ref_y) != len(r_lines):
            raise ValueError("ref_y needs one position per non-empty reference line")
        bins = np.clip((np.asarray(ref_y) * 10).astype(int), 0, 9)
        out["ned_by_height"] = [float(np.mean([line_ned[j] for j in range(len(r_lines)) if bins[j] == b]))
                                if (bins == b).any() else None for b in range(10)]
    return out
