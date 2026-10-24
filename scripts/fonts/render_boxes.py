"""Runs inside the mathocr-cdm image: every label of /work/labels.jsonl ({"id", "latex"}) rendered in the requested
fonts with each visible token in its own colour (CDM's tokenizer and colouring, token_add_color_RGB), in our
template (preview/tightpage, display style, the font packages of render.py), 300 dpi without anti-aliasing so the
colours are exact.  Per label and font:
  /work/png/<font>/<id>.png   -- grey ink coverage, 255 * min(R, G, B) / 255 (every colour has a zero channel, so a
                                 coloured core pixel is full ink and paper stays white);
  /work/box/<font>/<id>.json  -- {"tokens": CDM tokens, "boxes": [[x0, y0, x1, y1] | [] per token]} in pixels.

    docker run --rm --entrypoint python3 -e FONTS=cm,times -e NPROC=12 \
        -v <work>:/work -v <repo>/scripts/fonts:/scripts mathocr-cdm /scripts/render_boxes.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/opt/UniMERNet/cdm")
from modules.latex_processor import normalize_latex, token_add_color_RGB  # noqa: E402
from modules.tokenize_latex.tokenize_latex import tokenize_latex  # noqa: E402

WORK = Path("/work")
FONTS = {"cm": ("", ""), "times": (r"\usepackage{txfonts}", ""), "palatino": (r"\usepackage{pxfonts}", ""),
         "sans": (r"\usepackage{sansmath}", r"\sansmath")}
STEPS = range(0, 256, 15)
COLORS = [(r, g, b) for r in STEPS for g in STEPS for b in STEPS if min(r, g, b) == 0 and max(r, g, b) > 0]
DOC = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,upgreek}
\usepackage{xcolor}
%s
\usepackage[active,tightpage]{preview}\PreviewBorder=2pt
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{\protect\leavevmode\begingroup\color#1{#2}#3\endgroup}
\makeatother
\begin{preview}%s$\displaystyle %s $\end{preview}
\end{document}
"""


def colour(latex: str, tmp: str):
    ret, new = tokenize_latex(latex, middle_file=os.path.join(tmp, "tok.txt"))
    if not (ret and new):
        new = latex
    l_split = normalize_latex(new).strip().split(" ")
    tokens, idx = [], 0
    while idx < len(l_split):
        l_split, idx, tokens = token_add_color_RGB(l_split, idx, tokens)
    if len(tokens) > len(COLORS):
        return None, None
    out = " ".join(l_split)
    for k, (r, g, b) in enumerate(COLORS[: len(tokens)]):
        out = out.replace(f"<color_{k}>", f"{r},{g},{b}")
    return out, tokens


def render(job):
    uid, latex, font = job
    png, box = WORK / "png" / font / f"{uid}.png", WORK / "box" / font / f"{uid}.json"
    if png.exists() and box.exists():
        return {"id": uid, "ok": True}
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        coloured, tokens = colour(latex, d)
        if coloured is None:
            return {"id": uid, "ok": False}
        (Path(d) / "f.tex").write_text(DOC % (FONTS[font][0], FONTS[font][1], coloured), encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=d, capture_output=True, timeout=60)
        if r.returncode != 0:
            return {"id": uid, "ok": False}
        r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=png16m", "-r300", "-dTextAlphaBits=1",
                            "-dGraphicsAlphaBits=1", "-dFirstPage=1", "-dLastPage=1", "-o", "f.png", "f.pdf"],
                           cwd=d, capture_output=True, timeout=120)
        if r.returncode != 0:
            return {"id": uid, "ok": False}
        rgb = np.asarray(Image.open(Path(d) / "f.png").convert("RGB"))
        packed = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
        boxes = []
        for (cr, cg, cb) in COLORS[: len(tokens)]:
            ys, xs = np.nonzero(packed == ((cr << 16) | (cg << 8) | cb))
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(ys) else [])
        Image.fromarray(rgb.min(axis=2).astype(np.uint8), mode="L").save(png)
        box.write_text(json.dumps({"tokens": tokens, "boxes": boxes}), encoding="utf-8")
        return {"id": uid, "ok": True}
    except Exception:  # noqa: BLE001 -- one bad label must not stop the pool
        return {"id": uid, "ok": False}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    fonts = os.environ.get("FONTS", "cm").split(",")
    labels = [json.loads(x) for x in (WORK / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    with Pool(int(os.environ.get("NPROC", "8"))) as pool:
        for font in fonts:
            (WORK / "png" / font).mkdir(parents=True, exist_ok=True)
            (WORK / "box" / font).mkdir(parents=True, exist_ok=True)
            with open(WORK / f"render_{font}.jsonl", "w") as log:
                for i, res in enumerate(pool.imap_unordered(render, [(x["id"], x["latex"], font) for x in labels], chunksize=8)):
                    log.write(json.dumps(res) + "\n")
                    if i % 500 == 0:
                        log.flush()
                        print(font, i, "/", len(labels), flush=True)
    print("done", flush=True)
