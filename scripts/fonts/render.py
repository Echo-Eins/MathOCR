"""Runs inside the mathocr-render image (scripts/fonts/Dockerfile.render: the CDM image's TeX Live plus
texlive-fonts-extra, LuaLaTeX and OpenType math fonts; the older fonts also render in mathocr-cdm):

    docker run --rm --entrypoint python3 -e FONTS=times,sans -e NPROC=12 \
        -v <work>:/work -v <repo>/scripts/fonts:/scripts mathocr-render /scripts/render.py

Renders the labels of /work/labels.jsonl (or /work/$LABELS) ({"id", "latex"[, "font"][, "dpi"]}) with pdflatex or LuaLaTeX
(display style, page = formula box via preview/tightpage) and rasterizes with Ghostscript: /work/png/<font>/<id>.png.
A row without "font" is rendered in every font of FONTS (default times,sans); a row with "font" only in that one,
at its "dpi" (default 300).  Re-runnable: existing images are kept.  /work/render_<font>.jsonl logs which compiled.
"""

import json
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

WORK = Path("/work")
OTF = r"""\usepackage{unicode-math}\setmathfont{%s}"""
FONTS = {  # name: (engine, preamble, switch inside the formula box)
    "cm": ("pdflatex", "", ""),
    "times": ("pdflatex", r"\usepackage{txfonts}", ""),
    "palatino": ("pdflatex", r"\usepackage{pxfonts}", ""),  # held out: the fontswap test font
    "sans": ("pdflatex", r"\usepackage{sansmath}", r"\sansmath"),
    # texlive-fonts-extra (mathocr-render)
    "lmodern": ("pdflatex", r"\usepackage{lmodern}", ""),
    "mlmodern": ("pdflatex", r"\usepackage{mlmodern}", ""),
    "newtx": ("pdflatex", r"\usepackage{newtxtext,newtxmath}", ""),
    "mathptmx": ("pdflatex", r"\usepackage{mathptmx}", ""),
    "stix2": ("pdflatex", r"\usepackage{stix2}", ""),
    "libertinus": ("pdflatex", r"\usepackage{libertinus}", ""),
    "fourier": ("pdflatex", r"\usepackage{fourier}", ""),
    "kpfonts": ("pdflatex", r"\usepackage{kpfonts}", ""),
    "charter": ("pdflatex", r"\usepackage[charter]{mathdesign}", ""),
    "euler": ("pdflatex", r"\usepackage{eulervm}", ""),
    "concrete": ("pdflatex", r"\usepackage{concmath}", ""),
    "cmbright": ("pdflatex", r"\usepackage{cmbright}", ""),
    "arev": ("pdflatex", r"\usepackage{arev}", ""),
    "noto": ("pdflatex", r"\usepackage{notomath}", ""),
    "schoolbook": ("pdflatex", r"\usepackage{fouriernc}", ""),
    "cochineal": ("pdflatex", r"\usepackage{cochineal}\usepackage[cochineal]{newtxmath}", ""),
    "xcharter": ("pdflatex", r"\usepackage{XCharter}\usepackage[xcharter]{newtxmath}", ""),
    # OpenType math fonts (LuaLaTeX + unicode-math)
    "otf_lm": ("lualatex", OTF % "Latin Modern Math", ""),
    "otf_termes": ("lualatex", OTF % "TeX Gyre Termes Math", ""),
    "otf_bonum": ("lualatex", OTF % "TeX Gyre Bonum Math", ""),
    "otf_schola": ("lualatex", OTF % "TeX Gyre Schola Math", ""),
    "otf_dejavu": ("lualatex", OTF % "TeX Gyre DejaVu Math", ""),
    "otf_stix2": ("lualatex", OTF % "STIX Two Math", ""),
    "otf_xits": ("lualatex", OTF % "XITS Math", ""),
    "otf_libertinus": ("lualatex", OTF % "Libertinus Math", ""),
    "otf_fira": ("lualatex", OTF % "Fira Math", ""),
    "otf_asana": ("lualatex", OTF % "Asana Math", ""),
    "otf_garamond": ("lualatex", OTF % "Garamond-Math", ""),
    "otf_newcm": ("lualatex", OTF % "NewCMMath-Regular.otf", ""),
    "otf_pagella": ("lualatex", OTF % "TeX Gyre Pagella Math", ""),  # Palatino family: held out with "palatino"
}
DOC = r"""\documentclass[12pt]{article}
\usepackage{amsmath%s}
%s
\usepackage[active,tightpage]{preview}\PreviewBorder=2pt
\begin{document}
\begin{preview}%s$\displaystyle %s $\end{preview}
\end{document}
"""


def render(job):
    uid, latex, font, dpi = job
    out = WORK / "png" / font / f"{uid}.png"
    if out.exists():
        return {"id": uid, "ok": True}
    engine, preamble, switch = FONTS[font]
    extra = ",amssymb,upgreek" if engine == "pdflatex" else ""  # unicode-math carries these symbols itself
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        (Path(d) / "f.tex").write_text(DOC % (extra, preamble, switch, latex), encoding="utf-8")
        r = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=d,
                           capture_output=True, timeout=120)
        if r.returncode != 0 or not (Path(d) / "f.pdf").exists():
            return {"id": uid, "ok": False}
        # a font without some glyph compiles anyway and leaves a blank in its place (Asana: \int, minus):
        # the image would contradict its label
        if "Missing character" in (Path(d) / "f.log").read_text(encoding="latin-1"):
            return {"id": uid, "ok": False, "missing_glyph": True}
        r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pnggray", f"-r{dpi}",
                            "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-dFirstPage=1", "-dLastPage=1",
                            "-o", str(out), "f.pdf"], cwd=d, capture_output=True, timeout=120)
        return {"id": uid, "ok": r.returncode == 0 and out.exists()}
    except subprocess.TimeoutExpired:
        return {"id": uid, "ok": False}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    fonts = os.environ.get("FONTS", "times,sans").split(",")
    labels_file = WORK / os.environ.get("LABELS", "labels.jsonl")
    labels = [json.loads(x) for x in labels_file.read_text(encoding="utf-8").splitlines()]
    jobs = [(x["id"], x["latex"], f, 300) for f in fonts for x in labels if "font" not in x]
    jobs += [(x["id"], x["latex"], x["font"], int(x.get("dpi", 300))) for x in labels if "font" in x]
    for f in {j[2] for j in jobs}:
        (WORK / "png" / f).mkdir(parents=True, exist_ok=True)
    logs = {f: open(WORK / f"render_{f}.jsonl", "a") for f in {j[2] for j in jobs}}
    with Pool(int(os.environ.get("NPROC", "8"))) as pool:
        for i, (job, res) in enumerate(zip(jobs, pool.imap(render, jobs, chunksize=4))):
            logs[job[2]].write(json.dumps(res) + "\n")
            if i % 2000 == 0:
                for log in logs.values():
                    log.flush()
                print(i, "/", len(jobs), flush=True)
    for log in logs.values():
        log.close()
    print("done", flush=True)
