"""Runs inside the mathocr-cdm image (TeX Live + Ghostscript):

    docker run --rm --entrypoint python3 -e FONTS=times,sans -e NPROC=12 \
        -v <work>:/work -v <repo>/scripts/fonts:/scripts mathocr-cdm /scripts/render.py

Renders every label of /work/labels.jsonl ({"id", "latex"}) in each requested math font with pdflatex (display
style, page = formula box via preview/tightpage) and rasterizes at 300 dpi: /work/png/<font>/<id>.png.
Re-runnable: existing images are kept.  /work/render_<font>.jsonl logs which labels compiled."""

import json
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

WORK = Path("/work")
FONTS = {  # (preamble, switch inside the formula box)
    "cm": ("", ""),
    "times": (r"\usepackage{txfonts}", ""),
    "palatino": (r"\usepackage{pxfonts}", ""),
    "sans": (r"\usepackage{sansmath}", r"\sansmath"),
}
DOC = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,upgreek}
%s
\usepackage[active,tightpage]{preview}\PreviewBorder=2pt
\begin{document}
\begin{preview}%s$\displaystyle %s $\end{preview}
\end{document}
"""


def render(job):
    uid, latex, font = job
    out = WORK / "png" / font / f"{uid}.png"
    if out.exists():
        return {"id": uid, "ok": True}
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        (Path(d) / "f.tex").write_text(DOC % (FONTS[font][0], FONTS[font][1], latex), encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=d,
                           capture_output=True, timeout=60)
        if r.returncode != 0 or not (Path(d) / "f.pdf").exists():
            return {"id": uid, "ok": False}
        r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pnggray", "-r300",
                            "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-dFirstPage=1", "-dLastPage=1",
                            "-o", str(out), "f.pdf"], cwd=d, capture_output=True, timeout=120)
        return {"id": uid, "ok": r.returncode == 0 and out.exists()}
    except subprocess.TimeoutExpired:
        return {"id": uid, "ok": False}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    fonts = os.environ.get("FONTS", "times,sans").split(",")
    labels = [json.loads(x) for x in (WORK / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    with Pool(int(os.environ.get("NPROC", "8"))) as pool:
        for font in fonts:
            (WORK / "png" / font).mkdir(parents=True, exist_ok=True)
            with open(WORK / f"render_{font}.jsonl", "w") as log:
                jobs = [(x["id"], x["latex"], font) for x in labels]
                for i, res in enumerate(pool.imap_unordered(render, jobs, chunksize=8)):
                    log.write(json.dumps(res) + "\n")
                    if i % 2000 == 0:
                        log.flush()
                        print(font, i, "/", len(jobs), flush=True)
    print("done", flush=True)
