"""Runs inside the mathocr-cdm image: render each label of /work/labels.jsonl in several math fonts with pdflatex
and rasterize with Ghostscript (300 dpi, page = formula box via preview/tightpage).
Output: /work/png/<font>/<id>.png, /work/render_log.jsonl."""
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
        return {"id": uid, "font": font, "ok": True}
    d = tempfile.mkdtemp(dir="/tmp")
    try:
        (Path(d) / "f.tex").write_text(DOC % (FONTS[font][0], FONTS[font][1], latex), encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=d,
                           capture_output=True, timeout=60)
        if r.returncode != 0 or not (Path(d) / "f.pdf").exists():
            return {"id": uid, "font": font, "ok": False}
        r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pnggray", "-r300",
                            "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", "-dFirstPage=1", "-dLastPage=1",
                            "-o", str(out), "f.pdf"], cwd=d, capture_output=True, timeout=120)
        return {"id": uid, "font": font, "ok": r.returncode == 0 and out.exists()}
    except subprocess.TimeoutExpired:
        return {"id": uid, "font": font, "ok": False}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for f in FONTS:
        (WORK / "png" / f).mkdir(parents=True, exist_ok=True)
    labels = [json.loads(x) for x in (WORK / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    jobs = [(x["id"], x["latex"], f) for x in labels for f in FONTS]
    with Pool(int(os.environ.get("NPROC", "20"))) as pool, open(WORK / "render_log.jsonl", "w") as log:
        for i, res in enumerate(pool.imap_unordered(render, jobs, chunksize=4)):
            log.write(json.dumps(res) + "\n")
            log.flush()
            if i % 500 == 0:
                print(i, "/", len(jobs), flush=True)
    print("done")
