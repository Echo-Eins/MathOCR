# Official CDM evaluator (UniMERNet, arXiv:2409.03643) at a pinned commit.
# Lean variant of upstream cdm/DockerFile: Debian 13 ships ImageMagick 7
# (CDM calls `magick`), so neither texlive-full nor a source build of
# ImageMagick is needed; the LaTeX packages CDM's templates load are
# geometry, booktabs, multirow, amsmath, amssymb, upgreek, xcolor
# (CJK only for Chinese formulas, not used here).
FROM python:3.11-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
        imagemagick ghostscript nodejs git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Debian's ImageMagick policy forbids PDF input; CDM rasterizes PDFs.
RUN sed -i '/pattern="PDF"/d' /etc/ImageMagick-7/policy.xml && magick -version | head -1

RUN git clone https://github.com/opendatalab/UniMERNet.git /opt/UniMERNet \
    && git -C /opt/UniMERNet checkout 5a2c80d96b1d2dba447ff18d873e5fb73ba03c35
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 scikit-image==0.20.0 \
        opencv-python-headless==4.10.0.84 matplotlib==3.9.2 pillow==10.4.0 tqdm==4.66.5

WORKDIR /opt/UniMERNet/cdm
# Smoke test: the bundled example renders and scores.
RUN python evaluation.py -i assets/example/input_example.json -o /tmp/cdm_smoke -p 2 \
    && python -c "import json; print(json.load(open('/tmp/cdm_smoke/input_example/metrics_res.json'))['mean_score'])" \
    && rm -rf /tmp/cdm_smoke
ENTRYPOINT ["python", "evaluation.py"]
