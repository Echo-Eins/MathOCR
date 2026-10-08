"""Pinned upstream data: Hugging Face revisions and file checksums.

Every file is verified by sha256 before use, so a build either reads exactly
these bytes or fails.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class PinnedFile:
    repo: str
    revision: str
    filename: str
    sha256: str

    def fetch(self) -> Path:
        path = Path(hf_hub_download(self.repo, self.filename, repo_type="dataset", revision=self.revision))
        actual = sha256_file(path)
        if actual != self.sha256:
            raise RuntimeError(f"{self.repo}/{self.filename}@{self.revision}: sha256 {actual} != pinned {self.sha256}")
        return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


IM2LATEX_REPO, IM2LATEX_REV = "yuntian-deng/im2latex-100k", "a9561dba18ec3183f4922635de9946cfa18969de"
IM2LATEX = {
    "train": PinnedFile(IM2LATEX_REPO, IM2LATEX_REV, "data/train-00000-of-00001-93885635ef7c6898.parquet",
                        "159ae09f4fadf7bf06223ebdaee11a9c8cef93797dbc449354d312be3d961f61"),
    "val": PinnedFile(IM2LATEX_REPO, IM2LATEX_REV, "data/val-00000-of-00001-3f88ebb0c1272ccf.parquet",
                      "ba437d693904f4475e220c351ca5d431a391c46a74b05c4a4897c084704984f2"),
    "test": PinnedFile(IM2LATEX_REPO, IM2LATEX_REV, "data/test-00000-of-00001-fce261550cd3f5db.parquet",
                       "5ea0dfedd0f746811c7d564d1666edc4c1d1c72225f1762da77719d928e501e0"),
}

OLEEHYO_REPO, OLEEHYO_REV = "OleehyO/latex-formulas", "436791f14f92cbb7fdc1b27f2c316cf1f4e65445"
# Git-LFS object ids of the cleaned_formulas shards are their sha256.
OLEEHYO_SHARDS = [
    PinnedFile(OLEEHYO_REPO, OLEEHYO_REV, f"cleaned_formulas/train-0000{i}-of-00006.parquet", sha)
    for i, sha in enumerate([
        "a82ed31fc852c691cc5a853e2a0c42eacf95d64f410b5c0d29e4e26e6c049ace",
        "1b502509a370eb42bf73260136fd1cdeeb369fd38b859fc9fc4ef0ccd0fb1b62",
        "5bf5e047a988c85dc2c5559079273c1208c022900367b16b07e74f56763479ff",
        "c624c202255c372c38ba9b813d74fd540c76fd659eb172f46a28ff6d4413d4d6",
        "6ea143ae18e3de417d62de06674bfd1ec84563615e4e25b073a066752c815739",
        "5ba4fbf0eccdad785124f96ae8337180903d532e6990fe01f54269034003d7b7",
    ])
]

UNIMER_TEST = PinnedFile("wanderkid/UniMER_Dataset", "2343ddd963290469da36ca83e3a56c66e068add9", "UniMER-Test.zip",
                         "9bf370b8cac868fee84835f40dec26c477430611253e3feb681356a8149a3a90")
UNIMER_SUBSETS = ("cpe", "sce", "spe", "hwe")


def unimer_test_dir(cache_root: Path) -> Path:
    """Extract the pinned UniMER-Test zip once; return ``.../UniMER-Test``."""

    zip_path = UNIMER_TEST.fetch()
    target = cache_root / f"unimer_test_{UNIMER_TEST.sha256[:12]}"
    done = target / ".extracted"
    if not done.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                resolved = (target / member).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise RuntimeError(f"unsafe path in zip: {member}")
            zf.extractall(target)
        done.touch()
    return target / "UniMER-Test"
