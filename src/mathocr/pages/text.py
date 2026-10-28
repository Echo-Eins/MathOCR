"""Running text for the page stand: English Wikipedia (``wikimedia/wikipedia`` 20231101.en, CC BY-SA 4.0).

An article becomes a list of blocks, ``("heading", str)`` or ``("para", str)``, in its order:
* lines are normalized (NFC, unusual spaces -> space, zero-width characters dropped, whitespace collapsed, the
  ``"( ; "`` left behind by removed pronunciations closed up) and kept only if every character is in
  ``fonts.CHARSET`` -- a label must be exactly what is drawn, so nothing is transliterated;
* a heading is a short line (<= 60 characters, <= 8 words) that starts with a capital or digit and does not end
  in punctuation; a paragraph is prose (>= 60 characters, >= 8 words, ending in sentence punctuation); list items,
  table rows and captions are neither and are skipped;
* the article ends at its first reference-type section (References, See also, Notes, ...).

Articles go to splits by a sha256 bucket of the article id (``split_of``), so no article feeds two splits.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from mathocr.pages.fonts import CHARSET

_ALLOWED = frozenset(CHARSET)
_SPACES = re.compile("[            　]")
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_PRON = re.compile(r"\(\s*[;,]\s*")
_EMPTY_PARENS = re.compile(r"\(\s*\)")
_WS = re.compile(r"\s+")
_HEADING_END = re.compile(r"[.,;:!?]$")
_PARA_END = re.compile("[.!?\"”’)]$")
STOP_SECTIONS = frozenset({
    "references", "external links", "see also", "notes", "further reading", "citations", "bibliography", "sources",
    "footnotes", "gallery", "works cited", "notes and references", "references and notes", "primary sources",
    "secondary sources", "discography", "filmography", "selected works", "publications", "works", "literature",
})
SPLIT_BUCKETS = {"train": range(0, 90), "val": range(90, 95), "test": range(95, 100)}


def normalize(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = _ZERO_WIDTH.sub("", _SPACES.sub(" ", line))
    line = _EMPTY_PARENS.sub("", _PRON.sub("(", line))
    return _WS.sub(" ", line).strip()


def is_heading(line: str) -> bool:
    return (2 <= len(line) <= 60 and len(line.split()) <= 8 and (line[0].isupper() or line[0].isdigit())
            and not _HEADING_END.search(line))


def is_paragraph(line: str) -> bool:
    return len(line) >= 60 and len(line.split()) >= 8 and bool(_PARA_END.search(line))


def article_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for raw in text.split("\n"):
        line = normalize(raw)
        if not line:
            continue
        if line.lower() in STOP_SECTIONS:
            break
        if not set(line) <= _ALLOWED:
            continue
        if is_paragraph(line):
            blocks.append(("para", line))
        elif is_heading(line):
            blocks.append(("heading", line))
    # a heading must lead into prose: drop headings followed by another heading or by nothing
    return [b for i, b in enumerate(blocks)
            if b[0] == "para" or (i + 1 < len(blocks) and blocks[i + 1][0] == "para")]


def split_of(article_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"wiki-split\x00{article_id}".encode()).digest()[:8], "big") % 100
    return next(name for name, r in SPLIT_BUCKETS.items() if bucket in r)


def articles(paths: list[Path], split: str, min_chars: int = 800) -> Iterator[dict]:
    """Articles of ``split`` (id, title, blocks) with at least ``min_chars`` of paragraph text, in file order."""

    for path in paths:
        t = pq.read_table(path, columns=["id", "title", "text"])
        for aid, title, text in zip(t["id"].to_pylist(), t["title"].to_pylist(), t["text"].to_pylist()):
            if split_of(aid) != split:
                continue
            title = normalize(title)
            blocks = article_blocks(text)
            if not set(title) <= _ALLOWED or sum(len(s) for k, s in blocks if k == "para") < min_chars:
                continue
            yield {"id": aid, "title": title, "blocks": blocks}
