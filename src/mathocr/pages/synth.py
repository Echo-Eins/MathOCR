"""Synthetic printed pages with exact layout ground truth (the page stand, tasks/page_ocr_plan.md section 5a).

``render_page(seed, article_iter, fonts, spec)`` lays out English prose on a page and draws it; every coordinate
recorded in the layout is the one the text was drawn at -- nothing is recovered from the image afterwards:

* pages are drawn at their final size (multiples of 32), grayscale, paper 255, ink 0, so the model's input *is*
  this image (no resampling between label and pixels);
* text is set word by word with Pillow's basic layout, which neither kerns nor forms ligatures (``AV`` is ``A`` +
  ``V``), so a character's start is its word's origin plus the advance of the word's prefix -- exactly;
* Pillow rounds a text origin to whole pixels, so word origins and baselines are rounded *before* drawing and
  recorded as those integers;
* word boxes are ``ImageDraw.textbbox`` at the same origin (they contain the word's ink; tested);
* lines carry their role (header, title, heading, body, footnote, footer), block, column, baseline, font size and
  cap height, the metric box (origin .. end of the last advance, baseline - ascent .. baseline + descent), the ink
  box (union of word boxes), the words and the start x of every character (spaces included);
* the line list *is* the reading order: header, title, the body column by column, footnotes, footer; the label is
  the line texts joined by ``\\n``, words joined by single spaces whatever the drawn (justified) gap.

All random choices come from ``numpy.random.default_rng(seed)``.  A page is rejected (``None``) when it would carry
fewer than ``spec.min_body_lines`` body lines; the caller retries with another seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROLES = ("header", "title", "heading", "body", "footnote", "footer")


@dataclass(frozen=True)
class PageSpec:
    widths: tuple[int, ...] = (640, 704, 768, 832, 896)
    heights: tuple[int, ...] = (896, 960, 1024, 1088, 1152)
    margin: tuple[int, int] = (24, 72)
    p_two_columns: float = 0.35
    min_column_px: int = 240
    gutter: tuple[int, int] = (20, 48)
    cap_px: tuple[float, float] = (14.0, 22.0)  # body cap height; the formula model's glyph target is 18
    leading: tuple[float, float] = (1.15, 1.5)  # line advance / font size
    p_article_start: float = 0.3  # the page opens an article (title); else it starts at a random paragraph
    title_scale: tuple[float, float] = (1.4, 1.9)
    heading_scale: tuple[float, float] = (1.1, 1.3)
    small_scale: tuple[float, float] = (0.72, 0.85)  # header, footer, footnotes
    p_header: float = 0.35
    p_footer: float = 0.4
    p_footnote: float = 0.25
    p_justify: float = 0.6
    p_indent: float = 0.6
    indent_caps: tuple[float, float] = (1.0, 2.5)
    para_gap: tuple[float, float] = (0.3, 0.9)  # extra space between unindented paragraphs, in line advances
    p_italic_run: float = 0.15  # per paragraph: one run of 1..4 words in italic
    p_sans_headings: float = 0.3
    max_space_stretch: float = 2.5  # a justified line's gaps may grow to this many spaces, else it stays ragged
    min_body_lines: int = 5


@lru_cache(maxsize=256)
def _font(path: str, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@dataclass
class _Face:
    """One face at one size: the Pillow font and the metrics the layout needs."""

    family: str
    role: str
    path: str
    size: float
    cap: float
    font: ImageFont.FreeTypeFont
    ascent: int
    descent: int

    @classmethod
    def of(cls, family: str, role: str, face: dict, cap_px: float) -> "_Face":
        size = cap_px * face["upem"] / face["cap_height"]
        font = _font(face["path"], round(size, 2))
        ascent, descent = font.getmetrics()
        return cls(family, role, face["path"], round(size, 2), cap_px, font, ascent, descent)

    def width(self, text: str) -> float:
        return self.font.getlength(text)


class _Page:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.img = Image.new("L", (w, h), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.lines: list[dict] = []
        self.rules: list[list[int]] = []
        self.blocks = 0

    def new_block(self) -> int:
        self.blocks += 1
        return self.blocks - 1

    def set_line(self, words: list[tuple[str, _Face]], x: float, top: float, gaps: list[float], role: str,
                 block: int, column: int) -> dict:
        """Draw one line: ``words`` start at ``x``, word i+1 at word i's end + ``gaps[i]``; the baseline sits
        ``ascent`` below ``top`` (the tallest face of the line).  Returns the line record."""

        ascent = max(f.ascent for _, f in words)
        descent = max(f.descent for _, f in words)
        baseline = int(round(top + ascent))
        recs, char_x, texts = [], [], []
        cursor = x
        for i, (word, face) in enumerate(words):
            ox = int(round(cursor))
            self.draw.text((ox, baseline), word, font=face.font, fill=0, anchor="ls")
            box = [int(v) for v in self.draw.textbbox((ox, baseline), word, font=face.font, anchor="ls")]
            offset = sum(len(t) + 1 for t in texts)
            recs.append({"t": word, "c": offset, "x": ox, "box": box, "italic": face.role == "italic"})
            char_x.extend(round(ox + face.width(word[:k]), 2) for k in range(len(word)))
            end = ox + face.width(word)
            if i + 1 < len(words):
                char_x.append(round(end, 2))  # the space starts where the word's advance ends
                cursor = end + gaps[i]
            texts.append(word)
        text = " ".join(texts)
        assert len(char_x) == len(text)
        last_face = words[-1][1]
        x1 = recs[-1]["x"] + last_face.width(words[-1][0])
        body_face = words[0][1]
        line = {
            "text": text, "role": role, "block": block, "column": column, "baseline": baseline,
            "size": body_face.size, "cap": body_face.cap, "ascent": ascent, "descent": descent,
            "font": body_face.family, "face": body_face.role,
            "box": [recs[0]["x"], baseline - ascent, round(x1, 2), baseline + descent],
            "ink": [min(r["box"][0] for r in recs), min(r["box"][1] for r in recs),
                    max(r["box"][2] for r in recs), max(r["box"][3] for r in recs)],
            "words": recs, "char_x": char_x,
        }
        self.lines.append(line)
        return line


def _break(words: list[tuple[str, _Face]], space: float, width: float, first_indent: float) -> list[list[int]] | None:
    """Greedy line breaking: word indices per line; None if a word does not fit a line by itself."""

    lines, cur, cur_w = [], [], 0.0
    for i, (word, face) in enumerate(words):
        w = face.width(word)
        avail = width - (first_indent if not lines else 0.0)
        if w > avail:
            return None
        need = w if not cur else cur_w + space + w
        if cur and need > avail:
            lines.append(cur)
            cur, cur_w = [i], w
        else:
            cur, cur_w = cur + [i], need
    if cur:
        lines.append(cur)
    return lines


def _sentence(rng: np.random.Generator, paras: list[str]) -> str | None:
    pool = [s.strip() + "." for p in paras for s in p.split(". ")
            if 40 <= len(s.strip()) <= 200 and not re.search("[.!?\"”’)]$", s.strip())]
    return pool[int(rng.integers(len(pool)))] if pool else None


def render_page(seed: int, articles: list[dict], fonts: dict[str, dict[str, dict]], families: dict[str, list[str]],
                spec: PageSpec = PageSpec()) -> tuple[Image.Image, dict] | None:
    """One page whose text starts in ``articles[0]`` and continues, if the page has room, into ``articles[1:]``
    (``text.articles`` records; the caller gives every page its own main article).  ``fonts`` maps family ->
    {role: face} (``stand_fonts.check``), ``families`` lists the allowed family names per kind (``serif`` /
    ``sans`` / ``mono``).  Returns (image, layout) or None (too few body lines)."""

    rng = np.random.default_rng(seed)
    w, h = int(rng.choice(spec.widths)), int(rng.choice(spec.heights))
    ml, mr, mt, mb = (int(rng.integers(spec.margin[0], spec.margin[1] + 1)) for _ in range(4))
    kinds = [k for k in ("serif", "sans", "mono") if families.get(k)]
    weights = np.array([{"serif": 0.55, "sans": 0.35, "mono": 0.10}[k] for k in kinds])
    kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
    family = families[kind][int(rng.integers(len(families[kind])))]
    cap = float(rng.uniform(*spec.cap_px))
    reg = _Face.of(family, "regular", fonts[family]["regular"], cap)
    ita = _Face.of(family, "italic", fonts[family]["italic"], cap)
    head_family = family
    if kind == "serif" and families.get("sans") and rng.random() < spec.p_sans_headings:
        head_family = families["sans"][int(rng.integers(len(families["sans"])))]
    head = _Face.of(head_family, "bold", fonts[head_family]["bold"], cap * float(rng.uniform(*spec.heading_scale)))
    title = _Face.of(head_family, "bold", fonts[head_family]["bold"], cap * float(rng.uniform(*spec.title_scale)))
    small_cap = cap * float(rng.uniform(*spec.small_scale))
    small = _Face.of(family, "regular", fonts[family]["regular"], small_cap)
    small_it = _Face.of(family, "italic", fonts[family]["italic"], small_cap)
    advance = reg.size * float(rng.uniform(*spec.leading))
    space = reg.width(" ")

    page = _Page(w, h)
    x0, x1, top, bottom = ml, w - mr, mt, h - mb
    text_w = x1 - x0
    n_cols = 2 if rng.random() < spec.p_two_columns else 1
    gutter = int(rng.integers(spec.gutter[0], spec.gutter[1] + 1))
    col_w = (text_w - gutter * (n_cols - 1)) / n_cols
    if col_w < spec.min_column_px:
        n_cols, col_w = 1, float(text_w)
    columns = [[round(x0 + c * (col_w + gutter), 2), round(x0 + c * (col_w + gutter) + col_w, 2)] for c in range(n_cols)]
    justify = rng.random() < spec.p_justify
    indent_style = rng.random() < spec.p_indent
    indent = cap * float(rng.uniform(*spec.indent_caps))

    art_order = list(articles)
    opens = rng.random() < spec.p_article_start

    # header (drawn first: reading order), footer and footnote (drawn last, placed at the bottom)
    deferred: list = []
    if rng.random() < spec.p_header:
        face = small_it if rng.random() < 0.5 else small
        words = [(t, face) for t in art_order[0]["title"].split()]
        width = sum(f.width(t) for t, f in words) + face.width(" ") * (len(words) - 1)
        if width <= text_w:
            align = rng.choice(["left", "center", "right"])
            hx = x0 if align == "left" else (x0 + (text_w - width) / 2 if align == "center" else x1 - width)
            gaps = [face.width(" ")] * (len(words) - 1)
            line = page.set_line(words, hx, top, gaps, "header", page.new_block(), -1)
            top = line["baseline"] + line["descent"] + int(round(0.6 * advance))
            if rng.random() < 0.3:
                page.rules.append([x0, top - int(round(0.3 * advance)), x1, top - int(round(0.3 * advance)) + 1])
    if rng.random() < spec.p_footer:
        num = str(int(rng.integers(1, 500)))
        nw = small.width(num)
        align = rng.choice(["center", "left", "right"])
        fx = x0 + (text_w - nw) / 2 if align == "center" else (x0 if align == "left" else x1 - nw)
        f_top = bottom - small.ascent - small.descent
        footer = ("footer", [(num, small)], fx, f_top)
        bottom = f_top - int(round(0.8 * advance))
    else:
        footer = None
    if rng.random() < spec.p_footnote:
        sent = _sentence(rng, [s for k, s in art_order[0]["blocks"] if k == "para"][3:])
        if sent:
            words = [(t, small) for t in ("1 " + sent).split()]
            sp = small.width(" ")
            lines = _break(words, sp, text_w, 0.0)
            if lines:
                fn_adv = small.size * 1.2
                fn_h = (len(lines) - 1) * fn_adv + small.ascent + small.descent
                fn_top = bottom - fn_h
                rule_y = int(round(fn_top - 0.5 * advance))
                for k, idx in enumerate(lines):
                    deferred.append(("footnote", [words[i] for i in idx], float(x0), fn_top + k * fn_adv))
                deferred.append(("rule", [x0, rule_y, int(round(x0 + 0.3 * text_w)), rule_y + 1]))
                bottom = rule_y - int(round(0.5 * advance))

    # title
    col, y = 0, float(top)
    blocks: list[tuple[str, str]] = []
    used: list[str] = []  # ids of the articles whose blocks were drawn, in order
    if opens:
        words = [(t, title) for t in art_order[0]["title"].split()]
        lines = _break(words, title.width(" "), text_w, 0.0)
        if lines and len(lines) <= 3:
            block = page.new_block()
            t_adv = title.size * 1.15
            for idx in lines:
                ws = [words[i] for i in idx]
                width = sum(f.width(t) for t, f in ws) + title.width(" ") * (len(ws) - 1)
                page.set_line(ws, x0 + (text_w - width) / 2, y, [title.width(" ")] * (len(ws) - 1), "title", block, -1)
                y += t_adv
            y += 0.6 * advance
            used.append(art_order[0]["id"])
        blocks = [(k, t, art_order[0]["id"]) for k, t in art_order[0]["blocks"]]
    else:
        paras = art_order[0]["blocks"]
        starts = [i for i, (k, _) in enumerate(paras) if k == "para"]
        blocks = [(k, t, art_order[0]["id"]) for k, t in paras[starts[int(rng.integers(len(starts)))]:]] if starts else []
    for art in art_order[1:]:
        if sum(len(t) for _, t, _ in blocks) > 40 * w * h / (cap * cap):  # far more text than the page holds
            break
        blocks.extend([("heading", art["title"], art["id"])] + [(k, t, art["id"]) for k, t in art["blocks"]])
    col_top = y

    body_extent = max(reg.ascent, ita.ascent) + max(reg.descent, ita.descent)

    def fits(at: float) -> bool:
        return at + body_extent <= bottom

    after_heading = True  # no indent right after a title or heading (and at the page start)
    full = False
    for kind_b, text, art_id in blocks:
        if full:
            break
        if kind_b == "heading":
            words = [(t, head) for t in text.split()]
            lines = _break(words, head.width(" "), col_w, 0.0)
            if not lines:
                continue
            gap_before = 0.0 if y <= col_top + 0.5 else advance * float(rng.uniform(0.5, 1.2))
            need = gap_before + len(lines) * head.size * 1.2 + 2 * advance  # keep with two body lines
            if y + need > bottom:
                col += 1
                if col >= n_cols:
                    break
                y = col_top
                gap_before = 0.0
            y += gap_before
            block = page.new_block()
            if not used or used[-1] != art_id:
                used.append(art_id)
            for idx in lines:
                ws = [words[i] for i in idx]
                page.set_line(ws, columns[col][0], y, [head.width(" ")] * (len(ws) - 1), "heading", block, col)
                y += head.size * 1.2
            y += advance * float(rng.uniform(0.2, 0.5))
            after_heading = True
            continue
        toks = text.split()
        faces = [reg] * len(toks)
        if len(toks) > 6 and rng.random() < spec.p_italic_run:
            run = int(rng.integers(1, 5))
            s0 = int(rng.integers(0, len(toks) - run))
            for i in range(s0, s0 + run):
                faces[i] = ita
        words = list(zip(toks, faces))
        ind = indent if (indent_style and not after_heading) else 0.0
        lines = _break(words, space, col_w, ind)
        if not lines:
            continue
        if not indent_style and not after_heading and y > col_top + 0.5:
            y += advance * float(rng.uniform(*spec.para_gap))
        block = page.new_block()
        for k, idx in enumerate(lines):
            if not fits(y):
                col += 1
                if col >= n_cols:
                    full = True
                    break
                y = col_top
            if not used or used[-1] != art_id:
                used.append(art_id)
            ws = [words[i] for i in idx]
            lx = columns[col][0] + (ind if k == 0 else 0.0)
            natural = sum(f.width(t) for t, f in ws) + space * (len(ws) - 1)
            gaps = [space] * (len(ws) - 1)
            last = k == len(lines) - 1
            if justify and not last and len(ws) > 1:
                extra = (columns[col][1] - lx) - natural
                if 0 <= extra / (len(ws) - 1) <= (spec.max_space_stretch - 1) * space:
                    gaps = [space + extra / (len(ws) - 1)] * (len(ws) - 1)
            page.set_line(ws, lx, y, gaps, "body", block, col)
            y += advance
        after_heading = False

    for item in deferred + ([footer] if footer else []):  # reading order: footnotes, then the footer
        if item[0] == "rule":
            page.rules.append(item[1])
            continue
        role, ws, lx, ly = item
        page.set_line(ws, lx, ly, [ws[0][1].width(" ")] * (len(ws) - 1), role, page.new_block(), -1)
    for r in page.rules:
        page.draw.rectangle([r[0], r[1], r[2] - 1, r[3] - 1], fill=0)

    body = sum(l["role"] == "body" for l in page.lines)
    if body < spec.min_body_lines:
        return None
    # reading order: the deferred footnotes / footer were set after the body, the header before it -- page.lines
    # is already in reading order; check the invariants every page must satisfy
    for l in page.lines:
        b = l["ink"]
        assert 0 <= b[0] and 0 <= b[1] and b[2] <= w and b[3] <= h, ("ink outside the page", l["text"][:40])
    layout = {
        "w": w, "h": h, "columns": columns, "text_area": [x0, mt, x1, h - mb], "rules": page.rules,
        "fonts": {"body": family, "kind": kind, "heading": head_family},
        "style": {"justify": bool(justify), "indent": bool(indent_style), "leading": round(advance / reg.size, 4),
                  "cap": round(cap, 3)},
        "articles": used,
        "lines": page.lines,
    }
    return page.img, layout


def label(layout: dict) -> str:
    return "\n".join(l["text"] for l in layout["lines"])
