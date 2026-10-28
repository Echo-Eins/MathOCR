"""The page stand: fonts, text, page synthesis, cell targets and reading metrics (tasks/page_ocr_plan.md 5a)."""

import json
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from mathocr.pages import stand_fonts
from mathocr.pages.metrics import ned, page_metrics
from mathocr.pages.synth import ROLES, PageSpec, label, render_page
from mathocr.pages.targets import cell_targets
from mathocr.pages.text import article_blocks, is_heading, is_paragraph, normalize

CATALOG = Path("D:/MathOCR/pages/font_catalog.json")
WIKI = sorted(Path("D:/MathOCR/pages/raw/wikipedia_en/20231101.en").glob("train-*.parquet"))
needs_assets = pytest.mark.skipif(not CATALOG.exists() or not WIKI, reason="stand fonts / Wikipedia not on this machine")


@pytest.fixture(scope="module")
def stand():
    from mathocr.pages.text import articles

    fonts = stand_fonts.check(json.loads(CATALOG.read_text(encoding="utf-8"))["faces"])
    families = {"serif": list(stand_fonts.SERIF), "sans": list(stand_fonts.SANS), "mono": list(stand_fonts.MONO)}
    arts = list(islice(articles(WIKI, "val"), 160))
    pages = []
    k = 0
    while len(pages) < 40:
        res = render_page(1000 + k, arts[k:k + 4], fonts, families, PageSpec())
        if res is not None:
            pages.append((1000 + k, arts[k:k + 4], res))
        k += 1
    return fonts, families, pages


# ------------------------------------------------------------------ text


def test_normalize_and_block_kinds():
    assert normalize("Lincoln ( ; 1809\u00a0\u2013 1865)\u200b was") == "Lincoln (1809 \u2013 1865) was"
    assert is_heading("Early life") and not is_heading("Early life.") and not is_heading("early life")
    assert is_paragraph("This sentence is long enough to count as a paragraph of running prose text.")
    blocks = article_blocks("Intro text that is long enough to be a paragraph of prose, yes it is.\nHistory\n"
                            "More prose that is long enough to be a paragraph, of course it is.\nReferences\n"
                            "Late prose that is long enough to be a paragraph but comes after references.")
    assert blocks == [("para", "Intro text that is long enough to be a paragraph of prose, yes it is."),
                      ("heading", "History"),
                      ("para", "More prose that is long enough to be a paragraph, of course it is.")]


# ------------------------------------------------------------------ fonts


@needs_assets
def test_every_stand_family_has_the_three_faces_and_the_whole_charset():
    fonts = stand_fonts.check(json.loads(CATALOG.read_text(encoding="utf-8"))["faces"])
    for fam, roles in fonts.items():
        for r in stand_fonts.ROLES:
            assert roles[r]["usable"] and roles[r]["missing"] == "", (fam, r)
    assert not set(stand_fonts.HELD_OUT) & set(stand_fonts.SERIF + stand_fonts.SANS + stand_fonts.MONO)


# ------------------------------------------------------------------ synthesis


@needs_assets
def test_page_is_a_pure_function_of_its_seed(stand):
    fonts, families, pages = stand
    seed, arts, (img, layout) = pages[0]
    img2, layout2 = render_page(seed, arts, fonts, families, PageSpec())
    assert img.tobytes() == img2.tobytes() and json.dumps(layout) == json.dumps(layout2)


@needs_assets
def test_all_ink_lies_in_word_boxes_and_every_word_box_has_ink(stand):
    for _, _, (img, layout) in stand[2]:
        a = np.asarray(img)
        allowed = np.zeros(a.shape, bool)
        for l in layout["lines"]:
            for wd in l["words"]:
                x0, y0, x1, y1 = wd["box"]
                assert (a[y0:y1, x0:x1] < 128).any(), ("word box without ink", wd["t"])
                allowed[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = True
        for x0, y0, x1, y1 in layout["rules"]:
            allowed[y0:y1, x0:x1] = True
        stray = (a < 255) & ~allowed
        assert not stray.any(), f"{int(stray.sum())} ink pixels outside every word box"


@needs_assets
def test_line_records_are_consistent(stand):
    for _, _, (img, layout) in stand[2]:
        assert label(layout) == "\n".join(l["text"] for l in layout["lines"])
        for l in layout["lines"]:
            text, cx = l["text"], l["char_x"]
            assert len(cx) == len(text) and all(b >= a for a, b in zip(cx, cx[1:])), l["text"][:40]
            assert " ".join(w["t"] for w in l["words"]) == text
            for wd in l["words"]:
                assert text[wd["c"]:wd["c"] + len(wd["t"])] == wd["t"] and cx[wd["c"]] == wd["x"]
            assert l["role"] in ROLES and 0 <= l["ink"][0] and l["ink"][2] <= layout["w"] and l["ink"][3] <= layout["h"]


@needs_assets
def test_reading_order(stand):
    rank = {"header": 0, "title": 1, "heading": 2, "body": 2, "footnote": 3, "footer": 4}
    for _, _, (img, layout) in stand[2]:
        lines = layout["lines"]
        assert [rank[l["role"]] for l in lines] == sorted(rank[l["role"]] for l in lines)
        flow = [l for l in lines if l["role"] in ("heading", "body")]
        assert [l["column"] for l in flow] == sorted(l["column"] for l in flow)
        for a, b in zip(flow, flow[1:]):
            if a["column"] == b["column"]:
                assert b["baseline"] > a["baseline"]
        for a, b in zip(lines, lines[1:]):
            if a["role"] == b["role"] and a["role"] in ("footnote", "title"):
                assert b["baseline"] > a["baseline"]


@needs_assets
def test_cap_bands_never_overlap_even_per_pixel(stand):
    for _, _, (img, layout) in stand[2][:12]:
        t = cell_targets(layout, (1, 1))  # raises on any shared pixel
        assert t["mask"].sum() > 0
        for k in (8, 4):
            cell_targets(layout, (k, k // 2))


@needs_assets
def test_stand_varies(stand):
    pages = [p[2][1] for p in stand[2]]
    assert len({p["fonts"]["body"] for p in pages}) >= 20
    assert {len(p["columns"]) for p in pages} == {1, 2}
    assert {p["style"]["justify"] for p in pages} == {True, False}
    roles = {l["role"] for p in pages for l in p["lines"]}
    assert roles == set(ROLES)


# ------------------------------------------------------------------ cell targets


def _toy_layout():
    # two lines of one column, cells of 4x2 px: line 0 'ab cd' at baseline 11, line 1 'efg' at baseline 23
    return {"h": 32, "w": 16, "lines": [
        {"text": "ab cd", "role": "body", "column": 0, "baseline": 11, "cap": 6.0,
         "char_x": [2.0, 3.5, 5.0, 6.0, 7.5], "box": [2, 3, 9.0, 13]},
        {"text": "efg", "role": "body", "column": 0, "baseline": 23, "cap": 6.0,
         "char_x": [4.0, 5.5, 7.0], "box": [4, 15, 8.5, 25]},
    ]}


def test_cell_targets_by_hand():
    t = cell_targets(_toy_layout(), (4, 2))
    assert t["line"].shape == (8, 8)
    # line 0 band [5, 11] holds the row centres 6 and 10 (rows 1, 2); x in [2, 9) holds centres 3, 5, 7 (cols 1-3)
    assert np.array_equal(np.argwhere(t["line"] == 0)[:, 0], [1, 1, 1, 2, 2, 2])
    assert np.array_equal(np.unique(np.argwhere(t["line"] == 0)[:, 1]), [1, 2, 3])
    # line 1 band [17, 23]: row centres 18, 22 (rows 4, 5); x in [4, 8.5): centres 5, 7 (cols 2, 3)
    assert np.array_equal(np.unique(np.argwhere(t["line"] == 1)[:, 0]), [4, 5])
    assert np.array_equal(np.unique(np.argwhere(t["line"] == 1)[:, 1]), [2, 3])
    assert t["n_lines"][1, 1] == 2 and t["line_in_col"][4, 2] == 1 and t["n_lines_col"][4, 2] == 2
    # chars starting at or left of x = 3: 'a' (2.0) -> 1; x = 7: 'a', 'b', ' ', 'c' (6.0) -> 4
    assert t["chars_before"][1, 1] == 1 and t["chars_before"][1, 3] == 4 and t["chars_after"][1, 3] == 1
    assert t["line_chars"][1, 1] == 5 and t["line_chars"][4, 2] == 3
    assert t["dist_start"][1, 1] == pytest.approx(1.0) and t["dist_end"][1, 3] == pytest.approx(2.0)
    assert t["rel_x"][1, 2] == pytest.approx((5 - 2) / 7)
    assert t["line"][0, 0] == -1 and not t["mask"][0, 0] and np.isnan(t["rel_x"][0, 0])


def test_cell_targets_refuse_overlapping_bands():
    lay = _toy_layout()
    lay["lines"][1]["baseline"] = 13  # band [7, 13] overlaps line 0's band [5, 11]
    with pytest.raises(ValueError, match="share cells"):
        cell_targets(lay, (4, 2))


# ------------------------------------------------------------------ metrics


REF = "the first line of text\nsecond line goes here\nthird and last line"


def test_metrics_perfect():
    m = page_metrics(REF, REF, ref_y=[0.1, 0.5, 0.9])
    assert m["cer"] == 0 and m["skipped"] == m["repeats"] == m["extra"] == 0 and m["order_tau"] == 1.0
    assert m["ned_by_height"][1] == 0.0 and m["ned_by_height"][0] is None


def test_metrics_skip_repeat_extra_order():
    a, b, c = REF.split("\n")
    m = page_metrics("\n".join([a, a, c]), REF)
    assert (m["read"], m["skipped"], m["repeats"], m["extra"]) == (2, 1, 1, 0)
    m = page_metrics("\n".join([a, "zzzz qqqq xxxx wwww", b, c]), REF)
    assert (m["read"], m["skipped"], m["repeats"], m["extra"]) == (3, 0, 0, 1)
    m = page_metrics("\n".join([c, b, a]), REF)
    assert m["read"] == 3 and m["order_tau"] == pytest.approx(-1.0)
    m = page_metrics("\n".join([a, c]), REF, ref_y=[0.05, 0.55, 0.95])
    assert m["skipped"] == 1 and m["line_ned"][1] == 1.0 and m["ned_by_height"][5] == 1.0
    assert ned("abc", "abd") == pytest.approx(1 / 3)
