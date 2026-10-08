import random

import pytest
import sacrebleu

from mathocr.metrics import corpus_bleu, normalize_text, normalized_levenshtein, text_metrics
from mathocr.normalize import normalize_batch, unwrap_display
from mathocr.tokenizer import Vocab, detokenize, tokenize


def test_control_space_round_trip():
    s = r"a \ b \, c \\ d"
    assert detokenize(tokenize(s)) == s
    assert tokenize(s) == ["a", "\\", "b", "\\,", "c", "\\\\", "d"]


def test_vocab_is_deterministic_and_encodes():
    forms = [["x", "+", "y"], ["x", "=", "1"], ["x"]]
    v1, v2 = Vocab.build(forms, 1), Vocab.build(list(reversed(forms)), 1)
    assert v1.itos == v2.itos and v1.itos[4] == "x"
    ids = v1.encode(["x", "?"])
    assert ids[0] == v1.bos_id and ids[-1] == v1.eos_id and ids[2] == v1.unk_id
    assert v1.decode(ids[1:] + [5]) == ["x", "<unk>"]


@pytest.mark.parametrize("raw, expected", [
    (r"\begin{align*} x = 1 \end{align*}", "x = 1"),
    (r"\begin{align*} a &= b \end{align*}", "a  = b"),
    (r"\begin{align*} x \label{e}\\ \end{align*}", "x"),
    (r"\begin{align*} a&=b\\ c&=d\end{align*}", r"\begin{aligned} a&=b\\ c&=d \end{aligned}"),
    (r"\begin{gather*} a\\ b\end{gather*}", r"\begin{array}{c} a\\ b \end{array}"),
    (r"\begin{align*} a &= \begin{cases} x & y \\ z \end{cases}\end{align*}", r"a  = \begin{cases} x & y \\ z \end{cases}"),
    (r"\begin{align} x \end{align}", None),  # numbered: the image shows "(1)"
    (r"\begin{align*} 50\% \end{align*}", None),
    (r"\begin{align*} x \tag{2} \end{align*}", None),
])
def test_unwrap_display(raw, expected):
    assert unwrap_display(raw) == expected


def test_normalizer_matches_im2markup_convention():
    out = normalize_batch([r"\frac{a}{b}", r"\left(x\right)^2", r"\operatorname{sin} x", r"\bad{"])
    assert out == [r"\frac { a } { b }", r"\left( x \right) ^ { 2 }", r"\sin x", None]


def test_bleu_matches_sacrebleu():
    rng = random.Random(1)
    toks = ["x", "y", "\\frac", "{", "}", "^", "_", "2", "+", "=", "\\alpha", "(", ")"]
    refs = [" ".join(rng.choices(toks, k=rng.randint(3, 30))) for _ in range(200)]
    preds = [" ".join(t if rng.random() > 0.2 else rng.choice(toks) for t in r.split()) for r in refs]
    p, r = [normalize_text(x) for x in preds], [normalize_text(x) for x in refs]
    ours = corpus_bleu(p, r)
    ref = sacrebleu.corpus_bleu(p, [r], smooth_method="none", tokenize="13a").score / 100
    assert ours == pytest.approx(ref, abs=1e-12)


def test_text_metrics_basic():
    m = text_metrics(["x + 1", "y"], ["x + 1", "z"])
    assert m["exact_match"] == 0.5 and m["bleu"] == 0.0
    assert normalized_levenshtein("abc", "abd") == pytest.approx(1 / 3)
    assert normalize_text(r"\frac { a } { b } + \mathrm { s i n } x") == r"\frac{a}{b}+\mathrm{sin}x"
