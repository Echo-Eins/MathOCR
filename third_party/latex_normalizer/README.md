# LaTeX normalizer (im2markup / KaTeX)

`preprocess_formula.js` and `third_party/` are copied verbatim from
[opendatalab/UniMERNet](https://github.com/opendatalab/UniMERNet),
`cdm/modules/tokenize_latex/` at commit
`b62e9274c2257986ee89fd6b95d3960125345a80` (checked out at repository
commit `5a2c80d96b1d2dba447ff18d873e5fb73ba03c35`).

It is the im2markup normalizer (Deng et al., harvardnlp/im2markup) built on a
patched KaTeX parser: it renders the KaTeX AST back to a canonical,
space-separated LaTeX string.  The im2latex-100k and UniMER-Test labels were
produced by this normalizer, so running raw LaTeX through it **once** puts
it into the same token convention.  It is not idempotent (each pass wraps
array cells in another `{ }`), so already-normalized labels are never
re-normalized.

Licenses: UniMERNet — Apache-2.0 (`LICENSE-UniMERNet`); im2markup — MIT;
KaTeX — MIT (`third_party/katex/package.json`).

The Python side of the upstream wrapper (`tokenize_latex.py`: environment
rewrites before, `\operatorname{sin}` -> `\sin` after) is reimplemented in
`src/mathocr/normalize.py`.
