"""Token convention shared by the data, the model and the metrics.

Every label in this project is an im2markup-normalized LaTeX string (see
``third_party/latex_normalizer``): tokens separated by single spaces, with
compound tokens such as ``\\left(`` and ``\\begin{array}`` already formed.
Splitting on whitespace is therefore the tokenizer, exactly as in the
im2latex / UniMER evaluation code.  A control space ``\\ `` survives the
round trip: it splits to the token ``\\`` and joins back to ``\\ <next>``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path


def tokenize(normalized: str) -> list[str]:
    return normalized.split()


def detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)


class Vocab:
    """Frozen token <-> id mapping; ids 0..3 are the special tokens."""

    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, tokens: Sequence[str]):
        if tuple(tokens[: len(SPECIALS)]) != SPECIALS:
            raise ValueError("vocabulary must start with the special tokens")
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens in vocabulary")
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """BOS + ids + EOS; unknown tokens map to UNK."""

        return [self.bos_id, *(self.stoi.get(t, self.unk_id) for t in tokens), self.eos_id]

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Ids -> tokens, stopping at EOS and skipping PAD/BOS."""

        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.bos_id):
                out.append(self.itos[i])
        return out

    @classmethod
    def build(cls, formulas: Iterable[Sequence[str]], min_freq: int) -> "Vocab":
        """Tokens seen ``>= min_freq`` times, ordered by count then name.

        The order is a pure function of the counts, so one training split
        always yields the same ids.
        """

        counts = Counter(t for f in formulas for t in f)
        kept = sorted((t for t, c in counts.items() if c >= min_freq and t not in SPECIALS),
                      key=lambda t: (-counts[t], t))
        return cls([*SPECIALS, *kept])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))
