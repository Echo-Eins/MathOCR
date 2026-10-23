"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0a, part 3: what does a readout fitted inside an OOD split fix?  From the cached representations of
stage0_memory_xfold.py: the residual probe on z fitted on one half of the split's formulas, applied to the other;
the deployed head's errors it fixes, as (reference token, deployed prediction) pairs, typed as
  glyph   -- a confusion of look-alike symbols (x/\\chi, \\alpha/a, case pairs, ...);
  style   -- font-style / annotation-convention commands (\\mathrm, \\mathbf, \\boldsymbol, \\operatorname, ...);
  notation-- braces, spacing, sizing, display style (no rendering change);
  eos     -- the end of the formula (stopping too early or too late);
  other.
Output: docs/probe_results/stage0_fix_types_<name>.json."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mathocr import determinism  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from notation_pools import SIZE, SPACES, STYLE  # noqa: E402
from stage0_memory_probe import dev, fit, predict  # noqa: E402
from stage0_memory_xfold import CACHE, subset  # noqa: E402

B = "\\"
GLYPH = {frozenset(p) for p in [("x", B + "chi"), (B + "alpha", "a"), (B + "gamma", "Y"), ("p", B + "rho"), (B + "nu", "V"),
                                ("n", B + "eta"), ("v", B + "upsilon"), ("v", B + "nu"), (B + "varphi", B + "phi"),
                                ("l", "c"), ("i", "l"), ("1", "l"), ("0", "O"), ("o", "0"), ("5", "S"), (B + "sum", B + "Sigma"),
                                (B + "hat", B + "widehat"), (B + "tilde", B + "widetilde"), (B + "epsilon", B + "varepsilon"),
                                (B + "theta", B + "vartheta"), ("w", B + "omega"), ("u", B + "mu"), ("k", B + "kappa"),
                                (B + "ell", "l"), ("t", "+"), (B + "prime", "'"), ("-", B + "-")]}
STYLECMD = {B + s for s in ("mathrm", "mathbf", "boldsymbol", "operatorname", "mathcal", "mathit", "mathsf", "mathtt",
                            "bf", "rm", "cal", "it", "sf", "pmb", "bm", "mathbb", "mathfrak", "mathscr", "text", "mbox")}


def kind(gt: str, pred: str) -> str:
    if gt == "<eos>" or pred == "<eos>":
        return "eos"
    if frozenset((gt, pred)) in GLYPH or (len(gt) == len(pred) == 1 and gt.lower() == pred.lower() and gt != pred):
        return "glyph"
    if gt in STYLECMD or pred in STYLECMD:
        return "style"
    invisible = {"{", "}"} | SPACES | STYLE
    if gt in invisible or pred in invisible or SIZE.match(gt) or SIZE.match(pred):
        return "notation"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    determinism.configure(dev)
    model, vocab, _, _ = load_checkpoint(args.ckpt, dev)
    tied = model.embed.weight.detach().float()
    itos = vocab.itos
    res = {}
    for split in ("ood_spe", "ood_sce", "ood_cpe"):
        s = torch.load(CACHE / f"{args.name}_{split}.pt")
        formulas = s["formula"].unique()
        perm = formulas[torch.randperm(len(formulas), generator=torch.Generator().manual_seed(1))]
        half = set(perm[: len(perm) // 2].tolist())
        a = torch.tensor([int(f) in half for f in s["formula"].tolist()])
        probe = fit("z", subset(s, a), len(vocab), tied, epochs=12)
        sb = subset(s, ~a)
        pred = predict(probe, sb, "z")
        with torch.no_grad():
            dep = torch.cat([torch.nn.functional.linear(sb["z"][j].to(dev).float(), tied).argmax(-1).cpu()
                             for j in torch.arange(len(sb["y"])).split(16384)])
        fixed = (dep != sb["y"]) & (pred == sb["y"])
        broken = (dep == sb["y"]) & (pred != sb["y"])
        pairs = Counter((itos[int(y)], itos[int(d)]) for y, d in zip(sb["y"][fixed], dep[fixed]))
        kinds = Counter()
        for (g, d), c in pairs.items():
            kinds[kind(g, d)] += c
        n_fix = int(fixed.sum())
        res[split] = {"deployed_errors": int((dep != sb["y"]).sum()), "fixed": n_fix, "broken": int(broken.sum()),
                      "fixed_by_kind": {k: v / n_fix for k, v in kinds.most_common()},
                      "top_fixed_pairs_ref_deployed": [[g, d, c] for (g, d), c in pairs.most_common(25)]}
        print(f"{args.name} {split}: deployed errors {res[split]['deployed_errors']}, fixed {n_fix}, broken {int(broken.sum())}; "
              f"by kind {[(k, round(v, 3)) for k, v in res[split]['fixed_by_kind'].items()]}", flush=True)
        print("   top (reference, deployed):", [(g, d, c) for (g, d), c in pairs.most_common(14)], flush=True)
    (ROOT / "docs" / "probe_results" / f"stage0_fix_types_{args.name}.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
