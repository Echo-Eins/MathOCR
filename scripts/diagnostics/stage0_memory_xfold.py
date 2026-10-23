"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0a, part 2: is the information present *in the OOD representations themselves*?  The train-fitted probes of
stage0_memory_probe.py measure what a method trained on in-domain data can extract; they cannot learn to fix
OOD-specific errors.  Here the probes are fitted inside each split: 2-fold by formula (fit on one half, with early
stopping on 5 % of it, score on the other half, then swap).  If the probe on [z; pooled memory] beats the probe on
z alone on the held-out half, the memory where the decoder reads carries information about the next token that the
final latent does not -- the decoder loses it.  Representations are cached in D:/MathOCR/stage0.

    python scripts/diagnostics/stage0_memory_xfold.py --name final_v2 --ckpt .../best.pt
Output: docs/probe_results/stage0_memory_xfold_<name>.json."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mathocr import determinism  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from notation_pools import SPACES, STYLE  # noqa: E402
from stage0_memory_probe import EVAL_DATA, boot_ci, collect, dev, fit, predict  # noqa: E402

CACHE = Path(r"D:\MathOCR\stage0")


def subset(s: dict, mask: torch.Tensor) -> dict:
    return {k: v[mask] for k, v in s.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    determinism.configure(dev)
    model, vocab, icfg, _ = load_checkpoint(args.ckpt, dev)
    tied = model.embed.weight.detach().float()
    invisible = torch.zeros(len(vocab), dtype=torch.bool)
    for t in {"{", "}"} | SPACES | STYLE:
        if t in vocab.stoi:
            invisible[vocab.stoi[t]] = True
    CACHE.mkdir(parents=True, exist_ok=True)
    kinds = ("z", "c_last", "z+c_last", "z+c_all")
    res = {}
    for split, n in (("val", 1500), ("ood_spe", 1500), ("ood_sce", 1500), ("ood_cpe", 600)):
        cache = CACHE / f"{args.name}_{split}.pt"
        if cache.exists():
            s = torch.load(cache)
        else:
            s = collect(model, vocab, icfg, EVAL_DATA / f"{split}.parquet", n)
            torch.save(s, cache)
        formulas = s["formula"].unique()
        perm = formulas[torch.randperm(len(formulas), generator=torch.Generator().manual_seed(1))]
        fold_of = {int(f): i % 2 for i, f in enumerate(perm.tolist())}
        fold = torch.tensor([fold_of[int(f)] for f in s["formula"].tolist()])
        hits = {k: torch.zeros(len(s["y"]), dtype=torch.bool) for k in kinds}
        for k_fold in (0, 1):
            tr, te = fold == k_fold, fold != k_fold
            str_, ste = subset(s, tr), subset(s, te)
            for k in kinds:
                probe = fit(k, str_, len(vocab), tied, epochs=12)
                hits[k][te] = predict(probe, ste, k) == ste["y"]
        y, ok = s["y"], s["tied_ok"]
        vis = ~invisible[y]
        row = {"tokens": int(len(y)), "tied": {"all": float(ok.float().mean()), "visible": float(ok[vis].float().mean())}}
        for k, h in hits.items():
            row[k] = {"all": float(h.float().mean()), "visible": float(h[vis].float().mean()),
                      "fixes_tied_errors": float(h[~ok].float().mean()), "breaks_tied_correct": float((~h[ok]).float().mean())}
            if k != "z":
                d = (h.float() - hits["z"].float()).numpy()
                dv = d[vis.numpy()]
                row[k]["vs_z_visible"] = [float(dv.mean()), *boot_ci(dv, s["formula"].numpy()[vis.numpy()])]
        res[split] = row
        print(f"  {split:8} tied {row['tied']['all']:.4f} (visible {row['tied']['visible']:.4f})", flush=True)
        for k in kinds:
            r = row[k]
            extra = ""
            if "vs_z_visible" in r:
                m, lo, hi = r["vs_z_visible"]
                extra = f" | visible vs z {m * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]"
            print(f"    {k:9} all {r['all']:.4f} visible {r['visible']:.4f} fixes {r['fixes_tied_errors']:.3f} "
                  f"breaks {r['breaks_tied_correct']:.4f}{extra}", flush=True)
    (ROOT / "docs" / "probe_results" / f"stage0_memory_xfold_{args.name}.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
