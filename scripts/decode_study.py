"""Decoding study: how much exact match decoding alone adds, on one trained checkpoint.

    python scripts/decode_study.py CHECKPOINT [--data DIR] [--splits val ood_cpe ...] [--limit N] --out OUT.json

Variants (all deterministic):

* ``greedy`` -- the canonical path;
* ``beam3`` / ``beam5`` -- beam search, GNMT length penalty (alpha 0.6);
* ``beam5_pairs`` -- beam 5 that cannot unbalance ``{}``, ``\\left``/``\\right`` or
  ``\\begin``/``\\end`` and cannot stop while a group is open;
* ``consensus`` -- greedy reads at glyph scales 0.9, 1.0, 1.12 plus the beam-5 read
  at 1.0; every candidate is scored by teacher forcing under all three views and
  the best mean length-normalized log-probability wins (multi-zoom agreement).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.tokenizer import detokenize  # noqa: E402
from probe import score  # noqa: E402

SCALES = (0.9, 1.0, 1.12)


def groups_of(store, icfg, indices, jitter):
    g = defaultdict(list)
    for i in indices:
        g[store.shape(i, icfg, jitter)].append(i)
    return g


def batches(store, icfg, indices, jitter, budget=1_000_000, max_batch=32):
    items = TrainItems(store, icfg, None, 0)
    for shape, idx in sorted(groups_of(store, icfg, indices, jitter).items()):
        size = max(1, min(max_batch, budget // (shape[0] * shape[1])))
        for s in range(0, len(idx), size):
            yield idx[s : s + size], collate([items[(i, jitter, 0)] for i in idx[s : s + size]], 0)


def pair_groups(vocab) -> list[tuple[list[int], list[int]]]:
    toks = vocab.itos
    opens = lambda pre: [i for i, t in enumerate(toks) if t.startswith(pre)]  # noqa: E731
    return [([vocab.stoi["{"]], [vocab.stoi["}"]]), (opens("\\left"), opens("\\right")),
            (opens("\\begin{"), opens("\\end{"))]


@torch.no_grad()
def seq_logprob(model, images, seqs, vocab, alpha=0.6, reverse=False):
    """Length-normalized teacher-forced log-probability of every sequence (ids without BOS/EOS);
    ``reverse``: read right to left (bidirectional models)."""

    dev = images.device
    tokens = [torch.tensor([vocab.bos_id, *(s[::-1] if reverse else s), vocab.eos_id]) for s in seqs]
    t = torch.full((len(seqs), max(len(x) for x in tokens)), vocab.pad_id, dtype=torch.long)
    for r, x in enumerate(tokens):
        t[r, : len(x)] = x
    t = t.to(dev)
    direction = torch.full((len(seqs),), int(reverse), dtype=torch.long, device=dev) if model.cfg.bidir else None
    logp = torch.log_softmax(model(images, t[:, :-1], direction=direction).float(), dim=-1)
    tgt = t[:, 1:]
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).masked_fill(tgt == vocab.pad_id, 0.0).sum(1)
    n = (tgt != vocab.pad_id).sum(1).float()
    return (lp / ((5 + n) / 6) ** alpha).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("/mnt/SSD/MathOCR/probe/data20k"))
    ap.add_argument("--splits", nargs="*", default=["val", "ood_cpe", "ood_sce", "ood_spe"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=["greedy", "beam3", "beam5", "beam5_pairs", "consensus"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(args.checkpoint, device)
    pairs = pair_groups(vocab)
    result = {"checkpoint": str(args.checkpoint), "splits": {}}
    for split in args.splits:
        store = CropStore(args.data / f"{split}.parquet", vocab)
        idx = list(range(len(store)))[: args.limit or None]
        gts = [store.labels[i] for i in idx]
        preds: dict[str, dict[int, list[int]]] = {v: {} for v in args.variants}
        need_beam5 = any(v in ("beam5", "consensus") for v in args.variants)
        for ids, batch in batches(store, icfg, idx, 1.0, max_batch=8):  # beams multiply the batch by 5
            x = batch["images"].to(device)
            if "greedy" in args.variants or "consensus" in args.variants:
                g = model.generate(x, vocab.bos_id, vocab.eos_id)
                for i, row in zip(ids, g.tolist()):
                    preds.setdefault("greedy", {})[i] = [t for t in row[: (row + [vocab.eos_id]).index(vocab.eos_id)]]
            if "beam3" in args.variants:
                for i, h in zip(ids, model.beam_generate(x, vocab.bos_id, vocab.eos_id, beam=3)):
                    preds["beam3"][i] = h
            if need_beam5:
                for i, h in zip(ids, model.beam_generate(x, vocab.bos_id, vocab.eos_id, beam=5)):
                    preds.setdefault("beam5", {})[i] = h
            if "beam5_pairs" in args.variants:
                for i, h in zip(ids, model.beam_generate(x, vocab.bos_id, vocab.eos_id, beam=5, pairs=pairs)):
                    preds["beam5_pairs"][i] = h
        if model.cfg.bidir or model.count is not None:  # rerank the beam-5 n-best list
            from mathocr.train import countable_mask

            mask = countable_mask(vocab)
            nb = {}
            for ids, batch in batches(store, icfg, idx, 1.0, max_batch=8):
                x = batch["images"].to(device)
                for i, hyps in zip(ids, model.beam_generate(x, vocab.bos_id, vocab.eos_id, beam=5, nbest=True)):
                    nb[i] = hyps[:5]
                if model.cfg.bidir:
                    for row, i in enumerate(ids):
                        seqs = [h[1] for h in nb[i]]
                        r2l = seq_logprob(model, x[row : row + 1].expand(len(seqs), -1, -1, -1), seqs, vocab, reverse=True)
                        nb[i] = [(l2r, seq, float(r)) for (l2r, seq), r in zip(nb[i], r2l)]
                if model.count is not None:
                    est = model._log_count(model.encode(x)).exp().cpu()
                    for row, i in enumerate(ids):
                        nb[i] = [(*h, float((torch.bincount(torch.tensor(h[1], dtype=torch.long), minlength=len(vocab))[mask].float()
                                              - est[row][mask]).abs().sum())) for h in nb[i]]
            if model.cfg.bidir:
                preds["beam5_r2l"] = {i: max(nb[i], key=lambda h: h[0] + h[2])[1] for i in idx}
            if model.count is not None:
                for lam in (0.05, 0.2, 0.5):
                    preds[f"beam5_count{lam}"] = {i: max(nb[i], key=lambda h: h[0] - lam * h[-1])[1] for i in idx}
            args.variants = list(dict.fromkeys([*args.variants, *[v for v in preds if v.startswith(("beam5_r2l", "beam5_count"))]]))
        if "consensus" in args.variants:
            cands = {i: [preds["greedy"][i], preds["beam5"][i]] for i in idx}
            for sc in (0.9, 1.12):
                for ids, batch in batches(store, icfg, idx, sc):
                    g = model.generate(batch["images"].to(device), vocab.bos_id, vocab.eos_id)
                    for i, row in zip(ids, g.tolist()):
                        cands[i].append(row[: (row + [vocab.eos_id]).index(vocab.eos_id)])
            for i in idx:  # unique candidates in a fixed order
                seen, uniq = set(), []
                for c in cands[i]:
                    if tuple(c) not in seen:
                        seen.add(tuple(c))
                        uniq.append(c)
                cands[i] = uniq
            total = {i: torch.zeros(len(cands[i])) for i in idx}
            for sc in SCALES:
                for ids, batch in batches(store, icfg, idx, sc, max_batch=8):
                    x = batch["images"].to(device)
                    for row, i in enumerate(ids):
                        xi = x[row : row + 1].expand(len(cands[i]), -1, -1, -1)
                        total[i] += seq_logprob(model, xi, cands[i], vocab)
            preds["consensus"] = {i: cands[i][int(total[i].argmax())] for i in idx}
        result["splits"][split] = {}
        for v in args.variants:
            toks = [vocab.decode(preds[v][i]) for i in idx]
            result["splits"][split][v] = score(toks, gts)
            print(json.dumps({"split": split, "variant": v, **{k: round(val, 4) for k, val in result["splits"][split][v].items()}}), flush=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
