"""(Paths below are those of the Windows laptop; edit them for another machine.)
Greedy + beam n-best of a checkpoint for random samples of processed splits, batched by image shape (beam search
on a batch may differ from single-image beams in the last bits; the deployed decoder is untouched).
One JSON line per sample: index, source, gt, greedy, hyps = [[normalized beam score, hypothesis], ...].

    python scripts/diagnostics/nbest_dump.py --run C:/Coding/Python/MathOCR_final/runs/final_v2_s1 \
        --data C:/Coding/Python/MathOCR_final/data/processed --out D:/MathOCR/nbest_final_v2_s1 \
        --split test_oleehyo=1000 --split ood_spe=400
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import decode_store, load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", action="append", required=True, metavar="NAME=N")
    ap.add_argument("--beam", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device("cuda")
    determinism.configure(dev)
    model, vocab, icfg, _ = load_checkpoint(args.run / "checkpoints" / args.ckpt, dev)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.split:
        split, n = spec.split("=")
        store = CropStore(args.data / f"{split}.parquet", vocab)
        pick = sorted(np.random.default_rng(0).choice(len(store), size=min(int(n), len(store)), replace=False).tolist())
        greedy = dict(zip(pick, decode_store(model, store, icfg, vocab, dev, 400_000, 32, indices=pick)))
        items = TrainItems(store, icfg, aug_cfg=None, seed=0)
        rows = {}
        with torch.no_grad():
            for keys in ShapeBatchSampler(store, icfg, 300_000, 8, seed=0, train=False, indices=pick,
                                          max_tokens=model.cfg.max_len):
                batch = collate([items[k] for k in keys], vocab.pad_id)
                nbs = model.beam_generate(batch["images"].to(dev), vocab.bos_id, vocab.eos_id, beam=args.beam, nbest=True)
                for i, nb in zip(batch["index"].tolist(), nbs):
                    rows[i] = {"index": i, "source": store.sources[i], "gt": store.labels[i], "greedy": greedy[i],
                               "hyps": [[float(s), " ".join(vocab.decode(h))] for s, h in nb[: args.beam]]}
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i in pick:
                f.write(json.dumps(rows[i]) + "\n")
        print(split, len(rows), flush=True)


if __name__ == "__main__":
    main()
