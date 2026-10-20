"""Precompute soft targets for distillation (the memory used at training time only).

    python scripts/soft_targets.py TEACHER.pt --mode knn|self [--data DIR] [--k 16] [--tau 0.05] --out FILE.npz

For every training formula (store order) and every predicted position (the
targets ``target[1:]`` of teacher forcing) stores a K-entry distribution:

* ``self`` -- the teacher's own top-K next-token probabilities (plain knowledge
  distillation, the control);
* ``knn``  -- how the K nearest decoder states of *other* training formulas
  continued (exact cosine search over the whole training datastore, the
  query's own formula excluded), weighted by ``softmax(sim / tau)``.

Output: ``ids`` (N, K) int32, ``w`` (N, K) float32 (rows sum to 1), ``offsets``
(n + 1) with N = sum of (len(target) - 1).  Deterministic (exact top-k).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from decode_study import batches  # noqa: E402
from knn_decode import hidden_states  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher", type=Path)
    ap.add_argument("--mode", choices=["knn", "self"], required=True)
    ap.add_argument("--data", type=Path, default=Path("/mnt/SSD/MathOCR/probe/data20k"))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(args.teacher, device)
    store = CropStore(args.data / "train.parquet", vocab)
    n = len(store)
    lens = np.diff(store.target_offsets) - 1  # predicted positions per formula
    offsets = np.concatenate([[0], np.cumsum(lens)])
    total, k = int(offsets[-1]), args.k
    keys = torch.zeros(total, model.cfg.d_model, dtype=torch.float16, device=device)
    vals = torch.zeros(total, dtype=torch.long, device=device)
    ids = np.zeros((total, k), dtype=np.int32)
    w = np.zeros((total, k), dtype=np.float32)
    with torch.no_grad():
        for idx, batch in batches(store, icfg, range(n), 1.0, max_batch=32):
            x, t = batch["images"].to(device), batch["tokens"].to(device)
            h = hidden_states(model, x, t[:, :-1]).float()
            if args.mode == "self":
                p = torch.softmax(torch.nn.functional.linear(h, model.embed.weight.float()), dim=-1)
                top_p, top_i = p.topk(k, dim=-1)
                top_p = top_p / top_p.sum(-1, keepdim=True)
            for row, i in enumerate(idx):
                lo, hi = offsets[i], offsets[i + 1]
                keys[lo:hi] = torch.nn.functional.normalize(h[row, : hi - lo], dim=-1).half()
                vals[lo:hi] = t[row, 1 : 1 + hi - lo]
                if args.mode == "self":
                    ids[lo:hi] = top_i[row, : hi - lo].cpu().numpy()
                    w[lo:hi] = top_p[row, : hi - lo].cpu().numpy()
        if args.mode == "knn":
            owner = torch.from_numpy(np.repeat(np.arange(n), lens)).to(device)
            margin = 64  # neighbours from the query's own formula are dropped after the search
            chunk = 512  # 512 x N fp16 similarities ~ 1.4 GiB
            for s in range(0, total, chunk):
                q = keys[s : s + chunk]
                sim, nb = (q @ keys.T).topk(k + margin, dim=-1)
                sim = sim.float()
                own = owner[nb] == owner[s : s + chunk, None]
                sim = sim.masked_fill(own, float("-inf"))
                order = sim.argsort(dim=-1, descending=True, stable=True)[:, :k]
                sim, nb = sim.gather(1, order), nb.gather(1, order)
                wt = torch.softmax(sim / args.tau, dim=-1).nan_to_num(0.0)
                ids[s : s + chunk] = vals[nb].cpu().numpy()
                w[s : s + chunk] = wt.cpu().numpy()
    np.savez(args.out, ids=ids, w=w, offsets=offsets)
    agree = float((w * (ids == vals.cpu().numpy()[:, None])).sum() / total)
    print({"positions": total, "mode": args.mode, "mass_on_true_token": round(agree, 3)})


if __name__ == "__main__":
    main()
