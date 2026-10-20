"""Memory-augmented decoding (kNN-LM): the decoder consults exact neighbours from the training set.

    python scripts/knn_decode.py CHECKPOINT [--data DIR] [--k 16] --out OUT.json

Datastore: for every training formula (teacher forcing, no augmentation) and every
position, the decoder's final hidden state (after the last LayerNorm,
L2-normalized) as key and the next token as value.  Decoding is greedy on

    p(v) = (1 - lam) p_model(v) + lam * sum_{j in kNN(h)} softmax_j(<h, k_j> / tau) [v_j = v]

with exact top-k search on the GPU (deterministic).  Notation that cannot be
read off the image (grouping braces, spacing, \\cal vs \\mathcal) is a habit of
the author; exact neighbours carry such habits where the averaged model cannot.
Only training formulas enter the datastore.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from decode_study import batches  # noqa: E402
from probe import score  # noqa: E402


def hidden_states(model, images, tokens_in):
    """Final decoder states (B, T, d) under teacher forcing (same path as ``forward``)."""

    feats = [] if model.deform is not None else None
    memory = model.encode(images, feats)
    mem_kv = model._memory_kv(memory)
    depth = model._depth_step(tokens_in).cumsum(dim=1) if model.depth_embed is not None else None
    x = model._embed(tokens_in, 0, depth)
    x = model._decode(x, images, mem_kv, None, None, valid=tokens_in != 0, feats=feats)
    return model.dec_norm(x)


@torch.no_grad()
def build_datastore(model, store, icfg, vocab, device, limit=0, basis=None):
    """Keys: L2-normalized final decoder states (projected on ``basis`` (d, r) and renormalized
    when given); values: next tokens."""

    keys, vals = [], []
    for _, batch in batches(store, icfg, range(limit or len(store)), 1.0, max_batch=32):
        x, t = batch["images"].to(device), batch["tokens"].to(device)
        h = torch.nn.functional.normalize(hidden_states(model, x, t[:, :-1]).float(), dim=-1)
        tgt = t[:, 1:]
        keep = tgt != vocab.pad_id
        h = h[keep]
        if basis is not None:
            h = torch.nn.functional.normalize(h @ basis, dim=-1)
        keys.append(h.half())
        vals.append(tgt[keep].int())
    return torch.cat(keys), torch.cat(vals).long()


@torch.no_grad()
def pca_basis(model, store, icfg, vocab, device, rank: int, sample: int = 3000) -> torch.Tensor:
    """Top-``rank`` principal directions of the normalized states of the first ``sample``
    formulas (covariance eigendecomposition in float64 on the CPU: deterministic)."""

    keys, _ = build_datastore(model, store, icfg, vocab, device, limit=sample)
    k = keys.double().cpu()
    cov = (k - k.mean(0)).T @ (k - k.mean(0)) / k.shape[0]
    _, vecs = torch.linalg.eigh(cov)
    return vecs[:, -rank:].flip(1).float().to(device)


@torch.no_grad()
def knn_greedy(model, images, vocab, keys, vals, k, tau, lams, max_len=None, basis=None):
    """Greedy decodes for several mixing weights at once -> {lam: list of id lists}."""

    out = {}
    for lam in lams:
        b = images.shape[0]
        feats = [] if model.deform is not None else None
        memory = model.encode(images, feats)
        mem_kv = model._memory_kv(memory)
        caches = [{} for _ in model.decoder]
        token = torch.full((b, 1), vocab.bos_id, dtype=torch.long, device=images.device)
        depth = torch.zeros(b, 1, dtype=torch.long, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        seqs = []
        for step in range(min(max_len or model.cfg.max_len, model.cfg.max_len) - 1):
            depth = depth + model._depth_step(token)
            x = model._decode(model._embed(token, step, depth), images, mem_kv, caches, None, feats=feats)
            h = model.dec_norm(x)[:, -1].float()
            p = torch.softmax(torch.nn.functional.linear(h, model.embed.weight.float()), dim=-1)
            if lam > 0:
                hq = torch.nn.functional.normalize(h, dim=-1)
                if basis is not None:
                    hq = torch.nn.functional.normalize(hq @ basis, dim=-1)
                sim = hq.half() @ keys.T  # (B, N)
                top_s, top_i = sim.float().topk(k, dim=-1)
                w = torch.softmax(top_s / tau, dim=-1)
                pk = torch.zeros_like(p).scatter_add_(1, vals[top_i], w)
                p = (1 - lam) * p + lam * pk
            token = p.argmax(dim=-1, keepdim=True).masked_fill(done[:, None], vocab.eos_id)
            seqs.append(token)
            done |= token[:, 0] == vocab.eos_id
            if bool(done.all()):
                break
        ids = torch.cat(seqs, dim=1).tolist()
        out[lam] = [row[: (row + [vocab.eos_id]).index(vocab.eos_id)] for row in ids]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("/mnt/SSD/MathOCR/probe/data20k"))
    ap.add_argument("--splits", nargs="*", default=["val", "ood_cpe", "ood_sce", "ood_spe"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lams", nargs="*", type=float, default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N formulas of every split (and of train)")
    ap.add_argument("--train-data", type=Path, default=None, help="datastore split dir (default: --data)")
    ap.add_argument("--limit-eval", type=int, default=0, help="first N formulas of every evaluated split")
    ap.add_argument("--pca", type=int, default=0, help="project keys on the top-r principal directions")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(args.checkpoint, device)
    train = CropStore((args.train_data or args.data) / "train.parquet", vocab)
    basis = pca_basis(model, train, icfg, vocab, device, args.pca) if args.pca else None
    keys, vals = build_datastore(model, train, icfg, vocab, device, args.limit, basis)
    print(json.dumps({"datastore_entries": int(keys.shape[0]), "gib": round(keys.numel() * 2 / 2**30, 2)}), flush=True)
    result = {"checkpoint": str(args.checkpoint), "k": args.k, "tau": args.tau, "splits": {}}
    for split in args.splits:
        store = CropStore(args.data / f"{split}.parquet", vocab)
        n = min(len(store), args.limit or args.limit_eval or len(store))
        preds = {lam: {} for lam in args.lams}
        for ids, batch in batches(store, icfg, range(n), 1.0, max_batch=16):
            for lam, seqs in knn_greedy(model, batch["images"].to(device), vocab, keys, vals, args.k, args.tau,
                                        args.lams, basis=basis).items():
                preds[lam].update(zip(ids, seqs))
        result["splits"][split] = {}
        result.setdefault("preds", {})[split] = {str(lam): [" ".join(vocab.decode(preds[lam][i])) for i in range(n)]
                                                 for lam in args.lams}
        for lam in args.lams:
            m = score([vocab.decode(preds[lam][i]) for i in range(n)], store.labels[:n])
            result["splits"][split][str(lam)] = m
            print(json.dumps({"split": split, "lam": lam, **{k: round(v, 4) for k, v in m.items()}}), flush=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
