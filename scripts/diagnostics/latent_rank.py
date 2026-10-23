"""(Paths below are those of the Windows laptop; edit them for another machine.)
How many directions does the decoder need to carry the formula to the output?  final_v2_s1 frozen, real greedy
decoding (CPU, no GPU load) of 400 stand val formulas with
  latent r -- the final latent (dec_norm output, what the head reads) projected on its top-r principal directions;
  memory r -- the encoder memory (the image "thought" every decoder layer reads) projected likewise;
  random r -- the final latent projected on a random r-dimensional subspace (does the PCA structure matter?).
PCA from teacher-forced train states.  EM and notation-normalized EM (the same rendered formula).
Output: docs/probe_results/latent_rank_final_v2_s1.json."""

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("PROBE_ROOT", r"D:\MathOCR\probe")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import probe  # noqa: E402
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.model import Im2Latex  # noqa: E402
from notation_pools import inv  # noqa: E402

torch.set_num_threads(10)
dev = torch.device("cpu")
model, vocab, icfg, _ = load_checkpoint(Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\checkpoints\best.pt"), dev)
DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
model.cfg = dataclasses.replace(model.cfg, max_len=256)  # decode cap: stand val formulas are all < 254 tokens; truncated latents that never
                         # emit EOS would otherwise run to 511 steps (slow on CPU), the scores do not change


@torch.no_grad()
def train_states(n_images: int = 1500):
    store = CropStore(DATA / "train.parquet", vocab)
    pick = [i for i in np.random.default_rng(0).permutation(len(store)).tolist()
            if len(store.labels[i].split()) < 200][:n_images]
    items = TrainItems(store, icfg, aug_cfg=None, seed=0)
    zs, ms = [], []
    for keys in ShapeBatchSampler(store, icfg, 400_000, 32, seed=0, train=False, indices=pick, max_tokens=model.cfg.max_len):
        b = collate([items[k] for k in keys], vocab.pad_id)
        inp = b["tokens"][:, :-1]
        memory = model.encode(b["images"])
        x = model._decode(model._embed(inp, 0), b["images"], model._memory_kv(memory), None, None)
        zs.append(model.dec_norm(x)[inp != vocab.pad_id])
        m = memory.reshape(-1, memory.shape[-1])
        ms.append(m[torch.randperm(len(m), generator=torch.Generator().manual_seed(len(ms)))[:400]])
    return torch.cat(zs).double(), torch.cat(ms).double()


def pca(x: torch.Tensor):
    mu = x.mean(0)
    _, s, vt = torch.linalg.svd(x - mu, full_matrices=False)
    return mu, vt, (s**2) / (s**2).sum()


def projector(mu, basis, r):
    u = basis[:r].T.float()  # (d, r)
    mu = mu.float()
    return lambda z: mu + ((z - mu) @ u) @ u.T


zt, mt = train_states()
mu_z, v_z, ev_z = pca(zt)
mu_m, v_m, ev_m = pca(mt)
print(f"PCA from {len(zt)} latents / {len(mt)} memory cells; variance in top 8/32/128: latent "
      f"{ev_z[:8].sum():.2f}/{ev_z[:32].sum():.2f}/{ev_z[:128].sum():.2f}, memory "
      f"{ev_m[:8].sum():.2f}/{ev_m[:32].sum():.2f}/{ev_m[:128].sum():.2f}", flush=True)
q, _ = torch.linalg.qr(torch.randn(512, 512, generator=torch.Generator().manual_seed(0), dtype=torch.float64))
rand_basis = q.T

val = CropStore(probe.DATA / "val.parquet", vocab)
idx = sorted(np.random.default_rng(1).choice(len(val), 400, replace=False).tolist())
gts = [val.labels[i] for i in idx]
orig_kv = Im2Latex._memory_kv.__get__(model)


def run(name: str, latent=None, memory=None) -> dict:
    model.__dict__.pop("_logits", None)
    model.__dict__.pop("_memory_kv", None)
    if latent is not None:
        model._logits = lambda x: F.linear(latent(model.dec_norm(x)), model.embed.weight)
    if memory is not None:
        model._memory_kv = lambda m: orig_kv(memory(m))
    preds = probe.decode(model, val, icfg, vocab, dev, indices=idx)
    toks = [[t for t in p if t != "<noeos>"] for p in preds]
    em = float(np.mean([p == g.split() for p, g in zip(toks, gts)]))
    em_inv = float(np.mean([inv(p) == inv(g.split()) for p, g in zip(toks, gts)]))
    runaway = float(np.mean([p[-1:] == ["<noeos>"] for p in preds]))
    print(f"{name:12} EM {em:.3f}  EM modulo notation {em_inv:.3f}  runaway {runaway:.3f}", flush=True)
    return {"em": em, "em_notation_free": em_inv, "runaway": runaway}


res = {"full": run("full")}
for r in (4, 8, 16, 32, 64, 128, 256):
    res[f"latent {r}"] = run(f"latent {r}", latent=projector(mu_z, v_z, r))
for r in (16, 64):
    res[f"random {r}"] = run(f"random {r}", latent=projector(mu_z, rand_basis, r))
for r in (8, 16, 32, 64, 128, 256):
    res[f"memory {r}"] = run(f"memory {r}", memory=projector(mu_m, v_m, r))
res["variance"] = {"latent": ev_z.cumsum(0)[[7, 15, 31, 63, 127, 255]].tolist(),
                   "memory": ev_m.cumsum(0)[[7, 15, 31, 63, 127, 255]].tolist()}
(ROOT / "docs" / "probe_results" / "latent_rank_final_v2_s1.json").write_text(json.dumps(res, indent=1),
                                                                              encoding="utf-8", newline="\n")
