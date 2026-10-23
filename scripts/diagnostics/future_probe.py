"""(Paths below are those of the Windows laptop; edit them for another machine.)
Does one decoder step carry more than its next token?  final_v2_s1 frozen (CPU), teacher-forced states at input
position t: z_t (dec_norm output, trained only for y_{t+1}) and, as the floor, x0_t (the decoder input: token y_t
+ position = what the text alone says).  Probes z_t -> y_{t+1+j}, j = 0..7 (linear and one-hidden-layer MLP),
fitted on train states, scored on val; accuracy over all targets and over "visible" ones (not { } or spacing).
Output: docs/probe_results/future_probe_final_v2_s1.json."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from notation_pools import SPACES, STYLE  # noqa: E402

torch.set_num_threads(8)
torch.manual_seed(0)
dev = torch.device("cpu")
model, vocab, icfg, _ = load_checkpoint(Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\checkpoints\best.pt"), dev)
DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
J = 8
V = len(vocab)
invisible = torch.zeros(V, dtype=torch.bool)
for t in {"{", "}"} | SPACES | STYLE:
    if t in vocab.stoi:
        invisible[vocab.stoi[t]] = True


@torch.no_grad()
def states(split: str, n: int):
    store = CropStore(DATA / f"{split}.parquet", vocab)
    pick = [i for i in np.random.default_rng(0).permutation(len(store)).tolist()
            if len(store.labels[i].split()) < 200][:n]
    items = TrainItems(store, icfg, aug_cfg=None, seed=0)
    zs, xs, ys = [], [], []
    for keys in ShapeBatchSampler(store, icfg, 400_000, 32, seed=0, train=False, indices=pick, max_tokens=model.cfg.max_len):
        b = collate([items[k] for k in keys], vocab.pad_id)
        tok = b["tokens"]
        inp = tok[:, :-1]
        x0 = model._embed(inp, 0)
        x = model._decode(x0, b["images"], model._memory_kv(model.encode(b["images"])), None, None)
        z = model.dec_norm(x)
        # targets y_{t+1+j}; pad (= ignore) past the end of each sequence
        tgt = torch.full((*inp.shape, J), vocab.pad_id, dtype=torch.long)
        for j in range(J):
            tgt[:, : tok.shape[1] - 1 - j, j] = tok[:, 1 + j :]
        m = inp != vocab.pad_id
        zs.append(z[m])
        xs.append(x0[m])
        ys.append(tgt[m])
    return torch.cat(zs), torch.cat(xs), torch.cat(ys)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, V))

    def forward(self, z):
        return self.net(z)


def fit(head, z, y, epochs=3, bs=4096):
    keep = y != vocab.pad_id
    z, y = z[keep], y[keep]
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for i in torch.randperm(len(y), generator=g).split(bs):
            loss = F.cross_entropy(head(z[i]), y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def acc(head, z, y):
    keep = y != vocab.pad_id
    z, y = z[keep], y[keep]
    hit = head(z).argmax(-1) == y
    vis = ~invisible[y]
    return float(hit.float().mean()), float(hit[vis].float().mean())


zt, xt, yt = states("train", 3000)
zv, xv, yv = states("val", 800)
print(f"states: train {len(yt)}, val {len(yv)}", flush=True)
res = {}
for j in range(J):
    r = {}
    r["z linear"] = acc(fit(nn.Sequential(nn.LayerNorm(512), nn.Linear(512, V)), zt, yt[:, j]), zv, yv[:, j])
    r["z mlp"] = acc(fit(MLP(512), zt, yt[:, j]), zv, yv[:, j])
    r["x0 mlp"] = acc(fit(MLP(512), xt, yt[:, j]), xv, yv[:, j])
    res[f"t+{1 + j}"] = r
    print(f"y_(t+{1 + j}): " + "  ".join(f"{k} {a:.3f} (visible {b:.3f})" for k, (a, b) in r.items()), flush=True)
(ROOT / "docs" / "probe_results" / "future_probe_final_v2_s1.json").write_text(json.dumps(res, indent=1),
                                                                               encoding="utf-8", newline="\n")
