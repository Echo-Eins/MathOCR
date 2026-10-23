"""(Paths below are those of the Windows laptop; edit them for another machine.)
Is the output head a bottleneck?  final_v2_s1 frozen; its final latents (dec_norm output, teacher forcing) on
train / val / SPE; heads trained on the train latents:
  tied   -- the model's own head (no training)
  linear -- the same head re-fit (init: tied embedding)
  cells  -- one small MLP per vocabulary cell on a copy of the latent (hidden k per cell), all cells connected
            to each other (a V x V lateral layer over the cell activations), added to the linear logits
  mlp    -- a plain d -> 4d -> V MLP added to the linear logits (capacity control for "cells")
Val/SPE cross-entropy and token accuracy; accuracy on the tokens the tied head gets wrong."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402

RUN = Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1")
DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
dev = torch.device("cuda")
determinism.configure(dev)
model, vocab, icfg, _ = load_checkpoint(RUN / "checkpoints" / "best.pt", dev)
V, d = len(vocab), model.cfg.d_model


@torch.no_grad()
def latents(split: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    store = CropStore(DATA / f"{split}.parquet", vocab)
    pool = [i for i in np.random.default_rng(0).permutation(len(store)).tolist()
            if len(store.labels[i].split()) < model.cfg.max_len - 2][:n]
    items = TrainItems(store, icfg, aug_cfg=None, seed=0)
    zs, ys = [], []
    for keys in ShapeBatchSampler(store, icfg, 400_000, 32, seed=0, train=False, indices=pool, max_tokens=model.cfg.max_len):
        b = collate([items[k] for k in keys], vocab.pad_id)
        images, tokens = b["images"].to(dev), b["tokens"].to(dev)
        inp, out = tokens[:, :-1], tokens[:, 1:]
        memory = model.encode(images)
        x = model._decode(model._embed(inp, 0), images, model._memory_kv(memory), None, None)
        m = out != vocab.pad_id
        zs.append(model.dec_norm(x)[m].half())
        ys.append(out[m])
    return torch.cat(zs), torch.cat(ys)


class Cells(nn.Module):
    def __init__(self, k: int = 16):
        super().__init__()
        self.lin = nn.Linear(d, V)
        self.w1 = nn.Parameter(torch.randn(d, V * k) * d**-0.5)
        self.b1 = nn.Parameter(torch.zeros(V * k))
        self.w2 = nn.Parameter(torch.zeros(V, k))
        self.lateral = nn.Parameter(torch.zeros(V, V))  # cell j -> cell i
        self.k = k

    def forward(self, z):
        h = F.gelu(z @ self.w1 + self.b1).view(-1, V, self.k)
        s = (h * self.w2).sum(-1)
        return self.lin(z) + s + F.gelu(s) @ self.lateral.T


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(d, V)

    def forward(self, z):
        return self.lin(z)


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(d, V)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, V))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, z):
        return self.lin(z) + self.mlp(z)


def fit(head: nn.Module, z, y, epochs: int = 4, lr: float = 1e-3, bs: int = 8192) -> nn.Module:
    with torch.no_grad():
        head.lin.weight.copy_(model.embed.weight)
        head.lin.bias.zero_()
    head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(y) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=dev, generator=g)
        for i in range(0, len(y), bs):
            idx = perm[i : i + bs]
            loss = F.cross_entropy(head(z[idx].float()), y[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return head.eval()


@torch.no_grad()
def score(f, z, y, wrong):
    logits = torch.cat([f(z[i : i + 16384].float()) for i in range(0, len(y), 16384)])
    ce = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float()
    return ce, acc.mean().item(), acc[wrong].mean().item()


if __name__ == "__main__":
    zt, yt = latents("train", 12000)
    zv, yv = latents("val", 2000)
    zs, ys = latents("ood_spe", 1500)
    print(f"latents: train {len(yt)}, val {len(yv)}, SPE {len(ys)} tokens", flush=True)
    tied = lambda z: F.linear(z, model.embed.weight)  # noqa: E731
    heads = {"tied": tied}
    heads["linear"] = fit(LinearHead(), zt, yt)
    heads["cells"] = fit(Cells(), zt, yt)
    heads["mlp"] = fit(MLPHead(), zt, yt)
    with torch.no_grad():
        wv = (tied(zv.float()).argmax(-1) != yv)
        ws = (tied(zs.float()).argmax(-1) != ys)
    print(f"tied head errors: val {wv.float().mean().item() * 100:.2f}% of tokens, SPE {ws.float().mean().item() * 100:.2f}%")
    for name, f in heads.items():
        n_par = sum(p.numel() for p in f.parameters()) if isinstance(f, nn.Module) else V * d
        tr = score(f, zt[:200000], yt[:200000], torch.ones(min(200000, len(yt)), dtype=torch.bool, device=dev))
        va, sp = score(f, zv, yv, wv), score(f, zs, ys, ws)
        print(f"{name:7} params {n_par / 1e6:5.2f}M | train CE {tr[0]:.4f} acc {tr[1]:.4f} | val CE {va[0]:.4f} acc {va[1]:.4f} "
              f"(on tied-wrong {va[2]:.3f}) | SPE CE {sp[0]:.4f} acc {sp[1]:.4f} (on tied-wrong {sp[2]:.3f})", flush=True)
