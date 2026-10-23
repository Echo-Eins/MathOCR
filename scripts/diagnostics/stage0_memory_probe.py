"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0a: does the decoder lose information that sits in the encoder memory where it reads?

Frozen model, teacher forcing.  For every target position (predicting y_{t+1}):
  z      -- the final latent (dec_norm output, what the output head reads; probes on it are residual on top of
            the deployed tied head, so they start exactly at it);
  c_last -- the encoder memory pooled with the last decoder layer's cross-attention (mean over heads);
  c_all  -- the memory pooled with each of the six layers' cross-attention, concatenated.
Identical MLP probes (LayerNorm -> 1024 -> V) on z, c_last, c_all, [z; c_last], [z; c_all], fitted on train
positions with early stopping on a held-out 5 % of the train formulas; val and OOD are only scored.  Reported per
split: accuracy (all / visible / notation tokens), the share of the deployed head's errors a probe fixes and of
its correct tokens it breaks, and paired differences against the z probe with a formula-level bootstrap.

    python scripts/diagnostics/stage0_memory_probe.py --name final_v2 \
        --ckpt C:/Coding/Python/MathOCR_final/runs/final_v2_s1/checkpoints/best.pt \
        --train C:/Coding/Python/MathOCR_final/data/processed/train.parquet
Output: docs/probe_results/stage0_memory_probe_<name>.json."""

import argparse
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
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from notation_pools import SPACES, STYLE  # noqa: E402

EVAL_DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
dev = torch.device("cuda")


@torch.no_grad()
def collect(model, vocab, icfg, path: Path, n: int, seed: int = 0) -> dict:
    store = CropStore(path, vocab)
    limit = model.cfg.max_len - 2
    pool = [i for i in np.random.default_rng(seed).permutation(len(store)).tolist()
            if len(store.labels[i].split()) <= limit][:n]
    items = TrainItems(store, icfg, aug_cfg=None, seed=0)
    zs, cl, ca, ys, ok, fid = [], [], [], [], [], []
    for keys in ShapeBatchSampler(store, icfg, 150_000, 8, seed=0, train=False, indices=pool, max_tokens=model.cfg.max_len):
        b = collate([items[k] for k in keys], vocab.pad_id)
        images, tokens = b["images"].to(dev), b["tokens"].to(dev)
        inp, out = tokens[:, :-1], tokens[:, 1:]
        memory = model.encode(images)
        x = model._embed(inp, 0)
        pooled = []
        for layer, kv in zip(model.decoder, model._memory_kv(memory)):
            x, w = layer(x, kv, None, return_cross=True)  # w: (B, T, N), mean over heads
            pooled.append(torch.bmm(w.to(memory.dtype), memory))
        z = model.dec_norm(x)
        pred = F.linear(z, model.embed.weight).argmax(-1)
        m = out != vocab.pad_id
        zs.append(z[m].half().cpu())
        cl.append(pooled[-1][m].half().cpu())
        ca.append(torch.cat(pooled, dim=-1)[m].half().cpu())
        ys.append(out[m].cpu())
        ok.append((pred == out)[m].cpu())
        fid.append(b["index"].to(dev)[:, None].expand_as(out)[m].cpu())
    return {"z": torch.cat(zs), "c_last": torch.cat(cl), "c_all": torch.cat(ca), "y": torch.cat(ys),
            "tied_ok": torch.cat(ok), "formula": torch.cat(fid)}


class Probe(nn.Module):
    """MLP probe.  ``tied`` (the model's output embedding) makes it residual on top of the deployed head: logits =
    tied(z) + MLP(features) with a zero-initialized last layer, so it starts exactly at the deployed head and any
    gain over the z-only residual probe (same capacity) comes from the extra inputs."""

    def __init__(self, d_in: int, vocab: int, tied: torch.Tensor | None = None):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 1024), nn.GELU(), nn.Linear(1024, vocab))
        self.register_buffer("tied", tied)
        if tied is not None:
            nn.init.zeros_(self.net[3].weight)
            nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        out = self.net(x)
        if self.tied is not None:
            out = out + F.linear(x[:, : self.tied.shape[1]], self.tied)
        return out


def features(s: dict, kind: str) -> torch.Tensor:
    parts = {"z": ["z"], "c_last": ["c_last"], "c_all": ["c_all"], "z+c_last": ["z", "c_last"], "z+c_all": ["z", "c_all"]}[kind]
    return torch.cat([s[p] for p in parts], dim=-1)


def fit(kind: str, tr: dict, vocab_size: int, tied: torch.Tensor | None, seed: int = 0, epochs: int = 6,
        bs: int = 4096) -> Probe:
    torch.manual_seed(seed)
    x, y = features(tr, kind), tr["y"]
    formulas = tr["formula"].unique()
    g = torch.Generator().manual_seed(seed)
    held = set(formulas[torch.randperm(len(formulas), generator=g)[: max(1, len(formulas) // 20)]].tolist())
    hold = torch.tensor([f in held for f in tr["formula"].tolist()])
    xt, yt, xh, yh = x[~hold], y[~hold], x[hold].to(dev).float(), y[hold].to(dev)
    probe = Probe(x.shape[1], vocab_size, tied if kind.startswith("z") else None).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=5e-4 if kind.startswith("z") else 1e-3, weight_decay=0.01)
    probe.eval()
    with torch.no_grad():  # epoch 0 = the starting point (for residual probes: the deployed head itself)
        best = torch.cat([(probe(xh[j]).argmax(-1) == yh[j]) for j in torch.arange(len(yh)).split(16384)]).float().mean().item()
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    for _ in range(epochs):
        probe.train()
        for i in torch.randperm(len(yt), generator=g).split(bs):
            loss = F.cross_entropy(probe(xt[i].to(dev).float()), yt[i].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            acc = torch.cat([(probe(xh[j]).argmax(-1) == yh[j]) for j in torch.arange(len(yh)).split(16384)]).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    return probe.eval()


@torch.no_grad()
def predict(probe: Probe, s: dict, kind: str) -> torch.Tensor:
    x = features(s, kind)
    return torch.cat([probe(x[j].to(dev).float()).argmax(-1).cpu() for j in torch.arange(len(x)).split(16384)])


def boot_ci(diff: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0):
    """Formula-level (cluster) bootstrap of the mean of a per-token paired difference."""
    rng = np.random.default_rng(seed)
    uniq, inv_idx = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv_idx, weights=diff)
    cnts = np.bincount(inv_idx)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        means.append(sums[pick].sum() / cnts[pick].sum())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--train-images", type=int, default=6000)
    ap.add_argument("--eval-scale", type=float, default=1.0, help="shrink the eval sets (smoke tests)")
    args = ap.parse_args()
    determinism.configure(dev)
    model, vocab, icfg, _ = load_checkpoint(args.ckpt, dev)
    invisible = torch.zeros(len(vocab), dtype=torch.bool)
    for t in {"{", "}"} | SPACES | STYLE:
        if t in vocab.stoi:
            invisible[vocab.stoi[t]] = True
    tr = collect(model, vocab, icfg, args.train, args.train_images)
    evals = {s: collect(model, vocab, icfg, EVAL_DATA / f"{s}.parquet", max(10, int(n * args.eval_scale)))
             for s, n in (("val", 1500), ("ood_spe", 1500), ("ood_sce", 1500), ("ood_cpe", 600))}
    print(f"{args.name}: train positions {len(tr['y'])}; " + ", ".join(f"{k} {len(v['y'])}" for k, v in evals.items()), flush=True)
    kinds = ("z", "c_last", "c_all", "z+c_last", "z+c_all")
    tied = model.embed.weight.detach().float()
    probes = {k: fit(k, tr, len(vocab), tied) for k in kinds}
    res = {}
    for split, s in evals.items():
        y, ok = s["y"], s["tied_ok"]
        vis = ~invisible[y]
        preds = {k: predict(p, s, k) for k, p in probes.items()}
        hit = {k: (p == y) for k, p in preds.items()}
        row = {"tokens": int(len(y)), "tied": {"all": float(ok.float().mean()), "visible": float(ok[vis].float().mean()),
                                                "notation": float(ok[~vis].float().mean())}}
        for k, h in hit.items():
            row[k] = {"all": float(h.float().mean()), "visible": float(h[vis].float().mean()),
                      "notation": float(h[~vis].float().mean()),
                      "fixes_tied_errors": float(h[~ok].float().mean()), "breaks_tied_correct": float((~h[ok]).float().mean())}
            if k != "z":
                d = (h.float() - hit["z"].float()).numpy()
                dv = d[vis.numpy()]
                row[k]["vs_z_all"] = [float(d.mean()), *boot_ci(d, s["formula"].numpy())]
                row[k]["vs_z_visible"] = [float(dv.mean()), *boot_ci(dv, s["formula"].numpy()[vis.numpy()])]
        res[split] = row
        print(f"  {split:8} tied {row['tied']['all']:.4f} (visible {row['tied']['visible']:.4f})", flush=True)
        for k in kinds:
            r = row[k]
            extra = ""
            if "vs_z_visible" in r:
                m, lo, hi = r["vs_z_visible"]
                extra = f" | visible vs z {m * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]"
            print(f"    {k:9} all {r['all']:.4f} visible {r['visible']:.4f} notation {r['notation']:.4f} "
                  f"fixes {r['fixes_tied_errors']:.3f} breaks {r['breaks_tied_correct']:.4f}{extra}", flush=True)
    out = ROOT / "docs" / "probe_results" / f"stage0_memory_probe_{args.name}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
