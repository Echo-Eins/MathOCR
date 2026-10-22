"""(Paths below are those of the Windows laptop; edit them for another machine.)
Singular-value spectra of the weights and of the decoder itself.

1. Every weight matrix (convs reshaped to out x in*kh*kw; kv split into K and V): stable rank
   ||W||_F^2 / s_max^2, effective rank exp(H(s / sum s)) (Roy & Vetterli), the number of singular values
   above the Marchenko-Pastur edge of a random matrix with the same shape and Frobenius norm, and the Hill
   tail exponent alpha of the eigenvalues s^2 (top 10 %; heavy-tailed self-regularization: 2..4 "well
   trained", < 2 over-fitted).  Muon final, the Muon probe, its AdamW twin, and the random init.
2. Decoder circuits per head: QK = W_Q^T W_K and OV = W_O W_V (with the pre-norm gain), self and cross
   attention; for self-attention OV (a map of the residual stream onto itself) also the eigenvalues:
   copying score sum Re(l) / sum |l| (> 0: the head writes back what it read).
3. The decoder as a function, at teacher-forced val states: Jacobian of every layer at the same position
   (d x_out[t] / d x_in[t], 512 x 512) and of the whole stack (d x_6[t] / d x_0[t]); singular values of J
   and of the residual branch J - I.  Plus the spectrum of the residual-stream covariance per layer
   (participation ratio = how many of the 512 dimensions the stream really uses).
Output: docs/probe_results/spectra.json and printed tables."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.model import Im2Latex  # noqa: E402

CKPTS = {
    "final_v2 Muon": Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\checkpoints\best.pt"),
    "probe Muon": Path(r"D:\MathOCR\probe\runs\win_m_lr6_s8x4_g18_s1\checkpoints\best.pt"),
    "probe AdamW": Path(r"D:\MathOCR\probe\runs\win_adamw_s8x4_g18_s1\checkpoints\best.pt"),
}
VAL = Path(r"C:\Coding\Python\MathOCR_final\data\processed\val.parquet")
OUT = Path(__file__).resolve().parents[2] / "docs" / "probe_results" / "spectra.json"
dev = torch.device("cuda")
determinism.configure(dev)


@torch.no_grad()
def matrix_stats(w: torch.Tensor) -> dict:
    w = w.detach().double().flatten(1)
    m, n = w.shape
    s = torch.linalg.svdvals(w)
    k = s.numel()
    fro2 = float((s**2).sum())
    p = s / s.sum()
    erank = math.exp(float(-(p * torch.log(p.clamp_min(1e-300))).sum()))
    edge = math.sqrt(fro2 / (m * n)) * (math.sqrt(m) + math.sqrt(n))
    lam = (s**2).cpu().numpy()
    kk = max(5, k // 10)
    alpha = 1.0 + kk / float(np.sum(np.log(lam[:kk] / lam[kk])))
    return {"shape": [m, n], "smax": float(s[0]), "stable_rank": fro2 / float(s[0]) ** 2 / k,
            "eff_rank": erank / k, "above_mp": int((s > edge).sum()), "alpha": alpha,
            "top10_energy": float((s[: max(1, k // 10)] ** 2).sum()) / fro2}


def named_matrices(model: Im2Latex) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        if name.endswith("kv.weight"):
            k, v = p.chunk(2, dim=0)
            out[name.replace("kv.weight", "k")], out[name.replace("kv.weight", "v")] = k, v
        else:
            out[name.removesuffix(".weight")] = p
    return out


def group_of(name: str) -> str:
    if name.startswith("backbone"):
        return "cnn conv"
    if name == "proj" or name == "embed":
        return name
    part = "enc" if name.startswith("encoder") else "dec"
    for key, label in ((".attn.", ""), (".self_attn.", "self "), (".cross_attn.", "cross ")):
        if key in name:
            return f"{part} {label}{name.rsplit('.', 1)[1]}".replace("  ", " ")
    if ".ff.0" in name:
        return f"{part} ff up"
    if ".ff.3" in name:
        return f"{part} ff down"
    return name


@torch.no_grad()
def head_circuits(model: Im2Latex) -> dict:
    d, h = model.cfg.d_model, model.cfg.n_heads
    dh = d // h
    res = defaultdict(list)
    for li, layer in enumerate(model.decoder):
        for kind, att, norm in (("self", layer.self_attn, layer.n1), ("cross", layer.cross_attn, layer.n2)):
            wq = att.q.weight.double()
            wk, wv = (t.double() for t in att.kv.weight.chunk(2, dim=0))
            wo = att.out.weight.double()
            g_q = norm.weight.double()  # the query side always reads the decoder stream through this norm
            g_kv = layer.n1.weight.double() if kind == "self" else None  # cross K/V read the encoder memory
            for hi in range(h):
                sl = slice(hi * dh, (hi + 1) * dh)
                q, k, v, o = wq[sl] * g_q, wk[sl], wv[sl], wo[:, sl]
                if g_kv is not None:
                    k, v = k * g_kv, v * g_kv
                qk = torch.linalg.svdvals(q.T @ k)[:dh]
                ov_m = o @ v
                ov = torch.linalg.svdvals(ov_m)[:dh]
                rec = {"layer": li, "head": hi, "qk_smax": float(qk[0]), "qk_eff": _erank(qk) / dh,
                       "ov_smax": float(ov[0]), "ov_eff": _erank(ov) / dh}
                if kind == "self":
                    lam = torch.linalg.eigvals(ov_m)
                    rec["copy"] = float(lam.real.sum() / lam.abs().sum())
                res[kind].append(rec)
    return res


def _erank(s: torch.Tensor) -> float:
    p = s / s.sum()
    return math.exp(float(-(p * torch.log(p.clamp_min(1e-300))).sum()))


def val_batches(model, vocab, icfg, n_images: int, max_tokens: int):
    store = CropStore(VAL, vocab)
    rng = np.random.default_rng(0)
    pool = [i for i in rng.permutation(len(store)).tolist() if len(store.labels[i].split()) < max_tokens - 2]
    pick = sorted(pool[:n_images])
    items = TrainItems(store, icfg, aug_cfg=None, seed=0)
    for keys in ShapeBatchSampler(store, icfg, 300_000, 16, seed=0, train=False, indices=pick):
        yield collate([items[k] for k in keys], vocab.pad_id)


@torch.no_grad()
def stream_states(model, batch, vocab):
    """Residual stream at the decoder input and after every layer (teacher forcing), valid positions only."""
    images, tokens = batch["images"].to(dev), batch["tokens"].to(dev)
    inp = tokens[:, :-1]
    valid = inp != vocab.pad_id
    memory = model.encode(images)
    kvs = model._memory_kv(memory)
    x = model._embed(inp, 0)
    states = [x[valid]]
    for layer, kv in zip(model.decoder, kvs):
        x = layer(x, kv)
        states.append(x[valid])
    return states, memory, kvs, inp, valid


def cov_stats(x: torch.Tensor) -> dict:
    x = x.double() - x.double().mean(0)
    lam = torch.linalg.eigvalsh(x.T @ x / (x.shape[0] - 1)).flip(0).clamp_min(0)
    c = torch.cumsum(lam, 0) / lam.sum()
    return {"participation": float(lam.sum() ** 2 / (lam**2).sum()), "dims90": int((c < 0.9).sum()) + 1,
            "dims99": int((c < 0.99).sum()) + 1, "top1_share": float(lam[0] / lam.sum())}


def jacobian_stats(model, batch_iter, vocab, n_pos: int = 160) -> dict:
    """Per-layer and whole-stack Jacobians at one position t, the rest of the sequence held fixed."""
    d = model.cfg.d_model
    eye = torch.eye(d, device=dev, dtype=torch.float64)
    acc = defaultdict(list)
    done = 0
    for batch in batch_iter:
        _, memory, kvs, inp, valid = stream_states(model, batch, vocab)
        x0 = model._embed(inp, 0)
        for b in range(inp.shape[0]):
            length = int(valid[b].sum())
            ts = sorted({1, length // 2, length - 1}) if length > 2 else [0]
            kv_b = [(k[b : b + 1], v[b : b + 1]) for k, v in kvs]
            xs = [x0[b : b + 1, :length]]
            with torch.no_grad():
                for layer, kv in zip(model.decoder, kv_b):
                    xs.append(layer(xs[-1], kv))
            for t in ts:
                # Incremental step (the generation path): position t alone, the keys/values of positions < t
                # fixed in the cache -- exactly the diagonal block d x_out[t] / d x_in[t].
                with torch.no_grad():
                    prefix = [layer.self_attn.project_kv(layer.n1(xs[li][:, :t])) if t else None
                              for li, layer in enumerate(model.decoder)]

                def step(layer, x, kv, pre):
                    return layer(x, kv, {} if pre is None else {"k": pre[0], "v": pre[1]})

                for li, (layer, kv) in enumerate(zip(model.decoder, kv_b)):
                    def f(xt, layer=layer, kv=kv, pre=prefix[li]):
                        return step(layer, xt[None, None], kv, pre)[0, 0]

                    j = jacrev(f)(xs[li][0, t]).double()
                    s = torch.linalg.svdvals(j)
                    r = torch.linalg.svdvals(j - eye)
                    acc[f"layer {li}"].append((float(s[0]), float(s[-1]), float(s.median()), float(r[0]),
                                               float(torch.linalg.norm(j - eye) / math.sqrt(d)), _erank(r) / d))

                def g(xt, prefix=prefix):
                    x = xt[None, None]
                    for layer, kv, pre in zip(model.decoder, kv_b, prefix):
                        x = step(layer, x, kv, pre)
                    return x[0, 0]

                j = jacrev(g)(xs[0][0, t]).double()
                s = torch.linalg.svdvals(j)
                r = torch.linalg.svdvals(j - eye)
                acc["stack"].append((float(s[0]), float(s[-1]), float(s.median()), float(r[0]),
                                     float(torch.linalg.norm(j - eye) / math.sqrt(d)), _erank(s) / d))
                done += 1
            if done >= n_pos:
                break
        if done >= n_pos:
            break
    keys = ("smax", "smin", "smed", "res_smax", "res_rms", "eff")
    return {k: dict(zip(keys, np.mean(np.array(v), axis=0).round(4).tolist())) for k, v in acc.items()}


def main() -> None:
    results = {}
    models = {}
    for name, path in CKPTS.items():
        if path.exists():
            models[name] = load_checkpoint(path, dev)
    ref_model, ref_vocab, ref_icfg, _ = models["final_v2 Muon"]
    torch.manual_seed(1)
    init = Im2Latex(ref_model.cfg).to(dev).eval()
    models["random init"] = (init, ref_vocab, ref_icfg, None)

    print("== 1. weight matrices: stable rank / eff. rank (share of min(m,n)), # above MP edge, alpha ==")
    for name, (model, *_rest) in models.items():
        groups = defaultdict(list)
        for mname, w in named_matrices(model).items():
            groups[group_of(mname)].append(matrix_stats(w))
        results.setdefault(name, {})["weights"] = {
            g: {k: float(np.mean([s[k] for s in v])) for k in ("smax", "stable_rank", "eff_rank", "above_mp", "alpha",
                                                              "top10_energy")} | {"n": len(v)}
            for g, v in groups.items()}
    order = list(results["final_v2 Muon"]["weights"])
    print(f"{'group':16}" + "".join(f"{n:>34}" for n in models))
    for g in order:
        cells = []
        for n in models:
            s = results[n]["weights"].get(g)
            cells.append("" if s is None else f"{s['stable_rank']:.3f} {s['eff_rank']:.2f} {s['above_mp']:6.1f} {s['alpha']:5.2f}")
        print(f"{g:16}" + "".join(f"{c:>34}" for c in cells))

    print("\n== 2. decoder heads: QK / OV effective rank (share of 64), self-attention copying score ==")
    for name, (model, *_rest) in models.items():
        c = head_circuits(model)
        results[name]["circuits"] = c
        by_layer = defaultdict(dict)
        for kind, recs in c.items():
            for li in range(model.cfg.dec_layers):
                rl = [r for r in recs if r["layer"] == li]
                by_layer[li][kind] = (np.mean([r["qk_eff"] for r in rl]), np.mean([r["ov_eff"] for r in rl]),
                                      np.mean([r["ov_smax"] for r in rl]),
                                      np.mean([r["copy"] for r in rl]) if kind == "self" else float("nan"),
                                      np.min([r["copy"] for r in rl]) if kind == "self" else float("nan"),
                                      np.max([r["copy"] for r in rl]) if kind == "self" else float("nan"))
        print(f"{name}:")
        for li, v in by_layer.items():
            s, x = v["self"], v["cross"]
            print(f"   L{li}: self QK {s[0]:.2f} OV {s[1]:.2f} |OV| {s[2]:6.2f} copy {s[3]:+.2f} [{s[4]:+.2f}..{s[5]:+.2f}]"
                  f"   cross QK {x[0]:.2f} OV {x[1]:.2f} |OV| {x[2]:6.2f}")

    print("\n== 3. the decoder as a function (val, teacher forcing) ==")
    for name, (model, vocab, icfg, _c) in models.items():
        cov = defaultdict(list)
        mem_states = []
        for bi, batch in enumerate(val_batches(model, vocab, icfg, 300, model.cfg.max_len)):
            states, memory, *_ = stream_states(model, batch, vocab)
            for li, s in enumerate(states):
                cov[li].append(s)
            mem_states.append(memory.reshape(-1, memory.shape[-1]))
        cs = {f"x{li}": cov_stats(torch.cat(v)) for li, v in cov.items()}
        cs["memory"] = cov_stats(torch.cat(mem_states))
        js = jacobian_stats(model, val_batches(model, vocab, icfg, 120, model.cfg.max_len), vocab)
        results[name]["stream"], results[name]["jacobian"] = cs, js
        print(f"{name}:")
        print("   stream covariance (participation / dims for 90 % / 99 % / top PC share): "
              + "  ".join(f"{k} {v['participation']:.0f}/{v['dims90']}/{v['dims99']}/{v['top1_share']:.2f}" for k, v in cs.items()))
        for k, v in js.items():
            print(f"   J {k:8}: s_max {v['smax']:.3f} s_med {v['smed']:.3f} s_min {v['smin']:.3f} | J-I: s_max {v['res_smax']:.3f} "
                  f"rms {v['res_rms']:.3f} eff {v['eff']:.2f}")
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
