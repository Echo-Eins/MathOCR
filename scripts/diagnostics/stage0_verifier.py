"""(Paths below are those of the Windows laptop; edit them for another machine.)
Stage 0c, step 2 (GPU): does a render-and-compare verifier pick better hypotheses than the model's own ranking?

Candidates (stage0_verifier_prep.py: greedy + beam-8, de-duplicated) are rendered by TeX (scripts/fonts/render.py;
in-distribution in CM, OOD in CM / Times / Palatino / sans -- the input's font is unknown, the best match counts).
Input crop and candidate render are both brought to glyph height 18 px and compared on the GPU by
  chamfer   -- symmetric mean nearest-ink distance (top-left and centroid alignment, the better one);
  deep_*    -- cosine of the CNN backbone features on a common canvas, averaged over cells with ink in either image
               (backbones: final_v2 and the font-robust F1 stand model);
  size      -- control: only the log-ratio of the ink-box width and height.
A candidate that does not compile cannot be verified.  Selection: verifier alone, or model score + lambda x
similarity (both z-scored within the sample), lambda chosen on val only and then fixed for the tests and OOD.
Sanity check of each measure: does it rank the rendered reference above every model candidate?
Metrics: EM, EM modulo notation, token edit; paired bootstrap against the deployed greedy output.
Output: docs/probe_results/stage0_verifier.json and WORK/picks.jsonl (for CDM)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mathocr import determinism  # noqa: E402
from mathocr.data.dataset import CropStore  # noqa: E402
from mathocr.image import Crop, analyze  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.metrics import normalized_levenshtein  # noqa: E402
from stage0_style import inv  # noqa: E402

WORK = Path(r"D:\MathOCR\verifier")
DATA = Path(r"C:\Coding\Python\MathOCR_final\data\processed")
BACKBONES = {"deep_final": Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\checkpoints\best.pt"),
             "deep_f1": Path(r"D:\MathOCR\probe_fonts\runs\win_v2_fonts_s1\checkpoints\best.pt")}
GROUP = {"val": "ind", "test_im2latex": "ind", "test_oleehyo": "ind", "ood_spe": "ood", "ood_sce": "ood", "ood_cpe": "ood"}
FONTS = {"ind": ("cm",), "ood": ("cm", "times", "palatino", "sans")}
MEASURES = ("chamfer", "deep_final", "deep_f1", "size")
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))
TARGET = 18.0
dev = torch.device("cuda")


def norm_ink(crop: Crop) -> np.ndarray | None:
    if crop.glyph_px <= 0:
        return None
    s = min(max(TARGET / crop.glyph_px, 0.1), 4.0)
    h, w = crop.pixels.shape
    im = Image.fromarray(crop.pixels, mode="L").resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC)
    return 1.0 - np.asarray(im, dtype=np.float32) / 255.0


def render_ink(group: str, font: str, cid: str) -> np.ndarray | None:
    p = WORK / group / "png" / font / f"{cid}.png"
    if not p.exists():
        return None
    with Image.open(p) as im:
        return norm_ink(analyze(im))


def points(a: np.ndarray, cap: int = 2500) -> torch.Tensor | None:
    ys, xs = np.nonzero(a > 0.5)
    if len(ys) == 0:
        return None
    step = max(1, len(ys) // cap)
    return torch.tensor(np.stack([ys[::step], xs[::step]], 1), dtype=torch.float32, device=dev)


def chamfer(pa: torch.Tensor, pb: torch.Tensor) -> float:
    best = float("inf")
    for shift in (torch.zeros(2, device=dev), pa.mean(0) - pb.mean(0)):
        d = torch.cdist(pa, pb + shift)
        best = min(best, (d.min(1).values.mean() + d.min(0).values.mean()).item())
    return best


@torch.no_grad()
def deep_sims(backbone, inp: np.ndarray, cands: list[np.ndarray], chunk: int = 6) -> list[float]:
    arrays = [inp] + cands
    hh = max(a.shape[0] for a in arrays) + 16
    ww = max(a.shape[1] for a in arrays) + 16
    hh, ww = -(-hh // 32) * 32, -(-ww // 32) * 32
    feats, masks = [], []
    for s in range(0, len(arrays), chunk):
        part = arrays[s : s + chunk]
        x = torch.zeros(len(part), 1, hh, ww, device=dev)
        for i, a in enumerate(part):
            x[i, 0, 8 : 8 + a.shape[0], 8 : 8 + a.shape[1]] = torch.from_numpy(a).to(dev)
        f = F.normalize(backbone(x).float(), dim=1)
        m = F.adaptive_avg_pool2d(x, f.shape[-2:]) > 0.02
        feats.append(f)
        masks.append(m)
    f, m = torch.cat(feats), torch.cat(masks)
    out = []
    for j in range(1, len(arrays)):
        union = (m[0] | m[j])[0]
        cos = (f[0] * f[j]).sum(0)
        out.append(float(cos[union].mean()) if union.any() else -1.0)
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return np.zeros_like(v)
    lo = v[finite].min() - 3 * (v[finite].std() + 1e-6)
    v = np.where(finite, v, lo)
    return (v - v.mean()) / max(v.std(), 1e-3)


def boot_ci(d: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    determinism.configure(dev)
    backbones = {}
    for name, ck in BACKBONES.items():
        model, _, _, _ = load_checkpoint(ck, dev)
        backbones[name] = model.backbone.eval()
    rows = [json.loads(x) for x in (WORK / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    stores = {s: CropStore(DATA / f"{s}.parquet") for s in {r["split"] for r in rows}}
    samples = []
    for n_row, r in enumerate(rows):
        group = GROUP[r["split"]]
        inp = norm_ink(stores[r["split"]].crop(r["index"]))
        ids = [c["id"] for c in r["cands"]] + [f"{r['split']}__{r['index']}__gt"]
        sims = {m: np.full((len(ids), len(FONTS[group])), -np.inf) for m in MEASURES}
        if inp is not None:
            p_in = points(inp)
            for fi, font in enumerate(FONTS[group]):
                inks = [render_ink(group, font, cid) for cid in ids]
                ok = [k for k, a in enumerate(inks) if a is not None and points(a) is not None]
                for k in ok:
                    a = inks[k]
                    sims["chamfer"][k, fi] = -chamfer(p_in, points(a))
                    sims["size"][k, fi] = -(abs(np.log(a.shape[1] / inp.shape[1])) + abs(np.log(a.shape[0] / inp.shape[0])))
                for name, bb in backbones.items():
                    vals = deep_sims(bb, inp, [inks[k] for k in ok])
                    for k, v in zip(ok, vals):
                        sims[name][k, fi] = v
        samples.append({"split": r["split"], "gt": r["gt"], "cands": r["cands"],
                        "sim": {m: s.max(axis=1) for m, s in sims.items()}})  # best font
        if n_row % 500 == 0:
            print("scored", n_row, "/", len(rows), flush=True)

    def pick(s, measure, lam):
        cands = s["cands"]
        sim = s["sim"][measure][: len(cands)]
        score = np.array([c["score"] for c in cands])
        if lam == float("inf"):
            order = sorted(range(len(cands)), key=lambda k: (-sim[k] if np.isfinite(sim[k]) else np.inf, -score[k]))
            return order[0] if np.isfinite(sim[order[0]]) else int(np.argmax([c["greedy"] for c in cands]))
        return int(np.argmax(zscore(score) + lam * zscore(sim)))

    def evaluate(split_samples, chooser):
        em, emi, ed = [], [], []
        for s in split_samples:
            h = s["cands"][chooser(s)]["latex"].split()
            g = s["gt"].split()
            em.append(h == g)
            emi.append(inv(h) == inv(g))
            ed.append(normalized_levenshtein(h, g))
        return np.array(em, float), np.array(emi, float), np.array(ed, float)

    by_split = defaultdict(list)
    for s in samples:
        by_split[s["split"]].append(s)
    greedy = lambda s: int(np.argmax([c["greedy"] for c in s["cands"]]))  # noqa: E731
    res = {"lambda_on_val": {}}
    for m in MEASURES:  # lambda on val only, by EM modulo notation (visible correctness)
        best = max(LAMBDAS, key=lambda lam: evaluate(by_split["val"], lambda s: pick(s, m, lam))[1].mean())
        res["lambda_on_val"][m] = best
    print("lambda chosen on val:", res["lambda_on_val"], flush=True)
    picks = []
    for split, ss in by_split.items():
        g_em, g_emi, g_ed = evaluate(ss, greedy)
        oracle_em = np.mean([any(c["latex"].split() == s["gt"].split() for c in s["cands"]) for s in ss])
        oracle_emi = np.mean([any(inv(c["latex"].split()) == inv(s["gt"].split()) for c in s["cands"]) for s in ss])
        compile_fail_greedy = np.mean([not np.isfinite(s["sim"]["chamfer"][greedy(s)]) for s in ss])
        row = {"n": len(ss), "greedy": {"em": g_em.mean(), "em_notation_free": g_emi.mean(), "edit": g_ed.mean()},
               "oracle": {"em": float(oracle_em), "em_notation_free": float(oracle_emi)},
               "greedy_does_not_compile": float(compile_fail_greedy)}
        line = [f"{split:13} n={len(ss):4} greedy EM {g_em.mean():.3f} / notation-free {g_emi.mean():.3f} / edit {g_ed.mean():.3f}"
                f" | oracle {oracle_em:.3f} / {oracle_emi:.3f} | greedy not compiling {compile_fail_greedy * 100:.1f}%"]
        for m in MEASURES:
            for tag, lam in (("alone", float("inf")), ("combined", res["lambda_on_val"][m])):
                em, emi, ed = evaluate(ss, lambda s: pick(s, m, lam))
                d = emi - g_emi
                lo, hi = boot_ci(d)
                row[f"{m} {tag}"] = {"lambda": lam, "em": em.mean(), "em_notation_free": emi.mean(), "edit": ed.mean(),
                                     "delta_notation_free": [float(d.mean()), lo, hi],
                                     "won": int(((emi == 1) & (g_emi == 0)).sum()), "lost": int(((emi == 0) & (g_emi == 1)).sum())}
                line.append(f"   {m:10} {tag:8} EM {em.mean():.3f} notation-free {emi.mean():.3f} "
                            f"({d.mean() * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}], +{row[f'{m} {tag}']['won']}/-{row[f'{m} {tag}']['lost']}) edit {ed.mean():.3f}")
        # sanity: does the measure put the rendered reference above every model candidate?
        for m in MEASURES:
            wins = []
            for s in ss:
                gt_sim = s["sim"][m][-1]
                others = [v for c, v in zip(s["cands"], s["sim"][m][:-1]) if inv(c["latex"].split()) != inv(s["gt"].split())]
                if np.isfinite(gt_sim) and others:
                    wins.append(gt_sim > max(others))
            row[f"sanity {m}"] = float(np.mean(wins)) if wins else None
        line.append("   reference ranked above all wrong candidates: " +
                    ", ".join(f"{m} {row[f'sanity {m}']:.3f}" for m in MEASURES if row[f"sanity {m}"] is not None))
        res[split] = row
        print("\n".join(line), flush=True)
        best_m = max(MEASURES, key=lambda m: row[f"{m} combined"]["em_notation_free"]) if split != "val" else None
        for s in ss:
            picks.append({"split": split, "gt": s["gt"], "greedy": s["cands"][greedy(s)]["latex"],
                          **{f"pick_{m}": s["cands"][pick(s, m, res["lambda_on_val"][m])]["latex"] for m in MEASURES}})
    with open(WORK / "picks.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for p in picks:
            f.write(json.dumps(p) + "\n")
    (ROOT / "docs" / "probe_results" / "stage0_verifier.json").write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
