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


def points(a: np.ndarray, cap: int = 1500) -> np.ndarray | None:
    ys, xs = np.nonzero(a > 0.5)
    if len(ys) == 0:
        return None
    step = max(1, len(ys) // cap)
    return np.stack([ys[::step], xs[::step]], 1).astype(np.float32)


@torch.no_grad()
def chamfer_many(p_in: np.ndarray, cands: list[np.ndarray]) -> list[float]:
    """Symmetric mean nearest-ink distance from the input to every candidate, one batched cdist per alignment
    (top-left, centroid) and a single device sync -- not thousands of tiny launches."""
    a = torch.from_numpy(p_in).to(dev)
    m = max(len(p) for p in cands)
    pts = torch.zeros(len(cands), m, 2, device=dev)
    mask = torch.zeros(len(cands), m, dtype=torch.bool, device=dev)
    for k, p in enumerate(cands):
        pts[k, : len(p)] = torch.from_numpy(p).to(dev)
        mask[k, : len(p)] = True
    means = (pts * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    best = None
    for shift in (torch.zeros_like(means), a.mean(0)[None] - means):
        d = torch.cdist(a[None].expand(len(cands), -1, -1), pts + shift[:, None])  # (K, n_in, m)
        d = d.masked_fill(~mask[:, None, :], float("inf"))
        to_cand = d.min(2).values.mean(1)
        to_in = d.min(1).values.masked_fill(~mask, 0.0).sum(1) / mask.sum(1)
        c = to_cand + to_in
        best = c if best is None else torch.minimum(best, c)
    return best.cpu().tolist()


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
    f, m = torch.cat(feats), torch.cat(masks)[:, 0]
    union = m[:1] | m[1:]  # (K, h, w)
    cos = (f[:1] * f[1:]).sum(1)  # (K, h, w)
    num = (cos * union).sum((1, 2))
    den = union.sum((1, 2))
    return torch.where(den > 0, num / den.clamp_min(1), torch.full_like(num, -1.0)).cpu().tolist()


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


def deep_ok(a: np.ndarray, inp: np.ndarray) -> bool:
    """A render more than twice the input's height or width is wrong anyway; comparing it would size the common
    canvas after it (a 13 000 px garbage render once asked for 11.5 Mpx x 6 images, overflowed the 24 GB and the
    driver's fall-back to system memory ended in a DPC_WATCHDOG_VIOLATION bugcheck, twice)."""
    return a.shape[0] <= 2 * inp.shape[0] + 32 and a.shape[1] <= 2 * inp.shape[1] + 32


def main() -> None:
    determinism.configure(dev)
    # Hard cap: past it PyTorch raises OutOfMemoryError instead of the driver paging GPU memory to system RAM.
    torch.cuda.set_per_process_memory_fraction(0.5, torch.cuda.current_device())
    backbones = {}
    for name, ck in BACKBONES.items():
        model, _, _, _ = load_checkpoint(ck, dev)
        backbones[name] = model.backbone.eval()
    rows = [json.loads(x) for x in (WORK / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    stores = {s: CropStore(DATA / f"{s}.parquet") for s in {r["split"] for r in rows}}
    # Phase 1: similarities, appended to WORK/sims.jsonl as they are computed (a crash loses nothing; re-runs resume).
    cache = WORK / "sims.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[(d["split"], d["index"])] = d["sim"]
    with open(cache, "a", encoding="utf-8", newline="\n") as log:
        for n_row, r in enumerate(rows):
            if (r["split"], r["index"]) in done:
                continue
            group = GROUP[r["split"]]
            inp = norm_ink(stores[r["split"]].crop(r["index"]))
            ids = [c["id"] for c in r["cands"]] + [f"{r['split']}__{r['index']}__gt"]
            sims = {m: np.full((len(ids), len(FONTS[group])), -np.inf) for m in MEASURES}
            p_in = points(inp) if inp is not None else None
            if p_in is not None:
                for fi, font in enumerate(FONTS[group]):
                    inks = [render_ink(group, font, cid) for cid in ids]
                    pts = [points(a) if a is not None else None for a in inks]
                    ok = [k for k, p in enumerate(pts) if p is not None]
                    if not ok:
                        continue
                    for k, c in zip(ok, chamfer_many(p_in, [pts[k] for k in ok])):
                        sims["chamfer"][k, fi] = -c
                        a = inks[k]
                        sims["size"][k, fi] = -(abs(np.log(a.shape[1] / inp.shape[1])) + abs(np.log(a.shape[0] / inp.shape[0])))
                    fit = [k for k in ok if deep_ok(inks[k], inp)]
                    for name, bb in backbones.items():
                        if not fit:
                            continue
                        try:
                            vals = deep_sims(bb, inp, [inks[k] for k in fit])
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"OOM on {r['split']}#{r['index']} ({name}); deep similarity left at -inf", flush=True)
                            continue
                        for k, v in zip(fit, vals):
                            sims[name][k, fi] = v
            best = {m: [float(v) for v in s.max(axis=1)] for m, s in sims.items()}  # best font
            done[(r["split"], r["index"])] = best
            log.write(json.dumps({"split": r["split"], "index": r["index"], "sim": best}) + "\n")
            if n_row % 250 == 0:
                log.flush()
                print("scored", n_row, "/", len(rows), flush=True)
    samples = [{"split": r["split"], "gt": r["gt"], "cands": r["cands"],
                "sim": {m: np.array(v, dtype=float) for m, v in done[(r["split"], r["index"])].items()}} for r in rows]

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
