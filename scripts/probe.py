"""Probe stand: one-factor mechanism comparisons on a fixed 20k-pair subset.

    python scripts/probe.py --name ctrl_s1 --seed 1 [--epochs 4] [--set model.encoder=aas ...]
    python scripts/probe.py --summary

A probe trains ``configs/base.yaml`` (plus overrides) on a fixed subset of
the training split for a few epochs -- the small-data regime -- and then
measures, on the final weights:

* validation by source (im2latex / arXiv) and three out-of-source UniMER
  subsets (CPE, SCE, SPE): exact match, edit distance, BLEU;
* robustness to a glyph-size normalization error: the validation set laid
  out at 0.6x .. 1.6x of the target glyph height;
* the character of the errors: runaway decodes (no EOS), unbalanced braces,
  and for wrong formulas whether the symbols are wrong or only the
  structure (same multiset of inked symbols, different layout tokens);
* cost: seconds, steps, peak GPU memory.

Subsets are chosen by the smallest sha256(uid) (deterministic, mixed
sources) and written once under ``PROBE_ROOT/data20k``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathocr import determinism  # noqa: E402
from mathocr.config import load_config  # noqa: E402
from mathocr.data.dataset import CropStore, TrainItems, collate  # noqa: E402
from mathocr.image import ImageConfig  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from mathocr.metrics import text_metrics  # noqa: E402
from mathocr.tokenizer import detokenize, tokenize  # noqa: E402
from mathocr.train import STRUCTURAL, train  # noqa: E402

PROBE_ROOT = Path(os.environ.get("PROBE_ROOT", "/mnt/SSD/MathOCR/probe"))
DATA = PROBE_ROOT / "data20k"
SCALES = (0.7, 1.4)
SIZES = {"train": 20000, "val_im2latex": 400, "val_oleehyo": 400, "ood_cpe": 300, "ood_sce": 300, "ood_spe": 300}
SHORT_MAX_TOKENS = 254  # the standard data's limit; longer rows exist only in a data.train_max_tokens build


def _rank(uids: list[str]) -> list[int]:
    return sorted(range(len(uids)), key=lambda i: hashlib.sha256(uids[i].encode()).digest())


def build_subsets(src: Path, extra_long: bool = False, extra_long_rows: int | None = None,
                  extra_long_repeat: int = 1) -> None:
    """Train subset: the ``SIZES['train']`` smallest-hash rows of at most ``SHORT_MAX_TOKENS`` tokens (on the
    standard data: all rows, so the subset is the usual one); ``extra_long`` adds the longer rows on top --
    all of them, or the ``extra_long_rows`` smallest-hash ones, each ``extra_long_repeat`` times (oversampling)."""

    if (DATA / "manifest.json").exists():
        return
    tmp = DATA.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t = pq.read_table(src / "train.parquet")
    uids = t["uid"].to_pylist()
    short = [i for i, label in enumerate(t["label"].to_pylist()) if len(label.split()) <= SHORT_MAX_TOKENS]
    keep = [short[j] for j in _rank([uids[i] for i in short])[: SIZES["train"]]]
    n_long = 0
    if extra_long:
        is_short = set(short)
        long_rows = [i for i in range(len(uids)) if i not in is_short]
        if extra_long_rows is not None:
            long_rows = [long_rows[j] for j in _rank([uids[i] for i in long_rows])[:extra_long_rows]]
        keep += long_rows * extra_long_repeat
        n_long = len(long_rows) * extra_long_repeat
    pq.write_table(t.take(sorted(keep)), tmp / "train.parquet", row_group_size=1024)
    v = pq.read_table(src / "val.parquet")
    srcs = v["source"].to_pylist()
    keep = []
    for s in ("im2latex", "oleehyo"):
        idx = [i for i in range(len(srcs)) if srcs[i] == s]
        keep += [idx[j] for j in _rank([v["uid"][i].as_py() for i in idx])[: SIZES[f"val_{s}"]]]
    pq.write_table(v.take(sorted(keep)), tmp / "val.parquet")
    for split in ("ood_cpe", "ood_sce", "ood_spe"):
        o = pq.read_table(src / f"{split}.parquet")
        pq.write_table(o.take(sorted(_rank(o["uid"].to_pylist())[: SIZES[split]])), tmp / f"{split}.parquet")
    shutil.copy(src / "vocab.json", tmp / "vocab.json")
    manifest = json.loads((src / "manifest.json").read_text())
    manifest["probe_subset"] = {"rule": "smallest sha256(uid) per split (val: per source)", "sizes": SIZES,
                                "source": str(src), "extra_long_train_rows": n_long, "extra_long_repeat": extra_long_repeat}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.rename(DATA)


@torch.no_grad()
def decode(model, store: CropStore, icfg: ImageConfig, vocab, device, jitter: float = 1.0,
           pixel_budget: int = 2_000_000, max_batch: int = 64, indices: list[int] | None = None) -> list[list[str]]:
    """Greedy token lists for the samples of ``store`` (all, or ``indices`` in that order)
    laid out at ``jitter`` x the glyph target."""

    items = TrainItems(store, icfg, None, 0)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    order = list(range(len(store))) if indices is None else list(indices)
    for i in order:
        groups[store.shape(i, icfg, jitter)].append(i)
    out: dict[int, list[str]] = {}
    for shape in sorted(groups):
        idx = groups[shape]
        size = max(1, min(max_batch, pixel_budget // (shape[0] * shape[1])))
        for s in range(0, len(idx), size):
            batch = collate([items[(i, jitter, 0)] for i in idx[s : s + size]], vocab.pad_id)
            ids = model.generate(batch["images"].to(device), vocab.bos_id, vocab.eos_id)
            for i, row in zip(idx[s : s + size], ids.tolist()):
                out[i] = vocab.decode(row) + (["<noeos>"] if vocab.eos_id not in row else [])
    return [out[i] for i in order]


def character(preds: list[list[str]], gts: list[str]) -> dict:
    """What kind of errors: runaway decodes, brace balance, symbols vs structure."""

    n = len(preds)
    runaway = sum(p[-1:] == ["<noeos>"] for p in preds)
    unbalanced = sum(p.count("{") != p.count("}") for p in preds)
    wrong = symbols_wrong = structure_only = 0
    ratio = []
    for p, g in zip(preds, gts):
        p = [t for t in p if t != "<noeos>"]
        gt = tokenize(g)
        ratio.append(len(p) / max(1, len(gt)))
        if p == gt:
            continue
        wrong += 1
        ink = lambda seq: Counter(t for t in seq if t not in STRUCTURAL)  # noqa: E731
        if ink(p) == ink(gt):
            structure_only += 1
        else:
            symbols_wrong += 1
    return {"runaway": runaway / n, "unbalanced_braces": unbalanced / n, "len_ratio": float(np.mean(ratio)),
            "wrong_symbols": symbols_wrong / n, "wrong_structure_only": structure_only / n}


def score(preds: list[list[str]], gts: list[str]) -> dict:
    strings = [detokenize(t for t in p if t != "<noeos>") for p in preds]
    m = text_metrics(strings, gts)
    return {"em": m["exact_match"], "edit": m["edit_distance"], "bleu": m["bleu"], **character(preds, gts)}


def run_probe(name: str, seed: int, epochs: int, overrides: list[str], src: Path | None = None,
              extra_long: bool = False, extra_long_rows: int | None = None, extra_long_repeat: int = 1) -> dict:
    build_subsets(src or ROOT / "data" / "processed", extra_long, extra_long_rows, extra_long_repeat)
    cfg = load_config(ROOT / "configs" / "base.yaml", [
        f"data.out_dir={DATA}", f"train.epochs={epochs}", "train.warmup_steps=300", "train.val_decode_samples=800",
        "train.log_every=50", "train.workers=2", *overrides])
    run = PROBE_ROOT / "runs" / name
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    (run / "overrides.json").write_text(json.dumps({"seed": seed, "epochs": epochs, "overrides": overrides}) + "\n")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train(cfg, seed, run, resume=False)
    info = {"train_seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    (run / "train_info.json").write_text(json.dumps(info) + "\n")
    return eval_probe(name)


def eval_probe(name: str) -> dict:
    """Evaluation protocol of the stand (re-runnable on a finished probe's last.pt)."""

    run = PROBE_ROOT / "runs" / name
    meta = json.loads((run / "overrides.json").read_text())
    info = json.loads((run / "train_info.json").read_text()) if (run / "train_info.json").exists() else \
        {k: json.loads((run / "probe.json").read_text())[k] for k in ("train_seconds", "peak_gib")}
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    curve = [{k: r[k] for k in ("epoch", "step", "val_loss", "val_first_token_acc", "val_exact_match", "val_bleu",
                                "val_edit_distance")} for r in log if "val_loss" in r]
    clip = [r["clip_frac"] for r in log if "clip_frac" in r]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(run / "checkpoints" / "last.pt", device)
    result = {"name": name, **meta, **info, "steps": log[-1]["step"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "clip_frac_mean": float(np.mean(clip)) if clip else None, "curve": curve, "splits": {}, "scale": {}}
    val = CropStore(DATA / "val.parquet", vocab)
    preds_dir = run / "preds"
    preds_dir.mkdir(exist_ok=True)
    val_preds = decode(model, val, icfg, vocab, device)
    for source in ("im2latex", "oleehyo"):
        idx = [i for i, s in enumerate(val.sources) if s == source]
        result["splits"][f"val_{source}"] = score([val_preds[i] for i in idx], [val.labels[i] for i in idx])
    for split in ("ood_cpe", "ood_sce", "ood_spe"):
        store = CropStore(DATA / f"{split}.parquet", vocab)
        preds = decode(model, store, icfg, vocab, device)
        result["splits"][split] = score(preds, store.labels)
        with (preds_dir / f"{split}.jsonl").open("w") as f:
            for uid, gt, p in zip(store.uids, store.labels, preds):
                f.write(json.dumps({"uid": uid, "gt": gt, "pred": detokenize(p)}, ensure_ascii=False) + "\n")
    with (preds_dir / "val.jsonl").open("w") as f:
        for uid, gt, p in zip(val.uids, val.labels, val_preds):
            f.write(json.dumps({"uid": uid, "gt": gt, "pred": detokenize(p)}, ensure_ascii=False) + "\n")
    # Scale robustness on every second validation formula (both sources).
    result["val_all"] = score(val_preds, val.labels)
    keep = list(range(0, len(val), 2))
    result["scale"]["1.0"] = score([val_preds[i] for i in keep], [val.labels[i] for i in keep])
    for sc in SCALES:
        preds = decode(model, val, icfg, vocab, device, jitter=sc, indices=keep)
        result["scale"][str(sc)] = score(preds, [val.labels[i] for i in keep])
    (run / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary_row(result)), flush=True)
    return result


BUCKETS = ((0, 20), (21, 50), (51, 100), (101, 10**9))


def length_buckets(run: Path, split: str = "val") -> dict:
    """Exact match by ground-truth length (tokens): where long-range structure matters."""

    rows = [json.loads(x) for x in (run / "preds" / f"{split}.jsonl").read_text().splitlines()]
    out = {}
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= len(r["gt"].split()) <= hi]
        key = f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"
        out[key] = (round(sum(r["pred"] == r["gt"] for r in sel) / len(sel), 3) if sel else None, len(sel))
    return out


def summary_row(r: dict) -> dict:
    sp, sc = r["splits"], r["scale"]
    last = r["curve"][-1]
    return {"name": r["name"], "params_M": round(r["parameters"] / 1e6, 2), "min": round(r["train_seconds"] / 60, 1),
            "peak_GiB": round(r["peak_gib"], 1), "val_loss": round(last["val_loss"], 4),
            "first_tok": round(last["val_first_token_acc"], 3),
            "curve_loss": [round(c["val_loss"], 3) for c in r["curve"]],
            "curve_first": [round(c["val_first_token_acc"], 3) for c in r["curve"]],
            "curve_edit": [round(c["val_edit_distance"], 3) for c in r["curve"]],
            **{f"{k.split('_', 1)[1]}_em": round(v["em"], 3) for k, v in sp.items()},
            **{f"{k.split('_', 1)[1]}_edit": round(v["edit"], 3) for k, v in sp.items()},
            "scale_edit": {k: round(v["edit"], 3) for k, v in sc.items()},
            "scale_em": {k: round(v["em"], 3) for k, v in sc.items()},
            "val_em": round(r["val_all"]["em"], 3), "val_edit": round(r["val_all"]["edit"], 3),
            "val_bleu": round(r["val_all"]["bleu"], 3),
            "runaway_val": round(r["val_all"]["runaway"], 3),
            "wrong_symbols_val": round(r["val_all"]["wrong_symbols"], 3),
            "wrong_struct_val": round(r["val_all"]["wrong_structure_only"], 3),
            "em_by_len_val": length_buckets(PROBE_ROOT / "runs" / r["name"])}


@torch.no_grad()
def fovea_diagnostics(name: str) -> dict:
    """Is the fovea used, and where does it look?  (1) validation CE with the glimpse tokens
    zeroed vs intact; (2) fixation statistics over greedy decodes; (3) a drawing of six decodes."""

    import subprocess

    run = PROBE_ROOT / "runs" / name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(device)
    model, vocab, icfg, _ = load_checkpoint(run / "checkpoints" / "last.pt", device)
    val = CropStore(DATA / "val.parquet", vocab)
    items = TrainItems(val, icfg, None, 0)
    keep = list(range(0, len(val), 2))
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in keep:
        groups[val.shape(i, icfg)].append(i)
    out = {}
    for mode in ("intact", "glimpses_zeroed"):
        if mode == "glimpses_zeroed":
            original = model.fovea.forward
            model.fovea.forward = lambda *a, **k: original(*a, **k) * 0.0  # noqa: E731
        tot, cnt = 0.0, 0
        for shape in sorted(groups):
            idx = groups[shape]
            for s0 in range(0, len(idx), 32):
                batch = collate([items[(i, 1.0, 0)] for i in idx[s0 : s0 + 32]], vocab.pad_id)
                x, t = batch["images"].to(device), batch["tokens"].to(device)
                logits = model(x, t[:, :-1]).float()
                ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1),
                                                       ignore_index=vocab.pad_id, reduction="sum")
                tot, cnt = tot + float(ce), cnt + int((t[:, 1:] != vocab.pad_id).sum())
        out[f"val_ce_{mode}"] = tot / cnt
        if mode == "glimpses_zeroed":
            model.fovea.forward = original
    right, zooms, steps = [], [], 0
    for i in keep[:64]:
        img, _, _ = items[(i, 1.0, 0)]
        trace: list = []
        ids = model.generate(img[None].to(device), vocab.bos_id, vocab.eos_id, trace=trace)
        n = len(vocab.decode(ids[0].tolist())) + 1
        fx = torch.cat(trace, dim=1)[0, :n].cpu().numpy()
        right.append(float(np.mean(np.diff(fx[:, 0]) > 0)) if len(fx) > 1 else 0.5)
        zooms.extend(fx[:, 2].tolist())
        steps += len(fx)
    out.update({"fixations_rightward_share": float(np.mean(right)), "zoom_mean": float(np.mean(zooms)),
                "zoom_p10_p90": [float(np.percentile(zooms, 10)), float(np.percentile(zooms, 90))], "steps": steps})
    subprocess.run([sys.executable, str(ROOT / "scripts" / "diagnostics" / "fovea_trace.py"), str(run / "checkpoints" / "last.pt"),
                    str(DATA / "val.parquet"), str(run / "fovea_trace.png"), "--n", "6", "--start", str(min(400, len(val) - 6))], check=True)
    (run / "fovea_diag.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    return out


def compare(a: str, b: str, show: int) -> None:
    """Per split: how many formulas only A / only B gets exactly right, with examples."""

    for split in ("val", "ood_cpe", "ood_sce", "ood_spe"):
        ra = [json.loads(x) for x in (PROBE_ROOT / "runs" / a / "preds" / f"{split}.jsonl").read_text().splitlines()]
        rb = [json.loads(x) for x in (PROBE_ROOT / "runs" / b / "preds" / f"{split}.jsonl").read_text().splitlines()]
        ok = lambda r: r["pred"].replace(" <noeos>", "") == r["gt"]  # noqa: E731
        only_a = [(x, y) for x, y in zip(ra, rb) if ok(x) and not ok(y)]
        only_b = [(x, y) for x, y in zip(ra, rb) if ok(y) and not ok(x)]
        print(f"== {split}: both {sum(ok(x) and ok(y) for x, y in zip(ra, rb))}, only {a} {len(only_a)}, "
              f"only {b} {len(only_b)}, n={len(ra)}")
        for tag, rows in ((f"only {a}", only_a), (f"only {b}", only_b)):
            for x, y in rows[:show]:
                print(f"  [{tag}] {x['uid']}\n    gt : {x['gt']}\n    {a}: {x['pred']}\n    {b}: {y['pred']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true", help="re-run the evaluation protocol on a finished probe")
    ap.add_argument("--data-src", type=Path, default=None, help="processed data to subset (default: data/processed)")
    ap.add_argument("--extra-long", action="store_true",
                    help="train subset = the 20k short rows + every row longer than 254 tokens (needs its own PROBE_ROOT)")
    ap.add_argument("--extra-long-rows", type=int, default=None, help="with --extra-long: only this many long rows (smallest hash)")
    ap.add_argument("--extra-long-repeat", type=int, default=1, help="with --extra-long: repeat each long row this many times")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare, args.show)
        return
    if args.eval_only:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        eval_probe(args.name)
        return
    if args.summary:
        for p in sorted((PROBE_ROOT / "runs").glob("*/probe.json"), key=lambda p: p.stat().st_mtime):
            r = json.loads(p.read_text())
            if "reeval_of" not in r:
                print(json.dumps(summary_row(r)))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.name.startswith("soft_"):  # queue job: soft_<self|knn>_<run> -> runs/<run>/soft_<mode>.npz
        import subprocess

        _, mode, run = args.name.split("_", 2)  # mode: self | knn | knn<tau x 100>, e.g. knn20 = tau 0.20
        target = PROBE_ROOT / "runs" / run
        extra = ["--tau", str(int(mode[3:]) / 100)] if mode.startswith("knn") and len(mode) > 3 else []
        subprocess.run([sys.executable, str(ROOT / "scripts" / "soft_targets.py"), str(target / "checkpoints" / "last.pt"),
                        "--mode", mode[:3] if mode.startswith("knn") else mode, "--data", str(DATA),
                        "--out", str(target / f"soft_{mode}.npz"), *extra], check=True)
        marker = PROBE_ROOT / "runs" / args.name
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "probe.json").write_text(json.dumps({"reeval_of": args.name}) + "\n")
        return
    if args.name.startswith(("knnp128_", "fknn_")):  # PCA-compressed kNN: on a probe, or on a final run (full train)
        import subprocess

        kind, _, rest = args.name.partition("_")
        common = ["--pca", "128", "--lams", "0", "0.25", "0.5"]
        if kind == "knnp128":
            target = PROBE_ROOT / "runs" / rest
            cmd = [str(target / "checkpoints" / "last.pt"), "--data", str(DATA), "--out", str(target / "knn_p128.json")]
        else:
            run = ROOT / "runs" / rest
            cmd = [str(run / "checkpoints" / "best.pt"), "--data", str(ROOT / "data" / "processed"),
                   "--splits", "test_im2latex", "test_oleehyo", "ood_cpe", "ood_sce", "ood_spe",
                   "--limit-eval", "2000", "--out", str(run / "eval" / "knn_p128.json")]
        subprocess.run([sys.executable, str(ROOT / "scripts" / "knn_decode.py"), *cmd, *common], check=True)
        marker = PROBE_ROOT / "runs" / args.name
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "probe.json").write_text(json.dumps({"reeval_of": args.name}) + "\n")
        return
    if args.name.startswith(("decode_", "knn_")):  # queue jobs: decoding study / memory-augmented decoding
        import subprocess

        kind, _, rest = args.name.partition("_")
        target = PROBE_ROOT / "runs" / rest
        script = "decode_study.py" if kind == "decode" else "knn_decode.py"
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), str(target / "checkpoints" / "last.pt"),
                        "--data", str(DATA), "--out", str(target / f"{kind}.json")], check=True)
        marker = PROBE_ROOT / "runs" / args.name
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "probe.json").write_text(json.dumps({"reeval_of": args.name}) + "\n")
        return
    if args.name.startswith("trace_"):  # queue job: fovea diagnostics of a finished probe
        fovea_diagnostics(args.name[len("trace_"):])
        marker = PROBE_ROOT / "runs" / args.name
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "probe.json").write_text(json.dumps({"reeval_of": args.name}) + "\n")
        return
    if args.name.startswith("reeval_"):  # queue job: re-run only the evaluation protocol of a finished probe
        target = args.name[len("reeval_"):]
        eval_probe(target)
        marker = PROBE_ROOT / "runs" / args.name
        marker.mkdir(parents=True, exist_ok=True)
        (marker / "probe.json").write_text(json.dumps({"reeval_of": target}) + "\n")
        return
    run_probe(args.name, args.seed, args.epochs, args.set, args.data_src, args.extra_long, args.extra_long_rows,
              args.extra_long_repeat)


if __name__ == "__main__":
    main()
