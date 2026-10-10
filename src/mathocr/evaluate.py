"""Evaluate a checkpoint on the test and out-of-source splits.

    python -m mathocr.evaluate --run-dir runs/base_s1

For each split writes ``<run>/eval/<split>.jsonl`` (uid, gt, pred),
``<split>.cdm.json`` (input of the official CDM evaluator, a fixed
deterministic subset, see ``cdm_subset``) and ``metrics.json`` with the text
metrics.  ``--check-canonical N`` also re-decodes N samples per split one
image at a time (the canonical ``predict`` path) and reports how many
batched predictions differ from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from mathocr import determinism
from mathocr.data.dataset import CropStore
from mathocr.infer import decode_store, load_checkpoint
from mathocr.metrics import text_metrics

EVAL_SPLITS = ("test_im2latex", "test_oleehyo", "ood_cpe", "ood_sce", "ood_spe", "ood_hwe")


def cdm_subset(uids: list[str], size: int) -> list[int]:
    """Indices of a fixed subset: the ``size`` smallest sha256(uid).

    Depends on the uids only, so every seed is scored on the same samples.
    """

    ranked = sorted(range(len(uids)), key=lambda i: hashlib.sha256(uids[i].encode()).digest())
    return sorted(ranked[:size])


def cdm_image_id(uid: str) -> str:
    """File-name-safe id.  CDM's ``evaluation.py`` lists its outputs with
    ``name.split(".")[0]``, so an id containing a dot is silently skipped."""

    return re.sub(r"[^A-Za-z0-9_-]", "_", uid)


def evaluate(run_dir: Path, data_dir: Path, splits: list[str], checkpoint: str, cdm_size: int,
             check_canonical: int, pixel_budget: int, max_batch: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = determinism.configure(device)
    model, vocab, icfg, ckpt = load_checkpoint(run_dir / "checkpoints" / checkpoint, device)
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    results = {"checkpoint": checkpoint, "epoch": ckpt.get("epoch"), "seed": ckpt.get("seed"), "determinism": det,
               "splits": {}}
    for split in splits:
        store = CropStore(data_dir / f"{split}.parquet", vocab)  # targets are needed to collate batches
        preds = decode_store(model, store, icfg, vocab, device, pixel_budget, max_batch)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for uid, gt, pred in zip(store.uids, store.labels, preds):
                f.write(json.dumps({"uid": uid, "gt": gt, "pred": pred}, ensure_ascii=False) + "\n")
        subset = cdm_subset(store.uids, cdm_size)
        (out_dir / f"{split}.cdm.json").write_text(json.dumps(
            [{"img_id": cdm_image_id(store.uids[i]), "gt": store.labels[i], "pred": preds[i]} for i in subset],
            ensure_ascii=False, indent=0), encoding="utf-8")
        metrics = text_metrics(preds, store.labels)
        if check_canonical:
            idx = cdm_subset(store.uids, check_canonical)
            single = [decode_store(model, store, icfg, vocab, device, pixel_budget, 1, [i])[0] for i in idx]
            metrics["canonical_checked"] = len(idx)
            metrics["canonical_mismatch"] = sum(single[k] != preds[i] for k, i in enumerate(idx))
        results["splits"][split] = metrics
        print(json.dumps({"split": split, **metrics}), flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained run on test and out-of-source splits.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=None, help="default: data.out_dir of the run config")
    ap.add_argument("--splits", nargs="*", default=list(EVAL_SPLITS))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--cdm-size", type=int, default=1000)
    ap.add_argument("--check-canonical", type=int, default=200)
    ap.add_argument("--pixel-budget", type=int, default=2_000_000)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args(argv)
    if args.data_dir is None:
        import yaml

        args.data_dir = Path(yaml.safe_load((args.run_dir / "config.yaml").read_text())["data"]["out_dir"])
    evaluate(args.run_dir, args.data_dir, args.splits, args.checkpoint, args.cdm_size, args.check_canonical,
             args.pixel_budget, args.max_batch)


if __name__ == "__main__":
    main()
