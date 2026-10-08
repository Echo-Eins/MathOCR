"""Train one seed.

    python -m mathocr.train --config configs/base.yaml --seed 1 --run-dir runs/base_s1

Writes into ``--run-dir``: ``config.yaml`` (resolved), ``run.json``
(seed, data manifest hashes, determinism manifest, versions, git commit),
``log.jsonl`` (per-step and per-epoch metrics), TensorBoard events,
``checkpoints/last.pt`` (full state, resumable at an epoch boundary) and
``checkpoints/best.pt`` (weights with the best validation exact match).

Re-running with the same config and seed reproduces the same weights;
``--resume`` continues from ``last.pt`` with the same result as an
uninterrupted run (data order and augmentation are keyed by epoch, RNG
states are restored).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mathocr import determinism
from mathocr.config import config_hash, load_config
from mathocr.data.dataset import AugmentConfig, CropStore, ShapeBatchSampler, TrainItems, collate
from mathocr.image import ImageConfig
from mathocr.infer import decode_store
from mathocr.metrics import text_metrics
from mathocr.model import Im2Latex, ModelConfig, count_parameters
from mathocr.tokenizer import Vocab


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True)
        return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _worker_init(_: int) -> None:
    os.nice(19)
    torch.set_num_threads(1)


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embed" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _loss(model: Im2Latex, batch: dict, device: torch.device, vocab: Vocab, smoothing: float,
          amp: bool) -> tuple[torch.Tensor, int]:
    images = batch["images"].to(device, non_blocking=True)
    tokens = batch["tokens"].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        logits = model(images, tokens[:, :-1])
    target = tokens[:, 1:]
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                           ignore_index=vocab.pad_id, label_smoothing=smoothing)
    return loss, int((target != vocab.pad_id).sum())


@torch.no_grad()
def validate(model: Im2Latex, store: CropStore, sampler: ShapeBatchSampler, items: TrainItems, vocab: Vocab,
             device: torch.device, tcfg: dict, icfg: ImageConfig) -> dict:
    model.eval()
    total, count = 0.0, 0
    for keys in sampler:
        batch = collate([items[k] for k in keys], vocab.pad_id)
        loss, n = _loss(model, batch, device, vocab, 0.0, tcfg["amp"])
        total += float(loss) * n
        count += n
    n_dec = min(len(store), int(tcfg["val_decode_samples"]))
    stride = len(store) / n_dec
    indices = sorted({int(i * stride) for i in range(n_dec)})
    preds = decode_store(model, store, icfg, vocab, device, tcfg["eval_pixel_budget"], tcfg["eval_max_batch"], indices)
    metrics = text_metrics(preds, [store.labels[i] for i in indices])
    return {"val_loss": total / count, **{f"val_{k}": v for k, v in metrics.items()}}


def train(cfg: dict, seed: int, run_dir: Path, resume: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = determinism.configure(device)
    determinism.seed_everything(seed)
    tcfg, icfg = cfg["train"], ImageConfig(**cfg["image"])
    aug = AugmentConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["augment"].items()})
    data_dir = Path(cfg["data"]["out_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    vocab = Vocab.load(data_dir / "vocab.json")

    train_store = CropStore(data_dir / "train.parquet", vocab)
    val_store = CropStore(data_dir / "val.parquet", vocab)
    mcfg = ModelConfig(vocab_size=len(vocab), **{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["model"].items()})
    model = Im2Latex(mcfg).to(device)

    sampler = ShapeBatchSampler(train_store, icfg, tcfg["pixel_budget"], tcfg["max_batch"], seed, train=True, aug_cfg=aug)
    items = TrainItems(train_store, icfg, aug, seed)
    loader = DataLoader(items, batch_sampler=sampler, num_workers=tcfg["workers"],
                        collate_fn=partial(collate, pad_id=vocab.pad_id), pin_memory=device.type == "cuda",
                        worker_init_fn=_worker_init, persistent_workers=False)
    val_sampler = ShapeBatchSampler(val_store, icfg, tcfg["eval_pixel_budget"], tcfg["eval_max_batch"], 0, train=False)
    val_items = TrainItems(val_store, icfg, None, 0)

    steps_per_epoch = len(sampler)
    if tcfg.get("max_steps_per_epoch"):
        steps_per_epoch = min(steps_per_epoch, int(tcfg["max_steps_per_epoch"]))
    total_steps = steps_per_epoch * tcfg["epochs"]
    opt = torch.optim.AdamW(_param_groups(model, tcfg["weight_decay"]), lr=tcfg["lr"], betas=(0.9, 0.98), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, partial(_lr_lambda, warmup=tcfg["warmup_steps"], total=total_steps, min_ratio=tcfg["min_lr_ratio"]))

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, step, best = 0, 0, {"val_exact_match": -1.0, "val_edit_distance": math.inf}
    if resume and (ckpt_dir / "last.pt").exists():
        state = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rng_cuda"])
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (run_dir / "run.json").write_text(json.dumps({
            "seed": seed,
            "config_sha256": config_hash(cfg),
            "git_commit": _git_commit(),
            "data_splits": {k: v["uid_label_sha256"] for k, v in manifest["splits"].items()},
            "vocab_sha256": manifest["vocab"]["sha256"],
            "parameters": count_parameters(model),
            "steps_per_epoch": steps_per_epoch,
            "determinism": det,
            "python": sys.version,
        }, indent=2) + "\n")

    common = {"model_config": asdict(mcfg), "image_config": asdict(icfg), "vocab": vocab.itos, "seed": seed,
              "config_sha256": config_hash(cfg)}
    writer = SummaryWriter(run_dir / "tb")
    log = (run_dir / "log.jsonl").open("a", encoding="utf-8")
    try:
        for epoch in range(start_epoch, tcfg["epochs"]):
            sampler.set_epoch(epoch)
            model.train()
            t0, seen_tokens, running = time.time(), 0, 0.0
            for i, batch in enumerate(loader):
                if i >= steps_per_epoch:
                    break
                loss, n = _loss(model, batch, device, vocab, tcfg["label_smoothing"], tcfg["amp"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                sched.step()
                step += 1
                seen_tokens += n
                running += float(loss)
                if step % tcfg["log_every"] == 0:
                    rec = {"step": step, "epoch": epoch, "loss": running / tcfg["log_every"],
                           "grad_norm": float(grad_norm), "lr": sched.get_last_lr()[0],
                           "tokens_per_s": seen_tokens / (time.time() - t0)}
                    running = 0.0
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    for k in ("loss", "grad_norm", "lr", "tokens_per_s"):
                        writer.add_scalar(f"train/{k}", rec[k], step)
                    print(json.dumps(rec), flush=True)
            val = validate(model, val_store, val_sampler, val_items, vocab, device, tcfg, icfg)
            rec = {"epoch": epoch, "step": step, "epoch_seconds": time.time() - t0, **val}
            improved = (val["val_exact_match"], -val["val_edit_distance"]) > (best["val_exact_match"], -best["val_edit_distance"])
            if improved:
                best = {"val_exact_match": val["val_exact_match"], "val_edit_distance": val["val_edit_distance"], "epoch": epoch}
                torch.save({**common, "model": model.state_dict(), "epoch": epoch, "val": val}, ckpt_dir / "best.pt")
            rec["best_epoch"] = best["epoch"]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            for k, v in val.items():
                writer.add_scalar(f"val/{k[4:]}", v, step)
            print(json.dumps(rec), flush=True)
            torch.save({**common, "model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "epoch": epoch, "step": step, "best": best,
                        "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None},
                       ckpt_dir / "last.pt.tmp")
            os.replace(ckpt_dir / "last.pt.tmp", ckpt_dir / "last.pt")
            stop = tcfg.get("stop_after_epoch")
            if stop is not None and epoch >= stop and epoch + 1 < tcfg["epochs"]:
                print(json.dumps({"stopped_after_epoch": epoch, "resume_with": "--resume"}), flush=True)
                break
    finally:
        log.close()
        writer.close()
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train one seed of the image-to-LaTeX model.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. train.epochs=1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(train(cfg, args.seed, args.run_dir, args.resume)))


if __name__ == "__main__":
    main()
