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
import copy
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mathocr import determinism
from mathocr.config import config_hash, load_config
from mathocr.data.dataset import AugmentConfig, CropStore, FontVariants, ShapeBatchSampler, TrainItems, collate
from mathocr.image import ImageConfig
from mathocr.infer import decode_store
from mathocr.metrics import text_metrics
from mathocr.model import Im2Latex, ModelConfig, count_parameters
from mathocr.optim import build_optimizer
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
    if hasattr(os, "nice"):  # POSIX only (Windows has none)
        os.nice(19)
    torch.set_num_threads(1)


def _lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# Tokens with no ink of their own: grouping, scripts, alignment, spacing,
# invisible delimiters and style switches.  The count head counts the rest.
STRUCTURAL = {"{", "}", "^", "_", "&", "\\\\", "\\", "~", "\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad",
              "\\left.", "\\right.", "\\displaystyle", "\\textstyle", "\\scriptstyle", "\\scriptscriptstyle",
              "\\limits", "\\nolimits", "\\begin{array}", "\\end{array}", "\\begin{aligned}", "\\end{aligned}"}


def countable_mask(vocab: Vocab) -> torch.Tensor:
    return torch.tensor([i >= 4 and t not in STRUCTURAL for i, t in enumerate(vocab.itos)])


def reverse_targets(tokens: torch.Tensor, direction: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Rows with ``direction == 1``: tokens between BOS and EOS in reverse order (padding kept)."""

    pos = torch.arange(tokens.shape[1], device=tokens.device)[None]
    n = (tokens != pad_id).sum(dim=1, keepdim=True)  # BOS + content + EOS
    inner = (pos >= 1) & (pos <= n - 2) & (direction[:, None] == 1)
    return torch.gather(tokens, 1, torch.where(inner, n - 1 - pos, pos))


def _loss(model: Im2Latex, batch: dict, device: torch.device, vocab: Vocab, smoothing: float,
          amp: bool, count_weight: float = 0.0, countable: torch.Tensor | None = None,
          token_dropout: float = 0.0, both_directions: bool = True,
          soft: tuple | None = None) -> tuple[torch.Tensor, int, torch.Tensor, dict]:
    images = batch["images"].to(device, non_blocking=True)
    tokens = batch["tokens"].to(device, non_blocking=True)
    tokens_in = tokens[:, :-1]
    if token_dropout > 0 and model.cfg.bidir:
        raise ValueError("token_dropout and bidir are not combined")
    if token_dropout > 0:
        # Hide part of the teacher-forced prefix (<unk>), so the next token has
        # to be read off the image rather than guessed from the LaTeX so far.
        # Drawn from the seeded global generator: reproducible.
        hide = (torch.rand(tokens_in.shape, device=device) < token_dropout) & (tokens_in > vocab.unk_id)
        tokens_in = tokens_in.masked_fill(hide, vocab.unk_id)
    direction = None
    if model.cfg.bidir and both_directions:
        # BTTR: half of the samples (seeded draw) read right to left -- content tokens
        # reversed between BOS and EOS -- with the direction as an input embedding.
        direction = (torch.rand(tokens.shape[0], device=device) < 0.5).long()
        tokens = reverse_targets(tokens, direction, vocab.pad_id)
        tokens_in = tokens[:, :-1]
    want_aux = count_weight > 0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        out = model(images, tokens_in, return_aux=want_aux, direction=direction)
    logits, aux = out if want_aux else (out, {})
    target = tokens[:, 1:]
    parts = {}
    if soft is not None and soft[2] == "alt":
        # Uniform smoothing stays; an extra term teaches the *alternative* continuations of
        # the neighbours (gold token removed, renormalized; rows without alternatives add 0).
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                               ignore_index=vocab.pad_id, label_smoothing=smoothing)
        (s_ids, s_w, s_off), beta, _ = soft
        k = s_ids.shape[1]
        ids = torch.zeros(target.shape[0], target.shape[1], k, dtype=torch.long)
        wts = torch.zeros(target.shape[0], target.shape[1], k)
        for r, i in enumerate(batch["index"].tolist()):
            lo, hi = int(s_off[i]), int(s_off[i + 1])
            ids[r, : hi - lo] = torch.from_numpy(s_ids[lo:hi].astype(np.int64))
            wts[r, : hi - lo] = torch.from_numpy(s_w[lo:hi])
        ids, wts = ids.to(device), wts.to(device)
        wts = wts * (ids != target.unsqueeze(-1)).float()
        wts = wts / wts.sum(-1, keepdim=True).clamp(min=1e-8)
        logp = torch.log_softmax(logits.float(), dim=-1)
        real = (target != vocab.pad_id).float() * (wts.sum(-1) > 0).float()
        alt_ce = -(logp.gather(-1, ids) * wts).sum(-1).mul(real).sum() / (target != vocab.pad_id).sum().clamp(min=1)
        parts["alt_ce"] = alt_ce.detach()
        loss = loss + beta * alt_ce
    elif soft is None:
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                               ignore_index=vocab.pad_id, label_smoothing=smoothing)
    else:
        # Distillation: the smoothing mass goes to a precomputed per-position
        # distribution (teacher top-K or the training-set neighbours' continuations).
        (s_ids, s_w, s_off), eps, _ = soft
        rows = batch["index"].tolist()
        k = s_ids.shape[1]
        ids = torch.zeros(target.shape[0], target.shape[1], k, dtype=torch.long)
        wts = torch.zeros(target.shape[0], target.shape[1], k)
        for r, i in enumerate(rows):
            lo, hi = int(s_off[i]), int(s_off[i + 1])
            ids[r, : hi - lo] = torch.from_numpy(s_ids[lo:hi].astype(np.int64))
            wts[r, : hi - lo] = torch.from_numpy(s_w[lo:hi])
        logp = torch.log_softmax(logits.float(), dim=-1)
        real = (target != vocab.pad_id).float()
        hard = -(logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * real).sum() / real.sum()
        soft_ce = -(logp.gather(-1, ids.to(device)) * wts.to(device)).sum(-1).mul(real).sum() / real.sum()
        loss = (1 - eps) * hard + eps * soft_ce
        parts["soft_ce"] = soft_ce.detach()
    if want_aux:
        # Poisson NLL of the true symbol counts, per image normalized by its
        # token count so the term has the scale of the per-token CE.
        counts = F.one_hot(target, logits.shape[-1]).sum(dim=1).float()[:, countable]
        log_rate = aux["log_count"][:, countable]
        nll = (log_rate.exp() - counts * log_rate).sum(dim=1)
        count_loss = (nll / (target != vocab.pad_id).sum(dim=1).clamp(min=1)).mean()
        parts["count_loss"] = count_loss.detach()
        loss = loss + count_weight * count_loss
    return loss, int((target != vocab.pad_id).sum()), logits, parts


@torch.no_grad()
def validate(model: Im2Latex, store: CropStore, sampler: ShapeBatchSampler, items: TrainItems, vocab: Vocab,
             device: torch.device, tcfg: dict, icfg: ImageConfig) -> dict:
    model.eval()
    total, count, first_ok, samples = 0.0, 0, 0, 0
    for keys in sampler:
        batch = collate([items[k] for k in keys], vocab.pad_id)
        loss, n, logits, _ = _loss(model, batch, device, vocab, 0.0, tcfg["amp"], both_directions=False)  # L2R
        total += float(loss) * n
        count += n
        # The first token has no prefix to lean on: it is read off the image
        # alone, so this tracks whether the decoder uses the encoder at all.
        first_ok += int((logits[:, 0].argmax(-1).cpu() == batch["tokens"][:, 1]).sum())
        samples += len(keys)
    n_dec = min(len(store), int(tcfg["val_decode_samples"]))
    stride = len(store) / n_dec
    indices = sorted({int(i * stride) for i in range(n_dec)})
    preds = decode_store(model, store, icfg, vocab, device, tcfg["eval_pixel_budget"], tcfg["eval_max_batch"], indices)
    metrics = text_metrics(preds, [store.labels[i] for i in indices])
    return {"val_loss": total / count, "val_first_token_acc": first_ok / samples,
            **{f"val_{k}": v for k, v in metrics.items()}}


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
    mcfg = ModelConfig(vocab_size=len(vocab), open_id=vocab.stoi.get("{", -1), close_id=vocab.stoi.get("}", -1),
                       **{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["model"].items()})
    model = Im2Latex(mcfg).to(device)
    ema_decay = float(tcfg.get("ema_decay", 0.0))
    # EMA of the weights (deployed and validated when enabled); updated after every
    # step with a warm-up decay min(d, (1 + t) / (10 + t)); buffers are copied.
    ema = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay > 0 else None
    if tcfg.get("init_from"):
        # Curriculum / fine-tuning: start from the weights of another run (same architecture);
        # optimizer, schedule and data order start fresh from this run's seed.
        model.load_state_dict(torch.load(tcfg["init_from"], map_location="cpu", weights_only=False)["model"])

    train_source = train_store
    if aug.p_font > 0 and aug.font_variants:  # same labels in other fonts, one variant per sample and epoch
        train_source = FontVariants(train_store, [CropStore(data_dir / f, vocab) for f in aug.font_variants], aug.p_font, seed)
    sampler = ShapeBatchSampler(train_source, icfg, tcfg["pixel_budget"], tcfg["max_batch"], seed, train=True, aug_cfg=aug,
                                max_tokens=mcfg.max_len)
    items = TrainItems(train_source, icfg, aug, seed)
    loader = DataLoader(items, batch_sampler=sampler, num_workers=tcfg["workers"],
                        collate_fn=partial(collate, pad_id=vocab.pad_id), pin_memory=device.type == "cuda",
                        worker_init_fn=_worker_init, persistent_workers=False)
    val_sampler = ShapeBatchSampler(val_store, icfg, tcfg["eval_pixel_budget"], tcfg["eval_max_batch"], 0, train=False)
    val_items = TrainItems(val_store, icfg, None, 0)

    steps_per_epoch = len(sampler)
    if tcfg.get("max_steps_per_epoch"):
        steps_per_epoch = min(steps_per_epoch, int(tcfg["max_steps_per_epoch"]))
    total_steps = steps_per_epoch * tcfg["epochs"]
    opt = build_optimizer(model, tcfg)
    count_weight = float(tcfg.get("count_loss", 0.0))
    soft = None
    if tcfg.get("soft_targets"):
        z = np.load(tcfg["soft_targets"])
        if len(z["offsets"]) != len(train_store) + 1:
            raise ValueError("soft targets do not match the training split")
        soft = ((z["ids"], z["w"], z["offsets"]), float(tcfg.get("soft_weight", 0.1)), tcfg.get("soft_mode", "replace"))
    countable = countable_mask(vocab).to(device) if count_weight > 0 else None
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, partial(_lr_lambda, warmup=tcfg["warmup_steps"], total=total_steps, min_ratio=tcfg["min_lr_ratio"]))

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, step, best = 0, 0, {"val_exact_match": -1.0, "val_edit_distance": math.inf}
    if resume and (ckpt_dir / "last.pt").exists():
        state = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rng_cuda"])
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
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
        }, indent=2) + "\n", encoding="utf-8", newline="\n")

    common = {"model_config": asdict(mcfg), "image_config": asdict(icfg), "vocab": vocab.itos, "seed": seed,
              "config_sha256": config_hash(cfg)}
    writer = SummaryWriter(run_dir / "tb")
    log = (run_dir / "log.jsonl").open("a", encoding="utf-8", newline="\n")
    try:
        for epoch in range(start_epoch, tcfg["epochs"]):
            sampler.set_epoch(epoch)
            model.train()
            t0, seen_tokens, running, running_n, clipped, aux_sums = time.time(), 0, 0.0, 0, 0, {}
            for i, batch in enumerate(loader):
                if i >= steps_per_epoch:
                    break
                loss, n, _, parts = _loss(model, batch, device, vocab, tcfg["label_smoothing"], tcfg["amp"],
                                          count_weight, countable, float(tcfg.get("token_dropout", 0.0)), soft=soft)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        d = min(ema_decay, (1 + step) / (10 + step))
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                        for be, bm in zip(ema.buffers(), model.buffers()):
                            be.copy_(bm)
                sched.step()
                step += 1
                seen_tokens += n
                running += float(loss.detach())
                running_n += 1
                # grad_norm is the pre-clip norm: the share of clipped steps shows
                # whether clipping, not the LR schedule, sets the step size.
                clipped += int(float(grad_norm) > tcfg["grad_clip"])
                for k, v in parts.items():
                    aux_sums[k] = aux_sums.get(k, 0.0) + float(v)
                if step % tcfg["log_every"] == 0:
                    rec = {"step": step, "epoch": epoch, "loss": running / running_n,
                           "grad_norm": float(grad_norm), "clip_frac": clipped / running_n, "lr": sched.get_last_lr()[0],
                           "tokens_per_s": seen_tokens / (time.time() - t0),
                           **{k: v / running_n for k, v in aux_sums.items()}}
                    running, running_n, clipped, aux_sums = 0.0, 0, 0, {}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    for k in ("loss", "grad_norm", "clip_frac", "lr", "tokens_per_s", *parts):
                        writer.add_scalar(f"train/{k}", rec[k], step)
                    print(json.dumps(rec), flush=True)
            deployed = ema if ema is not None else model
            val = validate(deployed, val_store, val_sampler, val_items, vocab, device, tcfg, icfg)
            rec = {"epoch": epoch, "step": step, "epoch_seconds": time.time() - t0, **val}
            improved = (val["val_exact_match"], -val["val_edit_distance"]) > (best["val_exact_match"], -best["val_edit_distance"])
            if improved:
                best = {"val_exact_match": val["val_exact_match"], "val_edit_distance": val["val_edit_distance"], "epoch": epoch}
                torch.save({**common, "model": deployed.state_dict(), "epoch": epoch, "val": val}, ckpt_dir / "best.pt")
            rec["best_epoch"] = best["epoch"]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            for k, v in val.items():
                writer.add_scalar(f"val/{k[4:]}", v, step)
            print(json.dumps(rec), flush=True)
            torch.save({**common, "model": model.state_dict(), **({"ema": ema.state_dict()} if ema is not None else {}),
                        "optimizer": opt.state_dict(),
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
