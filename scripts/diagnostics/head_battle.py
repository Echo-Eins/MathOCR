"""(Paths below are those of the Windows laptop; edit them for another machine.)
Real decoding with re-trained readouts of the frozen final_v2_s1: the heads of head_probe.py (fitted on the
model's teacher-forced train latents) replace the output layer inside greedy decoding (graphed: the step calls
model._logits), scored on the probe stand's evaluation sets with the stand's own decode/score functions
(val 800 by source, CPE / SCE / SPE 300 each).  Output: docs/probe_results/head_battle_final_v2_s1.json."""

import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("PROBE_ROOT", r"D:\MathOCR\probe")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import head_probe as hp  # noqa: E402  (loads final_v2_s1)
import probe  # noqa: E402
from mathocr.data.dataset import CropStore  # noqa: E402

model, vocab, icfg, dev = hp.model, hp.vocab, hp.icfg, hp.dev
zt, yt = hp.latents("train", 12000)
heads = {"tied": None, "linear": hp.fit(hp.LinearHead(), zt, yt), "cells": hp.fit(hp.Cells(), zt, yt),
         "mlp": hp.fit(hp.MLPHead(), zt, yt)}
del zt, yt
torch.cuda.empty_cache()
stores = {"val": CropStore(probe.DATA / "val.parquet", vocab)}
stores.update({s: CropStore(probe.DATA / f"{s}.parquet", vocab) for s in ("ood_cpe", "ood_sce", "ood_spe")})
results = {}
for name, head in heads.items():
    if head is None:
        model.__dict__.pop("_logits", None)
    else:
        model._logits = lambda x, head=head: head(model.dec_norm(x).float())  # the graphed step calls this
    res = {}
    preds = probe.decode(model, stores["val"], icfg, vocab, dev)
    for source in ("im2latex", "oleehyo"):
        idx = [i for i, s in enumerate(stores["val"].sources) if s == source]
        res[f"val_{source}"] = probe.score([preds[i] for i in idx], [stores["val"].labels[i] for i in idx])
    for split in ("ood_cpe", "ood_sce", "ood_spe"):
        res[split] = probe.score(probe.decode(model, stores[split], icfg, vocab, dev), stores[split].labels)
    results[name] = res
    print(f"{name:7} " + " | ".join(f"{k} EM {v['em']:.3f} edit {v['edit']:.3f}" for k, v in res.items()), flush=True)
(ROOT / "docs" / "probe_results" / "head_battle_final_v2_s1.json").write_text(json.dumps(results, indent=1),
                                                                              encoding="utf-8", newline="\n")
