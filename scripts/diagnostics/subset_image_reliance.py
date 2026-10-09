"""Diagnostic: does the model learn to use the image on a small subset?"""
import copy, json, shutil, sys, time
from pathlib import Path
import pyarrow.parquet as pq, torch
sys.path.insert(0, "/home/EchoEins/Coding/Python/MathOCR/src")
from mathocr.config import load_config
from mathocr.train import train
from mathocr.infer import load_checkpoint
from mathocr.data.dataset import CropStore, collate, TrainItems, ShapeBatchSampler

S = Path(sys.argv[1]); name = sys.argv[2]; overrides = sys.argv[3:]
src = Path("/home/EchoEins/Coding/Python/MathOCR/data/processed")
sub = S / "subset"
if not (sub / "train.parquet").exists():
    sub.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(src / "train.parquet")
    pq.write_table(t.slice(0, int(sys.argv[2].split("_n")[1]) if "_n" in name else 3000), sub / "train.parquet")
    pq.write_table(pq.read_table(src / "val.parquet").slice(0, 600), sub / "val.parquet")
    for f in ("vocab.json", "manifest.json"): shutil.copy(src / f, sub / f)
cfg = load_config(Path("/home/EchoEins/Coding/Python/MathOCR/configs/base.yaml"), overrides)
cfg["data"]["out_dir"] = str(sub)
run = S / f"run_{name}"; shutil.rmtree(run, ignore_errors=True)
t0 = time.time(); train(cfg, 1, run, resume=False); print("train s", round(time.time() - t0))
dev = torch.device("cuda")
m, v, icfg, _ = load_checkpoint(run / "checkpoints" / "last.pt", dev)
for split in ("train", "val"):
    st = CropStore(sub / f"{split}.parquet", v); items = TrainItems(st, icfg, None, 0)
    tot = {"real": [0, 0], "blank": [0, 0]}
    for keys in ShapeBatchSampler(st, icfg, 600_000, 32, 0, train=False).batches()[:20]:
        b = collate([items[k] for k in keys], 0); x = b["images"].to(dev); t = b["tokens"].to(dev)
        tgt = t[:, 1:]; mask = tgt != 0
        for nm, img in (("real", x), ("blank", torch.zeros_like(x))):
            with torch.no_grad(): lg = m(img, t[:, :-1])
            tot[nm][0] += int(((lg.argmax(-1) == tgt) & mask).sum()); tot[nm][1] += int(mask.sum())
    print(name, split, {k: round(a / b, 3) for k, (a, b) in tot.items()}, flush=True)
