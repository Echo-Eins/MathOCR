"""Backbone variants: first-token accuracy and greedy BLEU after a fixed budget."""
import shutil, sys, time
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc, torch
from torch import nn
sys.path.insert(0, "/home/EchoEins/Coding/Python/MathOCR/src")
import mathocr.model as M
from mathocr.config import load_config
from mathocr.train import train
from mathocr.infer import load_checkpoint, decode_store
from mathocr.metrics import text_metrics
from mathocr.data.dataset import CropStore, collate, TrainItems, ShapeBatchSampler

S = Path(sys.argv[1]); name = sys.argv[2]; variant = set(sys.argv[3].split("+")); overrides = sys.argv[4:]
src = Path("/home/EchoEins/Coding/Python/MathOCR/data/processed"); sub = S / "subset20k"
if not (sub / "train.parquet").exists():
    sub.mkdir(parents=True, exist_ok=True)
    pq.write_table(pq.read_table(src / "train.parquet").slice(0, 20000), sub / "train.parquet")
    v = pq.read_table(src / "val.parquet"); pq.write_table(v.filter(pc.equal(v["source"], "im2latex")).slice(0, 1000), sub / "val.parquet")
    for f in ("vocab.json", "manifest.json"): shutil.copy(src / f, sub / f)

if "resnetd" in variant:
    orig = M.BasicBlock.__init__
    def init(self, c_in, c_out, stride):
        orig(self, c_in, c_out, stride)
        if self.skip is not None:
            pool = nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False) if stride != (1, 1) else nn.Identity()
            self.skip = nn.Sequential(pool, nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
    M.BasicBlock.__init__ = init
if "nozero" in variant:
    M.Im2Latex._init_weights_orig = M.Im2Latex._init_weights
    def iw(self):
        self._init_weights_orig()
        for m in self.backbone.modules():
            if isinstance(m, M.BasicBlock): nn.init.ones_(m.bn2.weight)
    M.Im2Latex._init_weights = iw

cfg = load_config(Path("/home/EchoEins/Coding/Python/MathOCR/configs/base.yaml"), overrides)
cfg["data"]["out_dir"] = str(sub)
run = S / f"run2_{name}"; shutil.rmtree(run, ignore_errors=True)
t0 = time.time(); train(cfg, 1, run, resume=False); secs = time.time() - t0
dev = torch.device("cuda")
m, v, icfg, _ = load_checkpoint(run / "checkpoints" / "last.pt", dev)
st = CropStore(sub / "val.parquet", v); items = TrainItems(st, icfg, None, 0)
ok = n = 0
for keys in ShapeBatchSampler(st, icfg, 600_000, 32, 0, train=False).batches():
    b = collate([items[k] for k in keys], 0); x = b["images"].to(dev); t = b["tokens"].to(dev)
    with torch.no_grad(): ok += int((m(x, t[:, :1]).argmax(-1)[:, 0] == t[:, 1]).sum()); n += len(keys)
preds = decode_store(m, st, icfg, v, dev, 2_000_000, 64, list(range(300)))
tm = text_metrics(preds, st.labels[:300])
print(f"{name}: first_token={ok/n:.3f} EM={tm['exact_match']:.3f} BLEU={tm['bleu']:.3f} edit={tm['edit_distance']:.3f} train_s={secs:.0f}", flush=True)
