"""(Paths below are those of the GB10 probe stand; edit them for another machine.)
Do visual nearest neighbours share notation habits?  Encoder-pooled embeddings of the probe
train pool and val; for each habit, P(neighbour has habit | query has habit) vs base rate."""
import json, sys, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from mathocr import determinism
from mathocr.data.dataset import CropStore
from mathocr.infer import load_checkpoint
from decode_study import batches
torch.set_num_threads(8)
dev = torch.device("cpu"); determinism.configure(dev)
model, vocab, icfg, _ = load_checkpoint("/mnt/SSD/MathOCR/probe/runs/m_lr12_s1/checkpoints/last.pt", dev)
D = "/mnt/SSD/MathOCR/probe/data20k"
def embed(store, n):
    out = torch.zeros(n, model.cfg.d_model)
    with torch.no_grad():
        for ids, b in batches(store, icfg, range(n), 1.0, max_batch=64):
            out[ids] = torch.nn.functional.normalize(model.encode(b["images"]).mean(1), dim=-1)
    return out
tr, va = CropStore(f"{D}/train.parquet", vocab), CropStore(f"{D}/val.parquet", vocab)
et, ev = embed(tr, len(tr)), embed(va, len(va))
nn = (ev @ et.T).topk(4, dim=1).indices
habits = {"{ \\frac": lambda s: "{ \\frac" in s, "\\cal": lambda s: "\\cal " in s or s.endswith("\\cal"),
          "\\mathcal": lambda s: "\\mathcal" in s, "~": lambda s: " ~ " in f" {s} ", "\\,": lambda s: "\\," in s,
          "\;": lambda s: "\;" in s, "\\left(": lambda s: "\\left(" in s, "\\frac 1": lambda s: "\\frac 1 " in s,
          "\\operatorname": lambda s: "\\operatorname" in s, "\\mathrm": lambda s: "\\mathrm" in s}
res = {}
for name, f in habits.items():
    base = sum(f(s) for s in tr.labels) / len(tr)
    q = [i for i in range(len(va)) if f(va.labels[i])]
    nq = [i for i in range(len(va)) if not f(va.labels[i])]
    hit = lambda idx: sum(f(tr.labels[j]) for i in idx for j in nn[i].tolist()) / max(1, 4 * len(idx))
    res[name] = {"base": round(base, 3), "n_val_with": len(q), "nbr_has_if_query_has": round(hit(q), 3),
                 "nbr_has_if_query_not": round(hit(nq), 3)}
    print(name, res[name], flush=True)
json.dump(res, open("/mnt/SSD/MathOCR/style_neighbours.json", "w"), indent=1)
