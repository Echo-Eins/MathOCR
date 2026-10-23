"""(Paths below are those of the Windows laptop; edit them for another machine.)
Where does the decoder's loss sit?  final_v2_s1 frozen (CPU), teacher forcing on 800 val formulas: cross-entropy
and token errors split into notation tokens that do not change the rendering ({ }, spacing, style) and visible
ones; and whether formula-level EM is just per-token accuracy compounded over the length (independent errors)."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from mathocr.data.dataset import CropStore, ShapeBatchSampler, TrainItems, collate  # noqa: E402
from mathocr.infer import load_checkpoint  # noqa: E402
from notation_pools import SPACES, STYLE  # noqa: E402

torch.set_num_threads(6)
dev = torch.device("cpu")
model, vocab, icfg, _ = load_checkpoint(Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\checkpoints\best.pt"), dev)
store = CropStore(Path(r"D:\MathOCR\probe\data20k\val.parquet"), vocab)
invisible = torch.zeros(len(vocab), dtype=torch.bool)
for t in {"{", "}"} | SPACES | STYLE:
    if t in vocab.stoi:
        invisible[vocab.stoi[t]] = True
items = TrainItems(store, icfg, aug_cfg=None, seed=0)
ce_all, wrong_all, inv_all, per_formula = [], [], [], []
with torch.no_grad():
    for keys in ShapeBatchSampler(store, icfg, 400_000, 32, seed=0, train=False, max_tokens=model.cfg.max_len):
        b = collate([items[k] for k in keys], vocab.pad_id)
        inp, out = b["tokens"][:, :-1], b["tokens"][:, 1:]
        logits = model(b["images"], inp).float()
        ce = F.cross_entropy(logits.transpose(1, 2), out, reduction="none")
        wrong = logits.argmax(-1) != out
        m = out != vocab.pad_id
        ce_all.append(ce[m])
        wrong_all.append(wrong[m])
        inv_all.append(invisible[out[m]])
        for r in range(len(out)):
            per_formula.append((int(m[r].sum()), int((wrong[r] & m[r]).sum())))
ce, wrong, inv = torch.cat(ce_all), torch.cat(wrong_all), torch.cat(inv_all)
n = len(ce)
print(f"{n} target tokens; notation tokens {inv.float().mean() * 100:.1f}% of them")
print(f"CE: mean {ce.mean():.4f}; share of the total on notation tokens {ce[inv].sum() / ce.sum() * 100:.1f}%, "
      f"on visible {ce[~inv].sum() / ce.sum() * 100:.1f}%")
print(f"token errors (teacher forcing): {wrong.float().mean() * 100:.2f}% of tokens; "
      f"{wrong[inv].sum() / wrong.sum() * 100:.1f}% of them on notation tokens "
      f"(error rate there {wrong[inv].float().mean() * 100:.2f}% vs visible {wrong[~inv].float().mean() * 100:.2f}%)")
lens = np.array([a for a, _ in per_formula])
errs = np.array([b for _, b in per_formula])
eps = wrong.float().mean().item()
print(f"formulas without a teacher-forced error: {np.mean(errs == 0):.3f}; independent-errors prediction "
      f"mean (1 - eps)^L = {np.mean((1 - eps) ** lens):.3f} (eps {eps:.4f}, mean L {lens.mean():.1f})")
