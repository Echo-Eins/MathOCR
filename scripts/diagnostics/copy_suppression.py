"""(Paths below are those of the Windows laptop; edit them for another machine.)
Follow-up of spectra.py: the decoder's self-attention OV circuits have mostly negative eigenvalues.  In token
space: M_h = E diag(g_out) W_O^h W_V^h diag(g_n1) E_hat^T (V x V, tied embeddings, E_hat = row-normalized E):
column x = how attending to a token x shifts every logit.  Copy suppression = the diagonal M[x, x] below the
rest of its column.  Also: what the dominant singular direction of the embedding is."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mathocr.infer import load_checkpoint  # noqa: E402

CKPTS = {
    "final_v2 Muon": Path(r"C:\Coding\Python\MathOCR_final\runs\final_v2_s1\checkpoints\best.pt"),
    "probe Muon": Path(r"D:\MathOCR\probe\runs\win_m_lr6_s8x4_g18_s1\checkpoints\best.pt"),
    "probe AdamW": Path(r"D:\MathOCR\probe\runs\win_adamw_s8x4_g18_s1\checkpoints\best.pt"),
    "probe Muon 1 epoch": Path(r"D:\MathOCR\probe\runs\win_m_lr6_e1_s1\checkpoints\best.pt"),  # AdamW twin's quality
}
dev = torch.device("cpu")
for name, path in CKPTS.items():
    model, vocab, _, _ = load_checkpoint(path, dev)
    with torch.no_grad():
        e = model.embed.weight.double()
        u, s, vt = torch.linalg.svd(e, full_matrices=False)
        mean = e.mean(0)
        cos = float(torch.nn.functional.cosine_similarity(vt[0], mean, dim=0))
        ec = e - mean
        sc = torch.linalg.svdvals(ec)
        # logit bias carried by the shared direction vs log-frequency is not available here; report the geometry
        print(f"{name}: embed s1/s2 {float(s[0] / s[1]):.1f}, |cos(top direction, mean embedding)| {abs(cos):.3f}, "
              f"stable rank {float((s**2).sum() / s[0]**2):.1f} -> centred {float((sc**2).sum() / sc[0]**2):.1f} (of {e.shape[1]})")
        e_hat = e / e.norm(dim=1, keepdim=True)
        g_out = model.dec_norm.weight.double()
        d, h = model.cfg.d_model, model.cfg.n_heads
        dh = d // h
        rows = []
        for li, layer in enumerate(model.decoder):
            att = layer.self_attn
            wv = att.kv.weight.double().chunk(2, dim=0)[1]
            wo = att.out.weight.double()
            g1 = layer.n1.weight.double()
            zs, below = [], []
            for hi in range(h):
                sl = slice(hi * dh, (hi + 1) * dh)
                m = (e * g_out) @ (wo[:, sl] @ (wv[sl] * g1)) @ e_hat.T  # [y, x]: logit y after reading x
                diag = m.diagonal()
                z = (diag - m.mean(0)) / m.std(0)
                zs.append(float(z.median()))
                below.append(float((diag < m.median(0).values).double().mean()))
            rows.append((li, np.round(zs, 1).tolist(), float(np.mean(below))))
        for li, zs, b in rows:
            print(f"   L{li}: own-token logit z per head {zs}  share of tokens suppressed {b * 100:.0f}%")
