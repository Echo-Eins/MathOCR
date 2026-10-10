"""Fused Attention-as-Shift along one axis (Triton), deterministic forward and backward.

For rows ``r`` (all other axes flattened), positions ``p`` along the axis and
the shift dictionary ``s_a``::

    score_a(r, p) = <Q[r, p], K[r, p - s_a]> * scale + bias[head(r), a]    (-inf if p - s_a is off the axis)
    O[r, p]       = sum_a softmax_a(score)(r, p) * V[r, p - s_a]

Forward is an online softmax over the dictionary (no per-shift tensors).
Backward is written entirely in *gather* form -- every output element is
produced by one program, accumulating in a fixed order -- so it needs no
atomics and is bitwise reproducible:

    dQ[p]  = sum_a dS_a(p) K[p - s_a] * scale
    dK[p'] = sum_a dS_a(p' + s_a) Q[p' + s_a] * scale
    dV[p'] = sum_a P_a(p' + s_a) dO[p' + s_a]
    dbias[h, a] = sum over the head's rows and positions of dS_a   (torch sum of a per-shift buffer)

with ``P = exp(score - lse)``, ``dS = P * (<dO, V[p - s_a]> - <dO, O>)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(Q, K, V, O, LSE, SHIFTS, BIAS, n, A, rows_per_head, H, scale,
         D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    p = pb * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    pin = p < n
    base = r * n * D
    q = tl.load(Q + base + p[:, None] * D + d[None, :], mask=pin[:, None], other=0.0).to(tl.float32)
    h = (r // rows_per_head) % H
    m = tl.full([BLOCK], -float("inf"), tl.float32)
    l = tl.zeros([BLOCK], tl.float32)
    acc = tl.zeros([BLOCK, D], tl.float32)
    for a in range(A):
        s = tl.load(SHIFTS + a)
        src = p - s
        ok = pin & (src >= 0) & (src < n)
        k = tl.load(K + base + src[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + base + src[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(q * k, axis=1) * scale + tl.load(BIAS + h * A + a)
        sc = tl.where(ok, sc, -float("inf"))
        m_new = tl.maximum(m, sc)
        alpha = tl.where(m_new == -float("inf"), 0.0, tl.exp(m - m_new))
        w = tl.where(ok, tl.exp(sc - m_new), 0.0)
        l = l * alpha + w
        acc = acc * alpha[:, None] + w[:, None] * v
        m = m_new
    out = acc / tl.where(l > 0, l, 1.0)[:, None]
    tl.store(O + base + p[:, None] * D + d[None, :], out.to(O.dtype.element_ty), mask=pin[:, None])
    tl.store(LSE + r * n + p, m + tl.log(tl.where(l > 0, l, 1.0)), mask=pin)


@triton.jit
def _bwd_q(Q, K, V, DO, LSE, DELTA, DQ, DS, SHIFTS, BIAS, n, A, rows_per_head, H, scale,
           D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    p = pb * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    pin = p < n
    base = r * n * D
    q = tl.load(Q + base + p[:, None] * D + d[None, :], mask=pin[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + base + p[:, None] * D + d[None, :], mask=pin[:, None], other=0.0).to(tl.float32)
    lse = tl.load(LSE + r * n + p, mask=pin, other=0.0)
    delta = tl.load(DELTA + r * n + p, mask=pin, other=0.0)
    h = (r // rows_per_head) % H
    dq = tl.zeros([BLOCK, D], tl.float32)
    for a in range(A):
        s = tl.load(SHIFTS + a)
        src = p - s
        ok = pin & (src >= 0) & (src < n)
        k = tl.load(K + base + src[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + base + src[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(q * k, axis=1) * scale + tl.load(BIAS + h * A + a)
        prob = tl.where(ok, tl.exp(sc - lse), 0.0)
        ds = prob * (tl.sum(do * v, axis=1) - delta)
        dq += ds[:, None] * k
        tl.store(DS + (r * A + a) * n + p, ds, mask=pin)
    tl.store(DQ + base + p[:, None] * D + d[None, :], dq * scale, mask=pin[:, None])


@triton.jit
def _bwd_kv(Q, K, V, DO, LSE, DELTA, DK, DV, SHIFTS, BIAS, n, A, rows_per_head, H, scale,
            D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    pk = pb * BLOCK + tl.arange(0, BLOCK)  # key/value positions p'
    d = tl.arange(0, D)
    kin = pk < n
    base = r * n * D
    k = tl.load(K + base + pk[:, None] * D + d[None, :], mask=kin[:, None], other=0.0).to(tl.float32)
    v = tl.load(V + base + pk[:, None] * D + d[None, :], mask=kin[:, None], other=0.0).to(tl.float32)
    h = (r // rows_per_head) % H
    dk = tl.zeros([BLOCK, D], tl.float32)
    dv = tl.zeros([BLOCK, D], tl.float32)
    for a in range(A):
        s = tl.load(SHIFTS + a)
        p = pk + s  # the query position that reads p' through shift a
        ok = kin & (p >= 0) & (p < n)
        q = tl.load(Q + base + p[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        do = tl.load(DO + base + p[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        lse = tl.load(LSE + r * n + p, mask=ok, other=0.0)
        delta = tl.load(DELTA + r * n + p, mask=ok, other=0.0)
        sc = tl.sum(q * k, axis=1) * scale + tl.load(BIAS + h * A + a)
        prob = tl.where(ok, tl.exp(sc - lse), 0.0)
        dv += prob[:, None] * do
        ds = prob * (tl.sum(do * v, axis=1) - delta)
        dk += ds[:, None] * q
    tl.store(DK + base + pk[:, None] * D + d[None, :], dk * scale, mask=kin[:, None])
    tl.store(DV + base + pk[:, None] * D + d[None, :], dv, mask=kin[:, None])


BLOCK = 32


class AASFunction(torch.autograd.Function):
    """q, k, v: (R, n, D) contiguous; shifts: (A,) int32; bias: (H, A) fp32; rows r belong to head
    ``(r // rows_per_head) % H``."""

    @staticmethod
    def forward(ctx, q, k, v, shifts, bias, rows_per_head: int, scale: float):
        R, n, D = q.shape
        A, H = shifts.numel(), bias.shape[0]
        o = torch.empty_like(q)
        lse = torch.empty(R, n, device=q.device, dtype=torch.float32)
        grid = (R, triton.cdiv(n, BLOCK))
        _fwd[grid](q, k, v, o, lse, shifts, bias, n, A, rows_per_head, H, scale, D=D, BLOCK=BLOCK)
        ctx.save_for_backward(q, k, v, o, lse, shifts, bias)
        ctx.rows_per_head, ctx.scale = rows_per_head, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, shifts, bias = ctx.saved_tensors
        R, n, D = q.shape
        A, H = shifts.numel(), bias.shape[0]
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1).contiguous()
        dq = torch.empty(R, n, D, device=q.device, dtype=torch.float32)
        dk, dv = torch.empty_like(dq), torch.empty_like(dq)
        ds = torch.empty(R, A, n, device=q.device, dtype=torch.float32)
        grid = (R, triton.cdiv(n, BLOCK))
        args = (ctx.rows_per_head, H, ctx.scale)
        _bwd_q[grid](q, k, v, do, lse, delta, dq, ds, shifts, bias, n, A, *args, D=D, BLOCK=BLOCK)
        _bwd_kv[grid](q, k, v, do, lse, delta, dk, dv, shifts, bias, n, A, *args, D=D, BLOCK=BLOCK)
        # dbias[h, a]: rows are ordered (outer, head, rows_per_head); a fixed-order torch reduction.
        dbias = ds.view(-1, H, ctx.rows_per_head, A, n).sum(dim=(0, 2, 4))
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, dbias.to(bias.dtype), None, None


def aas_axis(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shifts: list[int], bias: torch.Tensor,
             dim: int) -> torch.Tensor:
    """Kernel path of ``AASAxis``: q, k, v (B, H, Dh, Y, X); shifts along ``dim`` (-1: X, -2: Y);
    bias (H, A) for these shifts.  Returns the context (B, H, Dh, Y, X)."""

    b, h, dh, y, x = q.shape
    perm = (0, 1, 3, 4, 2) if dim == -1 else (0, 1, 4, 3, 2)  # axis last but one, channels last
    rows_per_head = y if dim == -1 else x
    q2, k2, v2 = (t.permute(*perm).reshape(-1, x if dim == -1 else y, dh).contiguous() for t in (q, k, v))
    sh = torch.tensor(shifts, device=q.device, dtype=torch.int32)
    o = AASFunction.apply(q2, k2, v2, sh, bias.float().contiguous(), rows_per_head, dh**-0.5)
    o = o.view(b, h, y, x, dh) if dim == -1 else o.view(b, h, x, y, dh)
    return o.permute(0, 1, 4, 2, 3) if dim == -1 else o.permute(0, 1, 4, 3, 2)
