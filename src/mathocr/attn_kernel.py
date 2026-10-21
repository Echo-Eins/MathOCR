"""Fused attention (FlashAttention-2 style, Triton) with dropout; deterministic forward and backward.

For one (batch, head), queries ``i`` and keys ``j``, ``scale = D^-1/2``::

    S = Q K^T * scale  (causal: S_ij = -inf for j > i),   P = softmax_j(S)
    Z = keep / (1 - p_drop),   O = (P * Z) V

Forward is an online softmax over key blocks and stores O and the row
log-sum-exp.  The dropout mask is ``keep_ij = rand(seed + b H + h, i Lk + j) >=
p_drop`` -- Philox, counter-based -- so the backward regenerates exactly the
forward mask and the mask does not depend on the block schedule.  ``seed`` is
drawn from torch's CPU generator, which ``determinism.seed_everything`` seeds and
the trainer checkpoints, so runs and resumed runs draw the same masks.

Backward is written in *gather* form (no atomics; every output block is owned
by one program and accumulated in a fixed order), hence bitwise reproducible::

    delta_i = <dO_i, O_i>,   dS = P * (Z * (dO V^T) - delta)
    dV = (P * Z)^T dO,   dK = dS^T Q * scale     one program per key block, looping over query blocks
    dQ = dS K * scale                            one program per query block, looping over key blocks

Block sizes are fixed per call shape (no autotuning, which could pick a
different -- numerically different -- configuration from run to run).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit(do_not_specialize=["seed"])
def _fwd(Q, K, V, O, LSE, sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som,
         H, Lq, Lk, qk_scale, p_drop, seed,
         CAUSAL: tl.constexpr, DROPOUT: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_ok = offs_m < Lq
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=m_ok[:, None], other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = tl.minimum(Lk, (pid_m + 1) * BM) if CAUSAL else Lk
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        n_ok = offs_n < Lk
        kt = tl.load(K + b * skb + h * skh + offs_n[None, :] * skn + offs_d[:, None], mask=n_ok[None, :], other=0.0)
        s = tl.dot(q, kt) * qk_scale  # log2 units
        valid = n_ok[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_ref = tl.where(m_new == -float("inf"), 0.0, m_new)  # a row without a valid key yet
        alpha = tl.exp2(m_i - m_ref)
        p = tl.exp2(s - m_ref[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if DROPOUT:
            r = tl.rand(seed + bh, offs_m[:, None] * Lk + offs_n[None, :])
            p = tl.where(r >= p_drop, p, 0.0)  # the 1 / (1 - p_drop) is applied once, at the end
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    l_ok = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_ok[:, None]
    if DROPOUT:
        out = out / (1.0 - p_drop)
    tl.store(O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :], out.to(O.dtype.element_ty),
             mask=m_ok[:, None])
    tl.store(LSE + bh * Lq + offs_m, m_i + tl.log2(l_ok), mask=m_ok)  # log2-sum-exp2 of the scaled scores


@triton.jit(do_not_specialize=["seed"])
def _bwd_kv(Q, K, V, DO, LSE, DELTA, DK, DV, sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sdb, sdh, sdm, sgb, sgh, sgl,
            H, Lq, Lk, qk_scale, scale, p_drop, seed,
            CAUSAL: tl.constexpr, DROPOUT: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    n_ok = offs_n < Lk
    k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
    v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    inv_keep = 1.0 / (1.0 - p_drop)
    lo = (pid_n * BN // BM) * BM if CAUSAL else 0  # the first query block that sees these keys
    for start_m in range(lo, Lq, BM):
        offs_m = start_m + tl.arange(0, BM)
        m_ok = offs_m < Lq
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=m_ok[:, None], other=0.0)
        do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdm + offs_d[None, :], mask=m_ok[:, None], other=0.0)
        lse = tl.load(LSE + bh * Lq + offs_m, mask=m_ok, other=0.0)
        delta = tl.load(DELTA + bh * Lq + offs_m, mask=m_ok, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = m_ok[:, None] & n_ok[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        if DROPOUT:
            keep = tl.rand(seed + bh, offs_m[:, None] * Lk + offs_n[None, :]) >= p_drop
            pz = tl.where(keep, p * inv_keep, 0.0)
            dp = tl.where(keep, dp * inv_keep, 0.0)
        else:
            pz = p
        dv += tl.dot(tl.trans(pz.to(do.dtype)), do)
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    out = b * sgb + h * sgh + offs_n[:, None] * sgl + offs_d[None, :]  # DK and DV share one dense layout
    tl.store(DK + out, (dk * scale).to(DK.dtype.element_ty), mask=n_ok[:, None])
    tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=n_ok[:, None])


@triton.jit(do_not_specialize=["seed"])
def _bwd_q(Q, K, V, DO, LSE, DELTA, DQ, sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sdb, sdh, sdm, sgb, sgh, sgl,
           H, Lq, Lk, qk_scale, scale, p_drop, seed,
           CAUSAL: tl.constexpr, DROPOUT: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_ok = offs_m < Lq
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=m_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdm + offs_d[None, :], mask=m_ok[:, None], other=0.0)
    lse = tl.load(LSE + bh * Lq + offs_m, mask=m_ok, other=0.0)
    delta = tl.load(DELTA + bh * Lq + offs_m, mask=m_ok, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    inv_keep = 1.0 / (1.0 - p_drop)
    hi = tl.minimum(Lk, (pid_m + 1) * BM) if CAUSAL else Lk
    for start_n in range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        n_ok = offs_n < Lk
        k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = m_ok[:, None] & n_ok[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        if DROPOUT:
            keep = tl.rand(seed + bh, offs_m[:, None] * Lk + offs_n[None, :]) >= p_drop
            dp = tl.where(keep, dp * inv_keep, 0.0)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + b * sgb + h * sgh + offs_m[:, None] * sgl + offs_d[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=m_ok[:, None])


@triton.jit(do_not_specialize=["seed"])
def _mask(OUT, Lq, Lk, p_drop, seed, BM: tl.constexpr, BN: tl.constexpr):
    """keep mask (B*H, Lq, Lk) exactly as the kernels draw it (tests and diagnostics)."""
    pid_m, pid_n, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    ok = (offs_m[:, None] < Lq) & (offs_n[None, :] < Lk)
    keep = tl.rand(seed + bh, offs_m[:, None] * Lk + offs_n[None, :]) >= p_drop
    tl.store(OUT + (bh * Lq + offs_m[:, None]) * Lk + offs_n[None, :], keep.to(tl.int8), mask=ok)


def _strides(t: torch.Tensor) -> tuple[int, int, int]:
    return t.stride(0), t.stride(1), t.stride(2)


def _last_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


# (BM, BN, num_warps, num_stages) -- fixed; the fastest of the tried set on an RTX 5090 Laptop (sm_120) for the
# model's shapes (encoder self-attention 1.5k-4.6k cells, cross-attention 80-200 queries).
FWD_CFG = (64, 32, 4, 3)
BWD_CFG = (64, 32, 4, 2)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool, p_drop: float, seed: int | None):
        q, k, v = (_last_contiguous(t) for t in (q, k, v))
        B, H, Lq, D = q.shape
        Lk = k.shape[2]
        if causal and Lq != Lk:
            raise ValueError("causal attention needs Lq == Lk")
        if seed is None:
            seed = int(torch.randint(0, 2**30, ()).item()) if p_drop > 0 else 0
        o = torch.empty(B, H, Lq, D, device=q.device, dtype=q.dtype)
        lse = torch.empty(B * H * Lq, device=q.device, dtype=torch.float32)
        scale = D**-0.5
        bm, bn, warps, stages = FWD_CFG
        _fwd[(triton.cdiv(Lq, bm), B * H)](
            q, k, v, o, lse, *_strides(q), *_strides(k), *_strides(v), *_strides(o),
            H, Lq, Lk, scale * LOG2E, p_drop, seed,
            CAUSAL=causal, DROPOUT=p_drop > 0, D=D, BM=bm, BN=bn, num_warps=warps, num_stages=stages)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.p_drop, ctx.seed, ctx.scale = causal, p_drop, seed, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = _last_contiguous(do)
        B, H, Lq, D = q.shape
        Lk = k.shape[2]
        delta = (do.float() * o.float()).sum(-1).reshape(-1).contiguous()
        dq = torch.empty(B, H, Lq, D, device=q.device, dtype=q.dtype)
        dk = torch.empty(B, H, Lk, D, device=k.device, dtype=k.dtype)
        dv = torch.empty_like(dk)
        bm, bn, warps, stages = BWD_CFG
        inputs = (*_strides(q), *_strides(k), *_strides(v), *_strides(do))
        common = (H, Lq, Lk, ctx.scale * LOG2E, ctx.scale, ctx.p_drop, ctx.seed)
        meta = dict(CAUSAL=ctx.causal, DROPOUT=ctx.p_drop > 0, D=D, BM=bm, BN=bn, num_warps=warps, num_stages=stages)
        _bwd_kv[(triton.cdiv(Lk, bn), B * H)](q, k, v, do, lse, delta, dk, dv, *inputs, *_strides(dk), *common, **meta)
        _bwd_q[(triton.cdiv(Lq, bm), B * H)](q, k, v, do, lse, delta, dq, *inputs, *_strides(dq), *common, **meta)
        return dq, dk, dv, None, None, None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
                    dropout_p: float = 0.0, seed: int | None = None) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)`` for CUDA
    bf16/fp16 ``(B, H, L, D)`` (``D`` a power of two, 16..128); deterministic forward and backward."""

    return FlashAttention.apply(q, k, v, causal, float(dropout_p), seed)


def dropout_keep_mask(batch_heads: int, lq: int, lk: int, p_drop: float, seed: int, device="cuda") -> torch.Tensor:
    """(batch_heads, lq, lk) bool: the keep mask ``flash_attention`` draws for ``seed``."""

    out = torch.empty(batch_heads, lq, lk, device=device, dtype=torch.int8)
    _mask[(triton.cdiv(lq, 64), triton.cdiv(lk, 64), batch_heads)](out, lq, lk, p_drop, seed, BM=64, BN=64)
    return out.bool()
