"""First-order linear recurrence along a sequence (Triton), deterministic forward and backward.

For sequences ``n``, positions ``t`` and channels ``c`` (``a``, ``b``: (N, L, C), channels contiguous)::

    forward order:   h[t] = a[t] * h[t - 1] + b[t]      (h[-1] = 0)
    reverse order:   h[t] = a[t] * h[t + 1] + b[t]      (h[L] = 0)

A program owns one sequence and a block of channels and walks the sequence in tiles of ``TILE`` positions:
inside a tile a parallel prefix scan (``tl.associative_scan``, a fixed combination tree) of the pairs (a, b)
under ``(a1, b1) . (a2, b2) = (a1 a2, b1 a2 + b2)``, across tiles the carried state.  Each output element is
written once, in a fixed order -- no atomics, bitwise reproducible.  The backward pass is the same scan run the
other way on the adjoint::

    lam[t] = dh[t] + a[t + 1] * lam[t + 1]     (forward order; mirrored for reverse)
    db[t]  = lam[t],   da[t] = lam[t] * h[t - 1]

Arithmetic in fp32.  ``linear_scan`` falls back to a torch loop off CUDA (same maths; the tests' reference)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

TILE, BLOCK_C = 32, 64


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _combine(a1, b1, a2, b2):
        return a1 * a2, b1 * a2 + b2

    @triton.jit
    def _scan(A, B, H, L, C, SHIFT: tl.constexpr, REVERSE: tl.constexpr, TILE: tl.constexpr, BLOCK: tl.constexpr):
        """H[t] = A[t + SHIFT] * H[t -+ 1] + B[t]; A off the sequence reads as 0."""
        n = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        cm = c < C
        base = n * L * C
        rows = tl.arange(0, TILE)
        carry = tl.zeros([BLOCK], tl.float32)
        n_tiles = tl.cdiv(L, TILE)
        for k in range(n_tiles):
            tile = n_tiles - 1 - k if REVERSE else k
            t = tile * TILE + rows
            ta = t + SHIFT
            m = (t < L)[:, None] & cm[None, :]
            ma = ((ta >= 0) & (ta < L))[:, None] & cm[None, :]
            a = tl.load(A + base + ta[:, None] * C + c[None, :], mask=ma, other=0.0)
            b = tl.load(B + base + t[:, None] * C + c[None, :], mask=m, other=0.0)
            pa, pb = tl.associative_scan((a, b), axis=0, combine_fn=_combine, reverse=REVERSE)
            h = pa * carry[None, :] + pb
            tl.store(H + base + t[:, None] * C + c[None, :], h, mask=m)
            last = 0 if REVERSE else TILE - 1
            carry = tl.sum(tl.where((rows == last)[:, None], h, 0.0), axis=0)

    return triton, _scan


_KERNELS = None


def _scan(a: torch.Tensor, b: torch.Tensor, shift: int, reverse: bool) -> torch.Tensor:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _kernels()
    triton, kernel = _KERNELS
    n, length, c = b.shape
    h = torch.empty_like(b)
    kernel[(n, triton.cdiv(c, BLOCK_C))](a, b, h, length, c, SHIFT=shift, REVERSE=reverse, TILE=TILE, BLOCK=BLOCK_C)
    return h


class LinearScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, reverse: bool):
        a, b = a.float().contiguous(), b.float().contiguous()
        h = _scan(a, b, 0, reverse)
        ctx.save_for_backward(a, h)
        ctx.reverse = reverse
        return h

    @staticmethod
    def backward(ctx, dh: torch.Tensor):
        a, h = ctx.saved_tensors
        rev = ctx.reverse
        lam = _scan(a, dh.float().contiguous(), -1 if rev else 1, not rev)
        h_prev = F.pad(h[:, 1:], (0, 0, 0, 1)) if rev else F.pad(h[:, :-1], (0, 0, 1, 0))
        return lam * h_prev, lam, None


def linear_scan_reference(a: torch.Tensor, b: torch.Tensor, reverse: bool) -> torch.Tensor:
    a, b = a.float(), b.float()
    h = torch.zeros_like(b[:, 0])
    out = [None] * a.shape[1]
    order = range(a.shape[1] - 1, -1, -1) if reverse else range(a.shape[1])
    for t in order:
        h = a[:, t] * h + b[:, t]
        out[t] = h
    return torch.stack(out, dim=1)


def linear_scan(a: torch.Tensor, b: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """h (N, L, C) fp32 of ``h[t] = a[t] * h[t -+ 1] + b[t]`` for a, b (N, L, C)."""

    if a.is_cuda:
        return LinearScan.apply(a, b, reverse)
    return linear_scan_reference(a, b, reverse)
