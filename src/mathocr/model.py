"""Image-to-LaTeX encoder-decoder.

Encoder: ResNet-D-style CNN (BasicBlocks with average-pooling shortcuts,
total stride 16 vertically and 8 horizontally -- formulas are wide and sub/superscripts need horizontal
resolution) -> fixed 2D sinusoidal positions -> pre-LN Transformer encoder
for global context.

Decoder: pre-LN Transformer decoder (causal self-attention + cross-attention
to the image memory) with tied input/output embeddings.  ``generate`` does
greedy decoding with a key/value cache; ``argmax`` breaks ties toward the
lowest token id, so decoding is a pure function of the logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    cnn_channels: tuple[int, ...] = (64, 128, 256, 384)
    cnn_blocks: tuple[int, ...] = (2, 2, 2, 2)
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 3
    dec_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 256  # target tokens including BOS/EOS
    # Global mixer over the coarse feature map: "transformer" (dense
    # self-attention, the control), "aas" (attention over a dictionary of
    # exact shifts, linear in the map size) or "aas_sort" (tokens grouped by
    # sorting on a learned coordinate, windowed attention in that order).
    encoder: str = "transformer"
    aas_kernel: bool = False  # fused Triton AAS (deterministic gather-form backward)
    # "sdpa": F.scaled_dot_product_attention everywhere; "triton": fused deterministic attention
    # (attn_kernel.py) for bf16/fp16 CUDA calls with >= ATTN_KERNEL_MIN_KEYS keys (training and the
    # teacher-forced validation loss; fp32 inference stays on SDPA)
    attn_kernel: str = "sdpa"
    local_expansion: float = 1.5  # hidden width of the AAS local residuals (12 d^2 per block, as a Transformer layer)
    sort_window: int = 32
    count_head: bool = False  # per-position symbol densities summed into counts (auxiliary loss)
    count_feed: bool = False  # CAN-style: the predicted counts (log1p) enter every decoder input
    # Attention feedback between decoder layers (layer l >= 1 reads layer l-1's cross-attention):
    ring_bias: bool = False  # learned bias by log-radius ring x direction around layer l-1's focus (centroid)
    coverage: bool = False  # penalty on cells layer l-1 already read at earlier steps (WAP-style coverage)
    feedback_rings: int = 6
    # Ring deformable reading: the top ``deform_layers`` decoder layers sample the CNN's own fine
    # feature maps (strides 4 and 8) at ring points around their cross-attention focus.
    deform_layers: int = 0
    bidir: bool = False  # BTTR-style: a learned direction embedding; trained on L2R and reversed (R2L) targets
    # Encoder output stride (vertical x horizontal pixels per memory cell): 16x8 (default), 8x8, 8x4.
    enc_stride: str = "16x8"
    rings: bool = False  # ring branches (depthwise dilations 1, 2, 4, 8) after CNN stages 2-4
    inject: bool = False  # the raw image, pooled to each CNN stage's resolution, enters through a shallow conv
    depth_pos: bool = False  # brace depth of every target token as an extra decoder position
    # Foveal ring reader: from decoder layer ``fovea_layer`` on, every step
    # also reads a glimpse of rings around the point it attends to.
    coarse: int = 1  # the encoder sees the image average-pooled by this factor (the fovea always reads full resolution)
    fovea: bool = False
    fovea_layer: int = 3
    fovea_levels: int = 5
    fovea_size: int = 8
    fovea_base: float = 2.0  # pixels between the samples of the innermost ring
    fovea_channels: int = 64
    open_id: int = -1  # vocabulary ids of "{" and "}" (set by the trainer)
    close_id: int = -1
    # Output readout on top of the tied linear head (whose logits always stay, so a zero head = the control):
    # "tied" -- nothing else; "cells" -- one small MLP per vocabulary cell on a copy of the final latent, the
    # cells connected all-to-all (CellHead); "cells_ml" -- the cells read a learned mix of every decoder
    # layer's state; "cells_vis" -- the cells also read the image memory again (own query, last layer's K/V);
    # "cells_det" -- cells plus per-cell class evidence of the memory pooled by the last cross-attention.
    head: str = "tied"
    head_k: int = 16  # hidden units per vocabulary cell


# --------------------------------------------------------------------------- CNN


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: tuple[int, int]):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = None
        if stride != (1, 1) or c_in != c_out:
            # ResNet-D shortcut: average, then 1x1.  A strided 1x1 conv reads
            # one pixel in four and drops thin (1-2 px) formula strokes; with
            # it the model barely read the image (first-token accuracy 0.13 vs
            # 0.71 on a 20k-sample probe, scripts/diagnostics).
            pool = nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False) if stride != (1, 1) else nn.Identity()
            self.skip = nn.Sequential(pool, nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class RingBranch(nn.Module):
    """Rings of growing radius around every position: depthwise 3x3 taps at
    dilations 1, 2, 4, 8 (radii 1..8 cells), summed, then mixed by a 1x1.
    Enters the stage output at 0.1 of its size (a learned per-channel scale)."""

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.ModuleList(nn.Conv2d(c, c, 3, padding=d, dilation=d, groups=c, bias=False) for d in self.DILATIONS)
        self.mix = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.scale = nn.Parameter(torch.full((c,), 0.1))

    def init_extra(self) -> None:
        for conv in self.dw:  # fan_in of a depthwise tap set is 9, not 9 * channels
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        r = self.bn(self.mix(F.relu(sum(conv(x) for conv in self.dw))))
        return x + self.scale.view(1, -1, 1, 1).to(r.dtype) * r


class PyramidInject(nn.Module):
    """The raw image average- and max-pooled to a stage's resolution, read by a
    shallow 3x3 conv and added to the stage output (scale starts at 0.1): thin
    strokes reach the coarse levels without passing through the whole stack."""

    def __init__(self, c: int, stride: tuple[int, int]):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(2, c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.scale = nn.Parameter(torch.full((c,), 0.1))

    def forward(self, feat: Tensor, image: Tensor) -> Tensor:
        pooled = torch.cat([F.avg_pool2d(image, self.stride), F.max_pool2d(image, self.stride)], dim=1)
        return feat + self.scale.view(1, -1, 1, 1).to(feat.dtype) * self.bn(self.conv(pooled.to(feat.dtype)))


class ConvBackbone(nn.Module):
    # Per-stage strides after the stride-2 stem, by overall output stride.
    STRIDE_PLANS = {"16x8": ((1, 1), (2, 2), (2, 2), (2, 1)),
                    "8x8": ((1, 1), (2, 2), (2, 2), (1, 1)),
                    "8x4": ((1, 1), (2, 2), (2, 1), (1, 1))}

    def __init__(self, channels: tuple[int, ...], blocks: tuple[int, ...], rings: bool = False, inject: bool = False,
                 stride: str = "16x8"):
        super().__init__()
        self.STRIDES = self.STRIDE_PLANS[stride]
        assert len(channels) == len(blocks) == len(self.STRIDES)
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages, c_in = [], channels[0]
        for c_out, n, stride in zip(channels, blocks, self.STRIDES):
            layers = [BasicBlock(c_in, c_out, stride)]
            layers += [BasicBlock(c_out, c_out, (1, 1)) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.out_channels = c_in
        # Optional branches are built on a forked RNG and registered last, so
        # every other parameter starts exactly as without them.
        with torch.random.fork_rng(devices=[]):
            total, cum = [], (2, 2)
            for st in self.STRIDES:
                cum = (cum[0] * st[0], cum[1] * st[1])
                total.append(cum)
            self.inject = nn.ModuleList(PyramidInject(c, t) for c, t in zip(channels, total)) if inject else None
            self.rings = nn.ModuleList(RingBranch(c) for c in channels[1:]) if rings else None

    def forward(self, x: Tensor, stages_out: list | None = None) -> Tensor:
        """Final feature map; ``stages_out`` (a list) also collects every stage's output."""

        if self.inject is None and self.rings is None and stages_out is None:
            return self.stages(self.stem(x))
        y = self.stem(x)
        for k, stage in enumerate(self.stages):
            y = stage(y)
            if self.inject is not None:
                y = self.inject[k](y, x)
            if self.rings is not None and k >= 1:
                y = self.rings[k - 1](y)
            if stages_out is not None:
                stages_out.append(y)
        return y


def sinusoid_2d(h: int, w: int, dim: int, device: torch.device) -> Tensor:
    """(h*w, dim) fixed encoding: first half encodes rows, second half columns."""

    assert dim % 4 == 0
    quarter = dim // 4
    freq = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter))
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * freq
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * freq
    pe_y = torch.cat([ys.sin(), ys.cos()], dim=1)[:, None, :].expand(h, w, 2 * quarter)
    pe_x = torch.cat([xs.sin(), xs.cos()], dim=1)[None, :, :].expand(h, w, 2 * quarter)
    return torch.cat([pe_y, pe_x], dim=2).reshape(h * w, dim)


def sinusoid_1d(n: int, dim: int) -> Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


# ------------------------------------------------- linear-cost global mixers (AAS)


class ChannelRMSNorm2d(nn.Module):
    """RMS normalization over channels at every position (no batch statistics)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight.view(1, -1, 1, 1).to(x.dtype)


def shift_map(x: Tensor, k: int, dim: int) -> Tensor:
    """``out[..., p, ...] = x[..., p - k, ...]`` along ``dim`` (-1 or -2), zeros outside.

    Pad-and-slice only: forward and backward are exact copies, so the
    operator is deterministic and has no scatter in its gradient.
    """

    if k == 0:
        return x
    n = x.shape[dim]
    if abs(k) >= n:
        return torch.zeros_like(x)
    pad = (k, 0) if k > 0 else (0, -k)
    if dim == -2:
        pad = (0, 0, *pad)
    y = F.pad(x, pad)
    keep = slice(0, n) if k > 0 else slice(-k, None)
    return y[..., keep] if dim == -1 else y[..., keep, :]


def shift_dictionary(n: int, max_power: int = 12) -> list[int]:
    """0, +1, -1, +2, -2, +4, -4, ... below ``n``; the order fixes each shift's bias slot."""

    out = [0]
    for j in range(max_power):
        if 2**j >= n:
            break
        out += [2**j, -(2**j)]
    return out


class AASAxis(nn.Module):
    """Attention-as-Shift along one axis of a planar feature map.

    Port of ``ECRA5 era5c.models.ribbons.AttentionAsShiftAxis`` (the author's
    design) to a non-periodic plane::

        q = Wq RMS(h),  v = Wv RMS(h),  key(q) = q Wk      (keys share Wq)
        s_g(p) = <q(p), key(q)(p - g)> / sqrt(d_h) + b_g
        a_g(p) = softmax_g s_g(p),   out(p) = Wo sum_g a_g(p) v(p - g)

    over the shift dictionary g in {0, +-1, +-2, +-4, ...}.  No token-token
    matrix is formed: cost is linear in the map size times log2 of its side,
    and any distance is reachable by composing a few layers.  Shifts that
    leave the map are masked out of the softmax (the plane has no wrap-around).
    """

    def __init__(self, channels: int, heads: int, dim: int, max_power: int = 12, kernel: bool = False):
        super().__init__()
        assert channels % heads == 0 and dim in (-1, -2)
        self.heads, self.head_dim, self.dim, self.max_power = heads, channels // heads, dim, max_power
        self.kernel = kernel  # fused Triton path (aas_kernel.py) on CUDA; same math
        self.norm = ChannelRMSNorm2d(channels)
        self.content = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.action_bias = nn.Parameter(torch.zeros(heads, 1 + 2 * max_power))
        self.output = nn.Conv2d(channels, channels, 1)

    def init_extra(self) -> None:
        nn.init.zeros_(self.action_bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        n = x.shape[self.dim]
        hn = self.norm(x)
        q = self.content(hn).view(b, self.heads, self.head_dim, h, w)
        v = self.value(hn).view(b, self.heads, self.head_dim, h, w)
        k = torch.einsum("bhdyx,ed->bheyx", q, self.key.weight.to(q.dtype))
        pos = torch.arange(n, device=x.device)
        shifts = shift_dictionary(n, self.max_power)
        if self.kernel and x.is_cuda:
            from mathocr.aas_kernel import aas_axis

            ctx = aas_axis(q, k, v, shifts, self.action_bias[:, : len(shifts)], self.dim)
            return self.output(ctx.reshape(b, c, h, w))
        # Keys and values are zero-padded once; every shift is a slice (a view)
        # of the padded tensor, so neither the forward nor autograd keeps a
        # copy per shift.  Products in the activation dtype, sums in fp32.
        m = max(abs(s) for s in shifts)
        kp, vp = (F.pad(t, (m, m) if self.dim == -1 else (0, 0, m, m)) for t in (k, v))

        def shifted(t: Tensor, s: int) -> Tensor:  # out[p] = original[p - s]
            window = slice(m - s, m - s + n)
            return t[..., window] if self.dim == -1 else t[..., window, :]

        scores = []
        for a, s in enumerate(shifts):
            sc = (q * shifted(kp, s)).sum(dim=2, dtype=torch.float32) / math.sqrt(self.head_dim)
            sc = sc + self.action_bias[:, a].float().view(1, -1, 1, 1)
            if s:
                valid = ((pos - s) >= 0) & ((pos - s) < n)
                valid = valid.view(1, 1, 1, n) if self.dim == -1 else valid.view(1, 1, n, 1)
                sc = sc.masked_fill(~valid, -1e9)
            scores.append(sc)
        att = torch.softmax(torch.stack(scores, dim=2), dim=2).to(v.dtype)
        ctx = att[:, :, 0].unsqueeze(2) * v
        for a, s in enumerate(shifts[1:], start=1):
            ctx = ctx + att[:, :, a].unsqueeze(2) * shifted(vp, s)
        return self.output(ctx.reshape(b, c, h, w))


class LocalResidual(nn.Module):
    """Local companion branch: RMS -> depthwise 3x3 -> 1x1 expand -> SiLU -> 1x1 (ECRA5 RibbonLocalResidual)."""

    def __init__(self, channels: int, expansion: float):
        super().__init__()
        hidden = int(round(channels * expansion))
        self.norm = ChannelRMSNorm2d(channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.full((channels,), 0.05))

    def forward(self, x: Tensor) -> Tensor:
        r = self.project(F.silu(self.expand(self.depthwise(self.norm(x)))))
        return x + self.scale.view(1, -1, 1, 1).to(x.dtype) * r


class RelativeMerge(nn.Module):
    """``x + scale * RMSNorm(update) * rms(x)``: a global branch enters at a fixed
    fraction of the local signal's size (ECRA5 ``relative_rms`` merge)."""

    def __init__(self, channels: int, scale: float = 0.2):
        super().__init__()
        self.norm = ChannelRMSNorm2d(channels)
        self.scale = nn.Parameter(torch.full((channels,), scale))

    def forward(self, x: Tensor, update: Tensor) -> Tensor:
        ref = x.float().square().mean(dim=1, keepdim=True).add(1e-6).sqrt().detach().to(x.dtype)
        return x + self.scale.view(1, -1, 1, 1).to(x.dtype) * self.norm(update) * ref


class AASBlock(nn.Module):
    """x-axis AAS -> local residual -> y-axis AAS -> local residual (the ECRA5 block order)."""

    def __init__(self, d: int, heads: int, expansion: float, kernel: bool = False):
        super().__init__()
        self.ax, self.ay = AASAxis(d, heads, -1, kernel=kernel), AASAxis(d, heads, -2, kernel=kernel)
        self.mx, self.my = RelativeMerge(d), RelativeMerge(d)
        self.lx, self.ly = LocalResidual(d, expansion), LocalResidual(d, expansion)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lx(self.mx(x, self.ax(x)))
        return self.ly(self.my(x, self.ay(x)))


class _Permute(torch.autograd.Function):
    """Gather along dim 2 by a permutation; the gradient gathers by the inverse
    permutation (exact, deterministic, no scatter-add)."""

    @staticmethod
    def forward(ctx, x: Tensor, order: Tensor, inverse: Tensor) -> Tensor:
        ctx.save_for_backward(inverse)
        return torch.gather(x, 2, order.unsqueeze(-1).expand(*order.shape, x.shape[-1]))

    @staticmethod
    def backward(ctx, grad: Tensor):
        (inverse,) = ctx.saved_tensors
        return torch.gather(grad, 2, inverse.unsqueeze(-1).expand(*inverse.shape, grad.shape[-1])), None, None


class SortAttention(nn.Module):
    """Grouping attention: related states are moved next to each other, then mixed locally.

    Every head reads a scalar coordinate ``c`` off each state, the states are
    stably sorted by it (so states with close coordinates become neighbours,
    others land in other groups), split into windows of ``window`` states and
    attend within their window and the two adjacent ones; the result is put
    back in place.  The logit carries ``-beta (c_i - c_j)^2``, so attention
    gradients pull the coordinates of states that need each other together --
    that is what teaches the sort.  Cost is linear in the number of states.
    """

    def __init__(self, d: int, heads: int, window: int):
        super().__init__()
        self.heads, self.head_dim, self.window = heads, d // heads, window
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.coord = nn.Linear(d, heads)
        self.log_beta = nn.Parameter(torch.zeros(heads))
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:  # (B, N, d)
        b, n, d = x.shape
        h = self.norm(x)
        q = self.q(h).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        c = self.coord(h).float().transpose(1, 2)  # (B, H, N)
        order = torch.argsort(c, dim=-1, stable=True)
        inverse = torch.argsort(order, dim=-1, stable=True)
        qs, vs = _Permute.apply(q, order, inverse), _Permute.apply(v, order, inverse)
        cs = _Permute.apply(c.unsqueeze(-1), order, inverse).squeeze(-1)
        ks = qs @ self.key.weight.T.to(qs.dtype)
        wdw = self.window
        pad = (-n) % wdw
        valid = torch.ones(b, self.heads, n, dtype=cs.dtype, device=x.device)
        if pad:
            qs, ks, vs = (F.pad(t, (0, 0, 0, pad)) for t in (qs, ks, vs))
            cs, valid = F.pad(cs, (0, pad)), F.pad(valid, (0, pad))
        nw = (n + pad) // wdw

        def windows(t: Tensor) -> Tensor:  # (B, H, nw*W, ...) -> (B, H, nw, 3W, ...): previous, own, next
            t = t.view(b, self.heads, nw, wdw, *t.shape[3:])
            prev = F.pad(t, (0, 0) * (t.dim() - 4) + (0, 0, 1, 0))[:, :, :nw]
            nxt = F.pad(t, (0, 0) * (t.dim() - 4) + (0, 0, 0, 1))[:, :, 1:]
            return torch.cat([prev, t, nxt], dim=3)

        qw = qs.view(b, self.heads, nw, wdw, self.head_dim)
        kc, vc, cc, mc = windows(ks), windows(vs), windows(cs), windows(valid)
        cw = cs.view(b, self.heads, nw, wdw)
        logits = (qw.float() @ kc.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        beta = F.softplus(self.log_beta).view(1, -1, 1, 1, 1)
        logits = logits - beta * (cw.unsqueeze(-1) - cc.unsqueeze(-2)).square()
        logits = logits.masked_fill(mc.unsqueeze(-2) < 0.5, -1e9)
        y = (torch.softmax(logits, dim=-1).to(vc.dtype) @ vc).view(b, self.heads, nw * wdw, self.head_dim)[:, :, :n]
        y = _Permute.apply(y, inverse, order)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class SortBlock(nn.Module):
    """Grouping attention -> local residual, twice (same budget and merge as ``AASBlock``)."""

    def __init__(self, d: int, heads: int, expansion: float, window: int):
        super().__init__()
        self.a1, self.a2 = SortAttention(d, heads, window), SortAttention(d, heads, window)
        self.m1, self.m2 = RelativeMerge(d), RelativeMerge(d)
        self.l1, self.l2 = LocalResidual(d, expansion), LocalResidual(d, expansion)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w = x.shape
        tok = lambda t: t.flatten(2).transpose(1, 2)  # noqa: E731
        grid = lambda t: t.transpose(1, 2).reshape(b, d, h, w)  # noqa: E731
        x = self.l1(self.m1(x, grid(self.a1(tok(x)))))
        return self.l2(self.m2(x, grid(self.a2(tok(x)))))


# ------------------------------------------------------------------- Transformer


ATTN_KERNEL_MIN_KEYS = 256  # shorter key sets (decoder self-attention) are as fast on SDPA


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, kernel: str = "sdpa"):
        super().__init__()
        self.heads, self.dropout, self.kernel = heads, dropout, kernel
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def split(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def project_kv(self, x: Tensor) -> tuple[Tensor, Tensor]:
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def attend(self, x: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        q = self.split(self.q(x))
        p = self.dropout if self.training else 0.0
        if (self.kernel == "triton" and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
                and k.shape[2] >= ATTN_KERNEL_MIN_KEYS):  # a pure function of the call's shape and dtype
            from mathocr.attn_kernel import flash_attention

            y = flash_attention(q, k, v, causal, p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=causal)
        b, h, n, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(b, n, h * dh))


class FeedForward(nn.Sequential):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * mult, d))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, mult: int, dropout: float, attn_kernel: str = "sdpa"):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(d), Attention(d, heads, dropout, attn_kernel)
        self.n2, self.ff = nn.LayerNorm(d), FeedForward(d, mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.n1(x)
        x = x + self.drop(self.attn.attend(h, *self.attn.project_kv(h), causal=False))
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, mult: int, dropout: float, attn_kernel: str = "sdpa"):
        super().__init__()
        self.n1, self.self_attn = nn.LayerNorm(d), Attention(d, heads, dropout, attn_kernel)
        self.n2, self.cross_attn = nn.LayerNorm(d), Attention(d, heads, dropout, attn_kernel)
        self.n3, self.ff = nn.LayerNorm(d), FeedForward(d, mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mem_kv: tuple[Tensor, Tensor], cache: dict | None = None,
                return_cross: bool = False, cross_bias: Tensor | None = None):
        h = self.n1(x)
        k, v = self.self_attn.project_kv(h)
        if cache is None:
            x = x + self.drop(self.self_attn.attend(h, k, v, causal=True))
        else:
            # Incremental step: x is the newest position only; the cache
            # holds keys/values of every earlier position.
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            x = x + self.self_attn.attend(h, k, v, causal=False)
        if not return_cross and cross_bias is None:
            x = x + self.drop(self.cross_attn.attend(self.n2(x), *mem_kv, causal=False))
            return x + self.drop(self.ff(self.n3(x)))
        # Same attention, with the weights formed explicitly (plus an optional
        # additive bias): their mean over heads is where this layer reads the image.
        att = self.cross_attn
        q = att.split(att.q(self.n2(x)))
        logits = (q.float() @ mem_kv[0].float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        w = torch.softmax(logits if cross_bias is None else logits + cross_bias, dim=-1)
        y = F.dropout(w, att.dropout, self.training).to(mem_kv[1].dtype) @ mem_kv[1]
        b, h, n, dh = y.shape
        x = x + self.drop(att.out(y.transpose(1, 2).reshape(b, n, h * dh)))
        return x + self.drop(self.ff(self.n3(x))), w.mean(dim=1)


# ------------------------------------------------------- attention feedback (coupling)


def stride_hw(stride: str) -> tuple[int, int]:
    sh, sw = stride.split("x")
    return int(sh), int(sw)


def cell_centres(images: Tensor, coarse: int, stride: str = "16x8") -> Tensor:
    """(h*w, 2) pixel centres (x, y) of the encoder memory cells (``stride``, times ``coarse``)."""

    sh, sw = stride_hw(stride)
    gh, gw = images.shape[-2] // (sh * coarse), images.shape[-1] // (sw * coarse)
    ys = (torch.arange(gh, device=images.device, dtype=torch.float32) + 0.5) * (sh * coarse)
    xs = (torch.arange(gw, device=images.device, dtype=torch.float32) + 0.5) * (sw * coarse)
    return torch.stack([xs.repeat(gh), ys.repeat_interleave(gw)], dim=-1)


class AttentionFeedback(nn.Module):
    """Cross-attention bias of decoder layer l computed from layer l-1's attention at the same step.

    * ring bias (the author's ring idea moved into the coupling): around the
      focus of layer l-1 (centroid of its head-mean attention) every memory cell
      falls into a log-radius ring (8, 24, 56, 120, ... px) and one of four
      directions; a learned (head, ring, direction) table is added to the
      logits -- a foveal prior that refines where the previous layer looked;
    * coverage: ``- softplus(gamma_h) * sum_{s<t} a^{l-1}_s`` penalizes cells
      already read at earlier steps (repetition / omission on long formulas).

    Both start as no-ops (zero table, tiny gamma) and are pure functions of the
    detached attention of layer l-1, so training is parallel and generation
    only needs a running coverage sum per layer.
    """

    R0 = 8.0

    def __init__(self, heads: int, rings: int, ring_bias: bool, coverage: bool):
        super().__init__()
        self.rings = rings
        self.ring_table = nn.Parameter(torch.zeros(heads, rings * 4)) if ring_bias else None
        self.coverage_gamma = nn.Parameter(torch.full((heads,), -4.0)) if coverage else None

    def init_extra(self) -> None:
        if self.ring_table is not None:
            nn.init.zeros_(self.ring_table)

    def forward(self, prev: Tensor, pos: Tensor, cov: Tensor | None) -> Tensor:
        """(B, heads, T, N) bias from ``prev`` (B, T, N) head-mean attention of layer l-1."""

        bias = 0
        if self.ring_table is not None:
            focus = prev @ pos  # (B, T, 2)
            d = pos[None, None] - focus[:, :, None]  # (B, T, N, 2)
            r = d.norm(dim=-1)
            ring = torch.log2(1 + r / self.R0).floor().clamp(0, self.rings - 1).long()
            dx, dy = d[..., 0], d[..., 1]
            horiz = dx.abs() >= dy.abs()
            sector = torch.where(horiz, torch.where(dx >= 0, 0, 2), torch.where(dy >= 0, 1, 3))
            idx = ring * 4 + sector  # (B, T, N)
            bias = self.ring_table[:, idx].permute(1, 0, 2, 3)  # (B, heads, T, N)
        if self.coverage_gamma is not None:
            bias = bias - F.softplus(self.coverage_gamma).view(1, -1, 1, 1) * cov[:, None]
        return bias


# ------------------------------------------------------------ ring deformable reading


def sample_features(flat: Tensor, h: int, w: int, img: Tensor, gx: Tensor, gy: Tensor) -> Tensor:
    """Bilinear samples from channels-last maps ``flat`` (B*H*W, C) of images ``img`` (N,) at
    index coordinates gx, gy (N, P) -> (N, P, C); zeros outside.  Gathers along the flattened
    batch of maps (deterministic gradient in this PyTorch's deterministic mode)."""

    c = flat.shape[1]
    base = (img * (h * w)).view(-1, 1)
    x0, y0 = gx.floor(), gy.floor()
    fx, fy = (gx - x0).unsqueeze(-1), (gy - y0).unsqueeze(-1)
    out = 0
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            ok = ((xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)).unsqueeze(-1)
            idx = (base + yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).reshape(-1)
            out = out + (wx * wy * ok).to(flat.dtype) * flat[idx].view(*gx.shape, c)
    return out


class RingDeformRead(nn.Module):
    """Deformable reading at ring points around a focus, from the CNN's fine feature maps.

    Each query reads ``2 rings x 8 directions`` points per level (radius 1 and
    2.5 level strides) around its focus, shifted by learned offsets (zero at
    start), and mixes them with query-dependent weights per head (Deformable
    DETR, with the author's ring layout as the initial sampling pattern).  Unlike
    the separate glimpse channel this reads features the main loss trains, at a
    resolution the stride-16x8 memory does not have.  Maps are first reduced to
    ``reduced`` channels (1x1), and only real (non-padding) queries are read.
    """

    RINGS, DIRS = (1.0, 2.5), 8

    def __init__(self, d: int, heads: int, channels: tuple[int, ...], reduced: int = 64):
        super().__init__()
        self.heads, self.levels = heads, len(channels)
        self.points = len(self.RINGS) * self.DIRS
        ang = torch.arange(self.DIRS, dtype=torch.float32) * (2 * math.pi / self.DIRS)
        unit = torch.stack([ang.cos(), ang.sin()], dim=-1)
        self.register_buffer("ring", torch.cat([r * unit for r in self.RINGS]), persistent=False)  # (P, 2)
        self.norm = nn.LayerNorm(d)
        self.offset = nn.Linear(d, self.levels * self.points * 2)
        self.weight = nn.Linear(d, heads * self.levels * self.points)
        self.reduce = nn.ModuleList(nn.Conv2d(c, reduced, 1) for c in channels)
        self.value = nn.ModuleList(nn.Linear(reduced, d) for _ in channels)
        self.out = nn.Linear(d, d)

    def init_extra(self) -> None:
        for lin in (self.offset, self.weight):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: Tensor, feats: list[Tensor], strides: list[float], focus: Tensor,
                valid: Tensor | None = None) -> Tensor:
        b, t, d = x.shape
        img = torch.arange(b, device=x.device).repeat_interleave(t)
        q, f = x.reshape(b * t, d), focus.reshape(b * t, 2)
        sel = None
        if valid is not None:  # read only real queries (a deterministic ascending packing)
            sel = valid.reshape(-1).nonzero().squeeze(1)
            img, q, f = img[sel], q[sel], f[sel]
        n = q.shape[0]
        hq = self.norm(q)
        off = self.offset(hq).float().view(n, self.levels, self.points, 2).tanh() * 4.0  # <= 4 strides
        vals = []
        for lv, (fmap, st) in enumerate(zip(feats, strides)):
            red = self.reduce[lv](fmap)
            hh, ww = red.shape[-2:]
            flat = red.permute(0, 2, 3, 1).reshape(-1, red.shape[1])
            pts = f[:, None, :] + (self.ring + off[:, lv]) * st  # (N, P, 2) pixels
            g = sample_features(flat, hh, ww, img, pts[..., 0] / st - 0.5, pts[..., 1] / st - 0.5)
            vals.append(self.value[lv](g))
        v = torch.cat(vals, dim=1).view(n, self.levels * self.points, self.heads, d // self.heads)
        a = torch.softmax(self.weight(hq).float().view(n, self.heads, -1), dim=-1).to(v.dtype)
        y = self.out(torch.einsum("nhp,nphd->nhd", a, v).reshape(n, d))
        if sel is None:
            return y.view(b, t, d)
        slot = torch.zeros(b * t, dtype=torch.long, device=y.device)  # unpack: padding reads a zero row
        slot[sel] = torch.arange(1, sel.numel() + 1, device=y.device)
        return torch.cat([y.new_zeros(1, d), y])[slot].view(b, t, d)


# ------------------------------------------------------------------ foveal reader


def bilinear_gather(plane: Tensor, img: Tensor, gx: Tensor, gy: Tensor) -> Tensor:
    """Sample ``plane[img[n]]`` (planes: B, H, W) at index coordinates ``gx, gy``
    (N, ...) with zeros outside.  Plain gathers from a tensor that needs no
    gradient, so it is deterministic; gradients reach the coordinates through
    the bilinear weights."""

    _, h, w = plane.shape
    flat = plane.reshape(-1)
    base = (img * (h * w)).view(-1, *([1] * (gx.dim() - 1)))
    x0, y0 = gx.floor(), gy.floor()
    fx, fy = gx - x0, gy - y0
    out = 0
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            idx = base + yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)
            out = out + wx * wy * flat[idx.reshape(-1)].view_as(gx) * ok
    return out


class RingFovea(nn.Module):
    """The author's ring CNN as a fovea.

    Around a fixation, ``levels`` rings of ``size x size`` samples are taken
    with spacing ``base * 2^k`` pixels (x a predicted zoom): ring k covers
    ``size * base * 2^k`` pixels (8 x 2 px = 16 px, about one glyph, up to
    256 px at the defaults).  The main convolutional
    stream starts at the centre at full resolution and goes deeper level by
    level; at every level it is pooled 2x and placed in the centre of the next,
    coarser ring, whose own shallow features (the periphery) flow in around it.
    The same stem and block serve every level (one scale-recursive operator).
    Each level leaves 2x2 tokens, so the glimpse is centre-heavy: fine tokens
    for the fixated symbol, coarse ones for its surroundings.  Cost per glimpse
    does not depend on the image size.
    """

    def __init__(self, d: int, levels: int, size: int, ch: int, base: float):
        super().__init__()
        assert size % 4 == 0
        self.levels, self.size, self.base = levels, size, base
        # No normalization inside the ring stream: with thousands of glimpses per
        # batch the stream is memory-bound, and fp32 norm passes doubled its cost.
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        self.level_bias = nn.Parameter(torch.zeros(levels, ch))
        self.block = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1))
        self.checkpoint = False  # per-level recomputation; not needed since the stream is light (1.5 GiB peak)
        self.tok = nn.Linear(ch, d)
        self.tok_pos = nn.Parameter(torch.zeros(levels * 4, d))
        self.steer_norm = nn.LayerNorm(d)
        self.steer = nn.Linear(d, 3)  # dx, dy (pixels, bounded), log zoom

    def init_extra(self) -> None:
        for conv in (self.stem, self.block[0], self.block[2]):
            conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last)
        nn.init.zeros_(self.steer.weight)
        nn.init.zeros_(self.steer.bias)
        nn.init.zeros_(self.level_bias)
        nn.init.normal_(self.tok_pos, std=0.02)

    def fixate(self, weights: Tensor, x: Tensor, grid: tuple[int, int], cell: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Centroid of the cross-attention ``weights`` (B, T, h*w) over the cell
        centres, plus a learned bounded offset; and a zoom in [e^-0.7, e^0.7]."""

        gh, gw = grid
        ys = (torch.arange(gh, device=x.device, dtype=torch.float32) + 0.5) * cell[0]
        xs = (torch.arange(gw, device=x.device, dtype=torch.float32) + 0.5) * cell[1]
        pos = torch.stack([xs.repeat(gh), ys.repeat_interleave(gw)], dim=-1)  # (h*w, 2) as (x, y)
        steer = self.steer(self.steer_norm(x)).float()
        centre = weights @ pos + 16.0 * torch.tanh(steer[..., :2])
        zoom = torch.exp(0.7 * torch.tanh(steer[..., 2]))
        return centre, zoom

    def _level(self, ring: Tensor, main: Tensor | None, k: int) -> Tensor:
        r = self.stem(ring) + self.level_bias[k].view(1, -1, 1, 1).to(ring.dtype)
        if main is not None:  # the deeper centre stream enters the middle of the coarser ring
            q = self.size // 4
            r = r + F.pad(F.avg_pool2d(main, 2), (q, q, q, q))
        return F.gelu(r + self.block(r))

    def forward(self, pyramid: list[Tensor], img: Tensor, centre: Tensor, zoom: Tensor) -> Tensor:
        """(N, levels*4, d) glimpse tokens for N fixations ``centre`` (N, 2) px in images ``img`` (N,)."""

        n, s = centre.shape[0], self.size
        offs = torch.arange(s, device=centre.device, dtype=torch.float32) - (s - 1) / 2
        main, toks = None, []
        for k in range(self.levels):
            step = self.base * (2.0**k) * zoom[:, None]  # (N, 1) pixels between samples
            lv = min(k + int(math.log2(self.base)), len(pyramid) - 1)  # pyramid level near the sample spacing
            gx = ((centre[:, 0:1] + offs * step) / 2.0**lv - 0.5)[:, None, :].expand(n, s, s)
            gy = ((centre[:, 1:2] + offs * step) / 2.0**lv - 0.5)[:, :, None].expand(n, s, s)
            ring = bilinear_gather(pyramid[lv], img, gx, gy).unsqueeze(1).to(self.stem.weight.dtype)
            ring = ring.contiguous(memory_format=torch.channels_last)  # NHWC convs: no layout conversions
            if self.checkpoint and self.training and torch.is_grad_enabled():  # recompute per level: bounded memory
                from torch.utils.checkpoint import checkpoint

                main = checkpoint(self._level, ring, main, k, use_reentrant=False)
            else:
                main = self._level(ring, main, k)
            # 2x2 tokens per level (adaptive pooling has no deterministic backward)
            toks.append(F.avg_pool2d(main, s // 2).flatten(2).transpose(1, 2))
        return self.tok(torch.cat(toks, dim=1)) + self.tok_pos.to(main.dtype)


class GlimpseRead(nn.Module):
    """Every decoder position attends to the tokens of its own glimpse only."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        b, t, d = x.shape
        dh = d // self.heads
        q = self.q(self.norm(x)).view(b, t, self.heads, dh)
        k, v = self.kv(g).chunk(2, dim=-1)
        k, v = k.view(b, t, -1, self.heads, dh), v.view(b, t, -1, self.heads, dh)
        a = torch.softmax(torch.einsum("bthd,btghd->bthg", q.float(), k.float()) / math.sqrt(dh), dim=-1)
        return self.out(torch.einsum("bthg,btghd->bthd", a.to(v.dtype), v).reshape(b, t, d))


def image_pyramid(images: Tensor, levels: int) -> list[Tensor]:
    """Ink planes (B, H/2^k, W/2^k) for k < levels (2x2 average pooling)."""

    planes, x = [], images.float()
    for k in range(levels):
        if k:
            # ceil_mode keeps a level even when a side runs out (a 32 px tall
            # image still has 7 levels); the fixed divisor counts pixels past
            # the border as paper, so cell j always covers [j 2^k, (j+1) 2^k).
            x = F.avg_pool2d(x, 2, ceil_mode=True, divisor_override=4)
        planes.append(x[:, 0])
    return planes


# ------------------------------------------------------------------------- readout


class CellHead(nn.Module):
    """One MLP per vocabulary cell, all reading the same latent, plus all-to-all links between the cells:
    ``s_v = w2_v . gelu(W1_v z + b1_v)``, ``out = s + gelu(s) @ L^T`` (L: V x V).  Zero-initialized readout
    and links, so training starts exactly at the tied head."""

    def __init__(self, d_in: int, vocab: int, k: int):
        super().__init__()
        self.vocab, self.k = vocab, k
        self.w1 = nn.Parameter(torch.empty(d_in, vocab * k))
        self.b1 = nn.Parameter(torch.zeros(vocab * k))
        self.w2 = nn.Parameter(torch.empty(vocab, k))
        self.lateral = nn.Parameter(torch.empty(vocab, vocab))  # cell j -> cell i

    def init_extra(self) -> None:
        nn.init.normal_(self.w1, std=self.w1.shape[0] ** -0.5)
        nn.init.zeros_(self.w2)
        nn.init.zeros_(self.lateral)

    def forward(self, z: Tensor) -> Tensor:
        h = F.gelu(z @ self.w1.to(z.dtype) + self.b1.to(z.dtype))
        s = (h.view(*z.shape[:-1], self.vocab, self.k) * self.w2.to(z.dtype)).sum(-1)
        return s + F.gelu(s) @ self.lateral.to(z.dtype).T


class LayerMix(nn.Module):
    """Softmax-weighted mix of every decoder layer's normalized state (ELMo-style)."""

    def __init__(self, d: int, layers: int):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.mix = nn.Parameter(torch.zeros(layers))

    def forward(self, states: list[Tensor]) -> Tensor:
        w = torch.softmax(self.mix, dim=0)
        return sum(w[i].to(s.dtype) * n(s) for i, (n, s) in enumerate(zip(self.norms, states)))


class MemoryReread(nn.Module):
    """The readout's own query over the last decoder layer's cross-attention keys/values."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, kv: tuple[Tensor, Tensor]) -> Tensor:
        b, n, d = x.shape
        q = self.q(self.norm(x)).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = kv
        y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class DetectorMap(nn.Module):
    """Class evidence of every memory cell (memory -> V), read where the decoder's last layer attends:
    ``logits += w_t @ D(memory)``.  Zero-initialized."""

    def __init__(self, d: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(d, vocab)

    def init_extra(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, memory: Tensor) -> Tensor:
        return self.proj(memory)


# ------------------------------------------------------------------------- Model


class Im2Latex(nn.Module):
    MAX_DEPTH = 15

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.backbone = ConvBackbone(cfg.cnn_channels, cfg.cnn_blocks, cfg.rings, cfg.inject, cfg.enc_stride)
        self.proj = nn.Linear(self.backbone.out_channels, d)
        self.enc_drop = nn.Dropout(cfg.dropout)
        if cfg.encoder == "transformer":
            blocks = (EncoderLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout, cfg.attn_kernel)
                      for _ in range(cfg.enc_layers))
        elif cfg.encoder == "aas":
            blocks = (AASBlock(d, cfg.n_heads, cfg.local_expansion, cfg.aas_kernel) for _ in range(cfg.enc_layers))
        elif cfg.encoder == "aas_sort":
            blocks = (SortBlock(d, cfg.n_heads, cfg.local_expansion, cfg.sort_window) for _ in range(cfg.enc_layers))
        else:
            raise ValueError(f"unknown encoder {cfg.encoder!r}")
        self.encoder = nn.ModuleList(blocks)
        self.enc_norm = nn.LayerNorm(d)
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.register_buffer("tok_pos", sinusoid_1d(cfg.max_len, d), persistent=False)
        self.dec_drop = nn.Dropout(cfg.dropout)
        self.decoder = nn.ModuleList(DecoderLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout, cfg.attn_kernel)
                                     for _ in range(cfg.dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        # Registered last, so enabling it leaves the initialization of every
        # other parameter unchanged.
        if cfg.depth_pos and min(cfg.open_id, cfg.close_id) < 0:
            raise ValueError("depth_pos needs open_id and close_id")
        if cfg.count_feed and not cfg.count_head:
            raise ValueError("count_feed needs count_head")
        if cfg.deform_layers and cfg.enc_stride != "16x8":
            raise ValueError("deform reading assumes the 16x8 stage layout (fine maps at strides 4 and 8)")
        with torch.random.fork_rng(devices=[]):  # construction draws must not shift the other inits
            self.count = nn.Linear(d, cfg.vocab_size) if cfg.count_head else None
            self.count_proj = nn.Linear(cfg.vocab_size, d) if cfg.count_feed else None
            fine = tuple(cfg.cnn_channels[1:3])  # stage outputs at strides 4 and 8
            self.deform = (nn.ModuleList(RingDeformRead(d, cfg.n_heads, fine) for _ in range(cfg.deform_layers))
                           if cfg.deform_layers else None)
            self.feedback = (nn.ModuleList(AttentionFeedback(cfg.n_heads, cfg.feedback_rings, cfg.ring_bias, cfg.coverage)
                                           for _ in range(cfg.dec_layers - 1))
                             if cfg.ring_bias or cfg.coverage else None)
            self.depth_embed = nn.Embedding(self.MAX_DEPTH + 1, d) if cfg.depth_pos else None
            self.dir_embed = nn.Embedding(2, d) if cfg.bidir else None
            self.fovea = (RingFovea(d, cfg.fovea_levels, cfg.fovea_size, cfg.fovea_channels, cfg.fovea_base)
                          if cfg.fovea else None)
            self.fovea_read = (nn.ModuleList(GlimpseRead(d, cfg.n_heads) for _ in range(cfg.fovea_layer, cfg.dec_layers))
                               if cfg.fovea else None)
            if cfg.head not in ("tied", "cells", "cells_ml", "cells_vis", "cells_det"):
                raise ValueError(f"unknown head {cfg.head!r}")
            self.head = CellHead(d, cfg.vocab_size, cfg.head_k) if cfg.head != "tied" else None
            self.head_mix = LayerMix(d, cfg.dec_layers) if cfg.head == "cells_ml" else None
            self.head_read = MemoryReread(d, cfg.n_heads) if cfg.head == "cells_vis" else None
            self.head_det = DetectorMap(d, cfg.vocab_size) if cfg.head == "cells_det" else None
        self._readout: dict = {}  # side inputs of the readout, left by _decode for _logits
        self._det_map: Tensor | None = None  # (B, N, V) class evidence of the memory (cells_det), set by _memory_kv
        self._init_weights()

    LATE = ("count.", "count_proj.", "depth_embed.", "fovea.", "fovea_read.", "feedback.", "deform.", "dir_embed.",
            "head.", "head_mix.", "head_read.", "head_det.")  # optional parts, initialized after everything else

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and not name.startswith(("backbone", *self.LATE)):
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, std=self.cfg.d_model**-0.5)
        # Residual branches are NOT zero-initialized: at init the signal would
        # then pass only through the downsampling shortcuts, and the probe
        # learned the image more slowly (first-token accuracy 0.71 -> 0.76
        # with both fixes, 0.60 with this one alone).
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for name, p in self.named_parameters():
            if p.dim() > 1 and name.startswith(self.LATE):
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_extra"):
                m.init_extra()
        if self.count is not None:
            # Densities start near zero (expected count ~0.3 per symbol over a
            # typical map); the Poisson loss still pulls present symbols up.
            nn.init.constant_(self.count.bias, -8.0)

    # -- encoder
    def encode(self, images: Tensor, feats: list | None = None) -> Tensor:
        """(B, 1, H, W) ink intensity -> (B, h * w, d) memory (``enc_stride`` cells); ``feats`` (a list) also
        receives the stride-4 and stride-8 CNN maps for deformable reading."""

        if self.cfg.coarse > 1:
            images = F.avg_pool2d(images, self.cfg.coarse)
        stages: list | None = [] if feats is not None else None
        f = self.backbone(images, stages)
        if feats is not None:
            feats.extend(stages[1:3])
        b, c, h, w = f.shape
        x = self.proj(f.flatten(2).transpose(1, 2))
        x = self.enc_drop(x + sinusoid_2d(h, w, self.cfg.d_model, x.device).to(x.dtype))
        if self.cfg.encoder == "transformer":
            for layer in self.encoder:
                x = layer(x)
        else:  # map-shaped mixers
            x = x.transpose(1, 2).reshape(b, -1, h, w)
            for layer in self.encoder:
                x = layer(x)
            x = x.flatten(2).transpose(1, 2)
        return self.enc_norm(x)

    def _memory_kv(self, memory: Tensor) -> list[tuple[Tensor, Tensor]]:
        if self.head_det is not None:
            self._det_map = self.head_det(memory)
        return [layer.cross_attn.project_kv(memory) for layer in self.decoder]

    def _embed(self, tokens: Tensor, offset: int, depth: Tensor | None = None,
               direction: Tensor | None = None) -> Tensor:
        n = tokens.shape[1]
        x = self.embed(tokens) * math.sqrt(self.cfg.d_model)
        x = x + self.tok_pos[offset : offset + n].to(x.dtype)
        if self.depth_embed is not None:
            x = x + self.depth_embed(depth.clamp(0, self.MAX_DEPTH)).to(x.dtype)
        if self.dir_embed is not None:  # 0 = left to right (canonical), 1 = right to left
            if direction is None:
                direction = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
            x = x + self.dir_embed(direction)[:, None].to(x.dtype)
        return x

    def _depth_step(self, tokens: Tensor) -> Tensor:
        return (tokens == self.cfg.open_id).long() - (tokens == self.cfg.close_id).long()

    def _logits(self, x: Tensor) -> Tensor:
        z = self.dec_norm(x)
        logits = F.linear(z, self.embed.weight)
        if self.head is None:
            return logits
        if self.head_mix is not None:
            z = self.head_mix(self._readout["states"])
        elif self.head_read is not None:
            z = z + self.head_read(x, self._readout["kv"])
        elif self.head_det is not None:
            w = self._readout["w"]
            logits = logits + torch.matmul(w.to(self._det_map.dtype), self._det_map).to(logits.dtype)
        return logits + self.head(z)

    def _glimpses(self, pyramid: list[Tensor], centre: Tensor, zoom: Tensor, valid: Tensor | None) -> Tensor:
        """(B, T, G, d) glimpse tokens; computed only at ``valid`` positions (others get zeros)."""

        b, t, _ = centre.shape
        img = torch.arange(b, device=centre.device).repeat_interleave(t)
        c, z = centre.reshape(b * t, 2), zoom.reshape(b * t)
        if valid is None:
            sel = None
        else:
            sel = valid.reshape(-1).nonzero().squeeze(1)  # ascending: a deterministic packing
            img, c, z = img[sel], c[sel], z[sel]
        g = self.fovea(pyramid, img, c, z)
        if sel is not None:  # unpack: invalid positions read a zero row
            slot = torch.zeros(b * t, dtype=torch.long, device=g.device)
            slot[sel] = torch.arange(1, sel.numel() + 1, device=g.device)
            g = torch.cat([g.new_zeros(1, *g.shape[1:]), g])[slot]
        return g.view(b, t, *g.shape[1:])

    def _decode(self, x: Tensor, images: Tensor, mem_kv: list, caches: list | None, pyramid: list | None,
                trace: list | None = None, valid: Tensor | None = None, feats: list | None = None) -> Tensor:
        fl = self.cfg.fovea_layer if self.fovea is not None else len(self.decoder) + 1  # no fovea: plain layers
        fb = self.feedback
        first_deform = len(self.decoder) - self.cfg.deform_layers
        g, prev, pos = None, None, None
        states = []
        if self.head_read is not None:
            self._readout = {"kv": mem_kv[-1]}
        for i, (layer, kv) in enumerate(zip(self.decoder, mem_kv)):
            cache = caches[i] if caches is not None else None
            if fb is not None:
                bias = None
                if i >= 1:
                    pos = cell_centres(images, self.cfg.coarse, self.cfg.enc_stride) if pos is None else pos
                    cov = None
                    if self.cfg.coverage:
                        if cache is None:  # teacher forcing: exclusive prefix sum over the steps
                            cov = prev.cumsum(dim=1) - prev
                        else:  # generation: running sum of the earlier steps
                            cov = cache.get("cov", torch.zeros_like(prev))
                            cache["cov"] = cov + prev
                    bias = fb[i - 1](prev, pos, cov)
                x, w = layer(x, kv, cache, return_cross=True, cross_bias=bias)
                prev = w.detach()
            elif i == fl - 1 or i >= first_deform or (self.head_det is not None and i == len(self.decoder) - 1):
                x, w = layer(x, kv, cache, return_cross=True)
                if self.head_det is not None and i == len(self.decoder) - 1:
                    self._readout = {"w": w}
            else:
                x = layer(x, kv, cache)
            if i >= first_deform:  # read fine features around this layer's own focus
                pos = cell_centres(images, self.cfg.coarse, self.cfg.enc_stride) if pos is None else pos
                focus = w.detach() @ pos
                strides = [4.0 * self.cfg.coarse, 8.0 * self.cfg.coarse]
                x = x + self.dec_drop(self.deform[i - first_deform](x, feats, strides, focus, valid))
            if i == fl - 1:
                c = self.cfg.coarse
                sh, sw = stride_hw(self.cfg.enc_stride)
                grid, cell = (images.shape[-2] // (sh * c), images.shape[-1] // (sw * c)), (sh * c, sw * c)
                centre, zoom = self.fovea.fixate(w, x, grid, cell)
                if trace is not None:
                    trace.append(torch.cat([centre, zoom[..., None]], dim=-1))
                g = self._glimpses(pyramid, centre, zoom, valid)
            if i >= fl:
                x = x + self.dec_drop(self.fovea_read[i - fl](x, g))
            if self.head_mix is not None:
                states.append(x)
        if self.head_mix is not None:
            self._readout = {"states": states}
        return x

    # -- training
    def forward(self, images: Tensor, tokens_in: Tensor, return_aux: bool = False, direction: Tensor | None = None):
        """Teacher-forced logits (B, T, V) for ``tokens_in`` = BOS + target[:-1].

        With ``return_aux`` also returns a dict of auxiliary outputs:
        ``log_count`` (B, V) -- log of the expected count of every symbol,
        ``log sum_p exp(z_p)`` over the memory positions (count head only).
        """

        feats: list | None = [] if self.deform is not None else None
        memory = self.encode(images, feats)
        mem_kv = self._memory_kv(memory)
        log_count = self._log_count(memory) if self.count is not None else None
        depth = self._depth_step(tokens_in).cumsum(dim=1) if self.depth_embed is not None else None
        x = self.dec_drop(self._embed(tokens_in, 0, depth, direction) + self._count_feature(log_count))
        pyramid = image_pyramid(images, self.cfg.fovea_levels + 2) if self.fovea is not None else None
        x = self._decode(x, images, mem_kv, None, pyramid, valid=tokens_in != 0, feats=feats)  # 0 = <pad>
        logits = self._logits(x)
        if not return_aux:
            return logits
        aux = {"log_count": log_count} if log_count is not None else {}
        return logits, aux

    def _log_count(self, memory: Tensor) -> Tensor:
        return torch.logsumexp(self.count(memory).float().clamp(max=10.0), dim=1)

    def _count_feature(self, log_count: Tensor | None):
        """(B, 1, d) feature of the predicted symbol counts, or 0 without ``count_feed``."""

        if self.count_proj is None:
            return 0
        return self.count_proj(torch.log1p(log_count.exp()).to(self.count_proj.weight.dtype)).unsqueeze(1)

    # -- inference
    @torch.no_grad()
    def generate(self, images: Tensor, bos_id: int, eos_id: int, max_len: int | None = None,
                 trace: list | None = None) -> Tensor:
        """Greedy decoding -> (B, <=max_len) token ids without BOS, EOS-terminated.

        ``trace`` (a list) collects the foveal fixations (B, 1, 3: x, y, zoom) per step."""

        max_len = min(max_len or self.cfg.max_len, self.cfg.max_len) - 1
        feats: list | None = [] if self.deform is not None else None
        memory = self.encode(images, feats)
        mem_kv = self._memory_kv(memory)
        feed = self._count_feature(self._log_count(memory)) if self.count_proj is not None else 0
        pyramid = image_pyramid(images, self.cfg.fovea_levels + 2) if self.fovea is not None else None
        b = images.shape[0]
        caches: list[dict] = [{} for _ in self.decoder]
        token = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        depth = torch.zeros(b, 1, dtype=torch.long, device=images.device)
        out = []
        for step in range(max_len):
            depth = depth + self._depth_step(token)  # depth after the token being embedded, as in training
            x = self._embed(token, step, depth) + feed
            x = self._decode(x, images, mem_kv, caches, pyramid, trace, feats=feats)
            token = self._logits(x)[:, -1].argmax(dim=-1, keepdim=True)
            token = token.masked_fill(done[:, None], eos_id)
            out.append(token)
            done |= token[:, 0] == eos_id
            if bool(done.all()):
                break
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def beam_generate(self, images: Tensor, bos_id: int, eos_id: int, beam: int = 5, alpha: float = 0.6,
                      max_len: int | None = None, pairs: list[tuple[list[int], list[int]]] | None = None,
                      nbest: bool = False):
        """Deterministic beam search -> per image the best hypothesis (token ids, no BOS/EOS).

        Hypotheses are ranked by ``sum log p / ((5 + n) / 6) ** alpha`` (GNMT length
        penalty).  ``pairs`` lists (openers, closers) token-id groups -- e.g. ``{``/``}``,
        ``\\left*``/``\\right*``, ``\\begin{*}``/``\\end{*}`` -- whose balance is enforced: a
        closer is impossible at depth 0 and EOS is impossible while a group is open.
        Top-k runs on the CPU, so the choice and its tie-breaking are deterministic.
        """

        b, k = images.shape[0], beam
        max_len = min(max_len or self.cfg.max_len, self.cfg.max_len) - 1
        rep = lambda t: t.repeat_interleave(k, dim=0)  # noqa: E731
        feats: list | None = [] if self.deform is not None else None
        memory = self.encode(images, feats)
        feed = self._count_feature(self._log_count(memory)) if self.count_proj is not None else 0
        if torch.is_tensor(feed):
            feed = rep(feed)
        mem_kv = [(rep(kk), rep(vv)) for kk, vv in self._memory_kv(memory)]
        feats = [rep(f) for f in feats] if feats is not None else None
        pyramid = [rep(p) for p in image_pyramid(images, self.cfg.fovea_levels + 2)] if self.fovea is not None else None
        imgs = rep(images)
        dev = images.device
        caches: list[dict] = [{} for _ in self.decoder]
        token = torch.full((b * k, 1), bos_id, dtype=torch.long, device=dev)
        depth = torch.zeros(b * k, 1, dtype=torch.long, device=dev)
        level = torch.zeros(b * k, len(pairs or []), dtype=torch.long)  # open groups per pair type (CPU)
        scores = torch.full((b, k), float("-inf"))
        scores[:, 0] = 0.0
        hist = [[] for _ in range(b * k)]
        finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(b)]
        alive = [True] * b
        lp = lambda n: ((5 + n) / 6) ** alpha  # noqa: E731
        for step in range(max_len):
            depth = depth + self._depth_step(token)
            x = self._embed(token, step, depth) + feed
            x = self._decode(x, imgs, mem_kv, caches, pyramid, feats=feats)
            logp = torch.log_softmax(self._logits(x)[:, -1].float(), dim=-1).cpu()
            for j, (opens, closes) in enumerate(pairs or []):
                shut = level[:, j] == 0
                logp[shut[:, None] & torch.isin(torch.arange(logp.shape[1]), torch.tensor(closes))[None]] = float("-inf")
                logp[~shut, eos_id] = float("-inf")
            cand = (scores.view(-1, 1) + logp).view(b, -1)
            top_v, top_i = cand.topk(2 * k, dim=1)
            src = torch.zeros(b * k, dtype=torch.long)
            nxt = torch.full((b * k,), eos_id, dtype=torch.long)
            new_scores = torch.full((b, k), float("-inf"))
            for i in range(b):
                if not alive[i]:
                    src[i * k : (i + 1) * k] = i * k
                    continue
                filled = 0
                for v, idx in zip(top_v[i].tolist(), top_i[i].tolist()):
                    if v == float("-inf"):
                        break
                    bi, tok = divmod(idx, logp.shape[1])
                    if tok == eos_id:
                        finished[i].append((v / lp(step + 1), hist[i * k + bi]))
                        continue
                    src[i * k + filled], nxt[i * k + filled] = i * k + bi, tok
                    new_scores[i, filled] = v
                    filled += 1
                    if filled == k:
                        break
                best_alive = new_scores[i].max().item()
                best_done = max((f[0] for f in finished[i]), default=float("-inf"))
                # Alive scores only fall; at length max_len they cannot beat a finished one.
                if filled == 0 or (len(finished[i]) >= k and best_alive / lp(max_len) <= best_done):
                    alive[i] = False
            if not any(alive):
                break
            hist = [hist[s] + [t] for s, t in zip(src.tolist(), nxt.tolist())]
            scores = new_scores
            idx_dev = src.to(dev)
            for cache in caches:
                for key in cache:
                    cache[key] = cache[key].index_select(0, idx_dev)
            depth = depth.index_select(0, idx_dev)
            level = level.index_select(0, src)
            for j, (opens, closes) in enumerate(pairs or []):
                level[:, j] += torch.isin(nxt, torch.tensor(opens)).long() - torch.isin(nxt, torch.tensor(closes)).long()
            token = nxt.to(dev).view(-1, 1)
        out = []
        for i in range(b):
            if not finished[i]:  # nothing ended within max_len: the alive hypotheses
                finished[i] = [(float(scores[i, j]) / lp(max_len), hist[i * k + j]) for j in range(k)
                               if scores[i, j] > float("-inf")]
            ranked = sorted(finished[i], key=lambda f: -f[0])  # stable: ties keep discovery order
            out.append(ranked if nbest else ranked[0][1])
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
