"""Greedy decoding with CUDA graphs: the per-token decoder step captured once per batch key.

Eager ``Im2Latex.generate`` launches ~150 small kernels per token; on this
stack that is ~10 ms per step of pure launch overhead for microseconds of GPU
work.  Here the same step -- the same eager kernels, no compiler -- runs as a
replayed CUDA graph:

* the key/value cache of every decoder layer is a static ``(B, H, L, Dh)``
  buffer written at position ``t`` (``index_copy_``) and read in full under an
  additive mask (``0`` for positions ``<= t``, ``-inf`` beyond), so every step
  has one shape;
* attention for the single query of a step is two matmuls and a softmax: the
  fused attention kernels are tiled for 64 queries and spend 3.9 of a step's
  4.5 ms on one (eager decoding pays the same); keys are stored transposed and
  contiguous, ``(B, H, Dh, N)``, so no operand is copied per step -- what
  remains is reading the memory keys/values in fp32 (1.8 GB per step for 16
  images of 4 608 cells), the inherent cost of cross-attention over the image;
* the step position, the previous token, the done flags and the output row
  live in static device buffers that the graph reads and updates in place, so
  a step is ``graph.replay()`` and nothing else; ``done.all()`` is polled every
  few steps;
* one graph per key = (batch size, image height, image width), kept in an LRU
  cache under a byte budget (``cache_bytes``: the static buffers, dominated by
  the memory key/values of a large batch), so a big-batch evaluation holds one
  or two keys while single-image decoding keeps every shape it has seen; all
  captures share one memory pool (a graph's intermediates are transient and its
  state lives in its own buffers, so replays in any order are safe).  A capture
  costs a two-step warm-up and the recording, ~30 ms.

The arithmetic differs from the eager loop only in the attention over the
masked static cache instead of a growing one, so the outputs are compared as
strings (``evaluate --check-canonical`` and ``tests/test_graphs.py``).
Replaying a graph is deterministic: same kernels, same buffers.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from mathocr.model import Im2Latex


def supported(model: Im2Latex) -> bool:
    """Plain decoder layers only (no fovea, deformable reading, attention feedback, count feed, depth or
    direction embeddings): those variants keep per-step state the static step does not carry."""

    return all(getattr(model, name, None) is None for name in ("fovea", "deform", "feedback", "count_proj", "depth_embed",
                                                               "dir_embed", "head_mix", "head_read", "head_det"))


def enabled(model: Im2Latex, device: torch.device) -> bool:
    """On by default on CUDA for supported models; ``MATHOCR_CUDA_GRAPHS=0`` selects the eager loop."""

    return device.type == "cuda" and supported(model) and os.environ.get("MATHOCR_CUDA_GRAPHS", "1") != "0"


class GraphedGreedy:
    """``decode(images)`` == ``model.generate(images, bos, eos, max_len)`` up to the numerics above."""

    def __init__(self, model: Im2Latex, bos_id: int, eos_id: int, max_len: int | None = None, check_every: int = 4,
                 cache_bytes: int = 2 << 30):
        self.model, self.bos, self.eos = model, bos_id, eos_id
        self.max_len = min(max_len or model.cfg.max_len, model.cfg.max_len) - 1  # steps, as in ``generate``
        self.check_every, self.cache_bytes = check_every, cache_bytes
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self.cache: OrderedDict[tuple[int, int, int], tuple[torch.cuda.CUDAGraph, dict, int]] = OrderedDict()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.buf: dict = {}
        self.eager_fallbacks = 0  # decodes that ran eagerly because a capture ran out of memory twice

    # -- the static step (what the graph records)
    @staticmethod
    def _attend(attn, h: Tensor, k_t: Tensor, v: Tensor, bias: Tensor | None) -> Tensor:
        """One query per row against transposed keys ``k_t`` (B, H, Dh, N) and values ``v`` (B, H, N, Dh)."""

        q = attn.query(h)  # (B, H, 1, Dh)
        scores = torch.matmul(q, k_t) * q.shape[-1] ** -0.5
        if bias is not None:
            scores = scores + bias
        y = torch.matmul(torch.softmax(scores, dim=-1), v)
        return attn.finish(y, h)

    def _step(self) -> None:
        model, b = self.model, self.buf
        t, mem = b["t"], b["mem"]
        x = model.embed(b["token"]) * math.sqrt(model.cfg.d_model) + model.tok_pos.index_select(0, t).unsqueeze(0)
        mask = torch.where(b["arange"] <= t, 0.0, float("-inf")).view(1, 1, 1, -1)
        v_first = {} if model.cfg.value_residual else None
        for i, layer in enumerate(model.decoder):
            h = layer.n1(x)
            k, v = layer.self_attn.project_kv(h)
            v = layer.mix_values(v, v_first)
            kc, vc = b["kcache"][i], b["vcache"][i]  # (B, H, Dh, L) and (B, H, L, Dh)
            kc.index_copy_(3, t, k.transpose(-1, -2))
            vc.index_copy_(2, t, v)
            x = x + self._attend(layer.self_attn, h, kc, vc, mask)
            x = x + self._attend(layer.cross_attn, layer.n2(x), mem[i][0], mem[i][1], None)
            x = x + layer.ff(layer.n3(x))
        token = model._logits(x)[:, -1].argmax(dim=-1, keepdim=True)
        token = token.masked_fill(b["done"][:, None], self.eos)
        b["token"].copy_(token)
        b["done"].logical_or_(token[:, 0] == self.eos)
        b["out"].index_copy_(1, t, token)
        t.add_(1)

    def _reset(self) -> None:
        b = self.buf
        b["token"].fill_(self.bos)
        b["done"].zero_()
        b["t"].zero_()
        b["out"].fill_(self.eos)
        for c in (*b["kcache"], *b["vcache"]):  # masked positions must hold finite values (0 * inf = nan)
            c.zero_()

    def release(self) -> None:
        self.cache.clear()
        self.graph, self.buf = None, {}
        self.pool = torch.cuda.graph_pool_handle()  # the old pool dies with its graphs; never reuse it

    def _capture(self, key: tuple[int, int, int], mem_kv: list[tuple[Tensor, Tensor]]) -> None:
        model = self.model
        bsz, dev = key[0], mem_kv[0][0].device
        heads, dh, n_layers = model.cfg.n_heads, model.cfg.d_model // model.cfg.n_heads, len(model.decoder)
        buf = {
            "token": torch.full((bsz, 1), self.bos, dtype=torch.long, device=dev),
            "done": torch.zeros(bsz, dtype=torch.bool, device=dev),
            "t": torch.zeros(1, dtype=torch.long, device=dev),
            "arange": torch.arange(self.max_len, device=dev),
            "out": torch.full((bsz, self.max_len), self.eos, dtype=torch.long, device=dev),
            "kcache": [torch.zeros(bsz, heads, dh, self.max_len, device=dev) for _ in range(n_layers)],
            "vcache": [torch.zeros(bsz, heads, self.max_len, dh, device=dev) for _ in range(n_layers)],
            "mem": [(k.transpose(-1, -2).contiguous(), v.contiguous()) for k, v in mem_kv],
        }
        nbytes = sum(t.numel() * t.element_size() for t in (*buf["kcache"], *buf["vcache"], buf["out"], *sum(buf["mem"], ())))
        while self.cache and sum(b for _, _, b in self.cache.values()) + nbytes > self.cache_bytes:
            self.cache.popitem(last=False)  # least recently used
        self.buf = buf
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm-up on the capture stream: lazy init (cuBLAS workspace, kernels) stays outside the graph
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self._reset()
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()  # nothing pending while recording (torch.cuda.graph does this plus a gc.collect)
        with torch.cuda.stream(s):
            g.capture_begin(pool=self.pool)
            recorded = False
            try:
                self._step()
                recorded = True
            finally:
                if recorded:
                    g.capture_end()
                else:  # a launch failed mid-capture: still end it, so the stream is usable for the retry
                    try:
                        g.capture_end()
                    except RuntimeError:
                        pass
        torch.cuda.synchronize()
        self._reset()
        self.cache[key] = (g, buf, nbytes)
        self.graph = g

    @torch.no_grad()
    def decode(self, images: Tensor) -> Tensor:
        """(B, 1, H, W) -> (B, <= max_len) token ids without BOS, EOS-terminated (EOS after EOS)."""

        model = self.model
        memory = model.encode(images)  # eager: a few big kernels
        mem_kv = model._memory_kv(memory)
        key = (images.shape[0], images.shape[-2], images.shape[-1])
        if key in self.cache:
            self.cache.move_to_end(key)
            self.graph, self.buf, _ = self.cache[key]
            for (k, v), (sk, sv) in zip(mem_kv, self.buf["mem"]):
                sk.copy_(k.transpose(-1, -2))
                sv.copy_(v)
            self._reset()
        else:
            try:
                self._capture(key, mem_kv)
            except RuntimeError as e:  # e.g. the driver refusing to reserve address space for a new pool
                if "out of memory" not in str(e).lower():
                    raise
                self.release()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    self._capture(key, mem_kv)
                except RuntimeError as e2:
                    if "out of memory" not in str(e2).lower():
                        raise
                    self.release()
                    torch.cuda.empty_cache()
                    self.eager_fallbacks += 1
                    return model.generate(images, self.bos, self.eos, self.max_len + 1)  # same strings, slower
        steps = 0
        for step in range(self.max_len):
            self.graph.replay()
            steps = step + 1
            if steps % self.check_every == 0 and bool(self.buf["done"].all()):
                break
        return self.buf["out"][:, :steps].clone()
