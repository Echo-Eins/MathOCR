"""Optimizers: AdamW (control) and Muon with an AdamW fallback path.

Muon (Jordan et al. 2024) replaces the momentum update of a weight matrix with
its nearest semi-orthogonal matrix (quintic Newton-Schulz iteration), so every
singular direction of the update moves at the same rate.  The Moonshot recipe
(Liu et al. 2025) adds decoupled weight decay and rescales the orthogonal
update to RMS ``0.2 * sqrt(max(m, n))`` -- the RMS of a tuned AdamW update --
so Muon and AdamW share one learning rate, one weight decay and one schedule.
Ported from ``Mythos_P/open_mythos/muon.py``; convolution kernels are treated
as ``(out, in * kh * kw)`` matrices.

One optimizer object owns both kinds of groups (``use_muon``), so the
trainer's single state dict, LR schedule and resume logic stay unchanged.
Everything is plain deterministic tensor math.  A step costs a few dozen
kernel launches rather than thousands: element-wise updates use the
multi-tensor ``torch._foreach_*`` ops (same values as the per-tensor ops), and
the Newton-Schulz iteration runs once per matrix shape on a stacked batch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


# Polar Express (Amsel et al. 2025): a different minimax-optimal odd quintic per iteration, computed with the
# authors' optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2, cushion=0.02); the input
# is scaled by 1.01 x its Frobenius norm.  The first five suffice for five steps.
POLAR_EXPRESS = ((8.237312490495555, -23.157747414558187, 16.6805684114459),
                 (4.082441999064831, -2.8930477353325843, 0.5252849256975642),
                 (3.926347992254648, -2.854746803476524, 0.531802242289498),
                 (3.298218713308519, -2.42454198102671, 0.48632008358844137),
                 (2.297036943455259, -1.636625581259032, 0.4002628455953629))


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7, polar: bool = False) -> torch.Tensor:
    """Approximate orthogonal polar factor ``U V^T`` of a 2D ``g``, or of every matrix of a 3D
    batch of same-shape matrices (bf16, Keller Jordan's quintic, or the Polar Express sequence).  Each
    iteration is three GEMMs for the whole batch: ``b s + c s^2`` and ``a x + B x`` are fused into ``baddbmm``."""

    coeffs = [POLAR_EXPRESS[min(i, len(POLAR_EXPRESS) - 1)] for i in range(steps)] if polar else         [(3.4445, -4.7750, 2.0315)] * steps
    x = g.bfloat16()
    single = x.dim() == 2
    if single:
        x = x.unsqueeze(0)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * (1.01 if polar else 1.0) + eps)
    for a, b, c in coeffs:
        s = x @ x.mT
        x = torch.baddbmm(x, torch.baddbmm(s, s, s, beta=b, alpha=c), x, beta=a)
    if transposed:
        x = x.mT
    return x[0] if single else x


class Muon(torch.optim.Optimizer):
    """Moonshot-scaled Muon for ``use_muon`` groups, decoupled AdamW for the rest."""

    def __init__(self, param_groups: Iterable[dict[str, Any]], *, lr: float, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 betas: tuple[float, float] = (0.9, 0.98), eps: float = 1e-8, polar: bool = False,
                 neuron_norm: bool = False, beta2: float = 0.95):
        """``polar``: Polar Express coefficients instead of the fixed quintic; ``neuron_norm``: NorMuon (Li et
        al. 2025) -- the orthogonal update divided per output neuron by the root of an EMA (``beta2``) of its
        mean square, then rescaled to RMS 0.2 (the same as the plain Muon update)."""

        groups = list(param_groups)
        if any("use_muon" not in g for g in groups):
            raise ValueError("every Muon param group must set 'use_muon'")
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, betas=betas, eps=eps, polar=polar, neuron_norm=neuron_norm,
                                      beta2=beta2))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for group in self.param_groups:
            (self._muon if group["use_muon"] else self._adamw)(group)

    def _muon(self, group: dict[str, Any]) -> None:
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad.reshape(p.shape[0], -1) for p in params]
        bufs = []
        for p, g in zip(params, grads):
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            bufs.append(state["momentum_buffer"])
        torch._foreach_mul_(bufs, mom)
        torch._foreach_add_(bufs, grads)
        updates = torch._foreach_add(grads, bufs, alpha=mom) if group["nesterov"] else bufs
        if wd:
            torch._foreach_mul_(params, 1 - lr * wd)
        # One batched Newton-Schulz per (short side, long side): tall matrices join the wide ones
        # transposed (the iteration orthogonalizes either orientation).  Groups keep parameter order.
        by_shape: dict[tuple[int, int], list[int]] = {}
        for i, u in enumerate(updates):
            by_shape.setdefault((min(u.shape), max(u.shape)), []).append(i)
        for (_, long_side), idx in by_shape.items():
            tall = [updates[i].size(0) > updates[i].size(1) for i in idx]
            ortho = zeropower_via_newtonschulz5(
                torch.stack([updates[i].mT if t else updates[i] for i, t in zip(idx, tall)]), group["ns_steps"],
                polar=group["polar"])
            rows = [(o.mT if t else o) for t, o in zip(tall, ortho.unbind(0))]  # (out, in): a row per output neuron
            if not group["neuron_norm"]:
                torch._foreach_add_([params[i] for i in idx],
                                    [r.reshape(params[i].shape).to(params[i].dtype) for i, r in zip(idx, rows)],
                                    alpha=-lr * 0.2 * math.sqrt(long_side))
                continue
            for i, r in zip(idx, rows):
                state, r = self.state[params[i]], r.float()
                if "neuron_v" not in state:
                    state["neuron_v"] = torch.zeros(r.shape[0], 1, device=r.device)
                v = state["neuron_v"]
                v.mul_(group["beta2"]).add_(r.square().mean(dim=1, keepdim=True), alpha=1 - group["beta2"])
                r = r / (v.sqrt() + group["eps"])
                scale = 0.2 * math.sqrt(r.numel()) / r.norm().clamp_min(1e-12)
                params[i].add_((r * scale).reshape(params[i].shape).to(params[i].dtype), alpha=-lr)

    def _adamw(self, group: dict[str, Any]) -> None:
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        by_step: dict[int, list[torch.Tensor]] = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            by_step.setdefault(state["step"], []).append(p)
        for t, params in by_step.items():  # normally one bucket: every parameter steps every time
            grads = [p.grad for p in params]
            exp_avg = [self.state[p]["exp_avg"] for p in params]
            exp_avg_sq = [self.state[p]["exp_avg_sq"] for p in params]
            torch._foreach_mul_(exp_avg, b1)
            torch._foreach_add_(exp_avg, grads, alpha=1 - b1)
            torch._foreach_mul_(exp_avg_sq, b2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1 - b2)
            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)
            denom = torch._foreach_div(exp_avg_sq, 1 - b2**t)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(params, exp_avg, denom, value=-lr / (1 - b1**t))


def _no_decay(name: str, p: nn.Parameter) -> bool:
    return p.dim() < 2 or any(k in name for k in ("embed", "action_bias", "level_bias", "tok_pos", "ring_table"))


def _feedback_group(model: nn.Module, tcfg: dict) -> tuple[list[dict], set[int]]:
    """Attention-feedback tables/strengths get their own LR (``train.feedback_lr_mult``): a few
    dozen scalars that must grow to O(1) logits, which a 3e-4 Adam step cannot do in a short run."""

    mult = float(tcfg.get("feedback_lr_mult", 1.0))
    params = [p for n, p in model.named_parameters() if n.startswith("feedback.")]
    if not params or mult == 1.0:
        return [], set()
    return [{"params": params, "weight_decay": 0.0, "lr": tcfg["lr"] * mult}], {id(p) for p in params}


def build_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    """``train.optimizer``: ``adamw`` (control) or ``muon`` (hidden matrices; embeddings, the
    one-channel stem, norms, biases, per-action biases and ``dw_`` parameters -- depthwise kernels and
    block-diagonal gates, which are not one linear map -- stay on AdamW)."""

    lr, wd, kind = tcfg["lr"], tcfg["weight_decay"], tcfg.get("optimizer", "adamw")
    fb_groups, fb_ids = _feedback_group(model, tcfg)
    named = [(n, p) for n, p in model.named_parameters() if id(p) not in fb_ids]
    if kind == "adamw":
        decay, no_decay = [], []
        for name, p in named:
            (no_decay if _no_decay(name, p) else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0},
                                  *fb_groups], lr=lr, betas=(0.9, 0.98), eps=1e-8)
    if kind == "muon":
        # muon_scope: "all" matrices, only the CNN "backbone", or only the "transformer" part (the rest: AdamW)
        scope = tcfg.get("muon_scope", "all")
        muon, decay, no_decay = [], [], []
        for name, p in named:
            in_scope = scope == "all" or (scope == "backbone") == name.startswith("backbone.")
            if _no_decay(name, p):
                no_decay.append(p)
            elif name.startswith("backbone.stem") or not in_scope or ".dw_" in name:
                decay.append(p)
            else:
                muon.append(p)
        return Muon([{"params": muon, "use_muon": True, "weight_decay": wd},
                     {"params": decay, "use_muon": False, "weight_decay": wd},
                     {"params": no_decay, "use_muon": False, "weight_decay": 0.0},
                     *({**g, "use_muon": False} for g in fb_groups)],
                    lr=lr, momentum=float(tcfg.get("muon_momentum", 0.95)), polar=bool(tcfg.get("muon_polar", False)),
                    neuron_norm=bool(tcfg.get("muon_neuron_norm", False)))
    raise ValueError(f"unknown optimizer {kind!r}")
