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
Everything is plain deterministic tensor math.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonal polar factor ``U V^T`` of a 2D ``g`` (bf16, Keller Jordan's quintic)."""

    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Moonshot-scaled Muon for ``use_muon`` groups, decoupled AdamW for the rest."""

    def __init__(self, param_groups: Iterable[dict[str, Any]], *, lr: float, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 betas: tuple[float, float] = (0.9, 0.98), eps: float = 1e-8):
        groups = list(param_groups)
        if any("use_muon" not in g for g in groups):
            raise ValueError("every Muon param group must set 'use_muon'")
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for group in self.param_groups:
            (self._muon if group["use_muon"] else self._adamw)(group)

    def _muon(self, group: dict[str, Any]) -> None:
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.reshape(p.shape[0], -1)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(mom).add_(g)
            update = g.add(buf, alpha=mom) if group["nesterov"] else buf
            ortho = zeropower_via_newtonschulz5(update, group["ns_steps"]).to(p.dtype)
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(ortho.view_as(p), alpha=-lr * 0.2 * math.sqrt(max(g.shape)))

    def _adamw(self, group: dict[str, Any]) -> None:
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            state["exp_avg"].mul_(b1).add_(p.grad, alpha=1 - b1)
            state["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            if wd:
                p.mul_(1 - lr * wd)
            denom = (state["exp_avg_sq"] / (1 - b2**t)).sqrt_().add_(eps)
            p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - b1**t))


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
    one-channel stem, norms, biases and per-action biases stay on AdamW)."""

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
            elif name.startswith("backbone.stem") or not in_scope:
                decay.append(p)
            else:
                muon.append(p)
        return Muon([{"params": muon, "use_muon": True, "weight_decay": wd},
                     {"params": decay, "use_muon": False, "weight_decay": wd},
                     {"params": no_decay, "use_muon": False, "weight_decay": 0.0},
                     *({**g, "use_muon": False} for g in fb_groups)],
                    lr=lr, momentum=float(tcfg.get("muon_momentum", 0.95)))
    raise ValueError(f"unknown optimizer {kind!r}")
