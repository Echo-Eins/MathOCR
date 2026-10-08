"""Image-to-LaTeX encoder-decoder.

Encoder: ResNet-style CNN (BasicBlocks, total stride 16 vertically and 8
horizontally -- formulas are wide and sub/superscripts need horizontal
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
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class ConvBackbone(nn.Module):
    # Per-stage strides: overall (16, 8) after the stride-2 stem.
    STRIDES = ((1, 1), (2, 2), (2, 2), (2, 1))

    def __init__(self, channels: tuple[int, ...], blocks: tuple[int, ...]):
        super().__init__()
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

    def forward(self, x: Tensor) -> Tensor:
        return self.stages(self.stem(x))


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


# ------------------------------------------------------------------- Transformer


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
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
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=causal)
        b, h, n, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(b, n, h * dh))


class FeedForward(nn.Sequential):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * mult, d))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, mult: int, dropout: float):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(d), Attention(d, heads, dropout)
        self.n2, self.ff = nn.LayerNorm(d), FeedForward(d, mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.n1(x)
        x = x + self.drop(self.attn.attend(h, *self.attn.project_kv(h), causal=False))
        return x + self.drop(self.ff(self.n2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, mult: int, dropout: float):
        super().__init__()
        self.n1, self.self_attn = nn.LayerNorm(d), Attention(d, heads, dropout)
        self.n2, self.cross_attn = nn.LayerNorm(d), Attention(d, heads, dropout)
        self.n3, self.ff = nn.LayerNorm(d), FeedForward(d, mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mem_kv: tuple[Tensor, Tensor], cache: dict | None = None) -> Tensor:
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
        x = x + self.drop(self.cross_attn.attend(self.n2(x), *mem_kv, causal=False))
        return x + self.drop(self.ff(self.n3(x)))


# ------------------------------------------------------------------------- Model


class Im2Latex(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.backbone = ConvBackbone(cfg.cnn_channels, cfg.cnn_blocks)
        self.proj = nn.Linear(self.backbone.out_channels, d)
        self.enc_drop = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList(EncoderLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.enc_layers))
        self.enc_norm = nn.LayerNorm(d)
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.register_buffer("tok_pos", sinusoid_1d(cfg.max_len, d), persistent=False)
        self.dec_drop = nn.Dropout(cfg.dropout)
        self.decoder = nn.ModuleList(DecoderLayer(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and not name.startswith("backbone"):
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, std=self.cfg.d_model**-0.5)
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # Zero-init the last BN of each residual branch: blocks start as identity.
        for m in self.backbone.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    # -- encoder
    def encode(self, images: Tensor) -> Tensor:
        """(B, 1, H, W) ink intensity -> (B, H/16 * W/8, d) memory."""

        f = self.backbone(images)
        b, c, h, w = f.shape
        x = self.proj(f.flatten(2).transpose(1, 2))
        x = self.enc_drop(x + sinusoid_2d(h, w, self.cfg.d_model, x.device).to(x.dtype))
        for layer in self.encoder:
            x = layer(x)
        return self.enc_norm(x)

    def _memory_kv(self, memory: Tensor) -> list[tuple[Tensor, Tensor]]:
        return [layer.cross_attn.project_kv(memory) for layer in self.decoder]

    def _embed(self, tokens: Tensor, offset: int) -> Tensor:
        n = tokens.shape[1]
        x = self.embed(tokens) * math.sqrt(self.cfg.d_model)
        return x + self.tok_pos[offset : offset + n].to(x.dtype)

    def _logits(self, x: Tensor) -> Tensor:
        return F.linear(self.dec_norm(x), self.embed.weight)

    # -- training
    def forward(self, images: Tensor, tokens_in: Tensor) -> Tensor:
        """Teacher-forced logits (B, T, V) for ``tokens_in`` = BOS + target[:-1]."""

        mem_kv = self._memory_kv(self.encode(images))
        x = self.dec_drop(self._embed(tokens_in, 0))
        for layer, kv in zip(self.decoder, mem_kv):
            x = layer(x, kv)
        return self._logits(x)

    # -- inference
    @torch.no_grad()
    def generate(self, images: Tensor, bos_id: int, eos_id: int, max_len: int | None = None) -> Tensor:
        """Greedy decoding -> (B, <=max_len) token ids without BOS, EOS-terminated."""

        max_len = min(max_len or self.cfg.max_len, self.cfg.max_len) - 1
        mem_kv = self._memory_kv(self.encode(images))
        b = images.shape[0]
        caches: list[dict] = [{} for _ in self.decoder]
        token = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        out = []
        for step in range(max_len):
            x = self._embed(token, step)
            for layer, kv, cache in zip(self.decoder, mem_kv, caches):
                x = layer(x, kv, cache)
            token = self._logits(x)[:, -1].argmax(dim=-1, keepdim=True)
            token = token.masked_fill(done[:, None], eos_id)
            out.append(token)
            done |= token[:, 0] == eos_id
            if bool(done.all()):
                break
        return torch.cat(out, dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
