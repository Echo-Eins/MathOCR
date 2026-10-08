"""YAML experiment config: one file fixes data, image, model and training."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load ``path`` and apply ``section.key=value`` overrides (YAML values)."""

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides or []:
        dotted, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must be key=value: {item}")
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key: {dotted}")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return copy.deepcopy(cfg[name])
