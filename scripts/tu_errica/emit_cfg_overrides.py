#!/usr/bin/env python3
"""Emit main.py CLI overrides for one Errica HP grid entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_hp_module() -> object:
    """Load ``errica_hp_grid.py`` without requiring ``scripts`` as a package."""
    import importlib.util

    path = _REPO_ROOT / "scripts" / "tu_errica" / "errica_hp_grid.py"
    spec = importlib.util.spec_from_file_location("errica_hp_grid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load errica_hp_grid from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hp_module = _load_hp_module()
load_hp_config = _hp_module.load_hp_config

# Map grid keys → GraphGym cfg keys.
_KEY_MAP: dict[str, str] = {
    "batch_size": "train.batch_size",
    "base_lr": "optim.base_lr",
    "layers_mp": "gnn.layers_mp",
    "dim_inner": "gnn.dim_inner",
    "gin_train_eps": "gnn.gin_train_eps",
    "graph_pooling": "model.graph_pooling",
    "dropout": "gnn.dropout",
    "early_stop_use_loss": "train.early_stop_use_loss",
    "d_h": "gnn.hybrid.d_h",
}


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def emit_overrides(model: str, hp_id: int, *, canonical: bool) -> list[str]:
    """Return flat key/value list suitable for ``python main.py ...``."""
    cfg = load_hp_config(model, hp_id, canonical_only=canonical)
    args: list[str] = []
    for key, value in cfg.items():
        cfg_key = _KEY_MAP.get(key, key)
        args.extend([cfg_key, _format_value(value)])
    return args


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["gin", "graphsage", "sigma_hetero"])
    parser.add_argument("--hp-id", type=int, default=-1)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of CLI args")
    args = parser.parse_args()
    overrides = load_hp_config(args.model, args.hp_id, canonical_only=args.canonical)
    if args.json:
        json.dump(overrides, sys.stdout)
        return
    flat = emit_overrides(args.model, args.hp_id, canonical=args.canonical)
    sys.stdout.write(" ".join(flat))


if __name__ == "__main__":
    main()
