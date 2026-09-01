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

_SKIP_HP_KEYS = frozenset({"sigma_params", "gin_params_budget", "under_budget"})


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def emit_overrides_from_hp(hp: dict[str, Any]) -> list[str]:
    """Return flat CLI args from an HP dict."""
    args: list[str] = []
    for key, value in hp.items():
        if key in _SKIP_HP_KEYS:
            continue
        cfg_key = _KEY_MAP.get(key, key)
        args.extend([cfg_key, _format_value(value)])
    return args


def emit_overrides(model: str, hp_id: int, *, canonical: bool) -> list[str]:
    """Return flat key/value list suitable for ``python main.py ...``."""
    cfg = load_hp_config(model, hp_id, canonical_only=canonical)
    return emit_overrides_from_hp(cfg)


def load_selection_hp(selection_file: Path, ds_tag: str, fold: int) -> dict[str, Any]:
    """Load winning HP for grid_eval from a per-fold selection JSON."""
    with selection_file.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    selection = payload.get("selection", payload)
    entry = selection[ds_tag][str(fold)]
    hp = entry["hp"]
    if not isinstance(hp, dict):
        raise TypeError(f"Invalid hp entry for {ds_tag} fold {fold}")
    return hp


def load_sigma_grid_hp(grid_file: Path, hp_id: int) -> dict[str, Any]:
    """Load one SiGMA grid entry from a per-fold grid JSON."""
    with grid_file.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    grid = payload["grid"]
    return dict(grid[hp_id])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gin", "graphsage", "gcn", "gat", "sigma_hetero"])
    parser.add_argument("--hp-id", type=int, default=-1)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--ds-tag", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--sigma-grid-file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of CLI args")
    args = parser.parse_args()

    if args.selection_file is not None:
        if args.ds_tag is None or args.fold is None:
            parser.error("--selection-file requires --ds-tag and --fold")
        overrides = load_selection_hp(args.selection_file, args.ds_tag, args.fold)
    elif args.sigma_grid_file is not None:
        if args.hp_id < 0:
            parser.error("--sigma-grid-file requires --hp-id")
        overrides = load_sigma_grid_hp(args.sigma_grid_file, args.hp_id)
    else:
        if args.model is None:
            parser.error("--model is required without --selection-file / --sigma-grid-file")
        overrides = load_hp_config(args.model, args.hp_id, canonical_only=args.canonical)

    if args.json:
        json.dump(overrides, sys.stdout)
        return
    flat = emit_overrides_from_hp(overrides)
    sys.stdout.write(" ".join(flat))


if __name__ == "__main__":
    main()
