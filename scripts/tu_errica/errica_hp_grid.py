"""Errica GIN / GraphSAGE hyperparameter grids from gnn-comparison YAML configs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

# Mirrors config_GIN.yml from diningphil/gnn-comparison (ICLR 2020).
GIN_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.01],
    "layers_mp": [4],  # hidden_units variants mapped to uniform dim_inner
    "dim_inner": [64, 32],
    "gin_train_eps": [True, False],
    "graph_pooling": ["add", "mean"],
    "dropout": [0.5, 0.0],
    "early_stop_use_loss": [False, True],
}

# Canonical single config (common winner; use for smoke / fast repro).
GIN_CANONICAL: dict[str, Any] = {
    "batch_size": 128,
    "base_lr": 0.01,
    "layers_mp": 4,
    "dim_inner": 64,
    "gin_train_eps": True,
    "graph_pooling": "add",
    "dropout": 0.5,
    "early_stop_use_loss": False,
}

SAGE_GRID: dict[str, list[Any]] = {
    "batch_size": [32],
    "base_lr": [0.0001, 0.001, 0.01],
    "layers_mp": [3, 5],
    "dim_inner": [32, 64],
    "graph_pooling": ["add", "max", "mean"],
    "dropout": [0.0],
    "early_stop_use_loss": [False, True],
}

SAGE_CANONICAL: dict[str, Any] = {
    "batch_size": 32,
    "base_lr": 0.01,
    "layers_mp": 3,
    "dim_inner": 64,
    "graph_pooling": "mean",
    "dropout": 0.0,
    "early_stop_use_loss": False,
}

# Reduced SiGMA grid (Errica has no published SiGMA recipe).
SIGMA_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.001, 0.01],
    "layers_mp": [4, 12],
    "dim_inner": [64],
    "d_h": [16],
    "early_stop_use_loss": [False],
}

SIGMA_CANONICAL: dict[str, Any] = {
    "batch_size": 128,
    "base_lr": 0.001,
    "layers_mp": 12,
    "dim_inner": 64,
    "d_h": 16,
    "early_stop_use_loss": False,
}


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a hyperparameter grid."""
    keys = list(grid.keys())
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def write_grid_json(model: str, out_dir: Path) -> Path:
    """Write canonical + full grid JSON for a model family."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if model == "gin":
        canonical, grid = GIN_CANONICAL, expand_grid(GIN_GRID)
    elif model == "graphsage":
        canonical, grid = SAGE_CANONICAL, expand_grid(SAGE_GRID)
    elif model == "sigma_hetero":
        canonical, grid = SIGMA_CANONICAL, expand_grid(SIGMA_GRID)
    else:
        raise ValueError(f"Unknown model: {model}")

    payload = {"canonical": canonical, "grid": grid}
    path = out_dir / f"{model}_hp_grid.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {len(grid)} configs + canonical → {path}")
    return path


def load_hp_config(model: str, hp_id: int, *, canonical_only: bool = False) -> dict[str, Any]:
    """Load HP config by index from vendored JSON grid file."""
    path = Path(__file__).resolve().parents[2] / "configs" / "tu_errica" / f"{model}_hp_grid.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if canonical_only or hp_id < 0:
        cfg = payload["canonical"]
        if not isinstance(cfg, dict):
            raise TypeError("canonical entry must be a dict")
        return cfg
    grid = payload["grid"]
    if hp_id >= len(grid):
        raise IndexError(f"hp_id={hp_id} out of range (grid size {len(grid)})")
    return grid[hp_id]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "configs" / "tu_errica"
    for family in ("gin", "graphsage", "sigma_hetero"):
        write_grid_json(family, root)
