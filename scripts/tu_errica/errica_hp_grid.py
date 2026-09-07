"""Errica TU hyperparameter grids from gnn-comparison YAML configs.

GIN and GraphSAGE grids mirror diningphil/gnn-comparison. GCN and GAT use
GIN-isomorphic grids (Errica does not publish separate GCN/GAT configs).
"""

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

# GIN-isomorphic grid (64 combos) — Errica has no published GCN recipe.
GCN_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.01],
    "layers_mp": [4],
    "dim_inner": [64, 32],
    "graph_pooling": ["add", "mean"],
    "dropout": [0.5, 0.0],
    "early_stop_use_loss": [False, True],
}

GCN_CANONICAL: dict[str, Any] = {
    "batch_size": 128,
    "base_lr": 0.01,
    "layers_mp": 4,
    "dim_inner": 64,
    "graph_pooling": "add",
    "dropout": 0.5,
    "early_stop_use_loss": False,
}

# GIN-isomorphic grid (64 combos) — Errica has no published GAT recipe.
GAT_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.01],
    "layers_mp": [4],
    "dim_inner": [64, 32],
    "graph_pooling": ["add", "mean"],
    "dropout": [0.5, 0.0],
    "early_stop_use_loss": [False, True],
}

GAT_CANONICAL: dict[str, Any] = {
    "batch_size": 128,
    "base_lr": 0.01,
    "layers_mp": 4,
    "dim_inner": 64,
    "graph_pooling": "add",
    "dropout": 0.5,
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

# GIN-isomorphic 64-config SiGMA grid for a protocol-matched search.
# Same axes as GIN except ``gin_train_eps`` → ``d_h ∈ {8, 16}``.
# Count: 2×1×1×2×2×2×2×2 = 64 (same as GIN; GCN/GAT are 32 without eps).
SIGMA_FULL64_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.01],
    "layers_mp": [4],
    "dim_inner": [64, 32],
    "d_h": [8, 16],
    "graph_pooling": ["add", "mean"],
    "dropout": [0.5, 0.0],
    "early_stop_use_loss": [False, True],
}

SIGMA_CANONICAL: dict[str, Any] = {
    "batch_size": 128,
    "base_lr": 0.001,
    "layers_mp": 12,
    "dim_inner": 64,
    "d_h": 16,
    "early_stop_use_loss": False,
}

# Anchor-boost SiGMA grid for PROTEINS / REDDIT-BINARY under Errica CV.
# Includes the winning paper-table a2g4 recipe (random-split table):
#   L=12, H=64, d_h=16, lr=1e-3, bs=64 (PROTEINS) / bs=16 (REDDIT),
# plus nearby LR / depth / d_h / batch variants. Count: 2×2×2×1×3 = 24.
# Fixed8 missed bs=64 (only 32/128), which is the paper PROTEINS batch.
SIGMA_ANCHOR_BOOST_GRID: dict[str, list[Any]] = {
    "batch_size": [16, 64],
    "base_lr": [0.001, 0.01],
    "layers_mp": [8, 12],
    "dim_inner": [64],
    "d_h": [8, 16, 32],
    "early_stop_use_loss": [False],
}

# Tiny SiGMA a1g2 select grid (Errica has no published SiGMA recipe — small
# custom grids are protocol-OK). Paper depth/width fixed; only search batch × LR
# (the axes that differ between PROTEINS and REDDIT paper recipes).
# Count: 2×2×1×1×1 = 4 → 4 × 2 datasets × 10 folds = 80 select tasks.
SIGMA_A1G2_MICRO_GRID: dict[str, list[Any]] = {
    "batch_size": [16, 64],
    "base_lr": [0.001, 0.01],
    "layers_mp": [12],
    "dim_inner": [64],
    "d_h": [16],
    "early_stop_use_loss": [False],
}

# NCI1 a1g2 micro: same tiny search, bio-style batches (fixed8 axes).
# Count: 4 → 4 × 1 dataset × 10 folds = 40 select tasks.
SIGMA_A1G2_NCI1_MICRO_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.001, 0.01],
    "layers_mp": [12],
    "dim_inner": [64],
    "d_h": [16],
    "early_stop_use_loss": [False],
}

# Ultra-tiny PROTEINS refine around modal ``anchor_boost`` center
# (bs=16, lr=1e-3, L=12, d_h=8). Only the GCN-like regularization corner:
# dropout=0.5 × pool∈{add,mean}. Skips drop=0 (near yaml default 0.1 /
# prior boost runs). Count: 2 → 20 select / 30 eval.
SIGMA_ANCHOR_REFINE_GRID: dict[str, list[Any]] = {
    "batch_size": [16],
    "base_lr": [0.001],
    "layers_mp": [12],
    "dim_inner": [64],
    "d_h": [8],
    "dropout": [0.5],
    "graph_pooling": ["add", "mean"],
    "early_stop_use_loss": [False],
}

# Ultra-tiny NCI1 refine around best SiGMA family so far (fixed8 / a2g4 deep):
# bs=32, lr=1e-3, L=12, d_h=16 (fixed8 SIGMA_GRID deep end; a1g2 used same
# L/d_h). Only GCN-like corner: dropout=0.5 × pool∈{add,mean}.
# Count: 2 → 20 select / 30 eval. Goal: beat GraphSAGE 81.6.
SIGMA_NCI1_REFINE_GRID: dict[str, list[Any]] = {
    "batch_size": [32],
    "base_lr": [0.001],
    "layers_mp": [12],
    "dim_inner": [64],
    "d_h": [16],
    "dropout": [0.5],
    "graph_pooling": ["add", "mean"],
    "early_stop_use_loss": [False],
}

# ---------------------------------------------------------------------------
# native_fair — compact SiGMA fair search on PROTEINS / NCI1 / REDDIT.
#
# Why not full64? full64 locked GIN's L=4 + lr=0.01 and never varied MP heads.
# Launch partition: gpu_h200 (see submit_tu_errica_native_fair_select.sh).
#
# Compact prayer grid (no a1g4 / a0g4 full mixes):
#   • a1g2_gin_sage / a1g2_gcn_gin — best 2-MP specialists + 1 attn
#   • a0g2_gin_sage / a0g2_gcn_gin — same MP mixes, drop global attn
# Train sweep (bs=32, d_h=16, H=64 fixed): lr × layers_mp (MLP depth) only.
# Count: 4 × 2 × 2 = 16 → 16 × 3 × 10 = 480 select / 90 eval.
# ---------------------------------------------------------------------------
SIGMA_NATIVE_FAIR_MP_FAMILIES: list[dict[str, Any]] = [
    {
        "mp_family": "a1g2_gin_sage",
        "num_attn_heads": 1,
        "num_gnn_heads": 2,
        "gnn_types": "GIN,SAGE",
    },
    {
        "mp_family": "a1g2_gcn_gin",
        "num_attn_heads": 1,
        "num_gnn_heads": 2,
        "gnn_types": "GCN,GIN",
    },
    {
        "mp_family": "a0g2_gin_sage",
        "num_attn_heads": 0,
        "num_gnn_heads": 2,
        "gnn_types": "GIN,SAGE",
    },
    {
        "mp_family": "a0g2_gcn_gin",
        "num_attn_heads": 0,
        "num_gnn_heads": 2,
        "gnn_types": "GCN,GIN",
    },
]

SIGMA_NATIVE_FAIR_TRAIN_GRID: dict[str, list[Any]] = {
    "batch_size": [32],
    "base_lr": [0.001, 0.01],
    "layers_mp": [4, 12],
    "dim_inner": [64],
    "d_h": [16],
    "dropout": [0.5],
    "graph_pooling": ["add"],
    "early_stop_use_loss": [False],
}

# ---------------------------------------------------------------------------
# a0g_pnr — MP-only (drop global attention) on PROTEINS / NCI1 / REDDIT.
#
# Same specialist MP mixes as native_fair (a0g*), wider train (bs×d_h).
# Count: 3 × 16 = 48 → 48 × 3 × 10 = 1,440 select.
# ---------------------------------------------------------------------------
SIGMA_A0G_PNR_MP_FAMILIES: list[dict[str, Any]] = [
    {
        "mp_family": "a0g4_full",
        "num_attn_heads": 0,
        "num_gnn_heads": 4,
        "gnn_types": "GCN,GIN,SAGE,GAT",
    },
    {
        "mp_family": "a0g2_gin_sage",
        "num_attn_heads": 0,
        "num_gnn_heads": 2,
        "gnn_types": "GIN,SAGE",
    },
    {
        "mp_family": "a0g2_gcn_gin",
        "num_attn_heads": 0,
        "num_gnn_heads": 2,
        "gnn_types": "GCN,GIN",
    },
]

# Wider train than native_fair (kept independent — native_fair dropped bs/d_h).
SIGMA_A0G_PNR_TRAIN_GRID: dict[str, list[Any]] = {
    "batch_size": [32, 128],
    "base_lr": [0.001, 0.01],
    "layers_mp": [4, 12],
    "dim_inner": [64],
    "d_h": [8, 16],
    "dropout": [0.5],
    "graph_pooling": ["add"],
    "early_stop_use_loss": [False],
}

# ---------------------------------------------------------------------------
# tiny_pnr — ultra-tiny select on PROTEINS / NCI1 / REDDIT with *sensible*
# SiGMA HPs (contrast full64: lr=0.01 + L=4).
#
# Arch: a2g4 gated (yaml). Only search bs×d_h around the known deep recipe.
# Count: 4 → 4 × 3 × 10 = 120 select / 90 eval. Fast path to a competitive eval.
# ---------------------------------------------------------------------------
SIGMA_TINY_PNR_GRID: dict[str, list[Any]] = {
    "batch_size": [16, 32],
    "base_lr": [0.001],
    "layers_mp": [12],
    "dim_inner": [64],
    "d_h": [8, 16],
    "dropout": [0.5],
    "graph_pooling": ["add"],
    "early_stop_use_loss": [False],
}

# Dataset families for hybrid SiGMA search (Option 3).
BIO_DS_TAGS: frozenset[str] = frozenset({"enzymes", "proteins", "nci1", "dd"})
SOCIAL_DS_TAGS: frozenset[str] = frozenset({"imdb-b", "reddit-b", "collab"})
# Errica datasets to push with the paper a2g4 anchor recipe.
ANCHOR_BOOST_DS_TAGS: frozenset[str] = frozenset({"proteins", "reddit-b"})
# Same datasets for a1g2 micro select.
A1G2_MICRO_DS_TAGS: frozenset[str] = frozenset({"proteins", "reddit-b"})
A1G2_NCI1_MICRO_DS_TAGS: frozenset[str] = frozenset({"nci1"})
# Local refine around PROTEINS anchor_boost mode (a2g4 + dropout/pool).
ANCHOR_REFINE_DS_TAGS: frozenset[str] = frozenset({"proteins"})
# Local refine around NCI1 fixed8/a2g4 deep center + dropout/pool.
NCI1_REFINE_DS_TAGS: frozenset[str] = frozenset({"nci1"})
# native_fair / a0g_pnr / tiny_pnr: PROTEINS + NCI1 + REDDIT.
NATIVE_FAIR_DS_TAGS: frozenset[str] = frozenset({"proteins", "nci1", "reddit-b"})
NATIVE_FAIR_DS_ORDER: tuple[str, ...] = ("proteins", "nci1", "reddit-b")
A0G_PNR_DS_TAGS: frozenset[str] = NATIVE_FAIR_DS_TAGS
A0G_PNR_DS_ORDER: tuple[str, ...] = NATIVE_FAIR_DS_ORDER
TINY_PNR_DS_TAGS: frozenset[str] = NATIVE_FAIR_DS_TAGS
TINY_PNR_DS_ORDER: tuple[str, ...] = NATIVE_FAIR_DS_ORDER

DS_TAG_TO_NAME: dict[str, str] = {
    "enzymes": "ENZYMES",
    "proteins": "PROTEINS",
    "nci1": "NCI1",
    "dd": "DD",
    "imdb-b": "IMDB-BINARY",
    "reddit-b": "REDDIT-BINARY",
    "collab": "COLLAB",
}

MODEL_TAG_BY_KEY: dict[str, str] = {
    "gin": "GIN",
    "graphsage": "GraphSAGE",
    "gcn": "GCN",
    "gat": "GAT",
    "sigma_hetero": "SiGMA_hetero",
}


def build_bio_sigma_micro_grid(
    *,
    layers_mp: int,
    dim_inner: int,
    d_h_values: list[int],
) -> list[dict[str, Any]]:
    """Small SiGMA grid at GIN-matched L/H (bio datasets, Option 3)."""
    grid: list[dict[str, Any]] = []
    for batch_size in (32, 128):
        for base_lr in (0.001, 0.01):
            for d_h in d_h_values:
                grid.append(
                    {
                        "batch_size": batch_size,
                        "base_lr": base_lr,
                        "layers_mp": layers_mp,
                        "dim_inner": dim_inner,
                        "d_h": d_h,
                        "early_stop_use_loss": False,
                    }
                )
    return grid


def social_sigma_grid_entries() -> list[dict[str, Any]]:
    """Fixed 8-config SiGMA grid (used for all datasets under ``fixed8`` mode)."""
    return expand_grid(SIGMA_GRID)


def full64_sigma_grid_entries() -> list[dict[str, Any]]:
    """GIN-isomorphic 64-config SiGMA grid (``full64`` mode)."""
    return expand_grid(SIGMA_FULL64_GRID)


def anchor_boost_sigma_grid_entries() -> list[dict[str, Any]]:
    """Paper a2g4-centered SiGMA grid (``anchor_boost`` mode)."""
    return expand_grid(SIGMA_ANCHOR_BOOST_GRID)


def a1g2_micro_sigma_grid_entries() -> list[dict[str, Any]]:
    """Tiny a1g2 SiGMA grid (batch × LR only; ``a1g2_micro`` mode)."""
    return expand_grid(SIGMA_A1G2_MICRO_GRID)


def a1g2_nci1_micro_sigma_grid_entries() -> list[dict[str, Any]]:
    """Tiny a1g2 NCI1 SiGMA grid (``a1g2_nci1_micro`` mode)."""
    return expand_grid(SIGMA_A1G2_NCI1_MICRO_GRID)


def anchor_refine_sigma_grid_entries() -> list[dict[str, Any]]:
    """Tiny PROTEINS refine grid around anchor_boost mode (``anchor_refine``)."""
    return expand_grid(SIGMA_ANCHOR_REFINE_GRID)


def nci1_refine_sigma_grid_entries() -> list[dict[str, Any]]:
    """Tiny NCI1 refine grid around fixed8/a2g4 deep center (``nci1_refine``)."""
    return expand_grid(SIGMA_NCI1_REFINE_GRID)


def native_fair_sigma_grid_entries() -> list[dict[str, Any]]:
    """Compact SiGMA fair grid: a0g2/a1g2 specialists × lr × L (``native_fair``).

    Returns
    -------
    list[dict[str, Any]]
        Flattened configs (16 by default). Each entry includes ``mp_family``
        metadata plus ``num_attn_heads`` / ``num_gnn_heads`` / ``gnn_types``
        for ``emit_cfg_overrides``.
    """
    train = expand_grid(SIGMA_NATIVE_FAIR_TRAIN_GRID)
    combos: list[dict[str, Any]] = []
    for family in SIGMA_NATIVE_FAIR_MP_FAMILIES:
        for train_hp in train:
            row = dict(train_hp)
            row.update(family)
            combos.append(row)
    return combos


def a0g_pnr_sigma_grid_entries() -> list[dict[str, Any]]:
    """MP-only (a0g*) fair grid on PROTEINS/NCI1/REDDIT (``a0g_pnr``).

    Returns
    -------
    list[dict[str, Any]]
        Flattened configs (48 by default): a0g4 full + a0g2 specialists ×
        the same train axes as ``native_fair``.
    """
    train = expand_grid(SIGMA_A0G_PNR_TRAIN_GRID)
    combos: list[dict[str, Any]] = []
    for family in SIGMA_A0G_PNR_MP_FAMILIES:
        for train_hp in train:
            row = dict(train_hp)
            row.update(family)
            combos.append(row)
    return combos


def tiny_pnr_sigma_grid_entries() -> list[dict[str, Any]]:
    """Ultra-tiny sensible SiGMA grid on PROTEINS/NCI1/REDDIT (``tiny_pnr``).

    Returns
    -------
    list[dict[str, Any]]
        4 configs: bs∈{16,32} × d_h∈{8,16} at lr=1e-3, L=12 (a2g4 yaml).
    """
    return expand_grid(SIGMA_TINY_PNR_GRID)


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
    elif model == "gcn":
        canonical, grid = GCN_CANONICAL, expand_grid(GCN_GRID)
    elif model == "gat":
        canonical, grid = GAT_CANONICAL, expand_grid(GAT_GRID)
    elif model == "sigma_hetero":
        canonical, grid = SIGMA_CANONICAL, expand_grid(SIGMA_GRID)
    elif model == "sigma_hetero_full64":
        canonical, grid = SIGMA_CANONICAL, expand_grid(SIGMA_FULL64_GRID)
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
    for family in ("gin", "graphsage", "gcn", "gat", "sigma_hetero", "sigma_hetero_full64"):
        write_grid_json(family, root)
