"""Parameter counting and d_h budget matching for Errica SiGMA grids."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from yacs.config import CfgNode

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401

# TU dataset → (dummy dim_in, num classes) for param counting.
TU_DATASET_DIMS: dict[str, tuple[int, int]] = {
    "ENZYMES": (3, 6),
    "PROTEINS": (3, 2),
    "NCI1": (37, 2),
    "DD": (89, 2),
    "IMDB-BINARY": (1, 2),
    "REDDIT-BINARY": (1, 2),
    "COLLAB": (1, 3),
}

SIGMA_ERRICA_CFG = _REPO_ROOT / "configs/tu_errica/sigma-hetero-errica-base.yaml"
GIN_ERRICA_CFG = _REPO_ROOT / "configs/tu_errica/gin-errica-base.yaml"


def _dummy_batch(n: int, dim_in: int) -> Batch:
    """Build a tiny batched graph for a dry forward."""
    x = torch.randn(n, dim_in)
    src = torch.arange(0, n - 1)
    dst = torch.arange(1, n)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([0]))
    return Batch.from_data_list([data])


def _apply_overrides(overrides: Mapping[str, Any]) -> None:
    """Apply flat GraphGym-style overrides to global ``cfg``."""
    for key, value in overrides.items():
        parts = key.split(".")
        node: Any = cfg
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)


def count_trainable_params(
    cfg_path: Path,
    overrides: Mapping[str, Any],
    *,
    dim_in: int,
    dim_out: int,
) -> int:
    """Count trainable parameters for a YAML config plus CLI-style overrides."""
    set_cfg(cfg)
    opt = CfgNode({"cfg_file": str(cfg_path), "opts": []})
    load_cfg(cfg, opt)
    cfg.dataset.node_encoder = True
    cfg.accelerator = "cpu"
    _apply_overrides(overrides)
    model = create_model(dim_in=dim_in, dim_out=dim_out)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hp_to_gin_overrides(hp: Mapping[str, Any]) -> dict[str, Any]:
    """Map Errica HP dict keys to GraphGym override keys for GIN."""
    key_map = {
        "batch_size": "train.batch_size",
        "base_lr": "optim.base_lr",
        "layers_mp": "gnn.layers_mp",
        "dim_inner": "gnn.dim_inner",
        "gin_train_eps": "gnn.gin_train_eps",
        "graph_pooling": "model.graph_pooling",
        "dropout": "gnn.dropout",
        "early_stop_use_loss": "train.early_stop_use_loss",
    }
    out: dict[str, Any] = {}
    for key, value in hp.items():
        mapped = key_map.get(key, key)
        out[mapped] = value
    return out


def hp_to_sigma_overrides(hp: Mapping[str, Any]) -> dict[str, Any]:
    """Map Errica HP dict keys to GraphGym override keys for SiGMA hetero."""
    key_map = {
        "batch_size": "train.batch_size",
        "base_lr": "optim.base_lr",
        "layers_mp": "gnn.layers_mp",
        "dim_inner": "gnn.dim_inner",
        "d_h": "gnn.hybrid.d_h",
        "early_stop_use_loss": "train.early_stop_use_loss",
    }
    out: dict[str, Any] = {}
    for key, value in hp.items():
        mapped = key_map.get(key, key)
        out[mapped] = value
    return out


def gin_param_count(dataset_name: str, hp: Mapping[str, Any]) -> int:
    """Parameter count for GIN at a given Errica HP entry."""
    dim_in, dim_out = TU_DATASET_DIMS[dataset_name]
    overrides = hp_to_gin_overrides(hp)
    return count_trainable_params(GIN_ERRICA_CFG, overrides, dim_in=dim_in, dim_out=dim_out)


def sigma_param_count(dataset_name: str, hp: Mapping[str, Any]) -> int:
    """Parameter count for SiGMA hetero at a given HP entry."""
    dim_in, dim_out = TU_DATASET_DIMS[dataset_name]
    overrides = hp_to_sigma_overrides(hp)
    return count_trainable_params(SIGMA_ERRICA_CFG, overrides, dim_in=dim_in, dim_out=dim_out)


def d_h_candidates_under_budget(
    dataset_name: str,
    *,
    layers_mp: int,
    dim_inner: int,
    param_budget: int,
    candidates: Sequence[int] = (4, 8, 16, 32, 64),
) -> list[int]:
    """Return ``d_h`` values with SiGMA params <= ``param_budget`` at fixed L/H."""
    valid: list[int] = []
    for d_h in candidates:
        hp = {
            "layers_mp": layers_mp,
            "dim_inner": dim_inner,
            "d_h": d_h,
            "batch_size": 32,
            "base_lr": 0.001,
            "early_stop_use_loss": False,
        }
        if sigma_param_count(dataset_name, hp) <= param_budget:
            valid.append(d_h)
    return valid or [4]
