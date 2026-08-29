"""Load Errica et al. (ICLR 2020) fixed 10-fold TU splits from gnn-comparison."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from torch_geometric.graphgym.config import cfg

ErricaFeatureMode = Literal["chem", "social_constant", "social_degree"]

# PyG TUDataset name → (split subdir, split file stem, feature mode).
ERRICA_DATASET_REGISTRY: dict[str, tuple[str, str, ErricaFeatureMode]] = {
    "NCI1": ("CHEMICAL", "NCI1", "chem"),
    "DD": ("CHEMICAL", "DD", "chem"),
    "ENZYMES": ("CHEMICAL", "ENZYMES", "chem"),
    "PROTEINS": ("CHEMICAL", "PROTEINS_full", "chem"),
    "IMDB-BINARY": ("COLLABORATIVE_DEGREE", "IMDB-BINARY", "social_degree"),
    "REDDIT-BINARY": ("COLLABORATIVE_DEGREE", "REDDIT-BINARY", "social_degree"),
    "COLLAB": ("COLLABORATIVE_DEGREE", "COLLAB", "social_degree"),
    "IMDB-MULTI": ("COLLABORATIVE_DEGREE", "IMDB-MULTI", "social_degree"),
    "REDDIT-MULTI-5K": ("COLLABORATIVE_DEGREE", "REDDIT-MULTI-5K", "social_degree"),
}


@dataclass(frozen=True)
class ErricaFoldSplit:
    """Train/val/test graph indices for one Errica outer fold."""

    train: list[int]
    val: list[int]
    test: list[int]


def errica_splits_root() -> str:
    """Return directory containing vendored Errica split JSON files."""
    custom = getattr(cfg.dataset, "errica_split_dir", "")
    if custom:
        return str(custom)
    return os.path.join(cfg.dataset.split_dir, "errica")


def resolve_errica_split_path(dataset_name: str) -> str:
    """Resolve the JSON path for a PyG TU dataset name."""
    if dataset_name not in ERRICA_DATASET_REGISTRY:
        supported = ", ".join(sorted(ERRICA_DATASET_REGISTRY))
        raise ValueError(
            f"Dataset '{dataset_name}' has no Errica split mapping. "
            f"Supported: {supported}"
        )
    subdir, stem, _ = ERRICA_DATASET_REGISTRY[dataset_name]
    path = os.path.join(errica_splits_root(), subdir, f"{stem}_splits.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Errica split file not found: {path}. "
            "Run: python scripts/tu_errica/vendor_errica_splits.py"
        )
    return path


def errica_feature_mode(dataset_name: str) -> ErricaFeatureMode:
    """Return Errica node-feature protocol for a dataset."""
    override = getattr(cfg.dataset, "errica_feature_mode", "")
    if override:
        return override  # type: ignore[return-value]
    _, _, mode = ERRICA_DATASET_REGISTRY[dataset_name]
    return mode


def load_errica_fold(dataset_name: str, fold_index: int) -> ErricaFoldSplit:
    """Load train/val/test indices for one Errica outer fold (0–9)."""
    if fold_index < 0 or fold_index > 9:
        raise IndexError(f"Errica fold_index must be in [0, 9], got {fold_index}")

    path = resolve_errica_split_path(dataset_name)
    with open(path, encoding="utf-8") as handle:
        folds: list[dict[str, object]] = json.load(handle)

    if len(folds) != 10:
        raise ValueError(f"Expected 10 Errica folds in {path}, found {len(folds)}")

    fold = folds[fold_index]
    test_ids = list(fold["test"])  # type: ignore[arg-type]
    model_selection = fold["model_selection"]
    if not isinstance(model_selection, list) or not model_selection:
        raise ValueError(f"Missing model_selection in Errica fold {fold_index}")
    inner = model_selection[0]
    if not isinstance(inner, dict):
        raise ValueError(f"Unexpected model_selection entry in fold {fold_index}")

    train_ids = list(inner["train"])  # type: ignore[arg-type]
    val_ids = list(inner["validation"])  # type: ignore[arg-type]
    return ErricaFoldSplit(train=train_ids, val=val_ids, test=test_ids)
