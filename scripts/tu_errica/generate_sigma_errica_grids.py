#!/usr/bin/env python3
"""Build per-fold SiGMA grids for Errica hybrid search (Option 3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, rel_path: str) -> Any:
    import importlib.util

    path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hp = _load_module("errica_hp_grid", "scripts/tu_errica/errica_hp_grid.py")
_budget = _load_module("param_budget", "scripts/tu_errica/param_budget.py")

BIO_DS_TAGS = _hp.BIO_DS_TAGS
DS_TAG_TO_NAME = _hp.DS_TAG_TO_NAME
SOCIAL_DS_TAGS = _hp.SOCIAL_DS_TAGS
build_bio_sigma_micro_grid = _hp.build_bio_sigma_micro_grid
social_sigma_grid_entries = _hp.social_sigma_grid_entries
d_h_candidates_under_budget = _budget.d_h_candidates_under_budget
gin_param_count = _budget.gin_param_count
sigma_param_count = _budget.sigma_param_count

GRIDS_DIR = _REPO_ROOT / "configs/tu_errica/sigma_grids"
MANIFEST_PATH = GRIDS_DIR / "manifest.json"


def _load_gin_selection(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    selection = payload.get("selection", payload)
    if not isinstance(selection, dict):
        raise TypeError(f"Invalid selection file: {path}")
    return selection


def _annotate_grid(
    grid: list[dict[str, Any]],
    *,
    dataset_name: str,
    gin_params: int | None,
) -> list[dict[str, Any]]:
    """Attach param counts and budget metadata to each grid entry."""
    annotated: list[dict[str, Any]] = []
    for entry in grid:
        sigma_p = sigma_param_count(dataset_name, entry)
        row = dict(entry)
        row["sigma_params"] = sigma_p
        if gin_params is not None:
            row["gin_params_budget"] = gin_params
            row["under_budget"] = sigma_p <= gin_params
        annotated.append(row)
    return annotated


def build_manifest(
    gin_selection_path: Path,
    *,
    num_folds: int = 10,
) -> dict[str, Any]:
    """Create per-fold SiGMA grids and a flat task manifest for SLURM."""
    gin_sel = _load_gin_selection(gin_selection_path)
    tasks: list[dict[str, Any]] = []
    grid_files: dict[str, list[dict[str, Any]]] = {}

    for ds_tag, ds_name in DS_TAG_TO_NAME.items():
        fold_map = gin_sel.get(ds_tag, {})
        for fold in range(num_folds):
            fold_key = str(fold)
            if ds_tag in BIO_DS_TAGS:
                if fold_key not in fold_map:
                    print(f"[warn] missing GIN winner for {ds_tag} fold {fold}", file=sys.stderr)
                    continue
                gin_hp = fold_map[fold_key]["hp"]
                layers_mp = int(gin_hp["layers_mp"])
                dim_inner = int(gin_hp["dim_inner"])
                budget = gin_param_count(ds_name, gin_hp)
                d_h_vals = d_h_candidates_under_budget(
                    ds_name,
                    layers_mp=layers_mp,
                    dim_inner=dim_inner,
                    param_budget=budget,
                )
                grid = build_bio_sigma_micro_grid(
                    layers_mp=layers_mp,
                    dim_inner=dim_inner,
                    d_h_values=d_h_vals,
                )
                grid = _annotate_grid(grid, dataset_name=ds_name, gin_params=budget)
            elif ds_tag in SOCIAL_DS_TAGS:
                grid = social_sigma_grid_entries()
                grid = _annotate_grid(grid, dataset_name=ds_name, gin_params=None)
            else:
                raise ValueError(f"Unknown dataset tag: {ds_tag}")

            rel_name = f"{ds_tag}_f{fold}.json"
            grid_files[rel_name] = grid
            for hp_id in range(len(grid)):
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "ds_tag": ds_tag,
                        "dataset": ds_name,
                        "fold": fold,
                        "grid_file": rel_name,
                        "hp_id": hp_id,
                    }
                )

    return {
        "gin_selection": str(gin_selection_path),
        "num_tasks": len(tasks),
        "tasks": tasks,
        "grid_files": list(grid_files.keys()),
    }


def write_grids(manifest: dict[str, Any], gin_selection_path: Path, *, num_folds: int) -> None:
    """Write grid JSON files referenced by ``manifest``."""
    gin_sel = _load_gin_selection(gin_selection_path)
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    grids_sub = GRIDS_DIR / "grids"
    grids_sub.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()
    for task in manifest["tasks"]:
        rel_name = str(task["grid_file"])
        if rel_name in written:
            continue
        ds_tag = str(task["ds_tag"])
        fold = int(task["fold"])
        ds_name = DS_TAG_TO_NAME[ds_tag]
        fold_key = str(fold)

        if ds_tag in BIO_DS_TAGS:
            gin_hp = gin_sel[ds_tag][fold_key]["hp"]
            layers_mp = int(gin_hp["layers_mp"])
            dim_inner = int(gin_hp["dim_inner"])
            budget = gin_param_count(ds_name, gin_hp)
            d_h_vals = d_h_candidates_under_budget(
                ds_name,
                layers_mp=layers_mp,
                dim_inner=dim_inner,
                param_budget=budget,
            )
            grid = build_bio_sigma_micro_grid(
                layers_mp=layers_mp,
                dim_inner=dim_inner,
                d_h_values=d_h_vals,
            )
            grid = _annotate_grid(grid, dataset_name=ds_name, gin_params=budget)
        else:
            grid = social_sigma_grid_entries()
            grid = _annotate_grid(grid, dataset_name=ds_name, gin_params=None)

        out_path = grids_sub / rel_name
        out_path.write_text(json.dumps({"grid": grid}, indent=2), encoding="utf-8")
        written.add(rel_name)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gin-selection",
        type=Path,
        default=_REPO_ROOT / "configs/tu_errica/selections/gin_per_fold.json",
    )
    parser.add_argument("--num-folds", type=int, default=10)
    args = parser.parse_args()

    manifest = build_manifest(args.gin_selection, num_folds=args.num_folds)
    write_grids(manifest, args.gin_selection, num_folds=args.num_folds)
    print(f"Wrote {MANIFEST_PATH} with {manifest['num_tasks']} sigma_grid_select tasks")
    print(f"Grid files: {GRIDS_DIR / 'grids'}")


if __name__ == "__main__":
    main()
