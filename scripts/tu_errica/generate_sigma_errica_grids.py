#!/usr/bin/env python3
"""Build per-fold SiGMA grids for Errica hybrid search.

Default (``--mode fixed8``): every dataset/fold uses the fixed 8-config
``SIGMA_GRID`` (no GIN/GCN parameter ceiling).

Legacy (``--mode budget_bio``): bio folds lock depth/width to the GIN winner
and keep SiGMA params ≤ that GIN budget; social folds use ``SIGMA_GRID``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]

SigmaGridMode = Literal["fixed8", "budget_bio"]


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

BIO_DS_TAGS = _hp.BIO_DS_TAGS
DS_TAG_TO_NAME = _hp.DS_TAG_TO_NAME
SOCIAL_DS_TAGS = _hp.SOCIAL_DS_TAGS
build_bio_sigma_micro_grid = _hp.build_bio_sigma_micro_grid
social_sigma_grid_entries = _hp.social_sigma_grid_entries

GRIDS_DIR = _REPO_ROOT / "configs/tu_errica/sigma_grids"
MANIFEST_PATH = GRIDS_DIR / "manifest.json"


def _budget_module() -> Any:
    """Lazy-load param_budget (needs yacs) only for budget_bio mode."""
    return _load_module("param_budget", "scripts/tu_errica/param_budget.py")


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
    with_params: bool,
) -> list[dict[str, Any]]:
    """Optionally attach param counts / budget metadata to each grid entry."""
    if not with_params:
        return [dict(entry) for entry in grid]
    budget = _budget_module()
    annotated: list[dict[str, Any]] = []
    for entry in grid:
        sigma_p = budget.sigma_param_count(dataset_name, entry)
        row = dict(entry)
        row["sigma_params"] = sigma_p
        if gin_params is not None:
            row["gin_params_budget"] = gin_params
            row["under_budget"] = sigma_p <= gin_params
        annotated.append(row)
    return annotated


def _grid_for_fold(
    *,
    mode: SigmaGridMode,
    ds_tag: str,
    ds_name: str,
    fold: int,
    gin_sel: dict[str, dict[str, dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Return the SiGMA HP grid for one (dataset, fold)."""
    if mode == "fixed8":
        return _annotate_grid(
            social_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )

    # Legacy Option-3 budgeted bio search.
    budget = _budget_module()
    fold_key = str(fold)
    if ds_tag in BIO_DS_TAGS:
        if gin_sel is None:
            raise ValueError("budget_bio mode requires a GIN selection file")
        fold_map = gin_sel.get(ds_tag, {})
        if fold_key not in fold_map:
            raise KeyError(f"missing GIN winner for {ds_tag} fold {fold}")
        gin_hp = fold_map[fold_key]["hp"]
        layers_mp = int(gin_hp["layers_mp"])
        dim_inner = int(gin_hp["dim_inner"])
        gin_params = budget.gin_param_count(ds_name, gin_hp)
        d_h_vals = budget.d_h_candidates_under_budget(
            ds_name,
            layers_mp=layers_mp,
            dim_inner=dim_inner,
            param_budget=gin_params,
        )
        grid = build_bio_sigma_micro_grid(
            layers_mp=layers_mp,
            dim_inner=dim_inner,
            d_h_values=d_h_vals,
        )
        return _annotate_grid(
            grid,
            dataset_name=ds_name,
            gin_params=gin_params,
            with_params=True,
        )
    if ds_tag in SOCIAL_DS_TAGS:
        return _annotate_grid(
            social_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=True,
        )
    raise ValueError(f"Unknown dataset tag: {ds_tag}")


def build_manifest(
    gin_selection_path: Path | None,
    *,
    num_folds: int = 10,
    mode: SigmaGridMode = "fixed8",
) -> dict[str, Any]:
    """Create per-fold SiGMA grids and a flat task manifest for SLURM."""
    gin_sel: dict[str, dict[str, dict[str, Any]]] | None = None
    if mode == "budget_bio":
        if gin_selection_path is None or not gin_selection_path.is_file():
            raise FileNotFoundError(
                "budget_bio mode requires --gin-selection (gin_per_fold.json)"
            )
        gin_sel = _load_gin_selection(gin_selection_path)

    tasks: list[dict[str, Any]] = []
    grid_files: dict[str, list[dict[str, Any]]] = {}

    for ds_tag, ds_name in DS_TAG_TO_NAME.items():
        for fold in range(num_folds):
            try:
                grid = _grid_for_fold(
                    mode=mode,
                    ds_tag=ds_tag,
                    ds_name=ds_name,
                    fold=fold,
                    gin_sel=gin_sel,
                )
            except KeyError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                continue

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
        "mode": mode,
        "gin_selection": str(gin_selection_path) if gin_selection_path else None,
        "num_tasks": len(tasks),
        "tasks": tasks,
        "grid_files": list(grid_files.keys()),
    }


def write_grids(
    manifest: dict[str, Any],
    gin_selection_path: Path | None,
    *,
    num_folds: int,
    mode: SigmaGridMode,
) -> None:
    """Write grid JSON files referenced by ``manifest``."""
    gin_sel: dict[str, dict[str, dict[str, Any]]] | None = None
    if mode == "budget_bio":
        assert gin_selection_path is not None
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
        grid = _grid_for_fold(
            mode=mode,
            ds_tag=ds_tag,
            ds_name=ds_name,
            fold=fold,
            gin_sel=gin_sel,
        )
        out_path = grids_sub / rel_name
        out_path.write_text(json.dumps({"grid": grid}, indent=2), encoding="utf-8")
        written.add(rel_name)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fixed8", "budget_bio"),
        default="fixed8",
        help="fixed8: SIGMA_GRID on all folds (default). "
        "budget_bio: legacy GIN-budgeted bio micro-grid.",
    )
    parser.add_argument(
        "--gin-selection",
        type=Path,
        default=_REPO_ROOT / "configs/tu_errica/selections/gin_per_fold.json",
        help="Required for --mode budget_bio; ignored for fixed8.",
    )
    parser.add_argument("--num-folds", type=int, default=10)
    args = parser.parse_args()
    mode: SigmaGridMode = args.mode  # type: ignore[assignment]

    gin_path: Path | None = args.gin_selection if mode == "budget_bio" else None
    manifest = build_manifest(gin_path, num_folds=args.num_folds, mode=mode)
    write_grids(
        manifest,
        gin_path,
        num_folds=args.num_folds,
        mode=mode,
    )
    print(
        f"Wrote {MANIFEST_PATH} with {manifest['num_tasks']} "
        f"sigma_grid_select tasks (mode={mode})"
    )
    print(f"Grid files: {GRIDS_DIR / 'grids'}")


if __name__ == "__main__":
    main()
