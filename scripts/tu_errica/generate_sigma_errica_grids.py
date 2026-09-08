"""Build per-fold SiGMA grids for Errica hybrid search.

Default (``--mode fixed8``): every dataset/fold uses the fixed 8-config
``SIGMA_GRID`` (no GIN/GCN parameter ceiling). Writes
``configs/tu_errica/sigma_grids/``.

``--mode full64``: GIN-isomorphic 64-config grid (same search budget as GIN).
Writes ``configs/tu_errica/sigma_grids_full64/`` so it does not clobber fixed8.

``--mode anchor_boost``: paper a2g4-centered 24-config grid on PROTEINS +
REDDIT-BINARY only. Writes ``configs/tu_errica/sigma_grids_anchor_boost/``.

``--mode a1g2_micro``: tiny 4-config grid (bs×lr at L12/d_h16) for SiGMA a1g2
(GCN+GIN) on PROTEINS + REDDIT-BINARY. Writes
``configs/tu_errica/sigma_grids_a1g2_micro/``.

``--mode a1g2_nci1_micro``: tiny 4-config grid for SiGMA a1g2 (GIN+SAGE) on
NCI1 only. Writes ``configs/tu_errica/sigma_grids_a1g2_nci1_micro/``.

``--mode anchor_refine``: tiny 4-config refine around PROTEINS ``anchor_boost``
mode winner (bs=16, lr=1e-3, L=12, d_h=8 × dropout × pooling). Writes
``configs/tu_errica/sigma_grids_anchor_refine/``.

``--mode nci1_refine``: ultra-tiny 2-config refine around NCI1 fixed8/a2g4
deep center (bs=32, lr=1e-3, L=12, d_h=16 × dropout=0.5 × pooling).
Writes ``configs/tu_errica/sigma_grids_nci1_refine/``.

``--mode native_fair``: compact SiGMA fair on PROTEINS/NCI1/REDDIT (16 configs):
a0g2 + a1g2 specialists × (lr × L); bs=32, d_h=16 fixed. Writes
``configs/tu_errica/sigma_grids_native_fair/``.

``--mode native_fair_v2``: mweber companion (6 configs): UniGCN mixes at
lr=1e-3, L=12, d_h=32. Writes ``configs/tu_errica/sigma_grids_native_fair_v2/``.

``--mode a0g_pnr``: MP-only (drop global attention) on PROTEINS / NCI1 /
REDDIT-BINARY. Wider train than native_fair; a0g4 full + a0g2 specialists.
Writes ``configs/tu_errica/sigma_grids_a0g_pnr/``.

``--mode tiny_pnr``: ultra-tiny sensible SiGMA select on PROTEINS / NCI1 /
REDDIT (bs∈{16,32} × d_h∈{8,16}, lr=1e-3, L=12, a2g4). 120 select / 90 eval.
Writes ``configs/tu_errica/sigma_grids_tiny_pnr/``.

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

SigmaGridMode = Literal[
    "fixed8",
    "budget_bio",
    "full64",
    "anchor_boost",
    "a1g2_micro",
    "a1g2_nci1_micro",
    "anchor_refine",
    "nci1_refine",
    "native_fair",
    "native_fair_v2",
    "a0g_pnr",
    "tiny_pnr",
]


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

A1G2_MICRO_DS_TAGS = _hp.A1G2_MICRO_DS_TAGS
A1G2_NCI1_MICRO_DS_TAGS = _hp.A1G2_NCI1_MICRO_DS_TAGS
ANCHOR_BOOST_DS_TAGS = _hp.ANCHOR_BOOST_DS_TAGS
ANCHOR_REFINE_DS_TAGS = _hp.ANCHOR_REFINE_DS_TAGS
NCI1_REFINE_DS_TAGS = _hp.NCI1_REFINE_DS_TAGS
A0G_PNR_DS_TAGS = _hp.A0G_PNR_DS_TAGS
A0G_PNR_DS_ORDER = _hp.A0G_PNR_DS_ORDER
TINY_PNR_DS_TAGS = _hp.TINY_PNR_DS_TAGS
TINY_PNR_DS_ORDER = _hp.TINY_PNR_DS_ORDER
NATIVE_FAIR_DS_TAGS = _hp.NATIVE_FAIR_DS_TAGS
NATIVE_FAIR_DS_ORDER = _hp.NATIVE_FAIR_DS_ORDER
NATIVE_FAIR_V2_DS_TAGS = _hp.NATIVE_FAIR_V2_DS_TAGS
NATIVE_FAIR_V2_DS_ORDER = _hp.NATIVE_FAIR_V2_DS_ORDER
BIO_DS_TAGS = _hp.BIO_DS_TAGS
DS_TAG_TO_NAME = _hp.DS_TAG_TO_NAME
SOCIAL_DS_TAGS = _hp.SOCIAL_DS_TAGS
a1g2_micro_sigma_grid_entries = _hp.a1g2_micro_sigma_grid_entries
a1g2_nci1_micro_sigma_grid_entries = _hp.a1g2_nci1_micro_sigma_grid_entries
anchor_boost_sigma_grid_entries = _hp.anchor_boost_sigma_grid_entries
anchor_refine_sigma_grid_entries = _hp.anchor_refine_sigma_grid_entries
nci1_refine_sigma_grid_entries = _hp.nci1_refine_sigma_grid_entries
native_fair_sigma_grid_entries = _hp.native_fair_sigma_grid_entries
native_fair_v2_sigma_grid_entries = _hp.native_fair_v2_sigma_grid_entries
a0g_pnr_sigma_grid_entries = _hp.a0g_pnr_sigma_grid_entries
tiny_pnr_sigma_grid_entries = _hp.tiny_pnr_sigma_grid_entries
build_bio_sigma_micro_grid = _hp.build_bio_sigma_micro_grid
full64_sigma_grid_entries = _hp.full64_sigma_grid_entries
social_sigma_grid_entries = _hp.social_sigma_grid_entries


def grids_dir_for_mode(mode: SigmaGridMode) -> Path:
    """Return the on-disk grid directory for ``mode`` (non-fixed8 are separate)."""
    if mode == "full64":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_full64"
    if mode == "anchor_boost":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_anchor_boost"
    if mode == "a1g2_micro":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_a1g2_micro"
    if mode == "a1g2_nci1_micro":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_a1g2_nci1_micro"
    if mode == "anchor_refine":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_anchor_refine"
    if mode == "nci1_refine":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_nci1_refine"
    if mode == "native_fair":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_native_fair"
    if mode == "native_fair_v2":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_native_fair_v2"
    if mode == "a0g_pnr":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_a0g_pnr"
    if mode == "tiny_pnr":
        return _REPO_ROOT / "configs/tu_errica/sigma_grids_tiny_pnr"
    return _REPO_ROOT / "configs/tu_errica/sigma_grids"


def _dataset_items_for_mode(mode: SigmaGridMode) -> list[tuple[str, str]]:
    """Return ``(ds_tag, ds_name)`` pairs included in ``mode``."""
    if mode == "anchor_boost":
        tags = ANCHOR_BOOST_DS_TAGS
        order = ("proteins", "reddit-b")
    elif mode == "a1g2_micro":
        tags = A1G2_MICRO_DS_TAGS
        order = ("proteins", "reddit-b")
    elif mode == "a1g2_nci1_micro":
        tags = A1G2_NCI1_MICRO_DS_TAGS
        order = ("nci1",)
    elif mode == "anchor_refine":
        tags = ANCHOR_REFINE_DS_TAGS
        order = ("proteins",)
    elif mode == "nci1_refine":
        tags = NCI1_REFINE_DS_TAGS
        order = ("nci1",)
    elif mode == "a0g_pnr":
        tags = A0G_PNR_DS_TAGS
        order = A0G_PNR_DS_ORDER
    elif mode == "tiny_pnr":
        tags = TINY_PNR_DS_TAGS
        order = TINY_PNR_DS_ORDER
    elif mode == "native_fair":
        tags = NATIVE_FAIR_DS_TAGS
        order = NATIVE_FAIR_DS_ORDER
    elif mode == "native_fair_v2":
        tags = NATIVE_FAIR_V2_DS_TAGS
        order = NATIVE_FAIR_V2_DS_ORDER
    else:
        return list(DS_TAG_TO_NAME.items())
    return [(tag, DS_TAG_TO_NAME[tag]) for tag in order if tag in tags]


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
    if mode == "full64":
        return _annotate_grid(
            full64_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "anchor_boost":
        if ds_tag not in ANCHOR_BOOST_DS_TAGS:
            raise KeyError(f"anchor_boost skips dataset {ds_tag}")
        return _annotate_grid(
            anchor_boost_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "a1g2_micro":
        if ds_tag not in A1G2_MICRO_DS_TAGS:
            raise KeyError(f"a1g2_micro skips dataset {ds_tag}")
        return _annotate_grid(
            a1g2_micro_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "a1g2_nci1_micro":
        if ds_tag not in A1G2_NCI1_MICRO_DS_TAGS:
            raise KeyError(f"a1g2_nci1_micro skips dataset {ds_tag}")
        return _annotate_grid(
            a1g2_nci1_micro_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "anchor_refine":
        if ds_tag not in ANCHOR_REFINE_DS_TAGS:
            raise KeyError(f"anchor_refine skips dataset {ds_tag}")
        return _annotate_grid(
            anchor_refine_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "nci1_refine":
        if ds_tag not in NCI1_REFINE_DS_TAGS:
            raise KeyError(f"nci1_refine skips dataset {ds_tag}")
        return _annotate_grid(
            nci1_refine_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "native_fair":
        if ds_tag not in NATIVE_FAIR_DS_TAGS:
            raise KeyError(f"native_fair skips dataset {ds_tag}")
        return _annotate_grid(
            native_fair_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "native_fair_v2":
        if ds_tag not in NATIVE_FAIR_V2_DS_TAGS:
            raise KeyError(f"native_fair_v2 skips dataset {ds_tag}")
        return _annotate_grid(
            native_fair_v2_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "a0g_pnr":
        if ds_tag not in A0G_PNR_DS_TAGS:
            raise KeyError(f"a0g_pnr skips dataset {ds_tag}")
        return _annotate_grid(
            a0g_pnr_sigma_grid_entries(),
            dataset_name=ds_name,
            gin_params=None,
            with_params=False,
        )
    if mode == "tiny_pnr":
        if ds_tag not in TINY_PNR_DS_TAGS:
            raise KeyError(f"tiny_pnr skips dataset {ds_tag}")
        return _annotate_grid(
            tiny_pnr_sigma_grid_entries(),
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

    for ds_tag, ds_name in _dataset_items_for_mode(mode):
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
        "datasets": [tag for tag, _ in _dataset_items_for_mode(mode)],
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

    grids_dir = grids_dir_for_mode(mode)
    grids_dir.mkdir(parents=True, exist_ok=True)
    grids_sub = grids_dir / "grids"
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

    manifest_path = grids_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Vendored single-file copy of the shared grid (documentation / inspection).
    if mode == "anchor_boost" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_anchor_boost_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "anchor_boost",
                    "note": (
                        "Paper a2g4 anchor (L12/H64/d_h16/lr1e-3) + LR/depth/d_h/batch "
                        "variants; PROTEINS+REDDIT-BINARY Errica select only."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "a1g2_micro" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_a1g2_micro_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_a1g2",
                    "mode": "a1g2_micro",
                    "note": (
                        "SiGMA a1g2 (1 attn + GCN,GIN): L12/H64/d_h16 fixed; "
                        "search bs∈{16,64} × lr∈{1e-3,1e-2} only. "
                        "PROTEINS+REDDIT-BINARY Errica select (80 tasks)."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "a1g2_nci1_micro" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_a1g2_nci1_micro_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_a1g2_ginsage",
                    "mode": "a1g2_nci1_micro",
                    "note": (
                        "SiGMA a1g2 (1 attn + GIN,SAGE) on NCI1: L12/H64/d_h16 fixed; "
                        "search bs∈{32,128} × lr∈{1e-3,1e-2} only (40 select tasks)."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "anchor_refine" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_anchor_refine_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "anchor_refine",
                    "note": (
                        "PROTEINS refine around modal anchor_boost winner "
                        "(bs=16, lr=1e-3, L=12, d_h=8): dropout=0.5 × "
                        "pool∈{add,mean} (2 configs × 10 folds = 20 select)."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "nci1_refine" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_nci1_refine_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "nci1_refine",
                    "note": (
                        "NCI1 refine around fixed8/a2g4 deep center "
                        "(bs=32, lr=1e-3, L=12, d_h=16): dropout=0.5 × "
                        "pool∈{add,mean} (2 configs × 10 folds = 20 select)."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "native_fair" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_native_fair_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "native_fair",
                    "note": (
                        "Compact SiGMA fair on PROTEINS/NCI1/REDDIT (gpu_h200). "
                        "a1g2_{gin_sage,gcn_gin} + a0g2_{gin_sage,gcn_gin} × "
                        "lr∈{1e-3,1e-2} × L∈{4,12}; bs=32, d_h=16, H=64 fixed "
                        "→ 16 configs × 3 × 10 = 480 select / 90 eval."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "native_fair_v2" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_native_fair_v2_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "native_fair_v2",
                    "note": (
                        "PNR fair v2 on mweber_gpu: UniGCN mixes, compact. "
                        "a1g2_{gin_sage,gin_unigcn,gcn_gin} + "
                        "a0g2_{gin_sage,gcn_gin,gcn_unigcn}; "
                        "lr=1e-3, L=12, d_h=32, bs=32, H=64 fixed "
                        "→ 6 configs × 3 × 10 = 180 select / 90 eval."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "a0g_pnr" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_a0g_pnr_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "a0g_pnr",
                    "note": (
                        "MP-only Errica fair on PROTEINS/NCI1/REDDIT (drop global attn). "
                        "a0g4_full + a0g2_gin_sage + a0g2_gcn_gin × same train axes as "
                        "native_fair → 48 configs × 3 × 10 = 1440 select / 90 eval."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if mode == "tiny_pnr" and written:
        sample = next(iter(written))
        sample_grid = json.loads((grids_sub / sample).read_text(encoding="utf-8"))
        vendored = _REPO_ROOT / "configs/tu_errica/sigma_hetero_tiny_pnr_hp_grid.json"
        vendored.write_text(
            json.dumps(
                {
                    "model": "sigma_hetero",
                    "mode": "tiny_pnr",
                    "note": (
                        "Ultra-tiny sensible SiGMA on PROTEINS/NCI1/REDDIT (a2g4). "
                        "bs∈{16,32} × d_h∈{8,16}, lr=1e-3, L=12, drop=0.5, pool=add "
                        "→ 4 configs × 3 × 10 = 120 select / 90 eval. "
                        "Avoids full64's lr=0.01 + L=4."
                    ),
                    "grid": sample_grid["grid"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "fixed8",
            "budget_bio",
            "full64",
            "anchor_boost",
            "a1g2_micro",
            "a1g2_nci1_micro",
            "anchor_refine",
            "nci1_refine",
            "native_fair",
            "native_fair_v2",
            "a0g_pnr",
            "tiny_pnr",
        ),
        default="fixed8",
        help="fixed8: 8-config SIGMA_GRID (default). "
        "full64: GIN-isomorphic 64-config grid (shallow; prefer native_fair). "
        "native_fair: a0g2+a1g2 on P/NCI1/REDDIT, lr×L only (480 select). "
        "native_fair_v2: UniGCN mixes on P/NCI1/REDDIT, lr=1e-3 L=12 d_h=32 (180 select). "
        "a0g_pnr: MP-only a0g* on PROTEINS/NCI1/REDDIT (1440 select). "
        "tiny_pnr: ultra-tiny sensible a2g4 on P/NCI1/REDDIT (120 select). "
        "anchor_boost: paper a2g4-centered grid on PROTEINS+REDDIT. "
        "a1g2_micro: tiny bs×lr grid for SiGMA a1g2 on PROTEINS+REDDIT. "
        "a1g2_nci1_micro: tiny bs×lr grid for SiGMA a1g2 (GIN,SAGE) on NCI1. "
        "anchor_refine: tiny dropout×pool refine on PROTEINS (a2g4). "
        "nci1_refine: ultra-tiny dropout×pool refine on NCI1 (a2g4). "
        "budget_bio: legacy GIN-budgeted bio micro-grid.",
    )
    parser.add_argument(
        "--gin-selection",
        type=Path,
        default=_REPO_ROOT / "configs/tu_errica/selections/gin_per_fold.json",
        help="Required for --mode budget_bio; ignored otherwise.",
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
    grids_dir = grids_dir_for_mode(mode)
    print(
        f"Wrote {grids_dir / 'manifest.json'} with {manifest['num_tasks']} "
        f"sigma_grid_select tasks (mode={mode})"
    )
    print(f"Grid files: {grids_dir / 'grids'}")


if __name__ == "__main__":
    main()
