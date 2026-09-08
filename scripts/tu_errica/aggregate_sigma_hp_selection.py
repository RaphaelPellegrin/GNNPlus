#!/usr/bin/env python3
"""Pick best SiGMA HP per (dataset, fold) from sigma_grid_select W&B runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_aggregate_module() -> Any:
    import importlib.util

    path = _REPO_ROOT / "scripts/tu_errica/aggregate_hp_selection.py"
    spec = importlib.util.spec_from_file_location("aggregate_hp_selection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load aggregate_hp_selection from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_agg = _load_aggregate_module()
DEFAULT_ENTITY = _agg.DEFAULT_ENTITY
DEFAULT_PROJECT = _agg.DEFAULT_PROJECT
_parse_fold = _agg._parse_fold
_val_metric = _agg._val_metric

MANIFEST_PATH = _REPO_ROOT / "configs/tu_errica/sigma_grids/manifest.json"


def collect_sigma_runs(
    *,
    entity: str,
    project: str,
    manifest: dict[str, Any],
    states: list[str],
    campaign: str,
    grids_dir: Path,
    model_tag: str = "SiGMA_hetero",
) -> list[dict[str, Any]]:
    """Fetch sigma_grid_select runs listed in the manifest.

    Each row records ``grids_dir`` and ``campaign`` so multi-source merges can
    resolve ``hp_id`` against the correct grid file.
    """
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("wandb required: pip install wandb") from exc

    api = wandb.Api()
    path = f"{entity}/{project}"
    rows: list[dict[str, Any]] = []
    grids_dir_resolved = str(grids_dir.resolve())

    for task in manifest["tasks"]:
        ds_tag = str(task["ds_tag"])
        fold = int(task["fold"])
        hp_id = int(task["hp_id"])
        hp_tag = f"f{fold}_hp{hp_id}"
        group = f"tu_errica_{ds_tag}_{model_tag}_{campaign}_{hp_tag}"
        filters: dict[str, Any] = {"group": group, "state": {"$in": states}}
        try:
            runs = list(api.runs(path, filters=filters, per_page=10))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] W&B query failed {group}: {exc}", file=sys.stderr)
            continue
        for run in runs:
            summary = dict(run.summary or {})
            val = _val_metric(summary)
            if val is None:
                continue
            parsed_fold = _parse_fold(str(run.name), dict(run.config or {}))
            if parsed_fold is not None:
                fold = parsed_fold
            rows.append(
                {
                    "ds_tag": ds_tag,
                    "dataset": task["dataset"],
                    "fold": fold,
                    "hp_id": hp_id,
                    "grid_file": task["grid_file"],
                    "grids_dir": grids_dir_resolved,
                    "campaign": campaign,
                    "val_accuracy": val,
                    "run_id": run.id,
                }
            )
    return rows


def select_best_sigma(
    rows: list[dict[str, Any]],
    *,
    grids_dir: Path | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Nested ds_tag → fold → best sigma grid entry.

    Prefer each row's ``grids_dir`` (set by ``collect_sigma_runs``). Fall back to
    ``grids_dir`` for single-campaign calls that omit per-row paths.
    """
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["ds_tag"], int(row["fold"]))
        if key not in best or row["val_accuracy"] > best[key]["val_accuracy"]:
            best[key] = row

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (ds_tag, fold), row in sorted(best.items()):
        row_grids = Path(str(row["grids_dir"])) if row.get("grids_dir") else None
        if row_grids is None:
            if grids_dir is None:
                raise ValueError(
                    f"row missing grids_dir and no default for {ds_tag} fold {fold}"
                )
            row_grids = grids_dir
        grid_path = row_grids / str(row["grid_file"])
        with grid_path.open(encoding="utf-8") as handle:
            grid = json.load(handle)["grid"]
        hp = dict(grid[int(row["hp_id"])])
        entry: dict[str, Any] = {
            "hp_id": row["hp_id"],
            "grid_file": row["grid_file"],
            "grids_dir": str(row_grids),
            "val_accuracy": row["val_accuracy"],
            "run_id": row["run_id"],
            "hp": hp,
        }
        if row.get("campaign"):
            entry["source_campaign"] = row["campaign"]
        out.setdefault(ds_tag, {})[str(fold)] = entry
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--campaign",
        default="sigma_grid_select_fixed8",
        help="W&B campaign segment in group names "
        "(use sigma_grid_select for legacy budget_bio).",
    )
    parser.add_argument(
        "--also-campaign",
        action="append",
        default=[],
        help="Extra W&B select campaign to merge (repeatable; pair with "
        "--also-manifest). Used for joint L∈{4,12} native_fair_v2.",
    )
    parser.add_argument(
        "--also-manifest",
        action="append",
        default=[],
        type=Path,
        help="Manifest for each --also-campaign (same order / count).",
    )
    parser.add_argument(
        "--model-tag",
        default="SiGMA_hetero",
        help="W&B model tag in group names (SiGMA_hetero or SiGMA_ungated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "configs/tu_errica/selections/sigma_fixed8_per_fold.json",
    )
    parser.add_argument("--state", default="finished")
    args = parser.parse_args()

    if len(args.also_campaign) != len(args.also_manifest):
        parser.error(
            f"--also-campaign ({len(args.also_campaign)}) and "
            f"--also-manifest ({len(args.also_manifest)}) counts must match"
        )

    states = [s.strip() for s in args.state.split(",") if s.strip()]
    sources: list[tuple[str, Path]] = [(args.campaign, args.manifest)]
    sources.extend(zip(args.also_campaign, args.also_manifest, strict=True))

    rows: list[dict[str, Any]] = []
    for campaign, manifest_path in sources:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        grids_dir = manifest_path.resolve().parent / "grids"
        part = collect_sigma_runs(
            entity=args.entity,
            project=args.project,
            manifest=manifest,
            states=states,
            campaign=campaign,
            grids_dir=grids_dir,
            model_tag=args.model_tag,
        )
        print(
            f"[aggregate] {campaign}: {len(part)} finished runs "
            f"(manifest={manifest_path})",
            file=sys.stderr,
        )
        rows.extend(part)

    selection = select_best_sigma(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": "sigma_hetero_ungated"
        if args.model_tag == "SiGMA_ungated"
        else "sigma_hetero",
        "model_tag": args.model_tag,
        "campaign": args.campaign
        if not args.also_campaign
        else "merged:" + "+".join(c for c, _ in sources),
        "sources": [
            {"campaign": c, "manifest": str(m.resolve())} for c, m in sources
        ],
        "manifest": str(args.manifest),
        "selection": selection,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n_folds = sum(len(v) for v in selection.values())
    print(
        f"Wrote {args.out} ({n_folds} folds, model_tag={args.model_tag}, "
        f"sources={len(sources)})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
