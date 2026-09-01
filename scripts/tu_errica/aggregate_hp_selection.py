#!/usr/bin/env python3
"""Pick best Errica grid HP per (dataset, fold) from W&B grid_select runs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_errica_hp_grid() -> Any:
    import importlib.util

    path = _REPO_ROOT / "scripts/tu_errica/errica_hp_grid.py"
    spec = importlib.util.spec_from_file_location("errica_hp_grid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load errica_hp_grid from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hp = _load_errica_hp_grid()
DS_TAG_TO_NAME = _hp.DS_TAG_TO_NAME
MODEL_TAG_BY_KEY = _hp.MODEL_TAG_BY_KEY
load_hp_config = _hp.load_hp_config

DEFAULT_ENTITY = "weber-geoml-harvard-university"
DEFAULT_PROJECT = "GNNPlus"

VAL_METRIC_KEYS: tuple[str, ...] = (
    "best/val_accuracy",
    "best_val_perf",
    "val_accuracy",
)

FOLD_RE = re.compile(r"_f(\d+)_")


def _parse_fold(run_name: str, config: dict[str, Any]) -> int | None:
    """Extract fold index from run name or config."""
    match = FOLD_RE.search(run_name)
    if match:
        return int(match.group(1))
    cfg = config or {}
    for path in (("dataset", "split_index"), ("cfg", "dataset", "split_index")):
        node: Any = cfg
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if node is not None:
            try:
                return int(node)
            except (TypeError, ValueError):
                pass
    return None


def _val_metric(summary: dict[str, Any]) -> float | None:
    for key in VAL_METRIC_KEYS:
        if key in summary and summary[key] is not None:
            value = float(summary[key])
            return value * 100.0 if value <= 1.0 else value
    return None


def collect_grid_select_runs(
    *,
    entity: str,
    project: str,
    model: str,
    ds_tag: str,
    num_hp: int,
    states: list[str],
) -> list[dict[str, Any]]:
    """Fetch all grid_select runs for one dataset and model."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("wandb required: pip install wandb") from exc

    api = wandb.Api()
    path = f"{entity}/{project}"
    model_tag = MODEL_TAG_BY_KEY[model]
    rows: list[dict[str, Any]] = []

    for hp_id in range(num_hp):
        group = f"tu_errica_{ds_tag}_{model_tag}_grid_select_hp{hp_id}"
        filters: dict[str, Any] = {"group": group, "state": {"$in": states}}
        try:
            runs = api.runs(path, filters=filters, per_page=50)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] W&B query failed {group}: {exc}", file=sys.stderr)
            continue
        for run in runs:
            summary = dict(run.summary or {})
            val = _val_metric(summary)
            if val is None:
                continue
            fold = _parse_fold(str(run.name), dict(run.config or {}))
            if fold is None:
                continue
            rows.append(
                {
                    "ds_tag": ds_tag,
                    "dataset": DS_TAG_TO_NAME[ds_tag],
                    "fold": fold,
                    "hp_id": hp_id,
                    "val_accuracy": val,
                    "run_id": run.id,
                    "run_name": run.name,
                    "state": run.state,
                }
            )
    return rows


def select_best_per_fold(
    rows: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return nested ds_tag → fold → best entry with full HP dict."""
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["ds_tag"], int(row["fold"]))
        if key not in best or row["val_accuracy"] > best[key]["val_accuracy"]:
            best[key] = row

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (ds_tag, fold), row in sorted(best.items()):
        hp = load_hp_config(model, int(row["hp_id"]), canonical_only=False)
        entry = {
            "hp_id": row["hp_id"],
            "val_accuracy": row["val_accuracy"],
            "run_id": row["run_id"],
            "hp": hp,
        }
        out.setdefault(ds_tag, {})[str(fold)] = entry
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["gin", "graphsage", "gcn", "gat"])
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: configs/tu_errica/selections/<model>_per_fold.json)",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Comma-separated W&B states",
    )
    args = parser.parse_args()

    grid_path = _REPO_ROOT / "configs/tu_errica" / f"{args.model}_hp_grid.json"
    with grid_path.open(encoding="utf-8") as handle:
        num_hp = len(json.load(handle)["grid"])

    states = [s.strip() for s in args.state.split(",") if s.strip()]
    all_rows: list[dict[str, Any]] = []
    for ds_tag in DS_TAG_TO_NAME:
        rows = collect_grid_select_runs(
            entity=args.entity,
            project=args.project,
            model=args.model,
            ds_tag=ds_tag,
            num_hp=num_hp,
            states=states,
        )
        all_rows.extend(rows)
        print(f"{ds_tag}: {len(rows)} grid_select runs with val metric", file=sys.stderr)

    selection = select_best_per_fold(all_rows, model=args.model)
    out_path = args.out or (
        _REPO_ROOT / "configs/tu_errica/selections" / f"{args.model}_per_fold.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_hp": num_hp,
        "selection": selection,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({sum(len(v) for v in selection.values())} fold entries)")


if __name__ == "__main__":
    main()
