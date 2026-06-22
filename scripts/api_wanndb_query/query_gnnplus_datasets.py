#!/usr/bin/env python3
"""List which datasets have runs in the GNNPlus W&B project.

Reads run config (``dataset.name``, ``dataset.format``, ``model.type``) and
optional sweep metadata, then prints a per-dataset summary.

Auth: ``WANDB_API_KEY`` or ``~/.netrc`` (``wandb login``).

Example::

    cd /path/to/GNNPlus
    python scripts/api_wanndb_query/query_gnnplus_datasets.py

    python scripts/api_wanndb_query/query_gnnplus_datasets.py \\
        --entity weber-geoml-harvard-university --project GNNPlus \\
        --max-scan 5000 --csv results/wandb_gnnplus_datasets.csv

    python scripts/api_wanndb_query/query_gnnplus_datasets.py --require-gates
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Iterable, Mapping, Optional, Sequence

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "wandb is required (see requirements-cluster.txt). "
        "Install with: pip install wandb"
    ) from exc

DEFAULT_ENTITY = "weber-geoml-harvard-university"
DEFAULT_PROJECT = "GNNPlus"

# Map OGB / PyG internal names to yaml stems used in GNNPlus configs.
_NAME_ALIASES: dict[str, str] = {
    "subset": "zinc",
    "peptides-functional": "peptides-func",
    "peptides-structural": "peptides-struct",
    "molhiv": "hiv",
    "ogbg-molhiv": "hiv",
    "ogbg-molpcba": "pcba",
    "ogbg-ppa": "ppa",
    "ogbg-code2": "code2",
}

_FORMAT_SLUGS: dict[str, str] = {
    "PyG-ZINC": "zinc",
    "PyG-GNNBenchmarkDataset": "",
    "PyG-TUDataset": "",
    "OGB": "",
}


def _unwrap(raw: Any) -> Any:
    """Return scalar from W&B config/summary entries when wrapped."""
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _nested_get(cfg: Mapping[str, Any], *keys: str) -> Any:
    """Traverse a nested config dict."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return _unwrap(node)


def _slug_from_run_name(name: str) -> Optional[str]:
    """Guess dataset yaml stem from run name patterns."""
    lowered = name.lower()
    # hybrid sweep / paper names: mnist_..., peptides_func_..., zinc_gcn_seed0
    for stem in (
        "peptides-func",
        "peptides-struct",
        "peptides_func",
        "peptides_struct",
        "cifar10",
        "mnist",
        "mutag",
        "enzymes",
        "cluster",
        "pattern",
        "pcba",
        "coco",
        "voc",
        "zinc",
        "hiv",
        "ppa",
        "mal",
        "code2",
    ):
        if stem.replace("_", "-") in lowered or stem in lowered:
            return stem.replace("_", "-")
    return None


def normalize_dataset_slug(
    dataset_name: Optional[str],
    dataset_format: Optional[str],
    run_name: str,
    sweep_name: Optional[str],
) -> str:
    """Map W&B config fields to a GNNPlus-style dataset slug."""
    raw_name = str(dataset_name or "").strip()
    raw_format = str(dataset_format or "").strip()

    if raw_name.lower() in _NAME_ALIASES:
        return _NAME_ALIASES[raw_name.lower()]

    if raw_name and raw_name.lower() not in ("none", "subset"):
        return raw_name.lower()

    if raw_format in _FORMAT_SLUGS and _FORMAT_SLUGS[raw_format]:
        return _FORMAT_SLUGS[raw_format]

    if raw_format == "PyG-GNNBenchmarkDataset" and raw_name:
        return raw_name.lower()

    for hint in (run_name, sweep_name or ""):
        guessed = _slug_from_run_name(hint)
        if guessed:
            return guessed

    if raw_name:
        return raw_name.lower()
    if raw_format:
        return raw_format.lower().replace("pyg-", "").replace("ogb", "ogb").strip("-")
    return "unknown"


@dataclass
class RunRecord:
    """One W&B run reduced to fields we care about for coverage."""

    run_id: str
    run_name: str
    state: str
    dataset_slug: str
    dataset_name: str
    dataset_format: str
    model_type: str
    layer_type: str
    sweep_id: str
    sweep_name: str
    has_gates: bool
    created_at: str


@dataclass
class DatasetStats:
    """Aggregated counts for one dataset slug."""

    slug: str
    total: int = 0
    by_state: Counter[str] = field(default_factory=Counter)
    by_model: Counter[str] = field(default_factory=Counter)
    with_gates: int = 0
    sweep_names: set[str] = field(default_factory=set)


def _has_gate_metrics(summary: Mapping[str, Any]) -> bool:
    return any(
        str(k).startswith("gates/") and "gate_mean" in str(k)
        for k in summary
    )


def fetch_run_records(
    entity: str,
    project: str,
    *,
    max_scan: int,
    states: Optional[set[str]] = None,
    require_gates: bool = False,
) -> list[RunRecord]:
    """Pull runs from W&B and normalize dataset / model fields."""
    path = f"{entity}/{project}"
    api = wandb.Api(timeout=int(os.environ.get("WANDB_API_TIMEOUT", "120")))
    records: list[RunRecord] = []
    scanned = 0

    for run in api.runs(path, per_page=100, order="-created_at"):
        scanned += 1
        if scanned > max_scan:
            break

        state = str(getattr(run, "state", "") or "")
        if states and state not in states:
            continue

        summary = {str(k): _unwrap(v) for k, v in dict(run.summary or {}).items()}
        if require_gates and not _has_gate_metrics(summary):
            continue

        cfg = {str(k): _unwrap(v) for k, v in dict(run.config or {}).items()}
        # cfg_to_dict nests yacs keys at top level
        dataset_name = _nested_get(cfg, "dataset", "name")
        dataset_format = _nested_get(cfg, "dataset", "format")
        model_type = str(_nested_get(cfg, "model", "type") or "")
        layer_type = str(_nested_get(cfg, "gnn", "layer_type") or "")
        if model_type == "hybrid_gnn":
            ha = _nested_get(cfg, "gnn", "hybrid", "num_attn_heads")
            hg = _nested_get(cfg, "gnn", "hybrid", "num_gnn_heads")
            layer_type = f"hybrid_a{ha}g{hg}"

        run_name = str(getattr(run, "name", "") or "")
        sweep = getattr(run, "sweep", None)
        sweep_id = str(getattr(sweep, "id", "") or "") if sweep else ""
        sweep_name = str(getattr(sweep, "name", "") or "") if sweep else ""

        slug = normalize_dataset_slug(
            str(dataset_name) if dataset_name is not None else None,
            str(dataset_format) if dataset_format is not None else None,
            run_name,
            sweep_name or None,
        )

        records.append(
            RunRecord(
                run_id=str(run.id),
                run_name=run_name,
                state=state,
                dataset_slug=slug,
                dataset_name=str(dataset_name or ""),
                dataset_format=str(dataset_format or ""),
                model_type=model_type or "unknown",
                layer_type=layer_type,
                sweep_id=sweep_id,
                sweep_name=sweep_name,
                has_gates=_has_gate_metrics(summary),
                created_at=str(getattr(run, "created_at", "") or ""),
            )
        )

    return records


def aggregate_by_dataset(records: Iterable[RunRecord]) -> dict[str, DatasetStats]:
    """Build per-dataset counters from run records."""
    out: DefaultDict[str, DatasetStats] = defaultdict(
        lambda: DatasetStats(slug="")
    )
    for rec in records:
        stats = out[rec.dataset_slug]
        stats.slug = rec.dataset_slug
        stats.total += 1
        stats.by_state[rec.state] += 1
        model_key = rec.model_type
        if rec.layer_type:
            model_key = f"{rec.model_type}/{rec.layer_type}"
        stats.by_model[model_key] += 1
        if rec.has_gates:
            stats.with_gates += 1
        if rec.sweep_name:
            stats.sweep_names.add(rec.sweep_name)
    return dict(out)


def fetch_sweep_dataset_hints(
    entity: str,
    project: str,
    *,
    name_prefix: str = "GNNplus_hybriddgatedGNN-",
) -> dict[str, list[str]]:
    """Map dataset slug → sweep ids from sweep display names."""
    path = f"{entity}/{project}"
    api = wandb.Api(timeout=int(os.environ.get("WANDB_API_TIMEOUT", "120")))
    hints: DefaultDict[str, list[str]] = defaultdict(list)
    try:
        sweeps = api.sweeps(path)
    except Exception:
        return {}

    pat = re.compile(
        rf"^{re.escape(name_prefix)}(?P<slug>[\w-]+)$",
        re.IGNORECASE,
    )
    for sweep in sweeps:
        name = str(getattr(sweep, "name", "") or "")
        m = pat.match(name)
        if not m:
            continue
        slug = m.group("slug").replace("_", "-").lower()
        sid = str(getattr(sweep, "id", "") or "")
        if sid:
            hints[slug].append(sid)
    return dict(hints)


def print_summary(
    records: list[RunRecord],
    by_dataset: dict[str, DatasetStats],
    sweep_hints: Mapping[str, list[str]],
) -> None:
    """Print human-readable coverage table to stdout."""
    print(f"Scanned {len(records)} runs in project.")
    print()
    print(f"{'dataset':<18} {'runs':>6} {'finished':>9} {'running':>8} "
          f"{'failed':>7} {'gates':>6}  models / sweeps")
    print("-" * 90)

    for slug in sorted(by_dataset.keys()):
        stats = by_dataset[slug]
        finished = stats.by_state.get("finished", 0)
        running = stats.by_state.get("running", 0)
        failed = (
            stats.by_state.get("failed", 0)
            + stats.by_state.get("crashed", 0)
            + stats.by_state.get("killed", 0)
        )
        models = ", ".join(
            f"{k}({v})" for k, v in sorted(stats.by_model.items())
        )
        sweeps = stats.sweep_names or set(sweep_hints.get(slug, []))
        sweep_str = ", ".join(sorted(sweeps)[:2])
        if len(sweeps) > 2:
            sweep_str += f" +{len(sweeps) - 2}"
        print(
            f"{slug:<18} {stats.total:>6} {finished:>9} {running:>8} "
            f"{failed:>7} {stats.with_gates:>6}  {models[:40]}"
        )
        if sweep_str:
            print(f"{'':18} {'':>6} {'':>9} {'':>8} {'':>7} {'':>6}  sweeps: {sweep_str}")

    only_sweeps = sorted(set(sweep_hints.keys()) - set(by_dataset.keys()))
    if only_sweeps:
        print()
        print("Sweeps created but no runs logged yet:", ", ".join(only_sweeps))

    print()
    print("Datasets with at least one run:", len(by_dataset))
    print("Dataset slugs:", ", ".join(sorted(by_dataset.keys())))


def write_csv(path: Path, records: Sequence[RunRecord]) -> None:
    """Write one row per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "state",
        "dataset_slug",
        "dataset_name",
        "dataset_format",
        "model_type",
        "layer_type",
        "sweep_id",
        "sweep_name",
        "has_gates",
        "created_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "run_id": rec.run_id,
                    "run_name": rec.run_name,
                    "state": rec.state,
                    "dataset_slug": rec.dataset_slug,
                    "dataset_name": rec.dataset_name,
                    "dataset_format": rec.dataset_format,
                    "model_type": rec.model_type,
                    "layer_type": rec.layer_type,
                    "sweep_id": rec.sweep_id,
                    "sweep_name": rec.sweep_name,
                    "has_gates": rec.has_gates,
                    "created_at": rec.created_at,
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=3000,
        help="Maximum runs to scan (newest first).",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=None,
        help="Only include runs in this state (repeatable). "
        "e.g. --state finished --state running",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Only count runs with gates/* gate_mean in summary.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-run CSV.",
    )
    parser.add_argument(
        "--list-sweeps",
        action="store_true",
        help="Also list hybrid sweep ids from sweep names.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    states = set(args.state) if args.state else None

    print(
        f"Querying {args.entity}/{args.project} "
        f"(max_scan={args.max_scan})..."
    )
    records = fetch_run_records(
        args.entity,
        args.project,
        max_scan=args.max_scan,
        states=states,
        require_gates=args.require_gates,
    )
    by_dataset = aggregate_by_dataset(records)

    sweep_hints: dict[str, list[str]] = {}
    if args.list_sweeps:
        sweep_hints = fetch_sweep_dataset_hints(args.entity, args.project)
        if sweep_hints:
            print()
            print("Hybrid sweeps (by name):")
            for slug in sorted(sweep_hints):
                ids = ", ".join(sweep_hints[slug])
                print(f"  {slug}: {ids}")
            print()

    print_summary(records, by_dataset, sweep_hints)

    if args.csv is not None:
        write_csv(args.csv, records)
        print(f"Wrote {len(records)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
