#!/usr/bin/env python3
"""Aggregate paper-repro W&B runs (mean ± std over seeds).

Uses ``best_test_perf`` from run summary (test metric at val-best epoch).
Filter by W&B ``group`` and/or ``tag``.

Example::

    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
        --group paper_bestmodel_v1_cifar10_ulij45a2

    python scripts/api_wanndb_query/aggregate_paper_repro.py \\
        --tag bestmodel_v1 --dataset CIFAR10
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("wandb required: pip install wandb") from exc

DEFAULT_ENTITY = "weber-geoml-harvard-university"
DEFAULT_PROJECT = "GNNPlus"
DEFAULT_METRIC_KEYS = (
    "best_test_perf",
    "best/test_accuracy",
    "best/test_accuracy-SBM",
    "best/test_mae",
    "best/test_f1",
)


@dataclass(frozen=True)
class RunScore:
    """One finished run with seed and primary metric."""

    run_id: str
    name: str
    seed: Optional[int]
    metric: float
    metric_key: str
    state: str


def _unwrap(raw: Any) -> Any:
    """Return scalar from W&B config/summary entries when wrapped."""
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _parse_seed(run: Any) -> Optional[int]:
    """Extract training seed from run config."""
    cfg = run.config or {}
    for key in ("seed", "cfg_seed"):
        if key in cfg:
            try:
                return int(_unwrap(cfg[key]))
            except (TypeError, ValueError):
                continue
    nested = cfg.get("cfg", {})
    if isinstance(nested, dict) and "seed" in nested:
        try:
            return int(_unwrap(nested["seed"]))
        except (TypeError, ValueError):
            return None
    return None


def _pick_metric(summary: dict[str, Any], metric_key: Optional[str]) -> tuple[str, float]:
    """Return (key, value) for the first available metric in summary."""
    if metric_key:
        if metric_key not in summary:
            raise KeyError(f"Metric {metric_key!r} not in run summary")
        return metric_key, float(summary[metric_key])

    for key in DEFAULT_METRIC_KEYS:
        if key in summary and summary[key] is not None:
            return key, float(summary[key])
    raise KeyError(f"No known metric in summary (tried {DEFAULT_METRIC_KEYS})")


def _dataset_name(run: Any) -> str:
    """Return dataset name from run config."""
    cfg = run.config or {}
    for path in (
        ("dataset", "name"),
        ("cfg", "dataset", "name"),
    ):
        node: Any = cfg
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if node is not None:
            return str(_unwrap(node))
    return ""


def fetch_runs(
    *,
    entity: str,
    project: str,
    group: Optional[str],
    tag: Optional[str],
    dataset: Optional[str],
    states: Sequence[str],
    max_runs: int,
) -> list[Any]:
    """Fetch W&B runs matching filters."""
    api = wandb.Api()
    filters: dict[str, Any] = {"state": {"$in": list(states)}}
    if group:
        filters["group"] = group
    if tag:
        filters["tags"] = {"$in": [tag]}
    path = f"{entity}/{project}"
    runs = list(api.runs(path, filters=filters, per_page=max_runs, order="-created_at"))
    if dataset:
        ds_upper = dataset.upper()
        runs = [r for r in runs if _dataset_name(r).upper() == ds_upper]
    return runs


def aggregate_runs(
    runs: Sequence[Any],
    metric_key: Optional[str],
) -> list[RunScore]:
    """Build per-run scores; skip runs without a usable metric."""
    scores: list[RunScore] = []
    for run in runs:
        summary = dict(run.summary or {})
        try:
            key, value = _pick_metric(summary, metric_key)
        except KeyError:
            continue
        scores.append(
            RunScore(
                run_id=run.id,
                name=str(run.name),
                seed=_parse_seed(run),
                metric=value,
                metric_key=key,
                state=str(run.state),
            )
        )
    return scores


def print_empty_diagnostic(runs: Sequence[Any], metric_key: Optional[str]) -> None:
    """Print W&B runs in the filter when none have a usable summary metric."""
    if not runs:
        print(
            "No W&B runs matched the filter (check group/tag and that jobs reached wandb.init).",
            file=sys.stderr,
        )
        print(
            "On cluster: squeue -j <JOBID>  and  tail logs_gnnplus/<job>_<TASK>.log",
            file=sys.stderr,
        )
        return

    print(f"Found {len(runs)} run(s) but none with a summary metric yet.", file=sys.stderr)
    print(f"{'state':<10}  {'run_id':<10}  {'metric?':<8}  name", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    keys = (metric_key,) if metric_key else DEFAULT_METRIC_KEYS
    for run in runs:
        summary = dict(run.summary or {})
        has_metric = any(k in summary and summary[k] is not None for k in keys)
        metric_flag = "yes" if has_metric else "no"
        print(
            f"{str(run.state):<10}  {run.id:<10}  {metric_flag:<8}  {run.name}",
            file=sys.stderr,
        )
    print(
        "\nIf state=running and metric?=no: training started; wait for epoch 1 eval.",
        file=sys.stderr,
    )
    print(
        "If no runs listed: jobs may be pending (PD) or failed before W&B init — check SLURM logs.",
        file=sys.stderr,
    )


def print_report(
    scores: list[RunScore],
    metric_key: str,
    *,
    all_runs: Sequence[Any],
    requested_metric: Optional[str],
) -> None:
    """Print per-seed table and mean ± std."""
    if not scores:
        print_empty_diagnostic(all_runs, requested_metric)
        sys.exit(1)

    scores_sorted = sorted(scores, key=lambda s: (-1 if s.seed is None else s.seed))
    print(f"{'seed':>6}  {'metric':>12}  {'run_id':<10}  name")
    print("-" * 72)
    for s in scores_sorted:
        seed_str = str(s.seed) if s.seed is not None else "?"
        print(f"{seed_str:>6}  {s.metric:12.6f}  {s.run_id:<10}  {s.name}")

    values = [s.metric for s in scores]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    print()
    print(f"n={len(values)}  {metric_key}: {mean:.4f} ± {stdev:.4f}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Aggregate paper-repro W&B runs.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", default=None, help="W&B run group")
    parser.add_argument("--tag", default=None, help="W&B tag (e.g. bestmodel_v1)")
    parser.add_argument("--dataset", default=None, help="Filter by dataset.name")
    parser.add_argument("--metric", default=None, help="Summary key (default: auto)")
    parser.add_argument(
        "--state",
        default="finished,running,failed,crashed",
        help="Comma-separated run states to include (default includes failed for debugging)",
    )
    parser.add_argument("--max-runs", type=int, default=50)
    args = parser.parse_args()

    if not args.group and not args.tag:
        parser.error("Provide at least one of --group or --tag")

    states = [s.strip() for s in args.state.split(",") if s.strip()]
    runs = fetch_runs(
        entity=args.entity,
        project=args.project,
        group=args.group,
        tag=args.tag,
        dataset=args.dataset,
        states=states,
        max_runs=args.max_runs,
    )
    scores = aggregate_runs(runs, args.metric)
    metric_label = scores[0].metric_key if scores else (args.metric or "metric")
    print_report(scores, metric_label, all_runs=runs, requested_metric=args.metric)


if __name__ == "__main__":
    main()
