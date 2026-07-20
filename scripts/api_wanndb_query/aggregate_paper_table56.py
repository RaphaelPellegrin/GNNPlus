#!/usr/bin/env python3
"""Aggregate SiGMA paper Table 5 / Table 6 W&B groups (mean ± std over seeds).

Queries each 5-seed experiment group under ``paper_T5_*`` / ``paper_T6_*`` and
prints a fill-in table using ``best_test_perf``.

Examples::

    # Everything (Table 5 + Table 6 VOC + Table 6 1-MP)
    python scripts/api_wanndb_query/aggregate_paper_table56.py

    # Table 5 only (LRGB), finished runs
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5

    # Table 5 MNIST + CIFAR10
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5mc

    # Table 6 only (VOC + 1-MP campaigns)
    python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6

    # Include running jobs in the mean (not recommended for final numbers)
    python scripts/api_wanndb_query/aggregate_paper_table56.py --state finished,running
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("wandb required: install from pyproject / pip install wandb") from exc

DEFAULT_ENTITY = "weber-geoml-harvard-university"
DEFAULT_PROJECT = "GNNPlus"
DEFAULT_METRIC = "best_test_perf"

# Metric direction for display only (↓ = lower better).
HIGHER_BETTER: dict[str, bool] = {
    "peptides_func": True,
    "peptides_struct": False,
    "voc": True,
    "coco": True,
    "mnist": True,
    "cifar10": True,
}

TABLE5_DATASETS: tuple[str, ...] = (
    "peptides_func",
    "peptides_struct",
    "voc",
    "coco",
)
TABLE5_MNIST_CIFAR_DATASETS: tuple[str, ...] = (
    "mnist",
    "cifar10",
)
TABLE5_VARIANTS: tuple[str, ...] = (
    "SiGMA",
    "SiGMA_ungated",
    "Attn_only",
    "MP_only",
)

TABLE6_VOC_VARIANTS: tuple[str, ...] = (
    "SiGMA",
    "Hetero_MP",
    "Hetero_MP_ungated",
)

TABLE6_1MP_DATASETS: tuple[str, ...] = (
    "peptides_func",
    "peptides_struct",
    "coco",
)
TABLE6_1MP_VARIANTS: tuple[str, ...] = (
    "SiGMA",
    "Homog_MP",
    "Hetero_MP",
    "Homog_MP_ungated",
    "Hetero_MP_ungated",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One 5-seed experiment cell (W&B group)."""

    table: str
    dataset: str
    variant: str
    group: str


@dataclass(frozen=True)
class ExperimentAgg:
    """Aggregation result for one experiment group."""

    spec: ExperimentSpec
    n_finished: int
    n_other: int
    mean: Optional[float]
    std: Optional[float]
    seeds: tuple[Optional[int], ...]
    values: tuple[float, ...]
    run_ids: tuple[str, ...]


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
    return None


def _metric_from_summary(summary: dict[str, Any], metric_key: str) -> Optional[float]:
    """Return metric float if present in summary."""
    if metric_key not in summary or summary[metric_key] is None:
        return None
    try:
        return float(summary[metric_key])
    except (TypeError, ValueError):
        return None


def build_experiment_specs(*, tables: Sequence[str], prefix_t5: str, prefix_t6: str) -> list[ExperimentSpec]:
    """Build the list of W&B groups to query for the selected tables."""
    specs: list[ExperimentSpec] = []
    table_set = {t.lower() for t in tables}

    if "5" in table_set or "all" in table_set:
        for ds in TABLE5_DATASETS:
            for variant in TABLE5_VARIANTS:
                specs.append(
                    ExperimentSpec(
                        table="5",
                        dataset=ds,
                        variant=variant,
                        group=f"{prefix_t5}_{ds}_{variant}",
                    )
                )

    if "5mc" in table_set or "5_mnist_cifar" in table_set or "all" in table_set:
        for ds in TABLE5_MNIST_CIFAR_DATASETS:
            for variant in TABLE5_VARIANTS:
                specs.append(
                    ExperimentSpec(
                        table="5mc",
                        dataset=ds,
                        variant=variant,
                        group=f"{prefix_t5}_{ds}_{variant}",
                    )
                )

    if "6" in table_set or "all" in table_set:
        for variant in TABLE6_VOC_VARIANTS:
            specs.append(
                ExperimentSpec(
                    table="6_voc",
                    dataset="voc",
                    variant=variant,
                    group=f"{prefix_t6}_voc_{variant}",
                )
            )
        for ds in TABLE6_1MP_DATASETS:
            for variant in TABLE6_1MP_VARIANTS:
                specs.append(
                    ExperimentSpec(
                        table="6_1mp",
                        dataset=ds,
                        variant=variant,
                        group=f"{prefix_t6}_{ds}_{variant}",
                    )
                )

    return specs


def fetch_group_runs(
    api: Any,
    *,
    entity: str,
    project: str,
    group: str,
    states: Sequence[str],
    max_runs: int,
) -> list[Any]:
    """Fetch W&B runs for a single group."""
    filters: dict[str, Any] = {
        "group": group,
        "state": {"$in": list(states)},
    }
    path = f"{entity}/{project}"
    return list(api.runs(path, filters=filters, per_page=max_runs, order="-created_at"))


def aggregate_group(
    spec: ExperimentSpec,
    runs: Sequence[Any],
    *,
    metric_key: str,
    score_states: Sequence[str],
) -> ExperimentAgg:
    """Compute mean ± std over runs in ``score_states`` that have the metric."""
    score_state_set = {s.lower() for s in score_states}
    values: list[float] = []
    seeds: list[Optional[int]] = []
    run_ids: list[str] = []
    n_other = 0

    for run in runs:
        state = str(run.state).lower()
        summary = dict(run.summary or {})
        metric = _metric_from_summary(summary, metric_key)
        if state not in score_state_set or metric is None:
            n_other += 1
            continue
        values.append(metric)
        seeds.append(_parse_seed(run))
        run_ids.append(str(run.id))

    if not values:
        # n_other already counts every unscored run from the loop above.
        return ExperimentAgg(
            spec=spec,
            n_finished=0,
            n_other=n_other,
            mean=None,
            std=None,
            seeds=tuple(seeds),
            values=tuple(values),
            run_ids=tuple(run_ids),
        )

    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return ExperimentAgg(
        spec=spec,
        n_finished=len(values),
        n_other=n_other,
        mean=mean,
        std=std,
        seeds=tuple(seeds),
        values=tuple(values),
        run_ids=tuple(run_ids),
    )


def _fmt_cell(agg: ExperimentAgg, *, expected_n: int) -> str:
    """Format mean±std (n) for a table cell."""
    if agg.mean is None or agg.std is None:
        if agg.n_other > 0:
            return f"— (0/{expected_n}, {agg.n_other} other)"
        return f"— (0/{expected_n})"
    arrow = "" if HIGHER_BETTER.get(agg.spec.dataset, True) else " ↓"
    n_note = f"n={agg.n_finished}"
    if agg.n_finished != expected_n:
        n_note = f"n={agg.n_finished}/{expected_n}"
    return f"{agg.mean:.4f}±{agg.std:.4f} ({n_note}){arrow}"


def print_detail(aggs: Iterable[ExperimentAgg], *, expected_n: int) -> None:
    """Print one block per experiment with per-seed rows."""
    for agg in aggs:
        spec = agg.spec
        print()
        print(f"===== Table {spec.table} | {spec.dataset} | {spec.variant} =====")
        print(f"group: {spec.group}")
        if not agg.values:
            print(f"  no scored runs yet (other/pending={agg.n_other})")
            continue
        print(f"{'seed':>6}  {'metric':>12}  {'run_id':<10}")
        print("-" * 36)
        order = sorted(
            zip(agg.seeds, agg.values, agg.run_ids, strict=True),
            key=lambda t: (-1 if t[0] is None else t[0]),
        )
        for seed, value, run_id in order:
            seed_str = str(seed) if seed is not None else "?"
            print(f"{seed_str:>6}  {value:12.6f}  {run_id:<10}")
        assert agg.mean is not None and agg.std is not None
        n_note = f"{agg.n_finished}/{expected_n}" if agg.n_finished != expected_n else str(agg.n_finished)
        print(f"n={n_note}  mean±std = {agg.mean:.4f} ± {agg.std:.4f}")


def print_summary_tables(aggs: Sequence[ExperimentAgg], *, expected_n: int) -> None:
    """Print compact markdown-style summary tables."""
    by_table: dict[str, list[ExperimentAgg]] = {}
    for agg in aggs:
        by_table.setdefault(agg.spec.table, []).append(agg)

    if "5" in by_table:
        print("\n## Table 5 — LRGB (paper_T5_*)\n")
        datasets = TABLE5_DATASETS
        variants = TABLE5_VARIANTS
        _print_pivot(by_table["5"], datasets=datasets, variants=variants, expected_n=expected_n)

    if "5mc" in by_table:
        print("\n## Table 5 — MNIST + CIFAR10 (paper_T5_*)\n")
        _print_pivot(
            by_table["5mc"],
            datasets=TABLE5_MNIST_CIFAR_DATASETS,
            variants=TABLE5_VARIANTS,
            expected_n=expected_n,
        )

    if "6_voc" in by_table:
        print("\n## Table 6 — VOC (paper_T6_voc_*)\n")
        _print_pivot(
            by_table["6_voc"],
            datasets=("voc",),
            variants=TABLE6_VOC_VARIANTS,
            expected_n=expected_n,
        )

    if "6_1mp" in by_table:
        print("\n## Table 6 — 1-MP LRGB (paper_T6_*)\n")
        _print_pivot(
            by_table["6_1mp"],
            datasets=TABLE6_1MP_DATASETS,
            variants=TABLE6_1MP_VARIANTS,
            expected_n=expected_n,
        )


def _print_pivot(
    aggs: Sequence[ExperimentAgg],
    *,
    datasets: Sequence[str],
    variants: Sequence[str],
    expected_n: int,
) -> None:
    """Print variant × dataset pivot of mean±std cells."""
    index: dict[tuple[str, str], ExperimentAgg] = {
        (a.spec.variant, a.spec.dataset): a for a in aggs
    }
    header = "| Model | " + " | ".join(datasets) + " |"
    sep = "|-------|" + "|".join(["-------"] * len(datasets)) + "|"
    print(header)
    print(sep)
    for variant in variants:
        cells: list[str] = []
        for ds in datasets:
            agg = index.get((variant, ds))
            if agg is None:
                cells.append("—")
            else:
                cells.append(_fmt_cell(agg, expected_n=expected_n))
        print(f"| {variant} | " + " | ".join(cells) + " |")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Aggregate paper Table 5 / Table 6 W&B 5-seed experiment groups.",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--table",
        default="all",
        help="Which tables: all | 5 | 5mc | 6 | 5,5mc,6 (default: all)",
    )
    parser.add_argument("--prefix-t5", default="paper_T5", help="W&B group prefix for Table 5")
    parser.add_argument("--prefix-t6", default="paper_T6", help="W&B group prefix for Table 6")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Summary metric key")
    parser.add_argument(
        "--state",
        default="finished,running,crashed,failed",
        help="States to fetch from W&B (default includes non-finished for progress)",
    )
    parser.add_argument(
        "--score-state",
        default="finished",
        help="States included in mean±std (default: finished only)",
    )
    parser.add_argument("--expected-n", type=int, default=5, help="Expected seeds per group")
    parser.add_argument("--max-runs", type=int, default=50, help="Max runs fetched per group")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print per-seed rows for every group",
    )
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="Only print groups with expected_n scored runs",
    )
    args = parser.parse_args()

    tables = [t.strip() for t in args.table.split(",") if t.strip()]
    if not tables:
        parser.error("--table must be non-empty")

    fetch_states = [s.strip() for s in args.state.split(",") if s.strip()]
    score_states = [s.strip() for s in args.score_state.split(",") if s.strip()]
    specs = build_experiment_specs(
        tables=tables,
        prefix_t5=args.prefix_t5,
        prefix_t6=args.prefix_t6,
    )
    if not specs:
        parser.error(f"No experiment specs for --table {args.table!r}")

    api = wandb.Api()
    aggs: list[ExperimentAgg] = []
    print(
        f"Querying {len(specs)} groups from {args.entity}/{args.project} "
        f"(score_state={score_states}, metric={args.metric})",
        file=sys.stderr,
    )

    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.group}", file=sys.stderr)
        runs = fetch_group_runs(
            api,
            entity=args.entity,
            project=args.project,
            group=spec.group,
            states=fetch_states,
            max_runs=args.max_runs,
        )
        aggs.append(
            aggregate_group(
                spec,
                runs,
                metric_key=args.metric,
                score_states=score_states,
            )
        )

    if args.only_complete:
        aggs = [a for a in aggs if a.n_finished >= args.expected_n]

    print_summary_tables(aggs, expected_n=args.expected_n)
    if args.detail:
        print_detail(aggs, expected_n=args.expected_n)

    n_done = sum(1 for a in aggs if a.n_finished >= args.expected_n)
    n_partial = sum(1 for a in aggs if 0 < a.n_finished < args.expected_n)
    n_empty = sum(1 for a in aggs if a.n_finished == 0)
    print(
        f"\nProgress: {n_done} complete, {n_partial} partial, {n_empty} empty "
        f"(of {len(aggs)} groups; scoring states={score_states})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
