#!/usr/bin/env python3
"""Aggregate multi-seed W&B repro groups: mean ± std of test@best-val.

Reports ``best/test_*`` (test metric at the epoch that maximized/minimized
``best/val_*``) plus the corresponding ``best/val_*`` for each group.

Example::

    python scripts/api_wanndb_query/aggregate_seed_repro_groups.py \\
        --preset zinc_mal_top

    python scripts/api_wanndb_query/aggregate_seed_repro_groups.py \\
        --group seed_repro_zinc_gine --val-metric best/val_mae --test-metric best/test_mae
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

# (label, wandb_group, val_metric, test_metric)
PRESETS: dict[str, list[tuple[str, str, str, str]]] = {
    "zinc_mal_top": [
        ("zinc_gine", "seed_repro_zinc_gine", "best/val_mae", "best/test_mae"),
        (
            "mal_v4cytwe0",
            "seed_repro_mal_v4cytwe0",
            "best/val_accuracy",
            "best/test_accuracy",
        ),
        (
            "mal_zk6ihqi8",
            "seed_repro_mal_zk6ihqi8",
            "best/val_accuracy",
            "best/test_accuracy",
        ),
        (
            "mal_apiw6l3u",
            "seed_repro_mal_apiw6l3u",
            "best/val_accuracy",
            "best/test_accuracy",
        ),
        (
            "mal_5sx7r420",
            "seed_repro_mal_5sx7r420",
            "best/val_accuracy",
            "best/test_accuracy",
        ),
    ],
}


@dataclass(frozen=True)
class SeedRow:
    """One finished seed with val/test metrics at the val-best checkpoint."""

    run_id: str
    name: str
    seed: Optional[int]
    state: str
    val: float
    test: float


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


def _metric(summary: dict[str, Any], key: str) -> Optional[float]:
    """Return float metric or None if missing."""
    if key not in summary or summary[key] is None:
        return None
    try:
        return float(summary[key])
    except (TypeError, ValueError):
        return None


def fetch_group_runs(
    *,
    entity: str,
    project: str,
    group: str,
    states: Sequence[str],
    max_runs: int,
) -> list[Any]:
    """Fetch W&B runs in a group."""
    api = wandb.Api()
    filters: dict[str, Any] = {
        "group": group,
        "state": {"$in": list(states)},
    }
    path = f"{entity}/{project}"
    return list(api.runs(path, filters=filters, per_page=max_runs, order="-created_at"))


def collect_rows(
    runs: Sequence[Any],
    *,
    val_metric: str,
    test_metric: str,
) -> list[SeedRow]:
    """Build per-seed rows; skip runs missing val or test metric."""
    rows: list[SeedRow] = []
    for run in runs:
        summary = dict(run.summary or {})
        val = _metric(summary, val_metric)
        test = _metric(summary, test_metric)
        if val is None or test is None:
            continue
        rows.append(
            SeedRow(
                run_id=run.id,
                name=str(run.name),
                seed=_parse_seed(run),
                state=str(run.state),
                val=val,
                test=test,
            )
        )
    return rows


def _mean_std(vals: Sequence[float]) -> tuple[float, float]:
    """Sample mean and std (0 if n<2)."""
    mean = float(statistics.mean(vals))
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return mean, std


def print_group_report(
    *,
    label: str,
    group: str,
    val_metric: str,
    test_metric: str,
    rows: list[SeedRow],
    all_runs: Sequence[Any],
) -> None:
    """Print per-seed table and mean ± std for one group."""
    print("=" * 78)
    print(f"{label}  |  group={group}")
    print(f"val={val_metric}  test={test_metric}  (test at best-val checkpoint)")
    print("-" * 78)

    if not rows:
        print(f"No finished runs with both metrics yet (matched {len(all_runs)} run(s)).")
        for run in all_runs:
            summary = dict(run.summary or {})
            has_val = val_metric in summary and summary[val_metric] is not None
            has_test = test_metric in summary and summary[test_metric] is not None
            print(
                f"  state={run.state:<10} id={run.id:<10} "
                f"val?={'y' if has_val else 'n'} test?={'y' if has_test else 'n'}  {run.name}"
            )
        print()
        return

    rows_sorted = sorted(rows, key=lambda r: (-1 if r.seed is None else r.seed))
    print(f"{'seed':>6}  {'val':>12}  {'test':>12}  {'run_id':<10}  name")
    for row in rows_sorted:
        seed_str = str(row.seed) if row.seed is not None else "?"
        print(
            f"{seed_str:>6}  {row.val:12.6f}  {row.test:12.6f}  "
            f"{row.run_id:<10}  {row.name}"
        )

    val_mean, val_std = _mean_std([r.val for r in rows])
    test_mean, test_std = _mean_std([r.test for r in rows])
    print()
    print(f"n={len(rows)}  {val_metric}:  {val_mean:.4f} ± {val_std:.4f}")
    print(f"n={len(rows)}  {test_metric}: {test_mean:.4f} ± {test_std:.4f}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Named bundle of groups (e.g. zinc_mal_top).",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Extra W&B group (repeatable). Requires --val-metric/--test-metric.",
    )
    parser.add_argument("--val-metric", default=None)
    parser.add_argument("--test-metric", default=None)
    parser.add_argument(
        "--states",
        default="finished",
        help="Comma-separated W&B states (default: finished).",
    )
    parser.add_argument("--max-runs", type=int, default=200)
    args = parser.parse_args(list(argv) if argv is not None else None)

    jobs: list[tuple[str, str, str, str]] = []
    if args.preset:
        jobs.extend(PRESETS[args.preset])
    if args.group:
        if not args.val_metric or not args.test_metric:
            print("--group requires --val-metric and --test-metric", file=sys.stderr)
            return 2
        for group in args.group:
            jobs.append((group, group, args.val_metric, args.test_metric))
    if not jobs:
        print("Provide --preset and/or --group", file=sys.stderr)
        return 2

    states = [s.strip() for s in args.states.split(",") if s.strip()]
    for label, group, val_metric, test_metric in jobs:
        runs = fetch_group_runs(
            entity=args.entity,
            project=args.project,
            group=group,
            states=states,
            max_runs=args.max_runs,
        )
        rows = collect_rows(runs, val_metric=val_metric, test_metric=test_metric)
        print_group_report(
            label=label,
            group=group,
            val_metric=val_metric,
            test_metric=test_metric,
            rows=rows,
            all_runs=runs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
