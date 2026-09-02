#!/usr/bin/env python3
"""Paired two-tailed t-tests for GCN/GIN routing benchmark (gated vs baselines).

Reads ``per_run_metrics.csv`` produced by ``analyze_gcn_gin_routing_results.py`` and
writes paired-comparison tables for rebuttal / supplementary analysis.

Example::

    python scripts/synthetic/run_gcn_gin_routing_paired_ttests.py \\
        --metrics-csv results/gcn_gin_routing/analysis/per_run_metrics.csv \\
        --out-dir results/gcn_gin_routing/analysis/paired_ttests
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from scipy import stats

DEFAULT_METRICS_CSV = Path("results/gcn_gin_routing/analysis/per_run_metrics.csv")
DEFAULT_OUT_DIR = Path("results/gcn_gin_routing/analysis/paired_ttests")

METRICS: tuple[str, ...] = ("acc_all", "acc_tau0", "acc_tau1")
METRIC_LABELS: dict[str, str] = {
    "acc_all": "Acc all",
    "acc_tau0": "Acc tau=0 (GCN-type)",
    "acc_tau1": "Acc tau=1 (GIN-type)",
}

BASELINE_MODEL = "a0g2_gated"
BASELINE_LABEL = "SiGMA (gated)"

COMPARISONS: tuple[tuple[str, str], ...] = (
    ("a0g2_ungated", "SiGMA (ungated)"),
    ("a0g1_gcn", "GCN-only"),
    ("a0g1_gin", "GIN-only"),
)

MODEL_ORDER: tuple[str, ...] = (
    "a0g2_gated",
    "a0g2_ungated",
    "a0g1_gcn",
    "a0g1_gin",
)
MODEL_LABELS: dict[str, str] = {
    "a0g2_gated": "SiGMA (gated)",
    "a0g2_ungated": "SiGMA (ungated)",
    "a0g1_gcn": "GCN-only",
    "a0g1_gin": "GIN-only",
}


@dataclass(frozen=True)
class PairedTestResult:
    """Result of one paired two-tailed t-test."""

    track: str
    lr_tag: str
    metric: str
    baseline_model: str
    baseline_label: str
    variant_model: str
    variant_label: str
    n_pairs: int
    seeds: tuple[int, ...]
    mean_baseline_pct: float
    mean_variant_pct: float
    mean_delta_pct: float
    t_stat: float
    p_value: float
    significant_005: bool


def _read_metrics_csv(path: Path) -> list[dict[str, str]]:
    """Load per-run metrics rows from CSV."""
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _seed_metrics(
    rows: Sequence[dict[str, str]],
    *,
    track: str,
    lr_tag: str,
    model: str,
    metric: str,
) -> dict[int, float]:
    """Map seed -> metric (fraction) for one model cell."""
    out: dict[int, float] = {}
    for row in rows:
        if row["track"] != track or row["lr_tag"] != lr_tag or row["model"] != model:
            continue
        out[int(row["seed"])] = float(row[metric])
    return out


def _paired_test(
    baseline: dict[int, float],
    variant: dict[int, float],
    *,
    track: str,
    lr_tag: str,
    metric: str,
    baseline_model: str,
    baseline_label: str,
    variant_model: str,
    variant_label: str,
) -> Optional[PairedTestResult]:
    """Run paired t-test on matching seeds."""
    common = sorted(set(baseline) & set(variant))
    if len(common) < 2:
        return None

    b_vals = [baseline[s] * 100.0 for s in common]
    v_vals = [variant[s] * 100.0 for s in common]
    deltas = [b - v for b, v in zip(b_vals, v_vals, strict=True)]

    if statistics.pstdev(b_vals) == 0.0 and statistics.pstdev(v_vals) == 0.0:
        if all(math.isclose(d, 0.0, abs_tol=1e-12) for d in deltas):
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = float("inf"), 0.0
    else:
        t_stat, p_value = stats.ttest_rel(b_vals, v_vals)
        t_stat = float(t_stat)
        p_value = float(p_value)

    return PairedTestResult(
        track=track,
        lr_tag=lr_tag,
        metric=metric,
        baseline_model=baseline_model,
        baseline_label=baseline_label,
        variant_model=variant_model,
        variant_label=variant_label,
        n_pairs=len(common),
        seeds=tuple(common),
        mean_baseline_pct=statistics.mean(b_vals),
        mean_variant_pct=statistics.mean(v_vals),
        mean_delta_pct=statistics.mean(deltas),
        t_stat=t_stat,
        p_value=p_value,
        significant_005=p_value < 0.05,
    )


def _build_wide_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Pivot to one row per (track, lr_tag, seed) with all main models."""
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row["model"] not in MODEL_ORDER:
            continue
        key = (row["track"], row["lr_tag"], int(row["seed"]))
        if key not in index:
            index[key] = {
                "track": row["track"],
                "lr_tag": row["lr_tag"],
                "seed": int(row["seed"]),
            }
        model = row["model"]
        for metric in METRICS:
            frac = float(row[metric])
            index[key][f"{model}_{metric}"] = frac
            index[key][f"{model}_{metric}_pct"] = frac * 100.0
    return [index[k] for k in sorted(index)]


def _format_p(p: float) -> str:
    """Format p-value for console / LaTeX."""
    if math.isnan(p):
        return "n/a"
    if p < 0.001:
        return f"{p:.2e}"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def _stars(p: float) -> str:
    """Significance stars."""
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def run_all_tests(rows: Sequence[dict[str, str]]) -> list[PairedTestResult]:
    """Compute all configured paired comparisons."""
    tracks = sorted({row["track"] for row in rows})
    lr_tags = sorted({row["lr_tag"] for row in rows})
    results: list[PairedTestResult] = []

    for track in tracks:
        for lr_tag in lr_tags:
            baseline_by_metric = {
                m: _seed_metrics(rows, track=track, lr_tag=lr_tag, model=BASELINE_MODEL, metric=m)
                for m in METRICS
            }
            if not baseline_by_metric["acc_all"]:
                continue
            for variant_model, variant_label in COMPARISONS:
                for metric in METRICS:
                    variant = _seed_metrics(
                        rows,
                        track=track,
                        lr_tag=lr_tag,
                        model=variant_model,
                        metric=metric,
                    )
                    res = _paired_test(
                        baseline_by_metric[metric],
                        variant,
                        track=track,
                        lr_tag=lr_tag,
                        metric=metric,
                        baseline_model=BASELINE_MODEL,
                        baseline_label=BASELINE_LABEL,
                        variant_model=variant_model,
                        variant_label=variant_label,
                    )
                    if res is not None:
                        results.append(res)
    return results


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write dict rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: Sequence[PairedTestResult]) -> None:
    """Print human-readable summary to stdout."""
    print("\n=== GCN/GIN routing paired t-tests (SiGMA gated vs variant) ===\n")
    print(
        f"{'Track':<6} {'LR':<6} {'Metric':<22} {'Variant':<18} "
        f"{'n':>2} {'Δ(pp)':>8} {'t':>7} {'p':>10} {'sig':>5}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r.track:<6} {r.lr_tag:<6} {METRIC_LABELS[r.metric]:<22} {r.variant_label:<18} "
            f"{r.n_pairs:2d} {r.mean_delta_pct:8.3f} {r.t_stat:7.2f} "
            f"{_format_p(r.p_value):>10} {_stars(r.p_value):>5}"
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Paired t-tests for GCN/GIN routing benchmark.")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.metrics_csv.is_file():
        raise SystemExit(f"Metrics CSV not found: {args.metrics_csv}")

    rows = _read_metrics_csv(args.metrics_csv)
    results = run_all_tests(rows)
    wide_rows = _build_wide_rows(rows)

    result_rows = [
        {
            "track": r.track,
            "lr_tag": r.lr_tag,
            "metric": r.metric,
            "metric_label": METRIC_LABELS[r.metric],
            "baseline_model": r.baseline_model,
            "baseline_label": r.baseline_label,
            "variant_model": r.variant_model,
            "variant_label": r.variant_label,
            "n_pairs": r.n_pairs,
            "seeds": ",".join(str(s) for s in r.seeds),
            "mean_baseline_pct": r.mean_baseline_pct,
            "mean_variant_pct": r.mean_variant_pct,
            "mean_delta_pct": r.mean_delta_pct,
            "t_stat": r.t_stat,
            "p_value": r.p_value,
            "significant_005": r.significant_005,
        }
        for r in results
    ]

    out_dir = args.out_dir
    write_csv(out_dir / "routing_paired_ttest_results.csv", result_rows)
    write_csv(out_dir / "routing_paired_wide.csv", wide_rows)

    print_summary(results)
    print(f"\nWrote {out_dir / 'routing_paired_ttest_results.csv'}", file=sys.stderr)
    print(f"Wrote {out_dir / 'routing_paired_wide.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
