#!/usr/bin/env python3
"""Export per-seed Table 5 architecture ablation data for paired t-test analysis.

Fetches W&B ``best_test_perf`` for Full SiGMA baselines and each ablation variant,
then writes CSV files that can be consumed by R, Python, or Excel.

Examples::

    python scripts/api_wanndb_query/export_table5_paired_ttest_data.py

    python scripts/api_wanndb_query/export_table5_paired_ttest_data.py \\
        --out-dir results_summaries/table5_paired_ttest_export

Output files
------------
- ``table5_arch_ablation_long.csv`` — one row per (dataset, model, seed)
- ``table5_arch_ablation_wide.csv`` — one row per (dataset, seed); columns per model
- ``table5_arch_ablation_groups.csv`` — W&B group mapping and metric direction
- ``routing_toy_gated_vs_ungated.csv`` — synthetic Track A toy benchmark (local CSV)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("wandb required: install from pyproject / pip install wandb") from exc

DEFAULT_ENTITY = "weber-geoml-harvard-university"
DEFAULT_PROJECT = "GNNPlus"
DEFAULT_METRIC = "best_test_perf"
DEFAULT_OUT_DIR = Path("results_summaries/table5_paired_ttest_export")

VARIANTS: tuple[str, ...] = (
    "SiGMA",
    "MP_only",
    "Attn_only",
    "SiGMA_attn_gate",
    "SiGMA_ungated_attn",
    "SiGMA_ungated",
)

VARIANT_LABEL: dict[str, str] = {
    "SiGMA": "SiGMA",
    "MP_only": "MP-only",
    "Attn_only": "Attention-only",
    "SiGMA_attn_gate": "SiGMA, ung. MP",
    "SiGMA_ungated_attn": "SiGMA, ung. att.",
    "SiGMA_ungated": "SiGMA, ungated",
}


@dataclass(frozen=True)
class DatasetExportSpec:
    """One benchmark row in Table 5 architecture ablations."""

    key: str
    label: str
    higher_better: bool
    display_scale: float
    display_unit: str
    sigma_wandb_group: str
    variant_wandb_prefix: str
    sigma_extra_wandb_groups: tuple[str, ...] = ()
    skip_variants: frozenset[str] = frozenset()


DATASET_SPECS: tuple[DatasetExportSpec, ...] = (
    DatasetExportSpec(
        key="peptides_func",
        label="Peptides-func",
        higher_better=True,
        display_scale=1.0,
        display_unit="AP",
        sigma_wandb_group="paper_T6_peptides_func_Homog_MP",
        variant_wandb_prefix="paper_T5_peptides_func",
    ),
    DatasetExportSpec(
        key="peptides_struct",
        label="Peptides-struct",
        higher_better=False,
        display_scale=1.0,
        display_unit="MAE",
        sigma_wandb_group="paper_bestmodel_v2_peptides_struct_g3bsaq32_b7m0_ep250",
        variant_wandb_prefix="paper_T5_peptides_struct",
    ),
    DatasetExportSpec(
        key="voc",
        label="PascalVOC-SP",
        higher_better=True,
        display_scale=1.0,
        display_unit="F1",
        sigma_wandb_group="paper_bestmodel_v1_voc_j7ukyzdm",
        variant_wandb_prefix="paper_T5_voc",
    ),
    DatasetExportSpec(
        key="mnist",
        label="MNIST",
        higher_better=True,
        display_scale=100.0,
        display_unit="accuracy_pct",
        sigma_wandb_group="paper_bestmodel_v1_mnist_lcvbyyss",
        variant_wandb_prefix="paper_T5_mnist",
    ),
    DatasetExportSpec(
        key="cifar10",
        label="CIFAR10",
        higher_better=True,
        display_scale=100.0,
        display_unit="accuracy_pct",
        sigma_wandb_group="paper_bestmodel_v1_cifar10_ulij45a2",
        variant_wandb_prefix="paper_T5_cifar10",
    ),
    DatasetExportSpec(
        key="pattern",
        label="PATTERN",
        higher_better=True,
        display_scale=100.0,
        display_unit="accuracy_pct",
        sigma_wandb_group="paper_sigma_grit_attn_pattern_vn4",
        variant_wandb_prefix="paper_T5_pattern_gritvn4",
        sigma_extra_wandb_groups=("paper_T5_pattern_gritvn4_SiGMA",),
    ),
    DatasetExportSpec(
        key="cluster",
        label="CLUSTER",
        higher_better=True,
        display_scale=100.0,
        display_unit="accuracy_pct",
        sigma_wandb_group="paper_bestmodel_v1_cluster_ht9bntg2",
        variant_wandb_prefix="paper_T5_cluster",
        sigma_extra_wandb_groups=("paper_T5_cluster_SiGMA",),
        skip_variants=frozenset({"SiGMA_ungated_attn"}),
    ),
)


@dataclass(frozen=True)
class SeedMetricRow:
    """One finished run with a parsed seed and metric."""

    seed: int
    metric_raw: float
    metric_display: float
    run_id: str
    run_name: str
    wandb_group: str


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


def _run_priority(run_name: str) -> int:
    """Prefer primary runs over H200 retries when deduplicating seeds."""
    name = run_name.lower()
    return (2 if "_h200" in run_name else 0) + (1 if "retry" in name else 0)


def fetch_group_seed_metrics(
    api: Any,
    *,
    entity: str,
    project: str,
    group: str,
    metric_key: str,
    display_scale: float,
    higher_better: bool,
    seed_filter: Optional[set[int]] = None,
) -> dict[int, SeedMetricRow]:
    """Fetch one metric per seed for a W&B group."""
    filters: dict[str, Any] = {"group": group, "state": "finished"}
    runs = list(api.runs(f"{entity}/{project}", filters=filters, per_page=100, order="-created_at"))
    by_seed: dict[int, SeedMetricRow] = {}

    for run in runs:
        seed = _parse_seed(run)
        summary = dict(run.summary or {})
        if seed is None or metric_key not in summary or summary[metric_key] is None:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue

        metric_raw = float(summary[metric_key])
        row = SeedMetricRow(
            seed=seed,
            metric_raw=metric_raw,
            metric_display=metric_raw * display_scale,
            run_id=str(run.id),
            run_name=str(run.name or ""),
            wandb_group=group,
        )
        if seed not in by_seed:
            by_seed[seed] = row
            continue

        old = by_seed[seed]
        old_prio = _run_priority(old.run_name)
        new_prio = _run_priority(row.run_name)
        if new_prio < old_prio:
            by_seed[seed] = row
        elif new_prio == old_prio:
            if higher_better and row.metric_raw > old.metric_raw:
                by_seed[seed] = row
            elif not higher_better and row.metric_raw < old.metric_raw:
                by_seed[seed] = row

    return by_seed


def build_long_rows(
    api: Any,
    *,
    entity: str,
    project: str,
    metric_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build long-format rows and group-metadata rows."""
    long_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for spec in DATASET_SPECS:
        seed_filter: Optional[set[int]] = None
        if spec.key == "peptides_struct":
            # Match the 5-seed Table 5 cohort (seeds 0-4).
            seed_filter = {0, 1, 2, 3, 4}

        sigma_by_seed = fetch_group_seed_metrics(
            api,
            entity=entity,
            project=project,
            group=spec.sigma_wandb_group,
            metric_key=metric_key,
            display_scale=spec.display_scale,
            higher_better=spec.higher_better,
            seed_filter=seed_filter,
        )
        for extra_group in spec.sigma_extra_wandb_groups:
            extra = fetch_group_seed_metrics(
                api,
                entity=entity,
                project=project,
                group=extra_group,
                metric_key=metric_key,
                display_scale=spec.display_scale,
                higher_better=spec.higher_better,
                seed_filter=seed_filter,
            )
            sigma_by_seed.update(extra)
        sigma_groups = ", ".join((spec.sigma_wandb_group, *spec.sigma_extra_wandb_groups))
        group_rows.append(
            {
                "dataset_key": spec.key,
                "dataset_label": spec.label,
                "model_key": "SiGMA",
                "model_label": VARIANT_LABEL["SiGMA"],
                "is_sigma_baseline": True,
                "higher_better": spec.higher_better,
                "display_unit": spec.display_unit,
                "display_scale": spec.display_scale,
                "wandb_group": sigma_groups,
                "wandb_metric": metric_key,
            }
        )
        for seed, row in sorted(sigma_by_seed.items()):
            long_rows.append(
                {
                    "dataset_key": spec.key,
                    "dataset_label": spec.label,
                    "model_key": "SiGMA",
                    "model_label": VARIANT_LABEL["SiGMA"],
                    "is_sigma_baseline": True,
                    "higher_better": spec.higher_better,
                    "display_unit": spec.display_unit,
                    "seed": seed,
                    "metric_raw": row.metric_raw,
                    "metric_display": row.metric_display,
                    "wandb_group": row.wandb_group,
                    "wandb_run_id": row.run_id,
                    "wandb_run_name": row.run_name,
                }
            )

        for variant in VARIANTS:
            if variant == "SiGMA" or variant in spec.skip_variants:
                continue
            group = f"{spec.variant_wandb_prefix}_{variant}"
            group_rows.append(
                {
                    "dataset_key": spec.key,
                    "dataset_label": spec.label,
                    "model_key": variant,
                    "model_label": VARIANT_LABEL[variant],
                    "is_sigma_baseline": False,
                    "higher_better": spec.higher_better,
                    "display_unit": spec.display_unit,
                    "display_scale": spec.display_scale,
                    "wandb_group": group,
                    "wandb_metric": metric_key,
                }
            )
            var_by_seed = fetch_group_seed_metrics(
                api,
                entity=entity,
                project=project,
                group=group,
                metric_key=metric_key,
                display_scale=spec.display_scale,
                higher_better=spec.higher_better,
            )
            for seed, row in sorted(var_by_seed.items()):
                long_rows.append(
                    {
                        "dataset_key": spec.key,
                        "dataset_label": spec.label,
                        "model_key": variant,
                        "model_label": VARIANT_LABEL[variant],
                        "is_sigma_baseline": False,
                        "higher_better": spec.higher_better,
                        "display_unit": spec.display_unit,
                        "seed": seed,
                        "metric_raw": row.metric_raw,
                        "metric_display": row.metric_display,
                        "wandb_group": row.wandb_group,
                        "wandb_run_id": row.run_id,
                        "wandb_run_name": row.run_name,
                    }
                )

    return long_rows, group_rows


def build_wide_rows(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot long rows to one row per (dataset, seed) for paired comparisons."""
    by_dataset_seed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in long_rows:
        key = (row["dataset_key"], row["dataset_label"], int(row["seed"]))
        if key not in by_dataset_seed:
            by_dataset_seed[key] = {
                "dataset_key": row["dataset_key"],
                "dataset_label": row["dataset_label"],
                "seed": int(row["seed"]),
                "higher_better": row["higher_better"],
                "display_unit": row["display_unit"],
            }
        model_key = row["model_key"]
        by_dataset_seed[key][f"{model_key}_metric_display"] = row["metric_display"]
        by_dataset_seed[key][f"{model_key}_metric_raw"] = row["metric_raw"]
        by_dataset_seed[key][f"{model_key}_wandb_run_id"] = row["wandb_run_id"]

    wide_rows = [by_dataset_seed[k] for k in sorted(by_dataset_seed)]
    return wide_rows


def export_routing_toy_csv(
    out_path: Path,
    *,
    repo_root: Path,
) -> bool:
    """Copy synthetic toy routing per-seed metrics if available."""
    source = repo_root / "results/gcn_gin_routing/analysis/per_run_metrics.csv"
    if not source.exists():
        return False

    with source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, Any]] = []
        for row in reader:
            if row.get("track") != "toy" or row.get("lr_tag") != "lr001":
                continue
            if row.get("model") not in {"a0g2_gated", "a0g2_ungated"}:
                continue
            model = row["model"]
            row["model_label"] = "SiGMA (gated)" if model == "a0g2_gated" else "SiGMA (ungated)"
            acc_all = float(row["acc_all"])
            acc_tau0 = float(row["acc_tau0"])
            acc_tau1 = float(row["acc_tau1"])
            row["acc_all_pct"] = acc_all * 100.0
            row["acc_tau0_pct"] = acc_tau0 * 100.0
            row["acc_tau1_pct"] = acc_tau1 * 100.0
            rows.append(row)

    if not rows:
        return False

    rows.sort(key=lambda r: (r["model"], int(r["seed"])))
    cols = [
        "track",
        "model",
        "model_label",
        "lr_tag",
        "seed",
        "acc_all",
        "acc_all_pct",
        "acc_tau0",
        "acc_tau0_pct",
        "acc_tau1",
        "acc_tau1_pct",
        "run_dir",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dict rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Export per-seed Table 5 ablation data for paired t-tests.",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root for local routing benchmark CSV",
    )
    args = parser.parse_args()

    api = wandb.Api()
    print(
        f"Querying W&B {args.entity}/{args.project} (metric={args.metric})",
        file=sys.stderr,
    )
    long_rows, group_rows = build_long_rows(
        api,
        entity=args.entity,
        project=args.project,
        metric_key=args.metric,
    )
    wide_rows = build_wide_rows(long_rows)

    out_dir = args.out_dir
    long_path = out_dir / "table5_arch_ablation_long.csv"
    wide_path = out_dir / "table5_arch_ablation_wide.csv"
    groups_path = out_dir / "table5_arch_ablation_groups.csv"
    routing_path = out_dir / "routing_toy_gated_vs_ungated.csv"

    write_csv(long_path, long_rows)
    write_csv(wide_path, wide_rows)
    write_csv(groups_path, group_rows)

    routing_ok = export_routing_toy_csv(routing_path, repo_root=args.repo_root)

    print(f"Wrote {long_path} ({len(long_rows)} rows)", file=sys.stderr)
    print(f"Wrote {wide_path} ({len(wide_rows)} rows)", file=sys.stderr)
    print(f"Wrote {groups_path} ({len(group_rows)} rows)", file=sys.stderr)
    if routing_ok:
        print(f"Wrote {routing_path}", file=sys.stderr)
    else:
        print(f"Skipped {routing_path} (source per_run_metrics.csv not found)", file=sys.stderr)

    print("\nPaired t-test hint (Python):")
    print("  wide = pandas.read_csv('table5_arch_ablation_wide.csv')")
    print("  scipy.stats.ttest_rel(wide['SiGMA_metric_display'], wide['SiGMA_ungated_metric_display'])")


if __name__ == "__main__":
    main()
