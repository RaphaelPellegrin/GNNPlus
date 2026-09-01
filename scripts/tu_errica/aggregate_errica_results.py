#!/usr/bin/env python3
"""Aggregate Errica-protocol TU runs from W&B or local run directories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# Errica Table 3/4 GIN reference (10-fold, degree social) — accuracy in %.
ERRICA_GIN_REFERENCE: dict[str, tuple[float, float]] = {
    "ENZYMES": (29.5, 8.2),
    "PROTEINS": (73.3, 4.0),
    "NCI1": (80.0, 1.4),
    "DD": (76.6, 4.3),
    "IMDB-BINARY": (71.2, 3.9),
    "REDDIT-BINARY": (89.9, 1.9),
    "COLLAB": (75.6, 2.3),
}

ERRICA_DATASETS: list[tuple[str, str]] = [
    ("enzymes", "ENZYMES"),
    ("proteins", "PROTEINS"),
    ("nci1", "NCI1"),
    ("dd", "DD"),
    ("imdb-b", "IMDB-BINARY"),
    ("reddit-b", "REDDIT-BINARY"),
    ("collab", "COLLAB"),
]

ERRICA_MODELS: list[str] = ["GIN", "GraphSAGE", "GCN", "GAT", "SiGMA_hetero"]

WANDB_METRIC_KEYS: tuple[str, ...] = (
    "best_test_perf",
    "best/test_accuracy",
    "best/val_accuracy",
)

DEFAULT_ENTITY = "weber-geoml-harvard-university"
DEFAULT_PROJECT = "GNNPlus"


def _parse_train_meta(meta_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _read_local_test_acc_pct(run_dir: Path) -> float | None:
    """Best-effort test accuracy (%) from local artifacts."""
    stats = run_dir / "stats.json"
    if stats.is_file():
        try:
            payload = json.loads(stats.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("test_accuracy", "accuracy"):
                if key in payload and payload[key] is not None:
                    value = float(payload[key])
                    return value * 100.0 if value <= 1.0 else value
        if isinstance(payload, list):
            best_val: float | None = None
            best_test: float | None = None
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                acc = entry.get("accuracy")
                if acc is None:
                    continue
                acc_f = float(acc)
                val_acc = float(entry.get("val_accuracy", acc_f))
                if best_val is None or val_acc >= best_val:
                    best_val = val_acc
                    best_test = acc_f
            if best_test is not None:
                return best_test * 100.0 if best_test <= 1.0 else best_test

    summary = run_dir / "wandb-summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        for key in WANDB_METRIC_KEYS:
            if key in payload and payload[key] is not None:
                value = float(payload[key])
                return value * 100.0 if value <= 1.0 else value
    return None


def collect_local_runs(root: Path) -> list[dict[str, Any]]:
    """Scan run tree for ``train_meta.txt`` folders."""
    rows: list[dict[str, Any]] = []
    for meta_path in root.rglob("train_meta.txt"):
        meta = _parse_train_meta(meta_path)
        run_dir = meta_path.parent
        acc = _read_local_test_acc_pct(run_dir)
        rows.append(
            {
                "dataset": meta.get("dataset", ""),
                "model": meta.get("model", ""),
                "fold": int(meta.get("fold", "0")),
                "seed": int(meta.get("seed", "0")),
                "hp_id": meta.get("hp_id", "canonical"),
                "campaign": meta.get("campaign", ""),
                "test_acc": acc,
                "run_dir": str(run_dir),
            }
        )
    return rows


def _wandb_group(ds_tag: str, model: str, campaign: str = "canonical") -> str:
    return f"tu_errica_{ds_tag}_{model}_{campaign}_canonical"


def _pick_wandb_metric(summary: dict[str, Any]) -> float | None:
    for key in WANDB_METRIC_KEYS:
        if key in summary and summary[key] is not None:
            value = float(summary[key])
            return value * 100.0 if value <= 1.0 else value
    return None


def collect_wandb_runs(
    *,
    entity: str,
    project: str,
    campaign: str,
    states: Sequence[str],
    max_runs_per_group: int,
) -> list[dict[str, Any]]:
    """Fetch finished Errica canonical runs from W&B."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("wandb required: pip install wandb") from exc

    api = wandb.Api()
    path = f"{entity}/{project}"
    rows: list[dict[str, Any]] = []

    for ds_tag, ds_name in ERRICA_DATASETS:
        for model in ERRICA_MODELS:
            group = _wandb_group(ds_tag, model, campaign)
            filters: dict[str, Any] = {
                "group": group,
                "state": {"$in": list(states)},
            }
            try:
                runs = api.runs(
                    path,
                    filters=filters,
                    per_page=max_runs_per_group,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] W&B query failed for {group}: {exc}", file=sys.stderr)
                continue

            for run in runs:
                summary = dict(run.summary or {})
                acc = _pick_wandb_metric(summary)
                if acc is None:
                    continue
                fold = 0
                seed = 0
                name = str(run.name)
                for token in name.split("_"):
                    if token.startswith("f") and token[1:].isdigit():
                        fold = int(token[1:])
                    if token.startswith("seed") and token[4:].isdigit():
                        seed = int(token[4:])
                rows.append(
                    {
                        "dataset": ds_name,
                        "model": model,
                        "fold": fold,
                        "seed": seed,
                        "hp_id": "canonical",
                        "campaign": campaign,
                        "test_acc": acc,
                        "run_dir": f"wandb:{run.id}",
                        "wandb_group": group,
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± std of test accuracy (%) per (dataset, model)."""
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        acc = row.get("test_acc")
        if acc is None:
            continue
        bucket[(row["dataset"], row["model"])].append(float(acc))

    summary: list[dict[str, Any]] = []
    for (dataset, model), values in sorted(bucket.items()):
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ref = ERRICA_GIN_REFERENCE.get(dataset)
        entry: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "n_runs": len(values),
            "test_acc_mean": round(mean, 2),
            "test_acc_std": round(std, 2),
        }
        if ref and model.upper() == "GIN":
            entry["errica_gin_mean"] = ref[0]
            entry["errica_gin_std"] = ref[1]
            entry["delta_vs_errica"] = round(mean - ref[0], 2)
        summary.append(entry)
    return summary


def print_summary(summary: list[dict[str, Any]], *, n_rows: int, n_with_acc: int) -> None:
    """Print summary table or diagnostics when empty."""
    if not summary:
        print(
            f"No metrics found ({n_rows} run dirs / W&B runs scanned, "
            f"{n_with_acc} with test accuracy).",
            file=sys.stderr,
        )
        print(
            "Local runs usually lack stats.json test metrics — use:\n"
            "  python scripts/tu_errica/aggregate_errica_results.py --source wandb",
            file=sys.stderr,
        )
        return

    print(f"{'dataset':<16} {'model':<14} {'n':>4}  {'mean±std':>12}  {'vs Errica GIN':>14}")
    print("-" * 70)
    for row in summary:
        vs = ""
        if "delta_vs_errica" in row:
            sign = "+" if row["delta_vs_errica"] >= 0 else ""
            vs = f"{sign}{row['delta_vs_errica']:.1f}"
        print(
            f"{row['dataset']:<16} {row['model']:<14} {row['n_runs']:>4}  "
            f"{row['test_acc_mean']:.1f}±{row['test_acc_std']:.1f}  {vs:>14}"
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/tu_errica"),
        help="Root directory with run subfolders (local mode)",
    )
    parser.add_argument(
        "--source",
        choices=("wandb", "local", "auto"),
        default="wandb",
        help="wandb (default): W&B API; local: scan train_meta.txt; auto: local then wandb",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--campaign", default="canonical")
    parser.add_argument(
        "--state",
        default="finished",
        help="Comma-separated W&B run states (wandb/auto)",
    )
    parser.add_argument("--max-runs-per-group", type=int, default=100)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path for per-run CSV",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.source in ("local", "auto"):
        rows = collect_local_runs(args.root)
    if args.source == "wandb" or (args.source == "auto" and not any(r.get("test_acc") for r in rows)):
        states = [s.strip() for s in args.state.split(",") if s.strip()]
        rows = collect_wandb_runs(
            entity=args.entity,
            project=args.project,
            campaign=args.campaign,
            states=states,
            max_runs_per_group=args.max_runs_per_group,
        )

    n_with_acc = sum(1 for r in rows if r.get("test_acc") is not None)
    summary = summarize(rows)

    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print_summary(summary, n_rows=len(rows), n_with_acc=n_with_acc)


if __name__ == "__main__":
    main()
