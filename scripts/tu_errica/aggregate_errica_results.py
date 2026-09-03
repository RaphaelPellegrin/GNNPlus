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

# Errica et al. ICLR 2020 Tables 3/4 — published GIN mean±std (accuracy %).
# Chemical: Table 3 GIN row. Social: Table 4 "With Degree" GIN row.
# (Previously ENZYMES/DD were mis-copied from ECC/DGCNN.)
ERRICA_GIN_REFERENCE: dict[str, tuple[float, float]] = {
    "ENZYMES": (59.6, 4.5),
    "PROTEINS": (73.3, 4.0),
    "NCI1": (80.0, 1.4),
    "DD": (75.3, 2.9),
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
    """Build W&B group matching ``run_tu_errica_fair.sh`` naming."""
    if campaign in ("grid_eval", "sigma_grid_eval", "sigma_grid_eval_fixed8"):
        hp_tag = "selected"
    elif campaign == "canonical":
        hp_tag = "canonical"
    else:
        hp_tag = campaign
    return f"tu_errica_{ds_tag}_{model}_{campaign}_{hp_tag}"


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
    models: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch finished Errica runs from W&B for ``campaign``."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("wandb required: pip install wandb") from exc

    api = wandb.Api()
    path = f"{entity}/{project}"
    rows: list[dict[str, Any]] = []
    model_list = list(models) if models is not None else list(ERRICA_MODELS)

    for ds_tag, ds_name in ERRICA_DATASETS:
        for model in model_list:
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
                        "hp_id": "selected" if "selected" in group else "canonical",
                        "campaign": campaign,
                        "test_acc": acc,
                        "run_dir": f"wandb:{run.id}",
                        "wandb_group": group,
                    }
                )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    *,
    fold_then_seed: bool = True,
) -> list[dict[str, Any]]:
    """Mean ± std of test accuracy (%) per (dataset, model).

    If ``fold_then_seed`` (Errica-style), average seeds within each fold first,
    then report mean±std over folds. Otherwise pool all runs.
    """
    if fold_then_seed:
        fold_bucket: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in rows:
            acc = row.get("test_acc")
            if acc is None:
                continue
            fold_bucket[(row["dataset"], row["model"], int(row["fold"]))].append(
                float(acc)
            )
        bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (dataset, model, _fold), values in fold_bucket.items():
            bucket[(dataset, model)].append(statistics.mean(values))
    else:
        bucket = defaultdict(list)
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
            "n_folds": len(values),
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


def format_latex_table(
    summary: list[dict[str, Any]],
    *,
    columns: Sequence[str],
    caption: str,
    label: str,
) -> str:
    """Build a LaTeX table; missing (dataset, model) cells are ``--``."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        (row["dataset"], row["model"]): row for row in summary
    }
    datasets = [name for _, name in ERRICA_DATASETS]
    col_spec = "l" + "c" * len(columns)
    header = "Dataset & " + " & ".join(columns) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for dataset in datasets:
        cells: list[str] = [dataset]
        best_mean = -1.0
        best_models: list[str] = []
        for model in columns:
            row = by_key.get((dataset, model))
            if row is None:
                continue
            if row["test_acc_mean"] > best_mean + 1e-9:
                best_mean = float(row["test_acc_mean"])
                best_models = [model]
            elif abs(row["test_acc_mean"] - best_mean) <= 1e-9:
                best_models.append(model)
        for model in columns:
            row = by_key.get((dataset, model))
            if row is None:
                cells.append("--")
                continue
            cell = f"{row['test_acc_mean']:.2f}" r"{\pm}" f"{row['test_acc_std']:.2f}"
            if model in best_models:
                cell = r"$\mathbf{" + cell + "}$"
            else:
                cell = f"${cell}$"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(
    summary: list[dict[str, Any]], *, n_rows: int, n_with_acc: int
) -> None:
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

    print(
        f"{'dataset':<16} {'model':<14} {'n':>4}  {'mean±std':>12}  {'vs Errica GIN':>14}"
    )
    print("-" * 70)
    for row in summary:
        vs = ""
        if "delta_vs_errica" in row:
            sign = "+" if row["delta_vs_errica"] >= 0 else ""
            vs = f"{sign}{row['delta_vs_errica']:.1f}"
        n_val = row.get("n_folds", row.get("n_runs", 0))
        print(
            f"{row['dataset']:<16} {row['model']:<14} {n_val:>4}  "
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
        "--models",
        default="",
        help="Comma-separated model tags (default: all). " "Example: GIN,GraphSAGE,GCN",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Comma-separated W&B run states (wandb/auto)",
    )
    parser.add_argument("--max-runs-per-group", type=int, default=100)
    parser.add_argument(
        "--pool-runs",
        action="store_true",
        help="Pool all seeds/folds instead of Errica fold-then-seed aggregation",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path for per-run CSV",
    )
    parser.add_argument(
        "--out-latex",
        type=Path,
        default=None,
        help="Optional path for LaTeX table",
    )
    parser.add_argument(
        "--latex-columns",
        default="GCN,GIN,GraphSAGE,GAT,SiGMA_hetero",
        help="Comma-separated column order for --out-latex",
    )
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",") if m.strip()] or None

    rows: list[dict[str, Any]] = []
    if args.source in ("local", "auto"):
        rows = collect_local_runs(args.root)
    if args.source == "wandb" or (
        args.source == "auto" and not any(r.get("test_acc") for r in rows)
    ):
        states = [s.strip() for s in args.state.split(",") if s.strip()]
        rows = collect_wandb_runs(
            entity=args.entity,
            project=args.project,
            campaign=args.campaign,
            states=states,
            max_runs_per_group=args.max_runs_per_group,
            models=model_filter,
        )

    n_with_acc = sum(1 for r in rows if r.get("test_acc") is not None)
    summary = summarize(rows, fold_then_seed=not args.pool_runs)

    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print_summary(summary, n_rows=len(rows), n_with_acc=n_with_acc)

    if args.out_latex is not None:
        columns = [c.strip() for c in args.latex_columns.split(",") if c.strip()]
        caption = (
            "TU graph classification test accuracy (\\%). "
            "Errica et al.\\ protocol: 10-fold CV with per-fold HP selection; "
            "mean$\\pm$std over folds after averaging 3 seeds per fold. "
            "Missing columns (--) are not yet available for this campaign."
        )
        latex = format_latex_table(
            summary,
            columns=columns,
            caption=caption,
            label="tab:tu_errica_grid_eval",
        )
        args.out_latex.parent.mkdir(parents=True, exist_ok=True)
        args.out_latex.write_text(latex, encoding="utf-8")
        print(f"Wrote {args.out_latex}", file=sys.stderr)
        print(latex)


if __name__ == "__main__":
    main()
