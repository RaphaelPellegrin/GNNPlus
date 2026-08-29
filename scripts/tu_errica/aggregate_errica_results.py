#!/usr/bin/env python3
"""Aggregate Errica-protocol TU runs from W&B or local run directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# Errica Table 3/4 GIN reference (10-fold, degree social).
ERRICA_GIN_REFERENCE: dict[str, tuple[float, float]] = {
    "ENZYMES": (29.5, 8.2),
    "PROTEINS": (73.3, 4.0),
    "NCI1": (80.0, 1.4),
    "DD": (76.6, 4.3),
    "IMDB-BINARY": (71.2, 3.9),
    "REDDIT-BINARY": (89.9, 1.9),
    "COLLAB": (75.6, 2.3),
}


def _parse_train_meta(meta_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _read_best_test_acc(run_dir: Path) -> float | None:
    """Best-effort parse of test accuracy from stats or wandb offline."""
    stats = run_dir / "stats.json"
    if stats.is_file():
        with stats.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        test_key = "test_accuracy"
        if test_key in payload:
            values = payload[test_key]
            if isinstance(values, list) and values:
                return float(max(values))
    return None


def collect_local_runs(root: Path) -> list[dict[str, Any]]:
    """Scan ``results/tu_errica`` or ``GNNPLUS_OUT_DIR/tu_errica`` tree."""
    rows: list[dict[str, Any]] = []
    for meta_path in root.rglob("train_meta.txt"):
        meta = _parse_train_meta(meta_path)
        run_dir = meta_path.parent
        acc = _read_best_test_acc(run_dir)
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± std of test accuracy per (dataset, model), averaging folds then seeds."""
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        acc = row.get("test_acc")
        if acc is None:
            continue
        bucket[(row["dataset"], row["model"])].append(float(acc))

    summary: list[dict[str, Any]] = []
    for (dataset, model), values in sorted(bucket.items()):
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = var ** 0.5
        else:
            std = 0.0
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/tu_errica"),
        help="Root directory with run subfolders",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path for per-run CSV",
    )
    args = parser.parse_args()
    rows = collect_local_runs(args.root)
    summary = summarize(rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

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


if __name__ == "__main__":
    main()
