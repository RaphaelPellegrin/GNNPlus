#!/usr/bin/env python3
"""Batch-plot TU SiGMA homo/hetero per-graph gate dumps.

Expects run dirs from ``run_tu_sigma_homo_hetero.sh``::

  <root>/<ds>_{SiGMA_homo,SiGMA_hetero}_{lr001,lr01}_seed<s>/gate_values_per_graph.pt

Example — paper-table SiGMA hetero (a2g4, best LR per dataset), seed 2::

  python scripts/gate_viz/plot_tu_hh_gates_batch.py \\
    --root $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero \\
    --out_dir $GNNPLUS_OUT_DIR/gate_viz/tu_hh_hetero \\
    --datasets paper --variants SiGMA_hetero \\
    --seeds 2 --prefer-lr best_from_table --color-by-class

Example (local, after rsync)::

  python scripts/gate_viz/plot_tu_hh_gates_batch.py \\
    --root results/tu_sigma_homo_hetero \\
    --out_dir results/gate_viz/tu_hh_hetero \\
    --datasets paper --variants SiGMA_hetero \\
    --seeds 2 --color-by-class
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Prefer table LRs from Paper_tu / W&B aggregate (higher mean when n=5).
# SiGMA hetero TU recipe is a2g4 (2 attn + GCN,GIN,SAGE,GAT), L=12 — not a4g4.
BEST_LR: dict[tuple[str, str], str] = {
    ("mutag", "SiGMA_homo"): "lr001",
    ("mutag", "SiGMA_hetero"): "lr001",
    ("enzymes", "SiGMA_homo"): "lr001",
    ("enzymes", "SiGMA_hetero"): "lr001",
    ("proteins", "SiGMA_homo"): "lr01",
    ("proteins", "SiGMA_hetero"): "lr001",
    ("collab", "SiGMA_homo"): "lr01",
    ("collab", "SiGMA_hetero"): "lr01",
    ("imdb_binary", "SiGMA_homo"): "lr01",
    ("imdb_binary", "SiGMA_hetero"): "lr001",
    ("reddit_binary", "SiGMA_homo"): "lr001",  # provisional until both LRs finish
    ("reddit_binary", "SiGMA_hetero"): "lr001",
    ("nci1", "SiGMA_homo"): "lr001",
    ("nci1", "SiGMA_hetero"): "lr001",
    ("triangles", "SiGMA_homo"): "lr001",
    ("triangles", "SiGMA_hetero"): "lr001",
    ("dd", "SiGMA_homo"): "lr01",  # only n=1 finished; still plot if present
    ("dd", "SiGMA_hetero"): "lr001",
}

# Paper table set (Lukas / PyG stats) first; extras kept for optional plots.
DATASETS_DEFAULT = (
    "mutag",
    "enzymes",
    "proteins",
    "collab",
    "imdb_binary",
    "reddit_binary",
)
DATASETS_ALL = DATASETS_DEFAULT + ("dd", "nci1", "triangles")
VARIANTS_DEFAULT = ("SiGMA_homo", "SiGMA_hetero")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for batch gate plots."""
    parser = argparse.ArgumentParser(
        description="Plot gate_values_per_graph.pt for TU SiGMA homo/hetero runs.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root with <ds>_<variant>_<lr>_seed<s>/gate_values_per_graph.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/gate_viz/tu_hh",
        help="Directory for PNGs (default: results/gate_viz/tu_hh).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="2",
        help="Comma-separated seeds to plot (default: 2).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="paper",
        help=(
            "Comma-separated dataset tags, or 'paper' "
            "(mutag/enzymes/proteins/collab/imdb_binary/reddit_binary), or 'all'."
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="SiGMA_hetero",
        help="Comma-separated variants: SiGMA_homo,SiGMA_hetero (default: hetero only).",
    )
    parser.add_argument(
        "--prefer-lr",
        type=str,
        choices=("best_from_table", "both", "lr001", "lr01"),
        default="best_from_table",
        help="Which LR folders to plot (default: best_from_table).",
    )
    parser.add_argument(
        "--color-by-class",
        action="store_true",
        help="Pass --color-by-class to plot_per_graph_gates.py.",
    )
    parser.add_argument(
        "--sort-head",
        type=int,
        default=1,
        help="Shared-order head index (1=GIN for hetero GCN,GIN,SAGE,GAT).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned plots without running.",
    )
    return parser.parse_args(argv)


def _seeds(spec: str) -> list[int]:
    """Parse comma-separated seed list."""
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _datasets(spec: str) -> tuple[str, ...]:
    """Resolve dataset tag list from CLI."""
    raw = spec.strip().lower()
    if raw == "paper":
        return DATASETS_DEFAULT
    if raw == "all":
        return DATASETS_ALL
    return tuple(x.strip().lower() for x in spec.split(",") if x.strip())


def _variants(spec: str) -> tuple[str, ...]:
    """Resolve variant list from CLI."""
    out: list[str] = []
    for x in spec.split(","):
        t = x.strip()
        if not t:
            continue
        if t not in VARIANTS_DEFAULT:
            raise ValueError(f"Unknown variant {t!r}; expected {VARIANTS_DEFAULT}")
        out.append(t)
    return tuple(out) if out else VARIANTS_DEFAULT


def _lrs_for(ds: str, variant: str, prefer: str) -> list[str]:
    """Resolve which LR tags to plot for one family."""
    if prefer == "both":
        return ["lr001", "lr01"]
    if prefer in ("lr001", "lr01"):
        return [prefer]
    return [BEST_LR.get((ds, variant), "lr001")]


def _iter_targets(
    root: Path,
    datasets: Sequence[str],
    variants: Sequence[str],
    seeds: Iterable[int],
    prefer_lr: str,
) -> list[tuple[Path, Path]]:
    """Return (pt_path, out_subdir) pairs to attempt."""
    del root  # existence checked by caller
    out: list[tuple[Path, Path]] = []
    for ds in datasets:
        for variant in variants:
            for lr in _lrs_for(ds, variant, prefer_lr):
                for seed in seeds:
                    run_dir_name = f"{ds}_{variant}_{lr}_seed{seed}"
                    pt = Path(run_dir_name) / "gate_values_per_graph.pt"
                    out.append((pt, Path(run_dir_name)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Discover dumps and invoke plot_per_graph_gates.py for each."""
    args = _parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    seeds = _seeds(args.seeds)
    datasets = _datasets(args.datasets)
    variants = _variants(args.variants)
    plot_script = Path(__file__).resolve().parent / "plot_per_graph_gates.py"
    if not plot_script.is_file():
        logging.error("Missing %s", plot_script)
        return 1

    rel_targets = _iter_targets(root, datasets, variants, seeds, args.prefer_lr)
    targets = [(root / pt, sub) for pt, sub in rel_targets]
    found = [(pt, sub) for pt, sub in targets if pt.is_file()]
    missing = [(pt, sub) for pt, sub in targets if not pt.is_file()]

    logging.info("Root: %s", root)
    logging.info("Found %d / %d gate dumps", len(found), len(targets))
    for pt, _ in missing:
        logging.warning("Missing: %s", pt)

    if args.dry_run:
        for pt, sub in found:
            logging.info("Would plot %s → %s", pt, out_root / sub)
        return 0

    n_ok = 0
    for pt, sub in found:
        out_dir = out_root / sub
        cmd = [
            sys.executable,
            str(plot_script),
            "--pt",
            str(pt),
            "--out_dir",
            str(out_dir),
            "--sort-mode",
            "shared",
            "--sort-branch",
            "gnn",
            "--sort-layer",
            "-1",
            "--sort-head",
            str(int(args.sort_head)),
        ]
        if args.color_by_class:
            cmd.append("--color-by-class")
        logging.info("Plotting %s", pt)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            n_ok += 1
        else:
            logging.error("Plot failed (%s): %s", proc.returncode, pt)

    logging.info("Done: %d plots ok, %d missing dumps", n_ok, len(missing))
    return 0 if n_ok == len(found) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
