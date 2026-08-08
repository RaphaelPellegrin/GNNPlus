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
        "--sort-mode",
        type=str,
        choices=("shared", "per_panel"),
        default="shared",
        help="shared: one graph order for all panels; per_panel: sort each cell.",
    )
    parser.add_argument(
        "--sort-head",
        type=int,
        default=1,
        help="Shared-order head index (1=GIN for hetero GCN,GIN,SAGE,GAT).",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=("graph", "node", "both"),
        default="graph",
        help=(
            "Which dumps to plot: graph (gate_values_per_graph.pt), "
            "node (gate_values_per_node.pt mean+band + drawings), or both."
        ),
    )
    parser.add_argument(
        "--band",
        type=str,
        default="p10_p90",
        choices=("p10_p90", "p25_p75", "minmax", "std"),
        help="Node-band mode for plot_per_node_gates.py (default: p10_p90).",
    )
    parser.add_argument(
        "--n-draw",
        type=int,
        default=8,
        help="Graphs to draw for node-colored plots (default: 8).",
    )
    parser.add_argument(
        "--skip-draw",
        action="store_true",
        help="Skip network drawings when plotting node dumps.",
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
    datasets: Sequence[str],
    variants: Sequence[str],
    seeds: Iterable[int],
    prefer_lr: str,
    pt_name: str,
) -> list[tuple[Path, Path]]:
    """Return (relative pt_path, out_subdir) pairs to attempt."""
    out: list[tuple[Path, Path]] = []
    for ds in datasets:
        for variant in variants:
            for lr in _lrs_for(ds, variant, prefer_lr):
                for seed in seeds:
                    run_dir_name = f"{ds}_{variant}_{lr}_seed{seed}"
                    pt = Path(run_dir_name) / pt_name
                    out.append((pt, Path(run_dir_name)))
    return out


def _run_graph_plots(
    *,
    root: Path,
    out_root: Path,
    datasets: Sequence[str],
    variants: Sequence[str],
    seeds: Sequence[int],
    prefer_lr: str,
    sort_mode: str,
    sort_head: int,
    color_by_class: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Plot graph-level dumps; return ``(n_ok, n_fail, n_missing)``."""
    plot_script = Path(__file__).resolve().parent / "plot_per_graph_gates.py"
    if not plot_script.is_file():
        logging.error("Missing %s", plot_script)
        return 0, 0, 0

    rel_targets = _iter_targets(
        datasets, variants, seeds, prefer_lr, "gate_values_per_graph.pt"
    )
    targets = [(root / pt, sub) for pt, sub in rel_targets]
    found = [(pt, sub) for pt, sub in targets if pt.is_file()]
    missing = [(pt, sub) for pt, sub in targets if not pt.is_file()]
    logging.info("Graph dumps: found %d / %d", len(found), len(targets))
    for pt, _ in missing:
        logging.warning("Missing: %s", pt)
    if dry_run:
        for pt, sub in found:
            logging.info("Would plot graph %s → %s", pt, out_root / sub)
        return len(found), 0, len(missing)

    n_ok = 0
    n_fail = 0
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
            str(sort_mode),
            "--sort-branch",
            "gnn",
            "--sort-layer",
            "-1",
            "--sort-head",
            str(int(sort_head)),
        ]
        if color_by_class:
            cmd.append("--color-by-class")
        logging.info("Plotting graph %s", pt)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            n_ok += 1
        else:
            n_fail += 1
            logging.error("Plot failed (%s): %s", proc.returncode, pt)
    return n_ok, n_fail, len(missing)


def _run_node_plots(
    *,
    root: Path,
    out_root: Path,
    datasets: Sequence[str],
    variants: Sequence[str],
    seeds: Sequence[int],
    prefer_lr: str,
    sort_head: int,
    color_by_class: bool,
    band: str,
    n_draw: int,
    skip_draw: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Plot node-level dumps; return ``(n_ok, n_fail, n_missing)``."""
    plot_script = Path(__file__).resolve().parent / "plot_per_node_gates.py"
    if not plot_script.is_file():
        logging.error("Missing %s", plot_script)
        return 0, 0, 0

    rel_targets = _iter_targets(
        datasets, variants, seeds, prefer_lr, "gate_values_per_node.pt"
    )
    targets = [(root / pt, sub) for pt, sub in rel_targets]
    found = [(pt, sub) for pt, sub in targets if pt.is_file()]
    missing = [(pt, sub) for pt, sub in targets if not pt.is_file()]
    logging.info("Node dumps: found %d / %d", len(found), len(targets))
    for pt, _ in missing:
        logging.warning("Missing: %s", pt)
    if dry_run:
        for pt, sub in found:
            logging.info("Would plot node %s → %s", pt, out_root / sub)
        return len(found), 0, len(missing)

    n_ok = 0
    n_fail = 0
    for pt, sub in found:
        out_dir = out_root / sub
        cmd = [
            sys.executable,
            str(plot_script),
            "--pt-node",
            str(pt),
            "--out_dir",
            str(out_dir),
            "--band",
            str(band),
            "--sort-branch",
            "gnn",
            "--sort-layer",
            "-1",
            "--sort-head",
            str(int(sort_head)),
            "--draw-branch",
            "gnn",
            "--draw-layer",
            "-1",
            "--draw-head",
            str(int(sort_head)),
            "--n-draw",
            str(int(n_draw)),
        ]
        if color_by_class:
            cmd.append("--color-by-class")
        if skip_draw:
            cmd.append("--skip-draw")
        logging.info("Plotting node %s", pt)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            n_ok += 1
        else:
            n_fail += 1
            logging.error("Plot failed (%s): %s", proc.returncode, pt)
    return n_ok, n_fail, len(missing)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Discover dumps and invoke graph / node plot scripts."""
    args = _parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    seeds = _seeds(args.seeds)
    datasets = _datasets(args.datasets)
    variants = _variants(args.variants)
    level = str(args.level)

    logging.info("Root: %s", root)
    n_ok = 0
    n_fail = 0
    n_missing = 0

    if level in ("graph", "both"):
        ok, fail, miss = _run_graph_plots(
            root=root,
            out_root=out_root,
            datasets=datasets,
            variants=variants,
            seeds=seeds,
            prefer_lr=str(args.prefer_lr),
            sort_mode=str(args.sort_mode),
            sort_head=int(args.sort_head),
            color_by_class=bool(args.color_by_class),
            dry_run=bool(args.dry_run),
        )
        n_ok += ok
        n_fail += fail
        n_missing += miss

    if level in ("node", "both"):
        ok, fail, miss = _run_node_plots(
            root=root,
            out_root=out_root,
            datasets=datasets,
            variants=variants,
            seeds=seeds,
            prefer_lr=str(args.prefer_lr),
            sort_head=int(args.sort_head),
            color_by_class=bool(args.color_by_class),
            band=str(args.band),
            n_draw=int(args.n_draw),
            skip_draw=bool(args.skip_draw),
            dry_run=bool(args.dry_run),
        )
        n_ok += ok
        n_fail += fail
        n_missing += miss

    logging.info(
        "Done: %d plots ok, %d failed, %d missing dumps",
        n_ok,
        n_fail,
        n_missing,
    )
    if args.dry_run:
        return 0
    return 0 if n_fail == 0 and n_ok > 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
