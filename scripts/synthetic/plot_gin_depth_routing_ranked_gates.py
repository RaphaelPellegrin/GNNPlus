#!/usr/bin/env python3
"""Ranked root-gate profiles for GIN depth-routing (layer 0 vs layer 1).

Reads ``gate_graph_summary.csv`` from gated run dirs and plots ranked root γ
colored by τ — one panel per layer.

Example::

  python scripts/synthetic/plot_gin_depth_routing_ranked_gates.py \\
    --results-root results/gin_routing_depth \\
    --out-dir results/gin_routing_depth/analysis/ranked_gates \\
    --lr-tag lr001 --split test
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_NAME_RE = re.compile(
    r"^l2_a0g1_gated_lr(?P<lr_tag>\d+)_seed(?P<seed>\d+)$",
)
TAU_COLORS = {0: "#4C72B0", 1: "#DD8452"}
TAU_LABELS = {0: r"$\tau=0$ (shallow)", 1: r"$\tau=1$ (deep)"}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/gin_routing_depth",
        help="Parent of toy/ (or gates/toy/) with gated run dirs.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Single gate_graph_summary.csv (overrides batch scan).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gin_routing_depth/analysis/ranked_gates",
    )
    parser.add_argument("--track", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def _load_csv_rows(path: Path, split: str) -> list[dict[str, str]]:
    """Load gate_graph_summary rows filtered by split."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("split", "") == split]
    return rows


def _discover_csvs(results_root: Path, track: str, lr_tag: str) -> list[Path]:
    """Find gate_graph_summary.csv under gated runs."""
    candidates = [
        results_root / track,
        results_root / "gates" / track,
        results_root / "toy",
    ]
    found: list[Path] = []
    for base in candidates:
        if not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            match = RUN_NAME_RE.match(run_dir.name)
            if match is None:
                continue
            if f"lr{match.group('lr_tag')}" != lr_tag and match.group("lr_tag") != lr_tag.removeprefix("lr"):
                # Accept both lr001 and 001 forms via exact name prefix.
                if not run_dir.name.startswith(f"l2_a0g1_gated_{lr_tag}_"):
                    continue
            csv_path = run_dir / "gate_graph_summary.csv"
            if csv_path.is_file():
                found.append(csv_path)
    return found


def _plot_ranked(
    rows: Sequence[dict[str, str]],
    out_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    """Two-panel ranked root γ: layer 0 and layer 1."""
    if not rows:
        logging.warning("No rows to plot for %s", out_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, layer_key, layer_name in (
        (axes[0], "layer0_gamma_root", "Layer 0"),
        (axes[1], "layer1_gamma_root", "Layer 1"),
    ):
        for tau in (0, 1):
            subset = [r for r in rows if int(r["tau"]) == tau]
            vals = np.array([float(r[layer_key]) for r in subset], dtype=np.float64)
            if vals.size == 0:
                continue
            order = np.argsort(vals)
            ax.scatter(
                np.arange(vals.size),
                vals[order],
                s=6,
                alpha=0.55,
                color=TAU_COLORS[tau],
                label=f"{TAU_LABELS[tau]} (n={vals.size}, μ={mean(vals):.3f})",
            )
        ax.set_title(layer_name)
        ax.set_xlabel("Graphs ranked by root γ")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel(r"Root MP gate $\gamma$")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_paths = [Path(args.csv)]
    else:
        csv_paths = _discover_csvs(Path(args.results_root), args.track, args.lr_tag)

    if not csv_paths:
        raise SystemExit(
            f"No gate_graph_summary.csv under {args.results_root} "
            f"(track={args.track}, lr={args.lr_tag}). Run dump first.",
        )

    # Per-seed figures + pooled figure.
    pooled: list[dict[str, str]] = []
    for csv_path in csv_paths:
        rows = _load_csv_rows(csv_path, args.split)
        pooled.extend(rows)
        seed_tag = csv_path.parent.name
        out_path = out_dir / f"fig_ranked_gates_{seed_tag}_{args.split}.png"
        _plot_ranked(
            rows,
            out_path,
            title=f"Depth routing · ranked root gates · {seed_tag} · {args.split}",
            dpi=args.dpi,
        )
        logging.info("Wrote %s (%d graphs)", out_path, len(rows))

    pooled_path = out_dir / f"fig_ranked_gates_pooled_{args.lr_tag}_{args.split}.png"
    _plot_ranked(
        pooled,
        pooled_path,
        title=(
            f"Depth routing · ranked root gates · pooled "
            f"({len(csv_paths)} seeds, {args.lr_tag}, {args.split})"
        ),
        dpi=args.dpi,
    )
    paper = out_dir.parent / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    _plot_ranked(
        pooled,
        paper / "fig_ranked_gates_pooled.png",
        title=(
            f"Depth routing · ranked root gates · pooled "
            f"({len(csv_paths)} seeds, {args.lr_tag}, {args.split})"
        ),
        dpi=args.dpi,
    )
    print(f"Wrote {pooled_path}")
    print(f"Wrote {paper / 'fig_ranked_gates_pooled.png'}")


if __name__ == "__main__":
    main()
