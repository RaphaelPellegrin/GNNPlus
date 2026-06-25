#!/usr/bin/env python3
"""Regenerate heterogeneity profile plots from existing GNNPlus graph_dict pickles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from GNNPlus.experiments.track_avg_accuracy import load_and_plot_gnnplus_pickle


def main() -> None:
    """Scan ``results`` for ``*_graph_dict.pickle`` and write by_index / by_accuracy PNGs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory to search recursively for pickles.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    pickle_files = sorted(results_dir.rglob("*_graph_dict.pickle"))
    if not pickle_files:
        print(f"No pickles under {results_dir}")
        sys.exit(1)

    for pickle_path in pickle_files:
        print(f"Plotting {pickle_path}")
        by_index, by_acc = load_and_plot_gnnplus_pickle(
            str(pickle_path),
            output_dir=str(pickle_path.parent.parent),
        )
        print(f"  by_index: {by_index}")
        print(f"  by_accuracy: {by_acc}")


if __name__ == "__main__":
    main()
