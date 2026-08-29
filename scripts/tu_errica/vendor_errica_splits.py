#!/usr/bin/env python3
"""Vendor Errica et al. fixed TU splits from diningphil/gnn-comparison into GNNPlus."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "splits" / "errica"
DEFAULT_SRC = Path("/tmp/gnn-comparison/data_splits")
GNN_COMPARISON_URL = "https://github.com/diningphil/gnn-comparison.git"


def _clone_if_missing(src: Path) -> Path:
    """Return path to gnn-comparison data_splits, cloning if needed."""
    if src.is_dir():
        return src
    clone_root = src.parent / "gnn-comparison"
    if not (clone_root / "data_splits").is_dir():
        print(f"Cloning {GNN_COMPARISON_URL} → {clone_root}")
        subprocess.run(
            ["git", "clone", "--depth", "1", GNN_COMPARISON_URL, str(clone_root)],
            check=True,
        )
    return clone_root / "data_splits"


def vendor_errica_splits(src: Path, dest: Path) -> None:
    """Copy CHEMICAL + COLLABORATIVE_* split JSON trees into ``dest``."""
    src = _clone_if_missing(src)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    n_files = sum(1 for _ in dest.rglob("*.json"))
    print(f"Vendored {n_files} split files → {dest}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help="Path to gnn-comparison/data_splits (cloned if missing)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory (default: splits/errica)",
    )
    args = parser.parse_args()
    vendor_errica_splits(args.src, args.dest)


if __name__ == "__main__":
    main()
    sys.exit(0)
