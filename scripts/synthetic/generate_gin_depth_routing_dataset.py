#!/usr/bin/env python3
"""Generate GIN depth-routing synthetic dataset on disk.

Example::

  python scripts/synthetic/generate_gin_depth_routing_dataset.py \\
    --root results/gin_routing_depth/data/GinDepthRouting
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_dataset_class() -> Any:
    """Load ``GinDepthRoutingDataset`` without importing full ``GNNPlus`` package."""
    module_path = _REPO_ROOT / "GNNPlus" / "loader" / "dataset" / "gin_depth_routing.py"
    spec = importlib.util.spec_from_file_location(
        "gin_depth_routing_dataset",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GinDepthRoutingDataset


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="results/gin_routing_depth/data/GinDepthRouting",
        help="Dataset root (GinDepthRouting/ with raw/ and processed/).",
    )
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opposite-sign-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    """Write raw spec and process all splits."""
    args = _parse_args()
    root = Path(args.root)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "seed": args.seed,
        "opposite_sign_fraction": args.opposite_sign_fraction,
    }
    with (raw_dir / "spec.json").open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    for split in ("train", "val", "test"):
        dataset_cls = _load_dataset_class()
        ds = dataset_cls(str(root), split=split)  # type: ignore[arg-type]
        print(f"{split}: {len(ds)} graphs -> {ds.processed_paths[0]}")


if __name__ == "__main__":
    main()
