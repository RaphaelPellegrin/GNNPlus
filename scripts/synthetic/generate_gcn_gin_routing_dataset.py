#!/usr/bin/env python3
"""Generate GCN/GIN routing synthetic dataset on disk.

Example:
  python scripts/synthetic/generate_gcn_gin_routing_dataset.py \\
    --root results/gcn_gin_routing/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GNNPlus.loader.dataset.gcn_gin_routing import GcnGinRoutingDataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="results/gcn_gin_routing/data/GcnGinRouting",
        help="Dataset root (GcnGinRouting/ with raw/ and processed/).",
    )
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=2_000)
    parser.add_argument("--test", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opposite-sign-fraction", type=float, default=0.2)
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
        ds = GcnGinRoutingDataset(str(root), split=split)  # type: ignore[arg-type]
        print(f"{split}: {len(ds)} graphs -> {ds.processed_paths[0]}")


if __name__ == "__main__":
    main()
