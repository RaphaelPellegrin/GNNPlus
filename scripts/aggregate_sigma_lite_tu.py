"""Aggregate SiGMA-lite TU runs into a Table 17 / 18 style mean ± std summary.

Run directories are ``<out_root>/<dataset>_dh<d_h>_<gate_act>_lr<lr>_seed<seed>``
(as written by ``bash_interface/run_sigma_lite_tu_table17.sh`` and
:func:`GNNPlus.sigma_lite_tu.run_sweep`). See :mod:`GNNPlus.sigma_lite_tu`.

Usage:
    python scripts/aggregate_sigma_lite_tu.py results/sigma_lite_tu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GNNPlus.sigma_lite_tu import summarize  # noqa: E402


def main() -> None:
    """Parse arguments and print the summary."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("out_root", type=Path, help="Directory with one sub-directory per run")
    args = parser.parse_args()
    print(summarize(args.out_root))


if __name__ == "__main__":
    main()
