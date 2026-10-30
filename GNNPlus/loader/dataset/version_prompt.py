"""Non-interactive-safe prompt when a dataset version tag is missing."""

from __future__ import annotations

import os
import sys


def confirm_dataset_update(class_name: str) -> bool:
    """Return True if the on-disk dataset folder should be deleted and re-downloaded.

    Interactive sessions prompt ``(y/N)``. Batch / SLURM jobs (no TTY) default to
    ``N`` so training does not crash with ``EOFError``. Override with
    ``GNNPLUS_UPDATE_DATASET=1`` (or ``y`` / ``yes`` / ``true``).
    """
    print(f"{class_name} has been updated.")
    env = os.environ.get("GNNPLUS_UPDATE_DATASET", "").strip().lower()
    if env in {"1", "y", "yes", "true"}:
        print("GNNPLUS_UPDATE_DATASET set: removing existing dataset folder.")
        return True
    if not sys.stdin.isatty():
        print(
            "Non-interactive session: keeping existing dataset "
            "(default N; set GNNPLUS_UPDATE_DATASET=1 to force refresh)."
        )
        return False
    return input("Will you update the dataset now? (y/N)\n").lower() == "y"
