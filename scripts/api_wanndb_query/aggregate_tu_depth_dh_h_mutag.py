#!/usr/bin/env python3
"""Back-compat wrapper → ``aggregate_tu_depth_dh_h.py`` (same CLI)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("aggregate_tu_depth_dh_h.py")
    runpy.run_path(str(target), run_name="__main__")
