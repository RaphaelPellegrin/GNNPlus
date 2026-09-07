"""Shared track names and labels for GCN/GIN routing Appendix H (+ d_h fill).

Paper tracks: ``toy`` (Track A, d_h=1), ``sigma`` (Track B, d_h=4).
Width-fill stems live under the same ``$GNNPLUS_OUT_DIR/gcn_gin_routing/`` root.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PAPER_TRACKS: tuple[str, ...] = ("toy", "sigma")
DH_FILL_TRACKS: tuple[str, ...] = (
    "toy_dh2",
    "toy_dh3",
    "toy_dh4",
    "sigma_dh1",
    "sigma_dh2",
    "sigma_dh3",
)
TRACK_ORDER: tuple[str, ...] = PAPER_TRACKS + DH_FILL_TRACKS

TRACK_LABELS: dict[str, str] = {
    "toy": r"Track A (Toy, $d_h{=}1$)",
    "sigma": r"Track B (SiGMA, PyG GIN/GCN, $d_h{=}4$)",
    "toy_dh2": r"Toy, $d_h{=}2$",
    "toy_dh3": r"Toy, $d_h{=}3$",
    "toy_dh4": r"Toy, $d_h{=}4$",
    "sigma_dh1": r"SiGMA, $d_h{=}1$",
    "sigma_dh2": r"SiGMA, $d_h{=}2$",
    "sigma_dh3": r"SiGMA, $d_h{=}3$",
}

# Comma-separated defaults for CLI / env (SLURM export uses semicolons).
DEFAULT_TRACKS_CSV: str = ",".join(TRACK_ORDER)
DH_FILL_TRACKS_CSV: str = ",".join(DH_FILL_TRACKS)
PAPER_TRACKS_CSV: str = ",".join(PAPER_TRACKS)


def ordered_tracks(available: Sequence[str] | set[str]) -> list[str]:
    """Return tracks in canonical order, then any extras alphabetically."""
    avail = set(available)
    ordered = [t for t in TRACK_ORDER if t in avail]
    ordered.extend(sorted(avail - set(ordered)))
    return ordered


def subplot_grid(
    n_panels: int,
    *,
    ncols: int = 4,
    col_w: float = 5.5,
    row_h: float = 4.5,
) -> tuple[Figure, list[Axes], int, int]:
    """Create a grid of axes; unused cells are hidden.

    Returns:
        ``(fig, flat_axes, nrows, ncols)`` where ``flat_axes`` has length
        ``n_panels`` (only the used axes).
    """
    if n_panels < 1:
        raise ValueError("n_panels must be >= 1")
    ncols_use = min(max(ncols, 1), n_panels)
    nrows = (n_panels + ncols_use - 1) // ncols_use
    fig, axes_arr = plt.subplots(
        nrows,
        ncols_use,
        figsize=(col_w * ncols_use, row_h * nrows),
        squeeze=False,
    )
    flat_all = [axes_arr[r, c] for r in range(nrows) for c in range(ncols_use)]
    for ax in flat_all[n_panels:]:
        ax.set_visible(False)
    return fig, flat_all[:n_panels], nrows, ncols_use


def track_label(track: str) -> str:
    """Human-readable panel title for a track stem."""
    return TRACK_LABELS.get(track, track)
