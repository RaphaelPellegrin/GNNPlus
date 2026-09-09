"""Shared markers for attention-sink heatmaps (out-of-panel arrows)."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory


def annotate_sink_receiver_column(
    ax: Axes,
    sink_j: float,
    *,
    color: str = "red",
    side: str = "right",
) -> None:
    """Mark the argmax-α receiver column with an arrow outside the heatmap.

    The sink is the **key/receiver** column (vertical index in the attention
    matrix). The arrow is drawn in the axes margin so it does not cover colors.

    Args:
        ax: Matplotlib axes with an ``imshow`` attention matrix.
        sink_j: Column index in the same coordinate system as ``imshow``.
        color: Arrow color.
        side: ``right`` (default) or ``left`` margin for the arrow tail.
    """
    blend_data_axes = blended_transform_factory(ax.transData, ax.transAxes)
    blend_axes_axes = blended_transform_factory(ax.transAxes, ax.transAxes)
    if side == "left":
        text_xy = (-0.08, 0.5)
    else:
        text_xy = (1.06, 0.5)
    ax.annotate(
        "",
        xy=(sink_j, 0.5),
        xytext=text_xy,
        xycoords=blend_data_axes,
        textcoords=blend_axes_axes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.1,
            "shrinkA": 0,
            "shrinkB": 2,
            "mutation_scale": 10,
        },
        clip_on=False,
        zorder=10,
    )
