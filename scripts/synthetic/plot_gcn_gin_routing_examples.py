#!/usr/bin/env python3
"""Plot curated GCN/GIN routing synthetic graph examples with captions.

Writes one image per graph under ``<out_dir>/examples/`` and an optional combined
grid at ``<out_dir>/fig_example_graphs.png``.

Example:
  python scripts/synthetic/plot_gcn_gin_routing_examples.py
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GNNPlus.loader.dataset.gcn_gin_routing import (  # noqa: E402
    FEAT_SIGNAL,
    build_star_graph,
    curated_example_specs,
    spec_to_metadata,
)

CAPTION_WRAP_WIDTH: int = 88
CAPTION_FONT_SIZE: float = 9.0
CAPTION_LINE_HEIGHT: float = 0.115
TABLE_FONT_SIZE: float = 8.5


@dataclass(frozen=True)
class GraphDrawStyle:
    """Visual sizing for star-graph panels."""

    root_node_size: float = 900.0
    signal_node_size: float = 650.0
    dummy_node_size: float = 220.0
    root_label_font: float = 9.0
    signal_label_font: float = 9.0
    title_font: float = 10.5
    edge_width: float = 1.2
    compact_layout: bool = False


GRAPH_STYLE_DEFAULT = GraphDrawStyle()
GRAPH_STYLE_COMPACT = GraphDrawStyle(
    root_node_size=820.0,
    signal_node_size=580.0,
    dummy_node_size=28.0,
    root_label_font=8.5,
    signal_label_font=6.8,
    title_font=10.0,
    edge_width=1.0,
    compact_layout=True,
)

FEATURE_TABLE_TITLE: str = "Node features"
FEATURE_COL_LABELS: tuple[str, ...] = (
    "Node",
    "Signal feature x",
    "Type channel (2nd dim)",
)
FEATURE_ROWS: tuple[tuple[str, ...], ...] = (
    ("Root r", "0", "τ (0 or 1)"),
    ("Signal neighbors", "+1 or −1", "0"),
    ("Dummy leaves (gray)", "0", "0"),
)

RULE_TABLE_TITLE: str = "Graph type τ — label rules"
RULE_COL_LABELS: tuple[str, ...] = ("τ", "Type", "Label rule")
RULE_ROWS: tuple[tuple[str, ...], ...] = (
    (
        "0",
        "GCN-type",
        r"$y=\mathbb{1}\!\left[\sum_u \dfrac{x_u}{\sqrt{(d_r+1)(d_u+1)}} > 0\right]$",
    ),
    ("1", "GIN-type", r"$y=\mathbb{1}\!\left[\sum_u x_u > 0\right]$"),
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gcn_gin_routing",
        help="Output directory (individual plots -> examples/).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write the multi-panel grid fig_example_graphs.png.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Write only the star-graph panel (no tables/caption).",
    )
    parser.add_argument(
        "--test-idx",
        type=int,
        default=None,
        help="Load test-split graph by index (requires --dataset-dir).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/gcn_gin_routing/data/GcnGinRouting",
        help="GcnGinRouting root when using --test-idx.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Output stem under examples/ (default: auto from --test-idx or slug).",
    )
    parser.add_argument(
        "--example-id",
        type=int,
        default=None,
        help="1-based curated example index (1–7) for --graph-only.",
    )
    return parser.parse_args()


def _slugify(text: str) -> str:
    """Turn a short label into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "example"


def _wrap_caption(caption: str, width: int = CAPTION_WRAP_WIDTH) -> str:
    """Wrap caption text to a fixed character width."""
    return "\n".join(textwrap.wrap(caption, width=width, break_long_words=False))


def _star_layout(
    data,
    *,
    compact: bool = False,
) -> dict[int, tuple[float, float]]:
    """Place root at center, neighbors on a ring, dummies on short spokes."""
    roles = data.node_role.cpu().numpy()
    root = int(data.root_index.item())
    pos: dict[int, tuple[float, float]] = {root: (0.0, 0.0)}

    neighbors = [i for i, r in enumerate(roles) if r == 1]
    k = len(neighbors)
    radius = 1.55 if compact else 1.35
    for j, u in enumerate(neighbors):
        angle = 2.0 * np.pi * j / k
        pos[u] = (radius * np.cos(angle), radius * np.sin(angle))

    dummy_by_neighbor: dict[int, list[int]] = {u: [] for u in neighbors}
    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        if roles[src] == 1 and roles[dst] == 2:
            dummy_by_neighbor[int(src)].append(int(dst))
        elif roles[dst] == 1 and roles[src] == 2:
            dummy_by_neighbor[int(dst)].append(int(src))

    for u, leaves in dummy_by_neighbor.items():
        ux, uy = pos[u]
        base_angle = np.arctan2(uy, ux)
        n_leaves = len(leaves)
        if n_leaves == 0:
            continue

        # Single arc at fixed radius from each signal neighbor (same orbit).
        leaf_radius = 0.55
        arc = min(1.45, max(0.30, 0.17 * n_leaves))
        for li, leaf in enumerate(leaves):
            if n_leaves == 1:
                ang = base_angle
            else:
                ang = base_angle - arc / 2.0 + li * arc / (n_leaves - 1)
            pos[leaf] = (
                ux + leaf_radius * np.cos(ang),
                uy + leaf_radius * np.sin(ang),
            )
    return pos


def _graph_panel_title(data) -> str:
    """Compact title: type, label, and analytic scores."""
    tau = int(data.tau.item())
    y = int(data.y.item())
    gcn_s = float(data.gcn_score.item())
    gin_s = float(data.gin_score.item())
    rule = "GCN" if tau == 0 else "GIN"
    return (
        f"Type τ={tau} ({rule}), y={y}, "
        f"s_GCN={gcn_s:+.3f}, s_GIN={gin_s:+.3f}"
    )


def _draw_graph(
    ax: plt.Axes,
    data,
    *,
    title: str,
    use_compact_title: bool = False,
    style: GraphDrawStyle = GRAPH_STYLE_DEFAULT,
) -> None:
    """Draw the star graph and header (no caption box)."""
    roles = data.node_role.cpu().numpy()
    signals = data.x[:, FEAT_SIGNAL].cpu().numpy()
    tau = int(data.tau.item())
    y = int(data.y.item())
    gcn_s = float(data.gcn_score.item())
    gin_s = float(data.gin_score.item())

    g = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        g.add_edge(int(src), int(dst))

    pos = _star_layout(data, compact=style.compact_layout)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.22 if style.compact_layout else 0.18)

    node_colors: list[str] = []
    node_sizes: list[float] = []
    labels: dict[int, str] = {}
    for i in range(data.num_nodes):
        role = int(roles[i])
        if role == 0:
            node_colors.append("#4C78A8")
            node_sizes.append(style.root_node_size)
            labels[i] = f"r\nτ={tau}"
        elif role == 1:
            node_colors.append("#F58518" if signals[i] > 0 else "#E45756")
            node_sizes.append(style.signal_node_size)
            deg = g.degree[i]
            labels[i] = f"{int(signals[i]):+d}\nd={deg}"
        else:
            node_colors.append("#BAB0AC")
            node_sizes.append(style.dummy_node_size)
            labels[i] = ""

    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        width=style.edge_width,
        alpha=0.65,
        edge_color="#666666",
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.0,
        edgecolors="#333333",
    )
    for i, label in labels.items():
        if not label:
            continue
        role = int(roles[i])
        font_size = style.root_label_font if role == 0 else style.signal_label_font
        x_pos, y_pos = pos[i]
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color="white" if role in (0, 1) else "#333333",
            zorder=4,
        )

    if use_compact_title:
        header = _graph_panel_title(data)
    else:
        rule = "GCN" if tau == 0 else "GIN"
        header = (
            f"{title}\n"
            f"type τ={tau} ({rule})  ·  y={y}  ·  "
            f"s_GCN={gcn_s:+.2f}  ·  s_GIN={gin_s:+.0f}"
        )
    ax.set_title(header, fontsize=style.title_font, fontweight="bold", loc="left", pad=10)


def _caption_height_lines(wrapped: str) -> int:
    """Return number of lines in wrapped caption."""
    return max(1, wrapped.count("\n") + 1)


def _style_table(
    table,
    *,
    header_color: str,
    body_color: str,
    edge_color: str,
) -> None:
    """Apply consistent colors and font to a matplotlib table."""
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold", fontsize=TABLE_FONT_SIZE)
        else:
            cell.set_facecolor(body_color)
            cell.set_text_props(fontsize=TABLE_FONT_SIZE)
        cell.set_height(0.22)


def _draw_bordered_table(
    ax: plt.Axes,
    *,
    title: str,
    col_labels: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    facecolor: str,
    edgecolor: str,
    header_color: str,
    col_widths: tuple[float, ...] | None = None,
    footer: str | None = None,
) -> None:
    """Draw a titled table inside a rounded reference box."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    pad = 0.02
    box_bottom = 0.10 if footer else 0.06
    box_top = 0.90
    bbox = FancyBboxPatch(
        (pad, box_bottom),
        1.0 - 2.0 * pad,
        box_top - box_bottom,
        boxstyle="round,pad=0.012",
        transform=ax.transAxes,
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(bbox)
    ax.text(
        0.5,
        0.94,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=edgecolor,
        zorder=2,
    )

    table_bottom = 0.14 if footer else 0.10
    table = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=list(col_labels),
        loc="center",
        cellLoc="left",
        bbox=[0.05, table_bottom, 0.90, 0.68],
        colWidths=list(col_widths) if col_widths is not None else None,
    )
    table.auto_set_font_size(False)
    table.set_zorder(3)
    _style_table(
        table,
        header_color=header_color,
        body_color="white",
        edge_color=edgecolor,
    )
    table.scale(1.0, 1.65)

    if footer is not None:
        ax.text(
            0.5,
            0.04,
            footer,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#333333",
            zorder=2,
        )


def _draw_node_feature_box(ax: plt.Axes) -> None:
    """Draw the node-feature schema table."""
    _draw_bordered_table(
        ax,
        title=FEATURE_TABLE_TITLE,
        col_labels=FEATURE_COL_LABELS,
        rows=FEATURE_ROWS,
        facecolor="#F4F4F4",
        edgecolor="#666666",
        header_color="#666666",
        col_widths=(0.34, 0.28, 0.38),
    )


def _draw_label_rule_box(ax: plt.Axes) -> None:
    """Draw the τ / label-rule reference table."""
    _draw_bordered_table(
        ax,
        title=RULE_TABLE_TITLE,
        col_labels=RULE_COL_LABELS,
        rows=RULE_ROWS,
        facecolor="#EEF3FA",
        edgecolor="#4C78A8",
        header_color="#4C78A8",
        col_widths=(0.08, 0.20, 0.72),
        footer=(
            r"$d$ on nodes = degree; $\tau$ at root selects which rule applies; "
            r"$y$ is the label the model must predict."
        ),
    )


def _draw_caption_box(ax: plt.Axes, wrapped_caption: str) -> None:
    """Draw caption inside a rounded box that fits the wrapped text."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    n_lines = _caption_height_lines(wrapped_caption)
    pad_x = 0.03
    pad_y = 0.08
    line_h = CAPTION_LINE_HEIGHT
    box_h = min(0.96, pad_y * 2.0 + n_lines * line_h)
    box_y = 0.5 - box_h / 2.0

    bbox = FancyBboxPatch(
        (pad_x, box_y),
        1.0 - 2.0 * pad_x,
        box_h,
        boxstyle="round,pad=0.012",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor="#888888",
        facecolor="#F7F7F7",
        clip_on=False,
    )
    ax.add_patch(bbox)
    ax.text(
        pad_x + 0.02,
        box_y + box_h - pad_y,
        wrapped_caption,
        transform=ax.transAxes,
        fontsize=CAPTION_FONT_SIZE,
        va="top",
        ha="left",
        linespacing=1.25,
    )


def _figure_heights(n_caption_lines: int) -> tuple[float, float, float, float]:
    """Return (fig_height, graph_ratio, ref_ratio, caption_ratio)."""
    caption_ratio = 0.18 + 0.040 * n_caption_lines
    caption_ratio = min(caption_ratio, 0.48)
    ref_ratio = 0.72
    graph_ratio = 1.0
    fig_h = 7.8 + 0.15 * n_caption_lines
    return fig_h, graph_ratio, ref_ratio, caption_ratio


def save_graph_only(
    data,
    *,
    out_path: Path,
    dpi: int,
    style: GraphDrawStyle = GRAPH_STYLE_COMPACT,
) -> None:
    """Save only the star-graph panel (top section of full example figures)."""
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    _draw_graph(ax, data, title="", use_compact_title=True, style=style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_single_example(
    data,
    *,
    title: str,
    caption: str,
    out_path: Path,
    dpi: int,
) -> None:
    """Save one graph + caption to ``out_path``."""
    wrapped = _wrap_caption(caption)
    n_lines = _caption_height_lines(wrapped)
    fig_h, graph_ratio, ref_ratio, caption_ratio = _figure_heights(n_lines)

    fig = plt.figure(figsize=(8.4, fig_h))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[graph_ratio, ref_ratio, caption_ratio],
        width_ratios=[1.0, 1.0],
        hspace=0.14,
        wspace=0.10,
        top=0.97,
        bottom=0.03,
        left=0.04,
        right=0.98,
    )
    ax_graph = fig.add_subplot(gs[0, :])
    ax_feat = fig.add_subplot(gs[1, 0])
    ax_rule = fig.add_subplot(gs[1, 1])
    ax_cap = fig.add_subplot(gs[2, :])
    _draw_graph(ax_graph, data, title=title)
    _draw_node_feature_box(ax_feat)
    _draw_label_rule_box(ax_rule)
    _draw_caption_box(ax_cap, wrapped)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_combined_grid(
    panels: list[tuple[object, str, str]],
    out_path: Path,
    dpi: int,
) -> None:
    """Save all examples in one multi-panel figure."""
    n = len(panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    max_lines = max(
        _caption_height_lines(_wrap_caption(caption)) for _, _, caption in panels
    )
    row_h = 4.2 + 0.16 * max_lines

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13.5, row_h * nrows),
        constrained_layout=False,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (data, title, caption) in zip(axes_flat[:n], panels, strict=True):
        ax.set_aspect("equal")
        ax.axis("off")
        inner = ax.inset_axes([0.0, 0.50, 1.0, 0.48])
        feat_ax = ax.inset_axes([0.0, 0.28, 0.49, 0.20])
        rule_ax = ax.inset_axes([0.51, 0.28, 0.49, 0.20])
        cap_ax = ax.inset_axes([0.0, 0.0, 1.0, 0.26])
        _draw_graph(inner, data, title=title)
        _draw_node_feature_box(feat_ax)
        _draw_label_rule_box(rule_ax)
        _draw_caption_box(cap_ax, _wrap_caption(caption))

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.subplots_adjust(hspace=0.35, wspace=0.10, top=0.96, bottom=0.02)
    fig.suptitle(
        "GCN vs GIN routing synthetic stars — curated examples",
        fontsize=14,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate per-graph figures and optional combined grid."""
    args = _parse_args()
    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"

    if args.test_idx is not None:
        from GNNPlus.loader.dataset.gcn_gin_routing import GcnGinRoutingDataset

        ds = GcnGinRoutingDataset(str(args.dataset_dir), split="test")
        idx = int(args.test_idx)
        if idx < 0 or idx >= len(ds):
            raise ValueError(f"test-idx={idx} out of range for split size {len(ds)}")
        data = ds[idx]
        stem = args.out_name or f"fig_test_{idx:04d}_graph_only"
        out_path = examples_dir / f"{stem}.png"
        if args.graph_only:
            save_graph_only(data, out_path=out_path, dpi=args.dpi)
        else:
            tau = int(data.tau.item())
            y = int(data.y.item())
            save_single_example(
                data,
                title=f"Test graph {idx}",
                caption=(
                    f"Test index {idx}: τ={tau}, y={y}, "
                    f"s_GCN={float(data.gcn_score):+.3f}, "
                    f"s_GIN={float(data.gin_score):+.3f}."
                ),
                out_path=out_path,
                dpi=args.dpi,
            )
        print(f"Wrote {out_path}")
        return

    slugs = [
        "easy_aligned_gcn",
        "easy_gin_vote",
        "medium_degree_imbalance",
        "hard_opposite_sign_gcn",
        "hard_opposite_sign_gin",
        "hard_near_threshold",
        "medium_many_neighbors",
    ]
    specs = curated_example_specs()

    if args.example_id is not None:
        panel_idx = int(args.example_id)
        if panel_idx < 1 or panel_idx > len(specs):
            raise ValueError(f"example-id must be 1..{len(specs)}")
        spec, caption = specs[panel_idx - 1]
        slug = slugs[panel_idx - 1]
        data = build_star_graph(spec)
        stem = args.out_name or f"fig_example_{panel_idx:02d}_{slug}_graph_only"
        out_path = examples_dir / f"{stem}.png"
        if args.graph_only:
            save_graph_only(data, out_path=out_path, dpi=args.dpi)
        else:
            diff = str(spec.difficulty).replace("_", " ").title()
            save_single_example(
                data,
                title=f"Example {panel_idx} — {diff}",
                caption=caption,
                out_path=out_path,
                dpi=args.dpi,
            )
        print(f"Wrote {out_path}")
        return

    slugs = [
        "easy_aligned_gcn",
        "easy_gin_vote",
        "medium_degree_imbalance",
        "hard_opposite_sign_gcn",
        "hard_opposite_sign_gin",
        "hard_near_threshold",
        "medium_many_neighbors",
    ]

    panels: list[tuple[object, str, str]] = []
    for panel_idx, ((spec, caption), slug) in enumerate(
        zip(curated_example_specs(), slugs, strict=True),
        start=1,
    ):
        data = build_star_graph(spec)
        meta = spec_to_metadata(spec)
        diff = str(meta["difficulty"]).replace("_", " ").title()
        title = f"Example {panel_idx} — {diff}"
        panels.append((data, title, caption))

        out_path = examples_dir / f"fig_example_{panel_idx:02d}_{slug}.png"
        if args.graph_only:
            save_graph_only(data, out_path=out_path, dpi=args.dpi)
        else:
            save_single_example(
                data,
                title=title,
                caption=caption,
                out_path=out_path,
                dpi=args.dpi,
            )
        print(f"Wrote {out_path}")

    if args.combined:
        combined = out_dir / "fig_example_graphs.png"
        save_combined_grid(panels, combined, args.dpi)
        print(f"Wrote {combined}")


if __name__ == "__main__":
    main()
