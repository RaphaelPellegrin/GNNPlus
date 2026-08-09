#!/usr/bin/env python3
"""Plot per-node SiGMA gates: ranked mean+band profiles and colored graphs.

Requires ``gate_values_per_node.pt`` from::

  python scripts/gate_viz/dump_per_graph_gates.py --level both|node ...

1) Ranked profiles (same layout as ``plot_per_graph_gates.py``):
   for each graph, plot graph-mean γ and a band over node γ (percentiles).

2) Graph drawings: nodes colored by γ at one (L, head), selected by
   mean extremes and/or highest within-graph variance.

3) Dirichlet energy of the **gate field** γ on each graph (needs edges)::

     Dir(γ) = ½ mean_{(i,j)∈E} (γ_i − γ_j)²

   This is layer×head specific (same γ used for coloring). It is *not* the
   embedding Dirichlet logged in training (``batch.x`` after the forward).

Example::

  python scripts/gate_viz/plot_per_node_gates.py \\
    --pt-node path/to/gate_values_per_node.pt \\
    --out_dir results/gate_viz/tu_hh_hetero/mutag_SiGMA_hetero_lr001_seed2 \\
    --color-by-class \\
    --draw-select mean_extremes,high_var
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import colormaps

# Same-directory import when invoked as ``python scripts/gate_viz/...``.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_per_graph_gates import (  # noqa: E402
    _class_colors,
    _head_labels,
    resolve_sort_layer,
    shared_graph_order,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for per-node gate plots."""
    parser = argparse.ArgumentParser(
        description="Plot per-node gate bands and node-colored graphs.",
    )
    parser.add_argument(
        "--pt-node",
        type=str,
        required=True,
        help="Path to gate_values_per_node.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (default: <pt_parent>/plots_node).",
    )
    parser.add_argument(
        "--color-by-class",
        action="store_true",
        help="Color mean markers / band tint by graph label when available.",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="p10_p90",
        choices=("p10_p90", "p25_p75", "minmax", "std"),
        help="Node-band summary within each graph (default: p10_p90).",
    )
    parser.add_argument(
        "--sort-branch",
        type=str,
        choices=("gnn", "attn"),
        default="gnn",
        help="Branch for shared graph ranking (default: gnn).",
    )
    parser.add_argument(
        "--sort-layer",
        type=int,
        default=-1,
        help="Layer for shared ranking (-1 = last).",
    )
    parser.add_argument(
        "--sort-head",
        type=int,
        default=1,
        help="Head for shared ranking (default: 1 = GIN for a2g4 hetero).",
    )
    parser.add_argument(
        "--draw-branch",
        type=str,
        choices=("gnn", "attn"),
        default="gnn",
        help="Branch used for node-colored graph drawings.",
    )
    parser.add_argument(
        "--draw-layer",
        type=int,
        default=-1,
        help="Layer for graph drawings (-1 = last).",
    )
    parser.add_argument(
        "--draw-head",
        type=int,
        default=1,
        help="Head for graph drawings (default: 1).",
    )
    parser.add_argument(
        "--n-draw",
        type=int,
        default=8,
        help="Number of graphs to draw per selection mode (default: 8).",
    )
    parser.add_argument(
        "--draw-select",
        type=str,
        default="mean_extremes,high_var",
        help=(
            "Comma-separated drawing selectors: "
            "mean_extremes (top/bottom mean γ), high_var (largest within-graph "
            "std of γ). Default: both."
        ),
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG dpi.")
    parser.add_argument(
        "--skip-draw",
        action="store_true",
        help="Only write mean+band / Dirichlet grids (no network drawings).",
    )
    parser.add_argument(
        "--skip-dirichlet",
        action="store_true",
        help="Skip Dirichlet-energy ranked grids.",
    )
    return parser.parse_args(argv)


def _graph_means_from_nodes(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    n_graphs: int,
) -> np.ndarray:
    """Mean-pool node gates ``[N, L, H]`` → graph means ``[G, L, H]``."""
    n_layers, n_heads = int(node_vals.shape[1]), int(node_vals.shape[2])
    out = np.zeros((n_graphs, n_layers, n_heads), dtype=np.float64)
    for g in range(n_graphs):
        a, b = int(ptr[g]), int(ptr[g + 1])
        if b > a:
            out[g] = node_vals[a:b].mean(axis=0)
    return out


def _band_for_nodes(node_col: np.ndarray, mode: str) -> Tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` for one graph's node gate vector."""
    mean = float(node_col.mean()) if node_col.size else 0.0
    if node_col.size == 0:
        return mean, mean, mean
    if mode == "p10_p90":
        lo, hi = np.percentile(node_col, [10.0, 90.0])
    elif mode == "p25_p75":
        lo, hi = np.percentile(node_col, [25.0, 75.0])
    elif mode == "minmax":
        lo, hi = float(node_col.min()), float(node_col.max())
    elif mode == "std":
        std = float(node_col.std())
        lo, hi = mean - std, mean + std
    else:
        raise ValueError(f"Unknown band mode {mode!r}")
    return mean, float(lo), float(hi)


def plot_branch_bands(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    graph_means: np.ndarray,
    *,
    kind: str,
    head_names: Sequence[str],
    labels: Optional[np.ndarray],
    color_by_class: bool,
    order: np.ndarray,
    sort_key_label: str,
    band_mode: str,
    title: str,
    out_path: Path,
    dpi: int,
    ref_branch: Optional[str],
    ref_layer: Optional[int],
    ref_head: Optional[int],
) -> None:
    """Write ``L × H`` grid: graph-mean markers + within-graph node bands."""
    n_graphs, n_layers, n_heads = graph_means.shape
    fig_w = max(3.2 * n_heads, 10.0)
    fig_h = max(2.2 * n_layers, 8.0)
    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    use_class = bool(color_by_class and labels is not None)
    unique_classes: list[int] = []
    colors: list[str] = []
    if use_class:
        assert labels is not None
        unique_classes = sorted(int(c) for c in np.unique(labels))
        colors = _class_colors(len(unique_classes))

    ranks = np.arange(n_graphs)
    xlabel = f"Rank ({sort_key_label})"

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            means = np.zeros(n_graphs, dtype=np.float64)
            los = np.zeros(n_graphs, dtype=np.float64)
            his = np.zeros(n_graphs, dtype=np.float64)
            for rank_i, g in enumerate(order):
                a, b = int(ptr[g]), int(ptr[g + 1])
                node_col = node_vals[a:b, layer, head]
                _, lo, hi = _band_for_nodes(node_col, band_mode)
                # Prefer stored graph mean when available (matches prior plots).
                means[rank_i] = float(graph_means[g, layer, head])
                los[rank_i] = lo
                his[rank_i] = hi

            ax.fill_between(
                ranks,
                los,
                his,
                color="#9E9E9E",
                alpha=0.35,
                linewidth=0.0,
                label="node band" if layer == 0 and head == 0 else None,
            )
            if use_class:
                assert labels is not None
                y_lab = labels[order]
                for ci, cls in enumerate(unique_classes):
                    mask = y_lab == cls
                    ax.scatter(
                        ranks[mask],
                        means[mask],
                        s=12,
                        alpha=0.85,
                        c=colors[ci],
                        edgecolors="none",
                        zorder=3,
                        label=f"c{cls}" if layer == 0 and head == 0 else None,
                    )
            else:
                ax.scatter(
                    ranks,
                    means,
                    s=10,
                    alpha=0.75,
                    c="#4C72B0",
                    edgecolors="none",
                    zorder=3,
                    label="graph mean" if layer == 0 and head == 0 else None,
                )

            if (
                ref_branch == kind
                and ref_layer == layer
                and ref_head == head
            ):
                for spine in ax.spines.values():
                    spine.set_color("#C44E52")
                    spine.set_linewidth(1.5)

            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, linestyle="--")
            if layer == 0:
                ax.set_title(head_names[head], fontsize=10, fontweight="bold")
            if head == 0:
                ax.set_ylabel(f"L{layer}\nγ", fontsize=9)
            if layer == n_layers - 1:
                ax.set_xlabel(xlabel, fontsize=7)
            ax.text(
                0.98,
                0.05,
                f"μ={float(means.mean()):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.962))
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.968),
            ncol=min(6, len(legend_labels)),
            fontsize=9,
            framealpha=0.95,
            borderaxespad=0.0,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def _parse_draw_select(spec: str) -> list[str]:
    """Parse ``--draw-select`` into unique mode names."""
    allowed = {"mean_extremes", "high_var"}
    out: list[str] = []
    for part in spec.split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(
                f"Unknown draw-select {mode!r}; expected subset of {sorted(allowed)}"
            )
        if mode not in out:
            out.append(mode)
    if not out:
        raise ValueError("--draw-select produced an empty list")
    return out


def _select_mean_extreme_indices(order: np.ndarray, n_draw: int) -> list[int]:
    """Take half top-ranked and half bottom-ranked graph ids."""
    n_draw = max(2, int(n_draw))
    n_top = n_draw // 2
    n_bot = n_draw - n_top
    top = [int(g) for g in order[:n_top]]
    bot = [int(g) for g in order[-n_bot:]]
    seen: set[int] = set()
    out: list[int] = []
    for g in top + bot:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _per_graph_std(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    n_graphs: int,
    layer: int,
    head: int,
) -> np.ndarray:
    """Within-graph std of node γ at ``(layer, head)`` → ``[G]``."""
    out = np.zeros(n_graphs, dtype=np.float64)
    for g in range(n_graphs):
        a, b = int(ptr[g]), int(ptr[g + 1])
        col = node_vals[a:b, layer, head]
        out[g] = float(col.std()) if col.size > 1 else 0.0
    return out


def _select_high_var_indices(stds: np.ndarray, n_draw: int) -> list[int]:
    """Graph ids with largest within-graph gate std (ties broken by id)."""
    n_draw = max(1, min(int(n_draw), int(stds.size)))
    order = np.argsort(-stds, kind="stable")
    return [int(g) for g in order[:n_draw]]


def _local_undirected_edges(
    edge_index: np.ndarray,
    edge_ptr: np.ndarray,
    ptr: np.ndarray,
    graph_id: int,
) -> np.ndarray:
    """Return unique undirected local edges ``[2, E_u]`` for one graph."""
    n0, n1 = int(ptr[graph_id]), int(ptr[graph_id + 1])
    e0, e1 = int(edge_ptr[graph_id]), int(edge_ptr[graph_id + 1])
    ei = edge_index[:, e0:e1].astype(np.int64, copy=True) - n0
    n_g = n1 - n0
    pairs: set[tuple[int, int]] = set()
    for u, v in ei.T:
        ui, vi = int(u), int(v)
        if ui == vi or ui < 0 or vi < 0 or ui >= n_g or vi >= n_g:
            continue
        pairs.add((ui, vi) if ui < vi else (vi, ui))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.asarray(sorted(pairs), dtype=np.int64).T
    return arr


def gate_dirichlet_energy(gates: np.ndarray, edge_index_local: np.ndarray) -> float:
    """Dirichlet energy of a scalar gate field on one graph.

    Uses undirected edges and
    ``E = (1 / 2) * mean_{(i,j)∈E} (γ_i − γ_j)²``.

    This is **layer/head-specific**: pass γ at a fixed ``(L, head)``.
    High E ⇒ neighboring nodes disagree on the gate (rough field).
    """
    if gates.size == 0 or edge_index_local.size == 0:
        return 0.0
    src = edge_index_local[0]
    dst = edge_index_local[1]
    diff = gates[src].astype(np.float64) - gates[dst].astype(np.float64)
    return float(0.5 * np.mean(np.square(diff)))


def _per_graph_dirichlet(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    edge_index: np.ndarray,
    edge_ptr: np.ndarray,
    n_graphs: int,
    layer: int,
    head: int,
) -> np.ndarray:
    """Per-graph gate Dirichlet energy at ``(layer, head)`` → ``[G]``."""
    out = np.zeros(n_graphs, dtype=np.float64)
    for g in range(n_graphs):
        a, b = int(ptr[g]), int(ptr[g + 1])
        gates = node_vals[a:b, layer, head]
        ei = _local_undirected_edges(edge_index, edge_ptr, ptr, g)
        out[g] = gate_dirichlet_energy(gates, ei)
    return out


def _dirichlet_tensor(
    node_vals: np.ndarray,
    ptr: np.ndarray,
    edge_index: np.ndarray,
    edge_ptr: np.ndarray,
    n_graphs: int,
) -> np.ndarray:
    """Dirichlet energy for every graph×layer×head → ``[G, L, H]``."""
    n_layers, n_heads = int(node_vals.shape[1]), int(node_vals.shape[2])
    out = np.zeros((n_graphs, n_layers, n_heads), dtype=np.float64)
    for layer in range(n_layers):
        for head in range(n_heads):
            out[:, layer, head] = _per_graph_dirichlet(
                node_vals, ptr, edge_index, edge_ptr, n_graphs, layer, head
            )
    return out


def plot_branch_dirichlet(
    dirichlet: np.ndarray,
    *,
    kind: str,
    head_names: Sequence[str],
    labels: Optional[np.ndarray],
    color_by_class: bool,
    order: np.ndarray,
    sort_key_label: str,
    title: str,
    out_path: Path,
    dpi: int,
    ref_branch: Optional[str],
    ref_layer: Optional[int],
    ref_head: Optional[int],
) -> None:
    """Write ``L × H`` grid of per-graph gate Dirichlet energy vs shared rank."""
    n_graphs, n_layers, n_heads = dirichlet.shape
    fig_w = max(3.2 * n_heads, 10.0)
    fig_h = max(2.2 * n_layers, 8.0)
    fig, axes = plt.subplots(
        n_layers,
        n_heads,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    use_class = bool(color_by_class and labels is not None)
    unique_classes: list[int] = []
    colors: list[str] = []
    if use_class:
        assert labels is not None
        unique_classes = sorted(int(c) for c in np.unique(labels))
        colors = _class_colors(len(unique_classes))

    ranks = np.arange(n_graphs)
    xlabel = f"Rank ({sort_key_label})"
    ymax = float(np.nanmax(dirichlet)) if dirichlet.size else 1.0
    ymax = max(ymax * 1.05, 1e-6)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            vals = dirichlet[order, layer, head]
            if use_class:
                assert labels is not None
                y_lab = labels[order]
                for ci, cls in enumerate(unique_classes):
                    mask = y_lab == cls
                    ax.scatter(
                        ranks[mask],
                        vals[mask],
                        s=12,
                        alpha=0.85,
                        c=colors[ci],
                        edgecolors="none",
                        zorder=3,
                        label=f"c{cls}" if layer == 0 and head == 0 else None,
                    )
            else:
                ax.scatter(
                    ranks,
                    vals,
                    s=10,
                    alpha=0.75,
                    c="#4C72B0",
                    edgecolors="none",
                    zorder=3,
                )
            if (
                ref_branch == kind
                and ref_layer == layer
                and ref_head == head
            ):
                for spine in ax.spines.values():
                    spine.set_color("#C44E52")
                    spine.set_linewidth(1.5)
            ax.set_ylim(-0.02 * ymax, ymax)
            ax.grid(True, alpha=0.3, linestyle="--")
            if layer == 0:
                ax.set_title(head_names[head], fontsize=10, fontweight="bold")
            if head == 0:
                ax.set_ylabel(f"L{layer}\nDir(γ)", fontsize=9)
            if layer == n_layers - 1:
                ax.set_xlabel(xlabel, fontsize=7)
            ax.text(
                0.98,
                0.95,
                f"μ={float(vals.mean()):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.96),
            ncol=min(6, len(legend_labels)),
            fontsize=9,
            framealpha=0.95,
            borderaxespad=0.0,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def plot_colored_graphs(
    *,
    node_vals: np.ndarray,
    ptr: np.ndarray,
    edge_index: np.ndarray,
    edge_ptr: np.ndarray,
    graph_ids: Sequence[int],
    labels: Optional[np.ndarray],
    layer: int,
    head: int,
    head_name: str,
    branch: str,
    title: str,
    out_path: Path,
    dpi: int,
    subtitle_metrics: Optional[Mapping[int, str]] = None,
) -> None:
    """Draw selected graphs with nodes colored by gate γ."""
    n = len(graph_ids)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.1 * nrows),
        squeeze=False,
    )
    cmap = colormaps["viridis"]
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for i, g in enumerate(graph_ids):
        ax = axes[i // ncols, i % ncols]
        n0, n1 = int(ptr[g]), int(ptr[g + 1])
        gates = node_vals[n0:n1, layer, head]
        ei = _local_undirected_edges(edge_index, edge_ptr, ptr, g)
        n_g = n1 - n0
        G = nx.Graph()
        G.add_nodes_from(range(n_g))
        for u, v in ei.T:
            G.add_edge(int(u), int(v))
        if n_g == 0:
            ax.axis("off")
            continue
        pos = nx.spring_layout(G, seed=0, k=1.2 / max(np.sqrt(n_g), 1.0))
        node_colors = [cmap(norm(float(gates[u]))) for u in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, alpha=0.5, edge_color="#666666")
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=80,
            linewidths=0.3,
            edgecolors="#222222",
        )
        cls = int(labels[g]) if labels is not None else -1
        std = float(gates.std()) if gates.size > 1 else 0.0
        extra = ""
        if subtitle_metrics is not None and g in subtitle_metrics:
            extra = f" · {subtitle_metrics[g]}"
        ax.set_title(
            f"g{g}"
            + (f" · c{cls}" if cls >= 0 else "")
            + f" · mean={float(gates.mean()):.2f} σ={std:.2f}"
            + extra,
            fontsize=8,
        )
        ax.axis("off")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.90, top=0.88)
    cax = fig.add_axes((0.92, 0.15, 0.02, 0.65))
    fig.colorbar(sm, cax=cax, label=f"γ · {branch} {head_name}")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load per-node dump and write band profiles + optional graph drawings."""
    args = _parse_args(argv)
    pt_path = Path(args.pt_node).expanduser().resolve()
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload)}")

    # Node dump stores node tensors under attn/gnn (not attn_node).
    attn_n = payload["attn"].detach().cpu().float().numpy()
    gnn_n = payload["gnn"].detach().cpu().float().numpy()
    ptr = payload["ptr"].detach().cpu().long().numpy()
    n_graphs = int(payload.get("num_graphs", ptr.size - 1))

    y_t = payload.get("y")
    labels: Optional[np.ndarray] = None
    if y_t is not None:
        labels = y_t.detach().cpu().numpy().astype(np.int64).reshape(-1)

    meta = dict(payload.get("meta") or {})
    dataset = str(meta.get("dataset", "dataset"))

    attn_g = _graph_means_from_nodes(attn_n, ptr, n_graphs)
    gnn_g = _graph_means_from_nodes(gnn_n, ptr, n_graphs)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pt_path.parent / "plots_node"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_names = _head_labels(meta, "attn", int(attn_n.shape[-1]))
    gnn_names = _head_labels(meta, "gnn", int(gnn_n.shape[-1]))
    class_suffix = "_by_class" if args.color_by_class and labels is not None else ""

    order, _ = shared_graph_order(
        attn_g,
        gnn_g,
        branch=str(args.sort_branch),
        layer=int(args.sort_layer),
        head=int(args.sort_head),
    )
    n_l = int(gnn_g.shape[1] if args.sort_branch == "gnn" else attn_g.shape[1])
    ref_branch = str(args.sort_branch)
    ref_layer = resolve_sort_layer(n_l, int(args.sort_layer))
    ref_head = int(args.sort_head)
    names = gnn_names if args.sort_branch == "gnn" else attn_names
    sort_key_label = f"L{ref_layer} {names[ref_head]} γ↓"

    band_tag = str(args.band)
    base = (
        f"Node-band gate | {dataset} | SiGMA · {n_graphs} graphs · "
        f"band={band_tag} · order by {sort_key_label}"
    )

    plot_branch_bands(
        attn_n,
        ptr,
        attn_g,
        kind="attn",
        head_names=attn_names,
        labels=labels,
        color_by_class=args.color_by_class,
        order=order,
        sort_key_label=sort_key_label,
        band_mode=band_tag,
        title=f"{base} · attention",
        out_path=out_dir
        / f"{dataset.lower()}_gates_attn_nodeband_shared_order{class_suffix}.png",
        dpi=int(args.dpi),
        ref_branch=ref_branch,
        ref_layer=ref_layer,
        ref_head=ref_head,
    )
    plot_branch_bands(
        gnn_n,
        ptr,
        gnn_g,
        kind="gnn",
        head_names=gnn_names,
        labels=labels,
        color_by_class=args.color_by_class,
        order=order,
        sort_key_label=sort_key_label,
        band_mode=band_tag,
        title=f"{base} · MP heads",
        out_path=out_dir
        / f"{dataset.lower()}_gates_gnn_nodeband_shared_order{class_suffix}.png",
        dpi=int(args.dpi),
        ref_branch=ref_branch,
        ref_layer=ref_layer,
        ref_head=ref_head,
    )

    has_edges = "edge_index" in payload and "edge_ptr" in payload
    edge_index: Optional[np.ndarray] = None
    edge_ptr: Optional[np.ndarray] = None
    if has_edges:
        edge_index = payload["edge_index"].detach().cpu().long().numpy()
        edge_ptr = payload["edge_ptr"].detach().cpu().long().numpy()

    if not args.skip_dirichlet:
        if not has_edges:
            logging.warning(
                "No edge_index/edge_ptr in dump — cannot compute Dirichlet energy."
            )
        else:
            assert edge_index is not None and edge_ptr is not None
            dir_base = (
                f"Gate Dirichlet energy Dir(γ)=½ mean_e (γ_i−γ_j)² | {dataset} | "
                f"SiGMA · {n_graphs} graphs · order by {sort_key_label}"
            )
            for kind, node_arr, names in (
                ("attn", attn_n, attn_names),
                ("gnn", gnn_n, gnn_names),
            ):
                dire = _dirichlet_tensor(
                    node_arr, ptr, edge_index, edge_ptr, n_graphs
                )
                plot_branch_dirichlet(
                    dire,
                    kind=kind,
                    head_names=names,
                    labels=labels,
                    color_by_class=args.color_by_class,
                    order=order,
                    sort_key_label=sort_key_label,
                    title=f"{dir_base} · {'attention' if kind == 'attn' else 'MP heads'}",
                    out_path=out_dir
                    / (
                        f"{dataset.lower()}_gates_{kind}_dirichlet_shared_order"
                        f"{class_suffix}.png"
                    ),
                    dpi=int(args.dpi),
                    ref_branch=ref_branch,
                    ref_layer=ref_layer,
                    ref_head=ref_head,
                )

    if not args.skip_draw:
        if not has_edges:
            logging.warning(
                "No edge_index/edge_ptr in dump — re-run dump with latest "
                "scripts/gate_viz/dump_per_graph_gates.py to enable drawings."
            )
        else:
            assert edge_index is not None and edge_ptr is not None
            draw_branch = str(args.draw_branch)
            vals = gnn_n if draw_branch == "gnn" else attn_n
            draw_names = gnn_names if draw_branch == "gnn" else attn_names
            d_layer = resolve_sort_layer(int(vals.shape[1]), int(args.draw_layer))
            d_head = int(args.draw_head)
            head_name = draw_names[d_head]
            stds = _per_graph_std(vals, ptr, n_graphs, d_layer, d_head)

            for mode in _parse_draw_select(str(args.draw_select)):
                if mode == "mean_extremes":
                    draw_ids = _select_mean_extreme_indices(order, int(args.n_draw))
                    sel_tag = f"top/bottom by {sort_key_label}"
                    fname = (
                        f"{dataset.lower()}_gates_{draw_branch}_L{d_layer}_"
                        f"{head_name}_node_graphs.png"
                    )
                elif mode == "high_var":
                    draw_ids = _select_high_var_indices(stds, int(args.n_draw))
                    sel_tag = f"highest within-graph σ(γ) at L{d_layer} {head_name}"
                    fname = (
                        f"{dataset.lower()}_gates_{draw_branch}_L{d_layer}_"
                        f"{head_name}_node_graphs_high_var.png"
                    )
                else:
                    raise RuntimeError(f"Unhandled draw mode {mode!r}")

                plot_colored_graphs(
                    node_vals=vals,
                    ptr=ptr,
                    edge_index=edge_index,
                    edge_ptr=edge_ptr,
                    graph_ids=draw_ids,
                    labels=labels,
                    layer=d_layer,
                    head=d_head,
                    head_name=head_name,
                    branch=draw_branch,
                    title=(
                        f"Nodes colored by γ | {dataset} · L{d_layer} {head_name} · "
                        f"{sel_tag}"
                    ),
                    out_path=out_dir / fname,
                    dpi=int(args.dpi),
                )

    logging.info("Done → %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
