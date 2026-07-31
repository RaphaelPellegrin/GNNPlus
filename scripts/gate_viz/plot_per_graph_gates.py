#!/usr/bin/env python3
"""Plot per-graph SiGMA gate values (hetero-profile style).

Default: sort graphs **once** by a reference head (last layer, GNN head 1 =
GIN for a4g4 GCN,GIN,SAGE,GAT) high→low, then reuse that order in every panel.

Optional ``--sort-mode per_panel`` restores independent ranking per panel.

Example::

  python scripts/gate_viz/plot_per_graph_gates.py \\
    --pt gate_values_per_graph.pt \\
    --out_dir results/gate_viz/enzymes_ogpkubk9_plateau_seed2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for gate-profile plots."""
    parser = argparse.ArgumentParser(
        description="Plot per-graph gates with shared or per-panel ranking.",
    )
    parser.add_argument(
        "--pt",
        type=str,
        required=True,
        help="Path to gate_values_per_graph.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (default: <pt_parent>/plots).",
    )
    parser.add_argument(
        "--color-by-class",
        action="store_true",
        help="Color points by graph label y when available.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG dpi (default: 150).",
    )
    parser.add_argument(
        "--sort-mode",
        type=str,
        choices=("shared", "per_panel"),
        default="shared",
        help="shared: one graph order for all panels; per_panel: sort each cell.",
    )
    parser.add_argument(
        "--sort-branch",
        type=str,
        choices=("gnn", "attn"),
        default="gnn",
        help="Branch used for shared sort key (default: gnn).",
    )
    parser.add_argument(
        "--sort-layer",
        type=int,
        default=-1,
        help="Layer index for shared sort (-1 = last layer).",
    )
    parser.add_argument(
        "--sort-head",
        type=int,
        default=1,
        help="Head index for shared sort (default: 1 = GIN for a4g4).",
    )
    return parser.parse_args(argv)


def _head_labels(meta: Mapping[str, Any], kind: str, n_heads: int) -> list[str]:
    """Human-readable head names for attn / gnn columns."""
    if kind == "gnn":
        raw = str(meta.get("gnn_types", "") or "")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == n_heads:
            return parts
        if len(parts) == 1 and n_heads > 1:
            return [f"{parts[0]}_{i}" for i in range(n_heads)]
    return [f"{kind}_{i}" for i in range(n_heads)]


def _class_colors(n: int) -> list[str]:
    """Stable class colors (ENZYMES has 6 classes)."""
    base = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
    ]
    return [base[i % len(base)] for i in range(n)]


def resolve_sort_layer(n_layers: int, sort_layer: int) -> int:
    """Map ``-1`` to last layer; validate absolute index."""
    layer = n_layers + sort_layer if sort_layer < 0 else sort_layer
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"sort_layer={sort_layer} out of range for L={n_layers}")
    return int(layer)


def shared_graph_order(
    attn: np.ndarray,
    gnn: np.ndarray,
    *,
    branch: str,
    layer: int,
    head: int,
) -> Tuple[np.ndarray, str]:
    """Argsort graphs by one (branch, layer, head) gate, high→low.

    Returns:
        ``(order, label)`` where ``order`` indexes graphs and ``label`` describes
        the sort key for titles/axes.
    """
    if branch == "gnn":
        values = gnn
        branch_tag = "gnn"
    elif branch == "attn":
        values = attn
        branch_tag = "attn"
    else:
        raise ValueError(f"Unknown branch {branch!r}")

    n_layers, n_heads = int(values.shape[1]), int(values.shape[2])
    layer_i = resolve_sort_layer(n_layers, layer)
    if head < 0 or head >= n_heads:
        raise ValueError(f"sort_head={head} out of range for H={n_heads}")
    key = values[:, layer_i, head]
    order = np.argsort(-key)
    label = f"L{layer_i} {branch_tag}[{head}] γ↓"
    return order, label


def plot_branch_grid(
    values: np.ndarray,
    *,
    kind: str,
    head_names: Sequence[str],
    labels: Optional[np.ndarray],
    color_by_class: bool,
    title: str,
    out_path: Path,
    dpi: int,
    sort_mode: str,
    shared_order: Optional[np.ndarray],
    sort_key_label: str,
    ref_branch: Optional[str] = None,
    ref_layer: Optional[int] = None,
    ref_head: Optional[int] = None,
) -> None:
    """Write ``L × H`` scatter grid with shared or per-panel ranking.

    Args:
        values: Float array ``[N, L, H]`` of per-graph mean gates.
        kind: ``attn`` or ``gnn`` (for axis labels only).
        head_names: Length-``H`` column titles.
        labels: Optional graph class labels ``[N]``.
        color_by_class: If True and labels given, color by class.
        title: Figure super-title.
        out_path: PNG path.
        dpi: Output resolution.
        sort_mode: ``shared`` or ``per_panel``.
        shared_order: Graph permutation for ``shared`` mode.
        sort_key_label: Short description of the shared sort key.
        ref_branch: Branch of the shared sort key (for panel highlight).
        ref_layer: Layer of the shared sort key.
        ref_head: Head of the shared sort key.
    """
    n_graphs, n_layers, n_heads = values.shape
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

    if sort_mode == "shared":
        if shared_order is None:
            raise ValueError("shared_order required for sort_mode=shared")
        order_fixed = shared_order
        xlabel = f"Rank ({sort_key_label})"
    else:
        order_fixed = None
        xlabel = "Rank (γ↓ per panel)"

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            col = values[:, layer, head]
            if order_fixed is not None:
                order = order_fixed
            else:
                order = np.argsort(-col)
            ranks = np.arange(n_graphs)
            y = col[order]
            if use_class:
                assert labels is not None
                y_lab = labels[order]
                for ci, cls in enumerate(unique_classes):
                    mask = y_lab == cls
                    ax.scatter(
                        ranks[mask],
                        y[mask],
                        s=10,
                        alpha=0.7,
                        c=colors[ci],
                        edgecolors="none",
                        label=f"c{cls}" if layer == 0 and head == 0 else None,
                    )
            else:
                ax.scatter(ranks, y, s=8, alpha=0.55, c="#4C72B0", edgecolors="none")
            if (
                sort_mode == "shared"
                and ref_branch == kind
                and ref_layer == layer
                and ref_head == head
            ):
                for spine in ax.spines.values():
                    spine.set_color("#C44E52")
                    spine.set_linewidth(1.5)
            mean = float(col.mean())
            std = float(col.std())
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
                f"{mean:.2f}±{std:.2f}",
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

    if use_class:
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                legend_labels,
                loc="upper right",
                ncol=min(6, len(legend_labels)),
                fontsize=9,
                framealpha=0.95,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def plot_mean_heatmap(
    attn: np.ndarray,
    gnn: np.ndarray,
    *,
    attn_names: Sequence[str],
    gnn_names: Sequence[str],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """Mean γ heatmap over graphs for attn and gnn heads."""
    mean_attn = attn.mean(axis=0)  # [L, Na]
    mean_gnn = gnn.mean(axis=0)  # [L, Ng]
    n_layers = mean_attn.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, max(4.0, 0.45 * n_layers)))

    for ax, mat, names, _kind in (
        (axes[0], mean_attn, attn_names, "attn"),
        (axes[1], mean_gnn, gnn_names, "gnn"),
    ):
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(list(names), rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
        ax.set_title(f"mean γ · {_kind}", fontsize=11, fontweight="bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if mat[i, j] < 0.55 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load gate dump and write ranked profile PNGs."""
    args = _parse_args(argv)
    pt_path = Path(args.pt).expanduser().resolve()
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload)}")

    attn = payload["attn"].detach().cpu().float().numpy()
    gnn = payload["gnn"].detach().cpu().float().numpy()
    y_t = payload.get("y")
    labels: Optional[np.ndarray] = None
    if y_t is not None:
        labels = y_t.detach().cpu().numpy().astype(np.int64).reshape(-1)

    meta = dict(payload.get("meta") or {})
    dataset = str(meta.get("dataset", "ENZYMES"))
    epoch = meta.get("epoch", "?")
    seed = meta.get("seed", "?")
    n_graphs = int(attn.shape[0])

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pt_path.parent / "plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_names = _head_labels(meta, "attn", int(attn.shape[-1]))
    gnn_names = _head_labels(meta, "gnn", int(gnn.shape[-1]))
    class_suffix = "_by_class" if args.color_by_class and labels is not None else ""

    shared_order: Optional[np.ndarray] = None
    sort_key_label = "per panel"
    ref_branch: Optional[str] = None
    ref_layer: Optional[int] = None
    ref_head: Optional[int] = None
    if args.sort_mode == "shared":
        shared_order, _ = shared_graph_order(
            attn,
            gnn,
            branch=str(args.sort_branch),
            layer=int(args.sort_layer),
            head=int(args.sort_head),
        )
        n_l = int(gnn.shape[1] if args.sort_branch == "gnn" else attn.shape[1])
        ref_branch = str(args.sort_branch)
        ref_layer = resolve_sort_layer(n_l, int(args.sort_layer))
        ref_head = int(args.sort_head)
        names = gnn_names if args.sort_branch == "gnn" else attn_names
        head_name = names[ref_head]
        sort_key_label = f"L{ref_layer} {head_name} γ↓"
        order_tag = "shared_order"
        logging.info("Shared graph order from %s", sort_key_label)
    else:
        order_tag = "by_rank"

    base_title = (
        f"Per-graph gate γ | {dataset} | SiGMA · ep{epoch} seed{seed} · n={n_graphs}"
    )
    if args.sort_mode == "shared":
        base_title += f" · order fixed by {sort_key_label}"
    else:
        base_title += " · each panel sorted independently"

    plot_branch_grid(
        attn,
        kind="attn",
        head_names=attn_names,
        labels=labels,
        color_by_class=args.color_by_class,
        title=f"{base_title} · attention heads",
        out_path=out_dir / f"{dataset.lower()}_gates_attn_{order_tag}{class_suffix}.png",
        dpi=int(args.dpi),
        sort_mode=str(args.sort_mode),
        shared_order=shared_order,
        sort_key_label=sort_key_label,
        ref_branch=ref_branch,
        ref_layer=ref_layer,
        ref_head=ref_head,
    )
    plot_branch_grid(
        gnn,
        kind="gnn",
        head_names=gnn_names,
        labels=labels,
        color_by_class=args.color_by_class,
        title=f"{base_title} · MP heads",
        out_path=out_dir / f"{dataset.lower()}_gates_gnn_{order_tag}{class_suffix}.png",
        dpi=int(args.dpi),
        sort_mode=str(args.sort_mode),
        shared_order=shared_order,
        sort_key_label=sort_key_label,
        ref_branch=ref_branch,
        ref_layer=ref_layer,
        ref_head=ref_head,
    )
    plot_mean_heatmap(
        attn,
        gnn,
        attn_names=attn_names,
        gnn_names=gnn_names,
        title=f"Mean gate γ by layer × head | {dataset} · ep{epoch} seed{seed}",
        out_path=out_dir / f"{dataset.lower()}_gates_mean_heatmap.png",
        dpi=int(args.dpi),
    )
    logging.info("Done → %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
