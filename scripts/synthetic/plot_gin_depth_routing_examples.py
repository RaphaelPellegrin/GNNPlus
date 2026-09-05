#!/usr/bin/env python3
"""Plot curated GIN depth-routing examples (1-GIN vs 2-GIN labels).

Writes one image per graph under ``results/gin_routing_depth/examples/`` plus a
JSON sidecar with scores / labels.

Example::

  python scripts/synthetic/plot_gin_depth_routing_examples.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module() -> Any:
    """Load dataset helpers without importing full ``GNNPlus`` package."""
    module_path = _REPO_ROOT / "GNNPlus" / "loader" / "dataset" / "gin_depth_routing.py"
    spec = importlib.util.spec_from_file_location("gin_depth_routing_plot", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gin_routing_depth",
        help="Output root (examples/ written underneath).",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _depth2_layout(data: Any) -> dict[int, tuple[float, float]]:
    """Place root at center, mids on a ring, leaves outside their mid."""
    roles = data.node_role.cpu().numpy()
    root = int(data.root_index.item())
    pos: dict[int, tuple[float, float]] = {root: (0.0, 0.0)}

    mids = [i for i, r in enumerate(roles) if int(r) == 1]
    k = max(len(mids), 1)
    mid_radius = 1.45
    for j, u in enumerate(mids):
        angle = 2.0 * np.pi * j / k - np.pi / 2.0
        pos[u] = (mid_radius * np.cos(angle), mid_radius * np.sin(angle))

    leaf_by_mid: dict[int, list[int]] = {u: [] for u in mids}
    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        if roles[src] == 1 and roles[dst] == 2:
            leaf_by_mid[int(src)].append(int(dst))
        elif roles[dst] == 1 and roles[src] == 2:
            leaf_by_mid[int(dst)].append(int(src))

    for u, leaves in leaf_by_mid.items():
        ux, uy = pos[u]
        base_angle = float(np.arctan2(uy, ux))
        n_leaves = len(leaves)
        if n_leaves == 0:
            continue
        leaf_radius = 0.85
        arc = min(1.2, max(0.35, 0.28 * n_leaves))
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


def _draw_graph(ax: Any, data: Any, *, title: str) -> None:
    """Draw one depth-2 tree with signal labels."""
    roles = data.node_role.cpu().numpy()
    signals = data.x[:, 0].cpu().numpy()
    tau = int(data.tau.item())
    y = int(data.y.item())
    s1 = float(data.s1_score.item())
    s2 = float(data.s2_score.item())

    g = nx.Graph()
    n = int(data.num_nodes)
    g.add_nodes_from(range(n))
    edges = data.edge_index.cpu().numpy().T
    undirected = {(int(min(a, b)), int(max(a, b))) for a, b in edges}
    g.add_edges_from(undirected)
    pos = _depth2_layout(data)

    root_nodes = [i for i, r in enumerate(roles) if int(r) == 0]
    mid_nodes = [i for i, r in enumerate(roles) if int(r) == 1]
    leaf_nodes = [i for i, r in enumerate(roles) if int(r) == 2]

    nx.draw_networkx_edges(g, pos, ax=ax, width=1.2, edge_color="#666666")
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=root_nodes,
        node_size=1100,
        node_color="#f4a261",
        edgecolors="#1d3557",
        linewidths=1.8,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=mid_nodes,
        node_size=750,
        node_color="#a8dadc",
        edgecolors="#1d3557",
        linewidths=1.4,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=leaf_nodes,
        node_size=520,
        node_color="#e9c46a",
        edgecolors="#1d3557",
        linewidths=1.2,
        ax=ax,
    )

    labels: dict[int, str] = {}
    for i in root_nodes:
        labels[i] = f"r\nτ={tau}"
    for i in mid_nodes:
        labels[i] = f"u\n{signals[i]:+.0f}"
    for i in leaf_nodes:
        labels[i] = f"v\n{signals[i]:+.0f}"
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(title, fontsize=11)
    ax.text(
        0.02,
        0.02,
        f"S1(1-GIN)={s1:+.0f}  →  class {int(s1 > 0)}\n"
        f"S2(2-GIN)={s2:+.0f}  →  class {int(s2 > 0)}\n"
        f"label y={y}  (τ={tau}: {'1-GIN' if tau == 0 else '2-GIN'})",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_axis_off()
    ax.set_aspect("equal")


def main() -> None:
    """Write curated example figures and metadata JSON."""
    args = _parse_args()
    mod = _load_module()
    out_dir = Path(args.out_dir)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    curated = mod.curated_example_specs()
    meta_rows: list[dict[str, Any]] = []

    for idx, (spec, caption) in enumerate(curated, start=1):
        data = mod.build_depth2_tree(spec)
        meta = mod.spec_to_metadata(spec)
        tau_name = "1gin_shallow" if spec.tau == 0 else "2gin_deep"
        stem = f"fig_example_{idx:02d}_{tau_name}_{spec.difficulty}"
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 8.2),
            gridspec_kw={"height_ratios": [3.2, 1.0]},
        )
        title = (
            f"Ex {idx}: τ={spec.tau} ({'1-GIN / shallow' if spec.tau == 0 else '2-GIN / deep'})"
            f" · y={spec.label()} · opposite_sign={spec.scores_disagree()}"
        )
        _draw_graph(axes[0], data, title=title)
        axes[1].set_axis_off()
        wrapped = "\n".join(textwrap.wrap(caption, width=92))
        axes[1].text(
            0.0,
            0.95,
            wrapped,
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            wrap=True,
        )
        fig.tight_layout()
        png_path = examples_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        row = {
            "example_id": idx,
            "stem": stem,
            "png": str(png_path.relative_to(out_dir)),
            "caption": caption,
            **meta,
        }
        meta_rows.append(row)
        print(
            f"[{idx}] τ={spec.tau} y={spec.label()} "
            f"S1={spec.s1_score():+.0f} S2={spec.s2_score():+.0f} "
            f"opp={spec.scores_disagree()} -> {png_path}"
        )

    meta_path = examples_dir / "examples_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta_rows, fh, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
