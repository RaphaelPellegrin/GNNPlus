#!/usr/bin/env python3
"""Paper figure: vertical sink band + graph + Fesser diagnostics.

Panels:
  * degree-sorted attention (sink head, optional flat head)
  * graph with AS node colored
  * ‖v‖ per node (sink highlighted) + mechanism stats
  * centrality ranks of the sink (degree / PageRank / …)

Example::

  python scripts/attention_sinks/plot_paper_vertical_band.py \\
    --run-dir results/tu_attention_sinks/mutag_GPS_ungated_attn_lr001_seed2 \\
    --mech-csv results/tu_attention_sinks/analysis/mutag_GPS_ungated_attn_lr001_seed2_mech.csv \\
    --key layer0_attn0 \\
    --flat-key layer8_attn0 \\
    --out results/tu_attention_sinks/paper_figures/fig_vertical_band_mutag_gps_l0.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.patches import Patch


_SINK_COLOR = "#e63946"
_NODE_COLOR = "#c8d0d8"
_EDGE_COLOR = "#6b7280"
_VBAR_COLOR = "#4a6670"

_CENTRALITY_METRICS: Tuple[str, ...] = (
    "degree",
    "pagerank",
    "eigenvector",
    "closeness",
    "clustering",
    "kcore",
)


@dataclass(frozen=True)
class GraphAttnView:
    """Degree-sorted attention view plus original-index graph for one head."""

    attn_sorted: np.ndarray
    degrees_sorted: np.ndarray
    alpha_sorted: np.ndarray
    alpha_local: np.ndarray
    order: np.ndarray
    degrees_local: np.ndarray
    edge_index_local: np.ndarray
    sink_local: int
    n: int
    value_norms_local: Optional[np.ndarray]
    head_output_local: Optional[np.ndarray]
    vnorm_ratio: float
    av_stable_rank: float
    av_row_cosine: float
    mechanism: str


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for vertical-band paper figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mech-csv", type=Path, required=True)
    p.add_argument("--key", type=str, default="layer0_attn0")
    p.add_argument("--flat-key", type=str, default="")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--max-n", type=int, default=40)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument("--layout-seed", type=int, default=0)
    return p.parse_args(argv)


def stable_rank(matrix: np.ndarray) -> float:
    """Stable rank ``‖M‖_F² / ‖M‖₂²`` (≈1 ⇒ near rank-1 / broadcast-like)."""
    if matrix.ndim != 2 or matrix.size == 0:
        return float("nan")
    singular = np.linalg.svd(matrix, compute_uv=False)
    s2 = singular.astype(np.float64) ** 2
    peak = float(s2.max()) if s2.size else 0.0
    if peak <= 0.0:
        return 0.0
    return float(s2.sum() / peak)


def mean_row_cosine(matrix: np.ndarray) -> float:
    """Mean cosine of each row of ``M`` to the mean row (broadcast ⇒ high)."""
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return float("nan")
    mean_row = matrix.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_row))
    if mean_norm <= 1e-12:
        return float("nan")
    cosines: List[float] = []
    for row in matrix:
        rn = float(np.linalg.norm(row))
        if rn <= 1e-12:
            continue
        cosines.append(float(np.dot(row, mean_row) / (rn * mean_norm)))
    if not cosines:
        return float("nan")
    return float(np.mean(cosines))


def classify_sink_mechanism(
    *,
    vnorm_ratio: float,
    av_stable_rank: float,
    row_cosine: float,
    nop_vnorm_thresh: float = 0.25,
    broadcast_rank_thresh: float = 1.5,
    broadcast_cosine_thresh: float = 0.85,
) -> str:
    """Heuristic NOP / broadcast / ambiguous label (Fesser-style)."""
    nop_like = (not np.isnan(vnorm_ratio)) and vnorm_ratio < nop_vnorm_thresh
    broadcast_like = (
        (not np.isnan(av_stable_rank) and av_stable_rank <= broadcast_rank_thresh)
        and (not np.isnan(row_cosine) and row_cosine >= broadcast_cosine_thresh)
        and (np.isnan(vnorm_ratio) or vnorm_ratio >= nop_vnorm_thresh)
    )
    if nop_like and not broadcast_like:
        return "nop"
    if broadcast_like and not nop_like:
        return "broadcast"
    return "ambiguous"


def _parse_source_batch(source: str) -> Tuple[str, int]:
    """Extract split and batch_index from a mech ``source`` path."""
    name = Path(source).name
    m = re.search(r"_(train|val|test)_batch(\d+)_", name)
    if not m:
        raise ValueError(f"Cannot parse split/batch from {name}")
    return m.group(1), int(m.group(2))


def _local_degrees(edge_index: torch.Tensor, n: int, start: int) -> np.ndarray:
    """Undirected degree vector for local nodes ``[0, n)``."""
    deg = np.zeros(n, dtype=np.int64)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    for u, v in zip(src, dst):
        if start <= u < start + n:
            deg[int(u) - start] += 1
        if start <= v < start + n:
            deg[int(v) - start] += 1
    return deg


def _local_edges(edge_index: torch.Tensor, n: int, start: int) -> np.ndarray:
    """Local undirected edge list ``(E, 2)`` with nodes in ``[0, n)``."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edges: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for u, v in zip(src, dst):
        if not (start <= u < start + n and start <= v < start + n):
            continue
        a = int(u) - start
        b = int(v) - start
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def _graph_slice(batch: torch.Tensor, graph_index: int) -> Tuple[int, int]:
    """Return ``[start, end)`` node range for ``graph_index``."""
    nodes = torch.where(batch == graph_index)[0]
    start = int(nodes[0].item())
    end = int(nodes[-1].item()) + 1
    return start, end


def _sink_strength(attn: np.ndarray) -> np.ndarray:
    """Column-mean attention (sink strength α)."""
    return attn.mean(axis=0)


def _rank_desc(values: np.ndarray, index: int) -> int:
    """0-based rank of ``index`` when sorting ``values`` descending (0 = top)."""
    order = np.argsort(-values.astype(np.float64))
    return int(np.where(order == index)[0][0])


def compute_centrality_ranks(
    edge_index_local: np.ndarray,
    n: int,
    sink_local: int,
) -> Dict[str, Dict[str, float]]:
    """Compute centrality values and sink ranks (0 = highest).

    Args:
        edge_index_local: Undirected edges ``(E, 2)``.
        n: Number of nodes.
        sink_local: Sink node index.

    Returns:
        Mapping metric → ``{value, rank, n}``.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edge_index_local.tolist())

    degree = np.asarray([float(g.degree(i)) for i in range(n)], dtype=np.float64)

    try:
        pr = nx.pagerank(g, alpha=0.85, max_iter=200)
        pagerank = np.asarray([pr.get(i, 0.0) for i in range(n)], dtype=np.float64)
    except Exception:
        pagerank = degree.copy()

    try:
        ev = nx.eigenvector_centrality_numpy(g)
        eigenvector = np.asarray([ev.get(i, 0.0) for i in range(n)], dtype=np.float64)
    except Exception:
        try:
            ev = nx.eigenvector_centrality(g, max_iter=500)
            eigenvector = np.asarray([ev.get(i, 0.0) for i in range(n)], dtype=np.float64)
        except Exception:
            eigenvector = degree.copy()

    try:
        cl = nx.closeness_centrality(g)
        closeness = np.asarray([cl.get(i, 0.0) for i in range(n)], dtype=np.float64)
    except Exception:
        closeness = np.zeros(n, dtype=np.float64)

    try:
        clustering = np.asarray(
            [float(nx.clustering(g, i)) for i in range(n)],
            dtype=np.float64,
        )
    except Exception:
        clustering = np.zeros(n, dtype=np.float64)

    try:
        core = nx.core_number(g)
        kcore = np.asarray([float(core.get(i, 0.0)) for i in range(n)], dtype=np.float64)
    except Exception:
        kcore = np.zeros(n, dtype=np.float64)

    metrics = {
        "degree": degree,
        "pagerank": pagerank,
        "eigenvector": eigenvector,
        "closeness": closeness,
        "clustering": clustering,
        "kcore": kcore,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in metrics.items():
        out[name] = {
            "value": float(vals[sink_local]),
            "rank": float(_rank_desc(vals, sink_local)),
            "n": float(n),
        }
    return out


def _select_exemplar(
    mech: pd.DataFrame,
    *,
    key: str,
    split: str,
    min_n: int,
    max_n: int,
) -> pd.Series:
    """Pick a high-ratio τ-sink graph for ``key``."""
    sub = mech[
        (mech["layer_head"] == key)
        & (mech["tau_sink"] == 1)
        & (mech["n_g"] >= min_n)
        & (mech["n_g"] <= max_n)
    ].copy()
    if sub.empty:
        sub = mech[
            (mech["layer_head"] == key)
            & (mech["n_g"] >= min_n)
            & (mech["n_g"] <= max_n)
        ].copy()
    if sub.empty:
        raise RuntimeError(f"No candidate graphs for key={key}")
    prefer = sub[sub["split"] == split]
    pool = prefer if not prefer.empty else sub
    pool = pool.sort_values("ratio_vs_uniform", ascending=False)
    return pool.iloc[0]


def _load_graph_attn(
    pt_path: Path,
    *,
    key: str,
    graph_index: int,
) -> GraphAttnView:
    """Load degree-sorted attention, ‖v‖, AV diagnostics, and local graph."""
    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    if key not in bundle["attention"]:
        raise KeyError(f"{pt_path}: missing attention key {key}")
    batch = bundle["batch"]
    edge_index = bundle["edge_index"]
    start, end = _graph_slice(batch, graph_index)
    n = end - start
    deg_local = _local_degrees(edge_index, n, start)
    order = np.argsort(-deg_local)
    a_full = bundle["attention"][key].detach().cpu().float().numpy()
    a_g = a_full[start:end, start:end]
    a_sorted = a_g[np.ix_(order, order)]
    deg_sorted = deg_local[order]
    alpha_local = _sink_strength(a_g)
    alpha_sorted = alpha_local[order]
    sink_local = int(np.argmax(alpha_local)) if alpha_local.size else 0
    edges = _local_edges(edge_index, n, start)

    vn_local: Optional[np.ndarray] = None
    vnr = float("nan")
    value_norms = bundle.get("value_norms", {})
    if key in value_norms:
        vn_local = value_norms[key].detach().cpu().float().numpy()[start:end]
        vnr = float(vn_local[sink_local] / (float(vn_local.mean()) + 1e-8))

    av_local: Optional[np.ndarray] = None
    sr = float("nan")
    rc = float("nan")
    head_outputs = bundle.get("head_outputs", {})
    if key in head_outputs:
        av_local = head_outputs[key].detach().cpu().float().numpy()[start:end]
        sr = stable_rank(av_local)
        rc = mean_row_cosine(av_local)

    mech = classify_sink_mechanism(
        vnorm_ratio=vnr, av_stable_rank=sr, row_cosine=rc
    )
    return GraphAttnView(
        attn_sorted=a_sorted,
        degrees_sorted=deg_sorted,
        alpha_sorted=alpha_sorted,
        alpha_local=alpha_local,
        order=order,
        degrees_local=deg_local,
        edge_index_local=edges,
        sink_local=sink_local,
        n=n,
        value_norms_local=vn_local,
        head_output_local=av_local,
        vnorm_ratio=vnr,
        av_stable_rank=sr,
        av_row_cosine=rc,
        mechanism=mech,
    )


def _style() -> None:
    """Paper matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _fmt(x: float, digits: int = 2) -> str:
    """Format a float, or ``n/a`` if NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _plot_matrix(
    ax: plt.Axes,
    attn: np.ndarray,
    degrees: np.ndarray,
    alpha: np.ndarray,
    *,
    title: str,
    show_ylabel: bool,
) -> AxesImage:
    """Draw one attention matrix with sink column marker."""
    im = ax.imshow(attn, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0.0)
    sink_j = int(np.argmax(alpha))
    ax.axvline(sink_j, color=_SINK_COLOR, ls="--", lw=1.6)
    ax.axvspan(sink_j - 0.5, sink_j + 0.5, color=_SINK_COLOR, alpha=0.18, zorder=0)
    n = attn.shape[0]
    tick_idx = np.linspace(0, n - 1, num=min(6, n), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(int(degrees[i])) for i in tick_idx])
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(int(degrees[i])) for i in tick_idx])
    ax.set_xlabel("key (col, deg ↓)")
    if show_ylabel:
        ax.set_ylabel("query (row, deg ↓)")
    ratio = float(alpha.max() * n) if n > 0 else float("nan")
    ax.set_title(f"{title}\nα_max={alpha.max():.3f}  ({ratio:.1f}× uniform)")
    return im


def _plot_sink_graph(
    ax: plt.Axes,
    view: GraphAttnView,
    *,
    layout_seed: int,
    head_label: str,
) -> None:
    """Draw the graph with the attention-sink node colored."""
    g = nx.Graph()
    g.add_nodes_from(range(view.n))
    g.add_edges_from(view.edge_index_local.tolist())
    pos = nx.spring_layout(g, seed=layout_seed, k=1.2 / max(np.sqrt(view.n), 1.0))

    node_colors = [_NODE_COLOR] * view.n
    node_colors[view.sink_local] = _SINK_COLOR
    sizes = [80 + 16 * float(view.degrees_local[i]) for i in range(view.n)]
    sizes[view.sink_local] = max(sizes[view.sink_local] * 1.45, 200.0)

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=_EDGE_COLOR, alpha=0.45, width=0.9)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=sizes,
        linewidths=0.7,
        edgecolors="#1f2937",
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels={view.sink_local: "AS"},
        ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )
    ax.set_title(f"graph · sink from {head_label}", fontsize=11)
    ax.legend(
        handles=[
            Patch(facecolor=_SINK_COLOR, edgecolor="#1f2937", label="attention sink"),
            Patch(facecolor=_NODE_COLOR, edgecolor="#1f2937", label="other nodes"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_vnorm_bars(ax: plt.Axes, view: GraphAttnView) -> None:
    """Bar chart of ‖v‖ with sink highlighted; annotate Fesser ratios."""
    if view.value_norms_local is None:
        ax.text(0.5, 0.5, "‖v‖ not in dump", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Sort nodes by degree (same display convention as matrices)
    order = view.order
    vn = view.value_norms_local[order]
    sink_sorted = int(np.where(order == view.sink_local)[0][0])
    colors = [_VBAR_COLOR] * view.n
    colors[sink_sorted] = _SINK_COLOR
    x = np.arange(view.n)
    ax.bar(x, vn, color=colors, width=0.85, edgecolor="none")
    mean_vn = float(view.value_norms_local.mean())
    ax.axhline(mean_vn, color="#555555", ls="--", lw=0.9, label=f"mean ‖v‖={mean_vn:.3f}")
    ax.set_xlim(-0.5, view.n - 0.5)
    ax.set_xlabel("node (deg ↓)")
    ax.set_ylabel(r"$\|v_i\|$")
    ax.set_title(
        f"value norms · vnr={_fmt(view.vnorm_ratio)}  "
        f"(NOP << 1, here {'NOP-like' if view.mechanism == 'nop' else 'not NOP'})"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    # Sparse ticks
    tick_idx = np.linspace(0, view.n - 1, num=min(6, view.n), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(int(view.degrees_sorted[i])) for i in tick_idx])


def _plot_centrality_ranks(
    ax: plt.Axes,
    ranks: Dict[str, Dict[str, float]],
) -> None:
    """Horizontal bars: sink rank / (n−1) for each centrality (0 = hub/top)."""
    labels = list(_CENTRALITY_METRICS)
    # Normalized rank in [0, 1]: 0 = top/hub, 1 = bottom/periphery
    vals = []
    annot = []
    for name in labels:
        info = ranks[name]
        n = max(int(info["n"]) - 1, 1)
        rank = float(info["rank"])
        vals.append(rank / n)
        annot.append(f"#{int(rank)}/{int(info['n'])}")

    y = np.arange(len(labels))
    ax.barh(y, vals, color="#1f4e79", height=0.65, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("sink rank / (n−1)   (0 = hub/top)")
    ax.axvline(0.0, color="#555555", lw=0.6)
    ax.axvline(1.0, color="#555555", lw=0.6, ls=":")
    for yi, (v, a) in enumerate(zip(vals, annot)):
        ax.text(min(v + 0.02, 0.98), yi, a, va="center", ha="left", fontsize=8)
    ax.set_title("Where is the sink? (centrality ranks)")
    ax.invert_yaxis()


def _plot_mech_text(ax: plt.Axes, view: GraphAttnView, ranks: Dict[str, Dict[str, float]]) -> None:
    """Text panel with Fesser mechanism + key ranks."""
    ax.axis("off")
    deg_rank = int(ranks["degree"]["rank"])
    pr_rank = int(ranks["pagerank"]["rank"])
    lines = [
        "Fesser mechanism (this sink head)",
        f"  label:           {view.mechanism}",
        f"  ||v_sink||/mean||v||: {_fmt(view.vnorm_ratio)}   (NOP << 1)",
        f"  stable_rank(AV): {_fmt(view.av_stable_rank)}   (broadcast ~ 1)",
        f"  mean row-cos(AV): {_fmt(view.av_row_cosine)}   (broadcast high)",
        "",
        "Sink localization",
        f"  node:            {view.sink_local}  (n={view.n})",
        f"  α_max:           {view.alpha_local[view.sink_local]:.3f}  "
        f"({view.alpha_local[view.sink_local] * view.n:.1f}× unif)",
        f"  deg / deg-rank:  {int(view.degrees_local[view.sink_local])} / #{deg_rank}",
        f"  PageRank rank:   #{pr_rank}",
    ]
    if view.head_output_local is None:
        lines.append("")
        lines.append("  note: AV (head_outputs) missing in dump →")
        lines.append("        stable_rank / row-cos = n/a")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f5", "edgecolor": "#d0d0d0"},
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Write vertical-band + graph + Fesser diagnostic figure."""
    args = _parse_args(argv)
    _style()

    mech = pd.read_csv(args.mech_csv)
    row = _select_exemplar(
        mech,
        key=args.key,
        split=args.split,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    split, batch_index = _parse_source_batch(str(row["source"]))
    local_name = Path(str(row["source"])).name
    pt_path = args.run_dir / "attention_matrices" / local_name
    if not pt_path.is_file():
        epoch = int(row["epoch"])
        pattern = f"*_{split}_batch{batch_index:04d}_epoch{epoch:05d}.pt"
        matches = list((args.run_dir / "attention_matrices").glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing dump for {local_name} under {args.run_dir}")
        pt_path = matches[0]

    graph_index = int(row["graph_index"])
    sink_view = _load_graph_attn(pt_path, key=args.key, graph_index=graph_index)
    ranks = compute_centrality_ranks(
        sink_view.edge_index_local, sink_view.n, sink_view.sink_local
    )

    flat_key = args.flat_key.strip()
    panels: List[Tuple[str, GraphAttnView]] = [
        (f"sink head · {args.key}", sink_view),
    ]
    if flat_key:
        flat_view = _load_graph_attn(pt_path, key=flat_key, graph_index=graph_index)
        panels.append((f"flat head · {flat_key}", flat_view))

    n_panels = len(panels)
    fig_w = max(9.0, 4.2 * n_panels + 1.0)
    fig = plt.figure(figsize=(fig_w, 11.2), constrained_layout=True)
    # Row0: attn mats | Row1: graph + vnorm | Row2: ranks + mech text
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.05, 0.85], width_ratios=[1.15, 1.0])

    # Attention matrices across top
    if n_panels == 1:
        ax0 = fig.add_subplot(gs[0, :])
        axes_attn = [ax0]
    else:
        # nested gridspec for equal matrix panels
        gs0 = gs[0, :].subgridspec(1, n_panels)
        axes_attn = [fig.add_subplot(gs0[0, i]) for i in range(n_panels)]

    for i, (title, view) in enumerate(panels):
        im = _plot_matrix(
            axes_attn[i],
            view.attn_sorted,
            view.degrees_sorted,
            view.alpha_sorted,
            title=title,
            show_ylabel=(i == 0),
        )
        cbar = fig.colorbar(im, ax=axes_attn[i], fraction=0.046, pad=0.04)
        if i == n_panels - 1:
            cbar.set_label("attention weight")

    ax_g = fig.add_subplot(gs[1, 0])
    _plot_sink_graph(ax_g, sink_view, layout_seed=int(args.layout_seed), head_label=args.key)

    ax_v = fig.add_subplot(gs[1, 1])
    _plot_vnorm_bars(ax_v, sink_view)

    ax_r = fig.add_subplot(gs[2, 0])
    _plot_centrality_ranks(ax_r, ranks)

    ax_t = fig.add_subplot(gs[2, 1])
    _plot_mech_text(ax_t, sink_view, ranks)

    ds = args.run_dir.name.split("_")[0]
    fig.suptitle(
        f"{ds} · sink band + ‖v‖ / AV / centrality ranks · "
        f"n={sink_view.n} · {split} graph {graph_index} · mech={sink_view.mechanism}",
        fontsize=12,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)

    meta = {
        "pt": str(pt_path),
        "key": args.key,
        "flat_key": flat_key or None,
        "split": split,
        "graph_index": graph_index,
        "n_g": sink_view.n,
        "sink_local": sink_view.sink_local,
        "mechanism": sink_view.mechanism,
        "vnorm_ratio": sink_view.vnorm_ratio,
        "av_stable_rank": sink_view.av_stable_rank,
        "av_row_cosine": sink_view.av_row_cosine,
        "deg_rank": int(ranks["degree"]["rank"]),
        "pagerank_rank": int(ranks["pagerank"]["rank"]),
        "png": str(args.out.resolve()),
        "pdf": str(pdf.resolve()),
    }
    print("Wrote vertical-band + Fesser diagnostics figure:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
