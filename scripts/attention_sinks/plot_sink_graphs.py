#!/usr/bin/env python3
"""Plot graphs with attention-sink nodes highlighted; summarize sink traits.

Uses mid-train ``attention_batch_epXXXX.pt`` bundles (edge_index + batch + A).

Example:
  python scripts/attention_sinks/plot_sink_graphs.py \\
    --pt results/tu_attention_sinks/mutag_GPS_ungated_attn_lr001_seed2/attention_sinks/ep00999/attention_batch_ep00999.pt \\
    --out_dir results/tu_attention_sinks/sink_graph_plots/GPS_ungated \\
    --key layer0_attn0 --n-graphs 6
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Parse CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt", type=str, required=True, help="attention_batch_*.pt path")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--key",
        type=str,
        default="",
        help="Attention key (default: strongest max-α head in bundle).",
    )
    p.add_argument("--n-graphs", type=int, default=8)
    p.add_argument("--tau", type=float, default=1.5)
    return p.parse_args()


def _alpha(A: np.ndarray) -> np.ndarray:
    """Column-mean sink strength."""
    return A.mean(axis=0)


def _tau_sink(A: np.ndarray, tau: float) -> np.ndarray:
    """τ·μ sink mask on columns."""
    a_hat = A.sum(axis=0)
    mu = float(a_hat.mean()) if a_hat.size else 0.0
    if mu <= 0:
        return np.zeros_like(a_hat, dtype=bool)
    return a_hat > tau * mu


def _split_graphs(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> List[Tuple[torch.Tensor, int, int]]:
    """Return list of (local_edge_index, n_nodes, global_start)."""
    n_graphs = int(batch.max().item()) + 1
    out: List[Tuple[torch.Tensor, int, int]] = []
    for g in range(n_graphs):
        nodes = torch.where(batch == g)[0]
        start = int(nodes[0].item())
        n = int(nodes.numel())
        mask = (edge_index[0] >= start) & (edge_index[0] < start + n)
        ei = edge_index[:, mask] - start
        out.append((ei, n, start))
    return out


def _degree(ei: torch.Tensor, n: int) -> np.ndarray:
    """Undirected degree."""
    deg = np.zeros(n, dtype=np.float64)
    if ei.numel() == 0:
        return deg
    for a, b in ei.t().tolist():
        deg[a] += 1
        deg[b] += 1
    return deg


def _pick_key(attention: Dict[str, torch.Tensor], batch: torch.Tensor) -> str:
    """Choose layer/head with largest mean max-α over graphs in the batch."""
    splits = []
    n_graphs = int(batch.max().item()) + 1
    for g in range(n_graphs):
        nodes = torch.where(batch == g)[0]
        splits.append((int(nodes[0]), int(nodes[-1]) + 1))
    best_key = next(iter(attention))
    best_score = -1.0
    for key, A_t in attention.items():
        A = A_t.detach().cpu().float().numpy()
        scores = []
        for s, e in splits:
            block = A[s:e, s:e]
            if block.size == 0:
                continue
            scores.append(float(_alpha(block).max()))
        m = float(np.mean(scores)) if scores else -1.0
        if m > best_score:
            best_score = m
            best_key = key
    return best_key


def main() -> None:
    """Load bundle, plot sink-highlighted graphs, print commonality stats."""
    args = _parse_args()
    payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    attention: Dict[str, torch.Tensor] = payload["attention"]
    edge_index = payload["edge_index"]
    batch = payload["batch"]
    value_norms: Dict[str, torch.Tensor] = payload.get("value_norms", {})

    key = args.key.strip() or _pick_key(attention, batch)
    A_full = attention[key].detach().cpu().float().numpy()
    vn_full = (
        value_norms[key].detach().cpu().float().numpy()
        if key in value_norms
        else None
    )

    graphs = _split_graphs(edge_index, batch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    n_plot = min(args.n_graphs, len(graphs))
    ncols = min(4, n_plot)
    nrows = int(math.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)

    for gi, (ei, n, start) in enumerate(graphs):
        A = A_full[start : start + n, start : start + n]
        alpha = _alpha(A)
        sink = int(alpha.argmax())
        deg = _degree(ei, n)
        deg_rank = int((-deg).argsort().tolist().index(sink))  # 0 = highest deg
        uniform = 1.0 / max(n, 1)
        is_tau = bool(_tau_sink(A, args.tau)[sink])
        vnr = float("nan")
        if vn_full is not None:
            vn = vn_full[start : start + n]
            vnr = float(vn[sink] / (vn.mean() + 1e-8))
        rec = {
            "graph": gi,
            "n": n,
            "sink": sink,
            "max_alpha": float(alpha[sink]),
            "uniform": uniform,
            "ratio_vs_uniform": float(alpha[sink] / uniform),
            "deg": float(deg[sink]),
            "deg_mean": float(deg.mean()),
            "deg_rank": deg_rank,
            "is_hub": deg_rank == 0,
            "tau_sink": is_tau,
            "vnorm_ratio": vnr,
        }
        records.append(rec)

        if gi < n_plot:
            ax = axes[gi // ncols][gi % ncols]
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(ei.t().tolist())
            pos = nx.spring_layout(G, seed=0)
            node_colors = ["#cccccc"] * n
            node_colors[sink] = "#d62728"
            sizes = [180 + 40 * deg[i] for i in range(n)]
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.8)
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color=node_colors, node_size=sizes, linewidths=0.5, edgecolors="k"
            )
            ax.set_title(
                f"g{gi} n={n} sink={sink}\n"
                f"α={alpha[sink]:.2f} ({alpha[sink]/uniform:.1f}×unif) "
                f"deg_rank={deg_rank}",
                fontsize=8,
            )
            ax.axis("off")

    for j in range(n_plot, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Sink node (red) · {key} · {Path(args.pt).name}", fontsize=11)
    fig.tight_layout()
    fig_path = out_dir / f"sink_graphs_{key}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Commonality summary over all graphs in the batch
    n_rec = len(records)
    hub_frac = float(np.mean([r["is_hub"] for r in records]))
    tau_frac = float(np.mean([r["tau_sink"] for r in records]))
    mean_ratio = float(np.mean([r["ratio_vs_uniform"] for r in records]))
    mean_deg_rank = float(np.mean([r["deg_rank"] for r in records]))
    mean_vnr = float(np.nanmean([r["vnorm_ratio"] for r in records]))

    summary_path = out_dir / f"sink_commonality_{key}.txt"
    lines = [
        f"pt={args.pt}",
        f"key={key}",
        f"n_graphs={n_rec}",
        f"frac_sink_is_max_degree_hub={hub_frac:.3f}",
        f"mean_degree_rank_of_sink={mean_deg_rank:.2f}  (0=hub)",
        f"frac_tau_mu_sink_on_argmax={tau_frac:.3f}",
        f"mean_max_alpha_over_uniform={mean_ratio:.2f}x",
        f"mean_sink_vnorm_ratio={mean_vnr:.3f}",
        "",
        "per-graph:",
    ]
    for r in records:
        lines.append(
            f"  g{r['graph']}: n={r['n']} sink={r['sink']} α={r['max_alpha']:.3f} "
            f"({r['ratio_vs_uniform']:.1f}×) deg={r['deg']:.0f}/{r['deg_mean']:.1f} "
            f"rank={r['deg_rank']} hub={r['is_hub']} tau={r['tau_sink']} vnr={r['vnorm_ratio']:.2f}"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {fig_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
