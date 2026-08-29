#!/usr/bin/env python3
"""Plot fully worked forward traces for GCN/GIN routing synthetic graphs.

For each graph type ``tau in {0, 1}``, saves one **correct** and one **incorrect**
test example (under a trained SiGMA gated model) showing:

- Input node features
- Encoded features (LinearNode encoder)
- Per-head routing MP outputs (GIN sum / GCN norm-sum), gates, and gated outputs
- Fused layer features at the root
- Graph readout logits and prediction

Outputs (default):
  ``<out_dir>/fig_forward_tau{0,1}_{correct,incorrect}.png``

Example:
  python scripts/synthetic/plot_gcn_gin_routing_forward_trace.py \\
    --run-dir results/gcn_gin_routing/toy/a0g2_gated_lr001_seed0 \\
    --dataset-dir results/gcn_gin_routing/data
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.checkpoint import get_ckpt_epochs, load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401 — register modules
from GNNPlus.gcn_gin_routing_gate_tracking import hybrid_head_indices
from GNNPlus.hybrid_gate_tracking import _unwrap_model
from GNNPlus.layer.gated_hybrid_layer import GatedHybridGraphLayer
from GNNPlus.loader.dataset.gcn_gin_routing import FEAT_SIGNAL

from scripts.synthetic.analyze_gcn_gin_routing_results import (  # noqa: E402
    RunRef,
    _load_cfg_for_run,
    _parse_run_ref,
    _pick_best_epoch,
    _pred_labels_from_score,
    _resolve_run_config,
)

CaseKey = tuple[int, Literal["correct", "incorrect"]]
ROLE_NAMES: tuple[str, str, str] = ("root", "signal", "dummy")


@dataclass
class GraphForwardTrace:
    """Intermediate tensors for one single-graph forward pass."""

    graph_idx: int
    tau: int
    true_label: int
    pred_label: int
    correct: bool
    gcn_score: float
    gin_score: float
    node_roles: np.ndarray
    x_raw: np.ndarray
    x_encoded: np.ndarray
    gin_raw: np.ndarray
    gin_gamma: np.ndarray
    gin_out: np.ndarray
    gcn_raw: np.ndarray
    gcn_gamma: np.ndarray
    gcn_out: np.ndarray
    x_fused: np.ndarray
    graph_embed: np.ndarray
    logits: np.ndarray
    prob_class1: float
    gin_head_name: str
    gcn_head_name: str


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Trained run directory (config + ckpt/).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(_REPO_ROOT / "results/gcn_gin_routing/data"),
        help="Parent of GcnGinRouting/ for PyG loader.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO_ROOT / "results/gcn_gin_routing/analysis/forward_traces"),
        help="Directory for output PNG figures.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to search for examples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args(argv)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device from CLI choice."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_ref_from_dir(run_dir: Path) -> RunRef:
    """Build a :class:`RunRef` from a run directory path."""
    track = run_dir.parent.name
    ref = _parse_run_ref(track, run_dir)
    if ref is None:
        raise ValueError(f"Unrecognized run directory name: {run_dir.name}")
    if _resolve_run_config(run_dir) is None:
        raise FileNotFoundError(f"No config yaml in {run_dir}")
    return ref


def _routing_head_raw(
    head: torch.nn.Module,
    x: Tensor,
    edge_index: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return raw MP output and gate γ before multiplying."""
    signal = x[:, 0:1]
    if head.gate_proj is None:
        gamma = torch.ones(signal.size(0), 1, device=x.device, dtype=x.dtype)
    else:
        gamma = torch.sigmoid(head.gate_proj(x))
    raw = head.conv(signal, edge_index)
    return raw, gamma


@torch.no_grad()
def trace_single_graph(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    *,
    graph_idx: int,
    gin_idx: int,
    gcn_idx: int,
    gin_name: str,
    gcn_name: str,
) -> GraphForwardTrace:
    """Run a detailed forward trace on one graph."""
    core = _unwrap_model(model)
    batch = Batch.from_data_list([data.clone()]).to(device)
    x_raw = batch.x.detach().cpu().numpy().copy()

    (
        x_enc,
        batch_enc,
        _ei_attn,
        _ea_attn,
        edge_index_mp,
        edge_attr_mp,
        _ei,
        _ea,
    ) = core._encode_batch(batch)

    layer0 = core.layers[0]
    if not isinstance(layer0, GatedHybridGraphLayer):
        raise TypeError("Expected GatedHybridGraphLayer at layers[0]")

    gin_head = layer0.mp_heads[gin_idx]
    gcn_head = layer0.mp_heads[gcn_idx]
    gin_raw_t, gin_gamma_t = _routing_head_raw(gin_head, x_enc, edge_index_mp)
    gcn_raw_t, gcn_gamma_t = _routing_head_raw(gcn_head, x_enc, edge_index_mp)
    gin_out_t = gin_raw_t * gin_gamma_t
    gcn_out_t = gcn_raw_t * gcn_gamma_t

    layer_out = layer0(
        x_enc,
        edge_index_mp,
        batch_enc.batch,
        edge_attr_mp,
        edge_index_attn=_ei_attn,
        edge_attr_attn=_ea_attn,
        edge_index_mp=edge_index_mp,
        edge_attr_mp=edge_attr_mp,
    )
    x_fused_t = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    batch_enc.x = x_fused_t
    pred, true = core.post_mp(batch_enc)
    _loss, pred_score = compute_loss(pred, true)
    pred_label = int(_pred_labels_from_score(pred_score).view(-1)[0].item())
    true_label = int(true.view(-1)[0].item())
    logits = pred_score.detach().cpu().numpy().reshape(-1)
    if logits.size == 1:
        prob = float(1.0 / (1.0 + np.exp(-logits[0])))
    else:
        probs = torch.softmax(torch.as_tensor(logits), dim=0).numpy()
        prob = float(probs[1]) if probs.size > 1 else float(probs[0])

    graph_embed = x_fused_t[batch_enc.ptr[0].item()].detach().cpu().numpy()

    roles = data.node_role.cpu().numpy()
    return GraphForwardTrace(
        graph_idx=graph_idx,
        tau=int(data.tau.item()),
        true_label=true_label,
        pred_label=pred_label,
        correct=pred_label == true_label,
        gcn_score=float(data.gcn_score.item()),
        gin_score=float(data.gin_score.item()),
        node_roles=roles,
        x_raw=x_raw,
        x_encoded=x_enc.detach().cpu().numpy(),
        gin_raw=gin_raw_t.detach().cpu().numpy(),
        gin_gamma=gin_gamma_t.detach().cpu().numpy(),
        gin_out=gin_out_t.detach().cpu().numpy(),
        gcn_raw=gcn_raw_t.detach().cpu().numpy(),
        gcn_gamma=gcn_gamma_t.detach().cpu().numpy(),
        gcn_out=gcn_out_t.detach().cpu().numpy(),
        x_fused=x_fused_t.detach().cpu().numpy(),
        graph_embed=graph_embed.reshape(-1),
        logits=logits,
        prob_class1=prob,
        gin_head_name=gin_name,
        gcn_head_name=gcn_name,
    )


def _collect_examples(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    gin_idx: int,
    gcn_idx: int,
    gin_name: str,
    gcn_name: str,
    data_cache: dict[int, Data],
    graph_offset: int = 0,
    found: Optional[dict[CaseKey, GraphForwardTrace]] = None,
) -> dict[CaseKey, GraphForwardTrace]:
    """Find first correct/incorrect example per tau on the loader."""
    wanted: set[CaseKey] = {
        (0, "correct"),
        (0, "incorrect"),
        (1, "correct"),
        (1, "incorrect"),
    }
    out: dict[CaseKey, GraphForwardTrace] = {} if found is None else dict(found)

    model.eval()
    offset = graph_offset
    for batch in loader:
        batch = batch.to(device)
        ptr = batch.ptr.cpu().tolist()
        n_graphs = len(ptr) - 1
        for g in range(n_graphs):
            data = _extract_graph_from_batch(batch, g)
            data_cache[offset + g] = data.cpu()
            trace = trace_single_graph(
                model,
                data,
                device,
                graph_idx=offset + g,
                gin_idx=gin_idx,
                gcn_idx=gcn_idx,
                gin_name=gin_name,
                gcn_name=gcn_name,
            )
            tau = trace.tau
            key: CaseKey = (tau, "correct" if trace.correct else "incorrect")
            if key in wanted and key not in out:
                out[key] = trace
        offset += n_graphs
        if len(out) == len(wanted):
            break
    return out


def _extract_graph_from_batch(batch: Batch, graph_idx: int) -> Data:
    """Extract one graph from a batched PyG object."""
    data = batch.get_example(graph_idx)
    for key in ("tau", "gcn_score", "gin_score", "node_role", "root_index"):
        if hasattr(batch, key):
            val = getattr(batch, key)
            if val is None:
                continue
            if isinstance(val, Tensor) and val.dim() == 0:
                setattr(data, key, val)
            elif isinstance(val, Tensor):
                setattr(data, key, val[graph_idx])
    return data


def _node_labels(roles: np.ndarray) -> list[str]:
    """Human-readable node labels for tables."""
    labels: list[str] = []
    sig_i = 0
    dum_i = 0
    for role in roles:
        if role == 0:
            labels.append("r")
        elif role == 1:
            labels.append(f"u{sig_i}")
            sig_i += 1
        else:
            labels.append(f"d{dum_i}")
            dum_i += 1
    return labels


def _fmt(val: float, *, precision: int = 3) -> str:
    """Format a scalar for table display."""
    if abs(val) < 1e-4 and val != 0.0:
        return f"{val:.2e}"
    return f"{val:.{precision}f}"


def _star_layout(data: Data) -> dict[int, tuple[float, float]]:
    """Place root at center, neighbors on a ring, dummies on short spokes."""
    roles = data.node_role.cpu().numpy()
    root = int(data.root_index.item())
    pos: dict[int, tuple[float, float]] = {root: (0.0, 0.0)}

    neighbors = [i for i, r in enumerate(roles) if r == 1]
    k = len(neighbors)
    radius = 1.35
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
        spread = 0.22
        leaf_radius = 0.55
        n_leaves = len(leaves)
        for li, leaf in enumerate(leaves):
            offset = (li - (n_leaves - 1) / 2.0) * spread
            ang = base_angle + offset
            pos[leaf] = (
                ux + leaf_radius * np.cos(ang),
                uy + leaf_radius * np.sin(ang),
            )
    return pos


def _draw_graph_panel(ax: plt.Axes, data: Data, trace: GraphForwardTrace) -> None:
    """Draw the star graph with node annotations."""
    roles = trace.node_roles
    signals = data.x[:, FEAT_SIGNAL].cpu().numpy()
    tau = trace.tau
    g = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        g.add_edge(int(src), int(dst))

    pos = _star_layout(data)
    ax.set_aspect("equal")
    ax.axis("off")

    node_colors: list[str] = []
    node_sizes: list[float] = []
    labels: dict[int, str] = {}
    for i in range(data.num_nodes):
        role = int(roles[i])
        if role == 0:
            node_colors.append("#4C78A8")
            node_sizes.append(900.0)
            labels[i] = f"r\nτ={tau}"
        elif role == 1:
            node_colors.append("#F58518" if signals[i] > 0 else "#E45756")
            node_sizes.append(650.0)
            labels[i] = f"{int(signals[i]):+d}"
        else:
            node_colors.append("#BAB0AC")
            node_sizes.append(180.0)
            labels[i] = ""

    nx.draw_networkx_edges(g, pos, ax=ax, width=1.1, alpha=0.6, edge_color="#666666")
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.0,
        edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_weight="bold",
    )

    status = "CORRECT" if trace.correct else "INCORRECT"
    rule = "GCN-type" if tau == 0 else "GIN-type"
    ax.set_title(
        f"τ={tau} ({rule}) · y={trace.true_label} · ŷ={trace.pred_label} · {status}",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )


def _draw_table(
    ax: plt.Axes,
    *,
    title: str,
    col_labels: list[str],
    rows: list[list[str]],
    highlight_rows: Optional[set[int]] = None,
) -> None:
    """Draw a compact matplotlib table."""
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=4)
    if not rows:
        ax.text(0.5, 0.5, "(no rows)", ha="center", va="center")
        return
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    highlight_rows = highlight_rows or set()
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C78A8")
            cell.set_text_props(color="white", weight="bold")
        elif row - 1 in highlight_rows:
            cell.set_facecolor("#FFF3CD")
        else:
            cell.set_facecolor("#FAFAFA")


def _key_node_indices(roles: np.ndarray) -> list[int]:
    """Root + signal neighbors only (skip dummy leaves for readability)."""
    return [i for i, r in enumerate(roles) if r in (0, 1)]


def save_forward_trace_figure(
    data: Data,
    trace: GraphForwardTrace,
    out_path: Path,
    *,
    dpi: int,
) -> None:
    """Render one fully worked forward trace to PNG."""
    key_nodes = _key_node_indices(trace.node_roles)
    node_names = _node_labels(trace.node_roles)
    highlight = {key_nodes.index(i) for i in key_nodes if trace.node_roles[i] == 0}

    def row_for_node(i: int) -> list[str]:
        return [
            node_names[i],
            _fmt(trace.x_raw[i, FEAT_SIGNAL]),
            _fmt(trace.x_raw[i, FEAT_TYPE]),
            _fmt(trace.x_encoded[i, 0]),
            _fmt(trace.x_encoded[i, 1]),
            _fmt(trace.gin_raw[i, 0]),
            _fmt(trace.gin_gamma[i, 0]),
            _fmt(trace.gin_out[i, 0]),
            _fmt(trace.gcn_raw[i, 0]),
            _fmt(trace.gcn_gamma[i, 0]),
            _fmt(trace.gcn_out[i, 0]),
            _fmt(trace.x_fused[i, 0]),
            _fmt(trace.x_fused[i, 1]),
        ]

    stage_rows = [row_for_node(i) for i in key_nodes]
    stage_cols = [
        "node",
        "x₀",
        "x₁",
        "h₀⁽⁰⁾",
        "h₁⁽⁰⁾",
        f"{trace.gin_head_name} raw",
        "γ_GIN",
        "GIN out",
        f"{trace.gcn_head_name} raw",
        "γ_GCN",
        "GCN out",
        "z₀",
        "z₁",
    ]

    readout_rows = [
        ["Root embedding z", ", ".join(_fmt(v) for v in trace.graph_embed)],
        ["Logits", ", ".join(_fmt(v) for v in trace.logits)],
        ["P(y=1)", _fmt(trace.prob_class1)],
        ["Prediction ŷ", str(trace.pred_label)],
        ["Label y", str(trace.true_label)],
        ["Rule s_GIN (analytic)", _fmt(trace.gin_score)],
        ["Rule s_GCN (analytic)", _fmt(trace.gcn_score)],
    ]

    fig = plt.figure(figsize=(14.5, 11.5))
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.1, 1.6, 0.55],
        hspace=0.28,
        top=0.94,
        bottom=0.04,
        left=0.05,
        right=0.98,
    )
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[1, 0])
    ax_readout = fig.add_subplot(gs[2, 0])

    _draw_graph_panel(ax_graph, data, trace)
    _draw_table(
        ax_table,
        title=(
            "Forward trace (root r + signal neighbors uᵢ; dummy leaves omitted). "
            "Layer 0 = node encoder → hybrid MP heads → fuse."
        ),
        col_labels=stage_cols,
        rows=stage_rows,
        highlight_rows=highlight,
    )
    _draw_table(
        ax_readout,
        title="Graph readout (graph_token = root node z) and label-rule scores",
        col_labels=["quantity", "value"],
        rows=readout_rows,
    )

    fig.suptitle(
        f"GCN/GIN routing forward trace · graph #{trace.graph_idx}",
        fontsize=13,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _load_model_and_loaders(
    run_dir: Path,
    dataset_dir: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[Any], RunRef, int, int, str, str]:
    """Load cfg, model checkpoint, and all split loaders."""
    run_ref = _run_ref_from_dir(run_dir)
    _load_cfg_for_run(run_ref, dataset_dir)
    seed_everything(int(cfg.seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    loaders = list(create_loader())

    model = create_model()
    epoch = _pick_best_epoch(run_dir)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    model.eval()
    model.to(device)

    hybrid = getattr(cfg.gnn, "hybrid", None)
    gnn_types = str(getattr(hybrid, "gnn_types", "")) if hybrid is not None else ""
    gin_idx, gcn_idx, _ = hybrid_head_indices(gnn_types)
    parts = [p.strip() for p in gnn_types.split(",") if p.strip()]
    gin_name = parts[gin_idx] if gin_idx < len(parts) else "GIN"
    gcn_name = parts[gcn_idx] if gcn_idx < len(parts) else "GCN"
    return model, loaders, run_ref, gin_idx, gcn_idx, gin_name, gcn_name


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate forward-trace figures for four tau × correctness cases."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    device = _select_device(args.device)

    model, loaders, run_ref, gin_idx, gcn_idx, gin_name, gcn_name = _load_model_and_loaders(
        run_dir,
        args.dataset_dir,
        device,
    )
    logging.info(
        "Loaded %s epoch from %s (heads: %s idx=%d, %s idx=%d)",
        run_ref.model,
        run_dir,
        gin_name,
        gin_idx,
        gcn_name,
        gcn_idx,
    )

    split_order = [args.split]
    for fallback in ("test", "val", "train"):
        if fallback not in split_order:
            split_order.append(fallback)

    data_cache: dict[int, Data] = {}
    found: dict[CaseKey, GraphForwardTrace] = {}
    for split_name in split_order:
        split_idx = {"train": 0, "val": 1, "test": 2}[split_name]
        if split_idx >= len(loaders):
            continue
        logging.info("Scanning split=%s for examples", split_name)
        found = _collect_examples(
            model,
            loaders[split_idx],
            device,
            gin_idx=gin_idx,
            gcn_idx=gcn_idx,
            gin_name=gin_name,
            gcn_name=gcn_name,
            data_cache=data_cache,
            graph_offset=len(data_cache),
            found=found,
        )
        if len(found) == 4:
            break

    missing = [
        f"tau={tau} {kind}"
        for tau in (0, 1)
        for kind in ("correct", "incorrect")
        if (tau, kind) not in found
    ]
    if missing:
        logging.warning(
            "Missing cases after scanning splits: %s. "
            "Use toy-track a0g2_gated (sigma is often 100%% acc).",
            ", ".join(missing),
        )

    for (tau, kind), trace in sorted(found.items()):
        data = data_cache.get(trace.graph_idx)
        if data is None:
            logging.warning("Skipping graph #%d (data not in cache)", trace.graph_idx)
            continue
        out_path = out_dir / f"fig_forward_tau{tau}_{kind}.png"
        save_forward_trace_figure(data, trace, out_path, dpi=args.dpi)
        logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
