#!/usr/bin/env python3
"""Ranked graph-level gate profiles for GCN/GIN routing synthetic runs.

Loads ``gate_values_per_node.pt`` (routing or TU format), optionally restricts
to one split (default: test), and writes ranked scatter grids (stacked panels) —
no within-graph node bands.

Aggregations:
  - ``mean`` (default): mean γ over all nodes in each graph
  - ``root``: γ at the classification root only
  - ``both``: write mean and root figures

Color modes:
  - ``tau_y`` (default): four colors for ``(tau, y)`` ∈ {0,1}²
  - ``tau``: two colors by graph type
  - ``y``: two colors by binary label

Example::

  python scripts/synthetic/plot_gcn_gin_routing_ranked_gates.py \\
    --pt results/gcn_gin_routing/gates/toy/a0g2_gated_lr001_seed2/gate_values_per_node.pt \\
    --out-dir results/gate_viz/gcn_gin_routing/a0g2_gated_lr001_seed2 \\
    --split train,val,test --color-by tau_y --sort-head 0

  # Root γ at the classification node:
  --aggregation root

  # Mean + root in one pass:
  --aggregation both

  # Combined 14k graphs:
  --split all
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_VIZ = _REPO_ROOT / "scripts" / "gate_viz"
for _p in (_REPO_ROOT, _GATE_VIZ):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gate_payload_utils import (  # noqa: E402
    load_graph_level_gates,
    tau_y_color_map,
    tau_y_labels,
)
from gate_metrics import HeadPanelStats, head_panel_stats  # noqa: E402
from plot_per_graph_gates import (  # noqa: E402
    _class_colors,
    _head_labels,
    resolve_sort_layer,
    shared_graph_order,
)

ColorBy = Literal["none", "y", "tau", "tau_y"]
Aggregation = Literal["mean", "root"]
AggregationSpec = Literal["mean", "root", "both"]
RUN_NAME_RE = re.compile(r"^a0g2_gated_lr(?P<lr_tag>\d+)_seed(?P<seed>\d+)$")
VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test", "all"})


def _parse_split_list(split_spec: str) -> list[str]:
    """Parse ``--split`` into one or more split tags (``all`` = combined dump)."""
    spec = split_spec.strip().lower()
    if not spec:
        raise ValueError("--split must be non-empty.")
    if spec == "each":
        return ["train", "val", "test"]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if part not in VALID_SPLITS:
            raise ValueError(
                f"Unknown split {part!r}. Use train, val, test, all, or comma-separated.",
            )
    if "all" in parts and len(parts) > 1:
        raise ValueError("Combine 'all' only by itself (not with train/val/test).")
    return parts


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt",
        type=str,
        default="",
        help="Path to gate_values_per_node.pt (single-run mode).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Scan for gated runs under this root (batch mode).",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="toy",
        help="Track subdir when using --results-root (default: toy).",
    )
    parser.add_argument(
        "--lr-tag",
        type=str,
        default="lr001",
        help="Filter runs by lr tag when batching (default: lr001).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="2",
        help="Comma-separated seeds for batch mode (default: 2).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: results/gate_viz/gcn_gin_routing/<run>).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=(
            "Split filter: test | all | train,val,test (comma-separated). "
            "Use 'all' for train+val+test combined (~14k graphs). "
            "Default: test."
        ),
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=("mean", "root", "both"),
        help=(
            "Graph-level gate: mean over nodes, root only, or both (default: mean). "
            "Root plots are written with '_root_' in the filename."
        ),
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="tau_y",
        choices=("none", "y", "tau", "tau_y"),
        help="Point coloring (default: tau_y = four colors).",
    )
    parser.add_argument(
        "--sort-mode",
        type=str,
        default="shared",
        choices=("shared", "per_panel"),
        help="Graph ranking mode (default: shared).",
    )
    parser.add_argument(
        "--sort-branch",
        type=str,
        default="gnn",
        choices=("gnn", "attn"),
        help="Branch for shared ranking (default: gnn).",
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
        default=0,
        help="Head for shared ranking (0=GIN/ROUTING_SUM for a0g2).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG dpi.")
    return parser.parse_args(argv)


def _color_groups(
    color_by: ColorBy,
    *,
    y: Optional[np.ndarray],
    tau: Optional[np.ndarray],
) -> tuple[list[Any], list[str], list[str]]:
    """Return (group_keys, labels, colors) for scatter coloring."""
    if color_by == "none":
        return [None], [""], ["#4C72B0"]

    if color_by == "y":
        if y is None:
            raise ValueError("--color-by y requires 'y' in the dump.")
        keys = sorted(int(v) for v in np.unique(y))
        colors = _class_colors(len(keys))
        labels = [rf"$y={k}$" for k in keys]
        return keys, labels, colors

    if color_by == "tau":
        if tau is None:
            raise ValueError("--color-by tau requires 'tau' in the dump.")
        keys = sorted(int(v) for v in np.unique(tau))
        colors = ["#4C72B0", "#DD8452"][: len(keys)]
        labels = [rf"$\tau={k}$" for k in keys]
        return keys, labels, colors

    if y is None or tau is None:
        raise ValueError("--color-by tau_y requires both 'tau' and 'y' in the dump.")
    cmap = tau_y_color_map()
    labmap = tau_y_labels()
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = [labmap[k] for k in keys]
    colors = [cmap[k] for k in keys]
    return keys, labels, colors


def _mask_for_group(
    group: Any,
    *,
    color_by: ColorBy,
    y: Optional[np.ndarray],
    tau: Optional[np.ndarray],
    order: np.ndarray,
) -> np.ndarray:
    """Boolean mask over ranks for one color group."""
    if color_by == "none":
        return np.ones(len(order), dtype=bool)
    assert y is not None
    y_ord = y[order]
    if color_by == "y":
        return y_ord == int(group)
    assert tau is not None
    tau_ord = tau[order]
    if color_by == "tau":
        return tau_ord == int(group)
    t_g, y_g = group
    return (tau_ord == int(t_g)) & (y_ord == int(y_g))


def _format_mean_std(stats: HeadPanelStats) -> str:
    """Panel annotation: graph-mean γ only (no layer entropy)."""
    return f"{stats.mean:.2f}±{stats.std:.2f}"


def _gamma_ylabel(layer: int, aggregation: Aggregation) -> str:
    """Y-axis label for one panel."""
    if aggregation == "root":
        return rf"L{layer} $\gamma_{{\mathrm{{root}}}}$"
    return rf"L{layer} $\gamma$"


def plot_ranked_gate_grid(
    values: np.ndarray,
    *,
    kind: str,
    head_names: Sequence[str],
    title: str,
    out_path: Path,
    dpi: int,
    sort_mode: str,
    shared_order: Optional[np.ndarray],
    sort_key_label: str,
    color_by: ColorBy,
    aggregation: Aggregation,
    y: Optional[np.ndarray],
    tau: Optional[np.ndarray],
    ref_branch: Optional[str] = None,
    ref_layer: Optional[int] = None,
    ref_head: Optional[int] = None,
) -> None:
    """Write stacked ranked gate panels (scatter, no node bands)."""
    n_graphs, n_layers, n_heads = values.shape
    if n_heads == 0:
        logging.info("Skipping %s branch (H=0).", kind)
        return

    group_keys, group_labels, group_colors = _color_groups(
        color_by,
        y=y,
        tau=tau,
    )

    n_panels = n_layers * n_heads
    fig_w = max(6.0, 3.2)
    fig_h = max(2.4 * n_panels, 4.0)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

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
            panel_idx = layer * n_heads + head
            ax = axes[panel_idx, 0]
            col = values[:, layer, head]
            order = order_fixed if order_fixed is not None else np.argsort(-col)
            ranks = np.arange(n_graphs)
            y_vals = col[order]

            for gi, group in enumerate(group_keys):
                mask = _mask_for_group(
                    group,
                    color_by=color_by,
                    y=y,
                    tau=tau,
                    order=order,
                )
                if not bool(mask.any()):
                    continue
                label = group_labels[gi] if color_by != "none" else None
                ax.scatter(
                    ranks[mask],
                    y_vals[mask],
                    s=12,
                    alpha=0.75,
                    c=group_colors[gi],
                    edgecolors="none",
                    label=label if panel_idx == 0 else None,
                    zorder=3,
                )

            if (
                sort_mode == "shared"
                and ref_branch == kind
                and ref_layer == layer
                and ref_head == head
            ):
                for spine in ax.spines.values():
                    spine.set_color("#C44E52")
                    spine.set_linewidth(1.5)

            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_title(head_names[head], fontsize=10, fontweight="bold")
            ax.set_ylabel(_gamma_ylabel(layer, aggregation), fontsize=9)
            if panel_idx == n_panels - 1:
                ax.set_xlabel(xlabel, fontsize=8)

            panel_stats = head_panel_stats(values, layer=layer, head=head)
            ax.text(
                0.98,
                0.05,
                _format_mean_std(panel_stats),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                },
            )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    if color_by != "none":
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(4, len(labels)),
                fontsize=9,
                framealpha=0.95,
            )
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_path)


def _discover_pts(
    results_root: Path,
    track: str,
    lr_tag: str,
    seeds: Sequence[int],
) -> list[tuple[Path, str]]:
    """Find ``gate_values_per_node.pt`` for gated routing runs."""
    found: list[tuple[Path, str]] = []
    track_dir = results_root / track
    if not track_dir.is_dir():
        return found
    for run_dir in sorted(track_dir.iterdir()):
        if not run_dir.is_dir() or "_failed" in run_dir.name:
            continue
        match = RUN_NAME_RE.match(run_dir.name)
        if match is None:
            continue
        run_lr = f"lr{match.group('lr_tag')}"
        seed = int(match.group("seed"))
        if run_lr != lr_tag or seed not in seeds:
            continue
        pt = run_dir / "gate_values_per_node.pt"
        if pt.is_file():
            found.append((pt, run_dir.name))
    return found


def _ranked_output_name(
    track: str,
    kind: str,
    split_tag: str,
    order_tag: str,
    color_tag: str,
    aggregation: Aggregation,
) -> str:
    """Build PNG basename; root plots include ``_root_`` (mean keeps legacy names)."""
    agg_infix = "root_" if aggregation == "root" else ""
    return (
        f"{track}_gates_{kind}_ranked_{agg_infix}{split_tag}_{order_tag}_by_{color_tag}.png"
    )


def plot_one_run(
    pt_path: Path,
    out_dir: Path,
    *,
    split: str,
    aggregation: Aggregation,
    color_by: ColorBy,
    sort_mode: str,
    sort_branch: str,
    sort_layer: int,
    sort_head: int,
    dpi: int,
) -> list[Path]:
    """Generate ranked mean gate figures for one checkpoint dump."""
    split_arg = None if split == "all" else split  # type: ignore[arg-type]
    try:
        data = load_graph_level_gates(
            str(pt_path),
            split=split_arg,
            aggregation=aggregation,
        )
    except ValueError as exc:
        logging.warning("Skipping %s split for %s: %s", split, pt_path.parent.name, exc)
        return []
    attn = data["attn"]
    gnn = data["gnn"]
    meta = data["meta"]
    n_graphs = int(data["num_graphs"])
    y = data.get("y")
    tau = data.get("tau")

    track = str(meta.get("track", pt_path.parent.parent.name))
    seed = meta.get("seed", "?")
    dataset = str(meta.get("dataset", "gcn_gin_routing"))
    gamma_phrase = "graph-mean γ" if aggregation == "mean" else "root γ"

    attn_names = _head_labels(meta, "attn", int(attn.shape[-1]))
    gnn_names = _head_labels(meta, "gnn", int(gnn.shape[-1]))

    shared_order: Optional[np.ndarray] = None
    sort_key_label = "per panel"
    ref_branch: Optional[str] = None
    ref_layer: Optional[int] = None
    ref_head: Optional[int] = None
    if sort_mode == "shared":
        shared_order, _ = shared_graph_order(
            attn,
            gnn,
            branch=sort_branch,
            layer=sort_layer,
            head=sort_head,
        )
        n_l = int(gnn.shape[1] if sort_branch == "gnn" else attn.shape[1])
        ref_branch = sort_branch
        ref_layer = resolve_sort_layer(n_l, sort_layer)
        ref_head = sort_head
        names = gnn_names if sort_branch == "gnn" else attn_names
        sort_key_label = f"L{ref_layer} {names[ref_head]} γ↓"

    split_tag = split if split != "all" else "all_splits"
    color_tag = color_by if color_by != "none" else "plain"
    order_tag = "shared_order" if sort_mode == "shared" else "by_rank"
    base_title = (
        f"Ranked {gamma_phrase} | {track} | seed {seed} | "
        f"{split_tag} · {n_graphs} graphs · order: {sort_key_label}"
    )

    written: list[Path] = []
    for kind, arr, names, branch_title in (
        ("attn", attn, attn_names, "attention"),
        ("gnn", gnn, gnn_names, "MP heads"),
    ):
        if int(arr.shape[-1]) == 0:
            continue
        out_path = out_dir / _ranked_output_name(
            track,
            kind,
            split_tag,
            order_tag,
            color_tag,
            aggregation,
        )
        plot_ranked_gate_grid(
            arr,
            kind=kind,
            head_names=names,
            title=f"{base_title} · {branch_title}",
            out_path=out_path,
            dpi=dpi,
            sort_mode=sort_mode,
            shared_order=shared_order,
            sort_key_label=sort_key_label,
            color_by=color_by,
            aggregation=aggregation,
            y=y,
            tau=tau,
            ref_branch=ref_branch,
            ref_layer=ref_layer,
            ref_head=ref_head,
        )
        written.append(out_path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.pt:
        pt_paths = [(Path(args.pt).expanduser().resolve(), Path(args.pt).parent.name)]
    elif args.results_root:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        pt_paths = _discover_pts(
            Path(args.results_root).expanduser().resolve(),
            args.track,
            args.lr_tag,
            seeds,
        )
    else:
        raise SystemExit("Provide --pt or --results-root.")

    if not pt_paths:
        raise SystemExit("No gate_values_per_node.pt files found.")

    split_list = _parse_split_list(str(args.split))
    agg_spec: AggregationSpec = args.aggregation  # type: ignore[assignment]
    aggregations: list[Aggregation] = (
        ["mean", "root"] if agg_spec == "both" else [agg_spec]  # type: ignore[list-item]
    )
    viz_root = _REPO_ROOT / "results" / "gate_viz" / "gcn_gin_routing"
    batch_track = str(args.track) if args.results_root else ""
    for pt_path, run_name in pt_paths:
        if args.out_dir:
            out_dir = Path(args.out_dir).expanduser().resolve()
        else:
            track_name = batch_track or pt_path.parent.parent.name
            out_dir = viz_root / track_name / run_name
        print(f"\n{run_name} → {out_dir}")
        for split_name in split_list:
            for aggregation in aggregations:
                written = plot_one_run(
                    pt_path,
                    out_dir,
                    split=split_name,
                    aggregation=aggregation,
                    color_by=args.color_by,  # type: ignore[arg-type]
                    sort_mode=str(args.sort_mode),
                    sort_branch=str(args.sort_branch),
                    sort_head=int(args.sort_head),
                    sort_layer=int(args.sort_layer),
                    dpi=int(args.dpi),
                )
                for path in written:
                    agg_tag = "" if aggregation == "mean" else f" [{aggregation}]"
                    print(f"  [{split_name}]{agg_tag} {path}")


if __name__ == "__main__":
    main()
