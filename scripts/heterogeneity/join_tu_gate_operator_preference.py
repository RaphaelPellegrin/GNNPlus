#!/usr/bin/env python3
"""Join TU heterogeneity profiles with SiGMA hetero gate dumps.

For each graph, infer operator preference from standalone GCN/GIN/SAGE/GatedGCN
heterogeneity pickles (per-graph mean test accuracy), then test whether SiGMA's
learned MP gates at the readout layer align with that preference.

Example (after cluster pull)::

    python scripts/heterogeneity/join_tu_gate_operator_preference.py \\
      --dataset mutag \\
      --hetero-root results/heterogeneity/powerful_gnns/tu_gate_bridge \\
      --gate-pt results/tu_sigma_homo_hetero/mutag_SiGMA_hetero_lr001_seed2/gate_values_per_graph.pt \\
      --out-dir results/heterogeneity/tu_gate_bridge_analysis/mutag
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OPERATORS: Tuple[str, ...] = ("GCN", "GIN", "SAGE", "GATEDGCN")
OPERATOR_CFG_SUFFIX: Mapping[str, str] = {
    "GCN": "gcn",
    "GIN": "gin",
    "SAGE": "sage",
    "GATEDGCN": "gatedgcn",
}
# SiGMA hetero a2g4 MP heads (must match tu_sigma_homo_hetero training).
SIGMA_MP_HEADS: Tuple[str, ...] = ("GCN", "GIN", "SAGE", "GAT")


@dataclass(frozen=True)
class GraphOperatorRecord:
    """Per-graph operator accuracies and SiGMA gate vector."""

    graph_idx: int
    accuracies: Dict[str, float]
    preferred: str
    margin: float
    sigma_gates: Dict[str, float]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset tag (mutag, enzymes).",
    )
    parser.add_argument(
        "--hetero-root",
        type=str,
        required=True,
        help="Root with <ds>_<model>/ pickles from gate-bridge hetero jobs.",
    )
    parser.add_argument(
        "--gate-pt",
        type=str,
        required=True,
        help="SiGMA hetero gate_values_per_graph.pt (same dataset).",
    )
    parser.add_argument(
        "--gate-run-dir",
        type=str,
        default="",
        help="SiGMA run dir with config_used.yaml (default: parent of --gate-pt).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="Optional dataset root override when reloading SiGMA split.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory for CSV + PNG outputs.",
    )
    parser.add_argument(
        "--gate-layer",
        type=int,
        default=-1,
        help="MP layer index for gate readout (-1 = last).",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default="GCN,GIN,SAGE,GATEDGCN",
        help="Comma-separated operators to join (default: all four).",
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=1,
        help="Minimum test appearances required per graph (default: 1).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI (default: 200).",
    )
    return parser.parse_args(argv)


def _load_avg_accuracy(pickle_path: Path) -> Dict[int, float]:
    """Load per-graph mean test accuracy from a heterogeneity pickle."""
    with pickle_path.open("rb") as fh:
        payload = pickle.load(fh)
    graph_dict: Dict[int, List[int]] = payload["graph_dict"]
    out: Dict[int, float] = {}
    for gidx, vals in graph_dict.items():
        if not vals:
            continue
        out[int(gidx)] = float(np.mean(vals))
    return out


def _find_pickle(model_dir: Path, operator: str) -> Path:
    """Return the graph_dict pickle inside ``model_dir``."""
    layer_tag = operator
    candidates = sorted(model_dir.glob(f"*_{layer_tag}_L*_graph_dict.pickle"))
    if not candidates:
        candidates = sorted(model_dir.glob("*_graph_dict.pickle"))
    if not candidates:
        raise FileNotFoundError(f"No pickle in {model_dir} for operator {operator}")
    return candidates[0]


def _load_operator_profiles(
    hetero_root: Path,
    dataset: str,
    min_appearances: int,
    operators: Sequence[str],
) -> Dict[int, Dict[str, float]]:
    """Map graph_idx → {operator: avg_accuracy}."""
    merged: Dict[int, Dict[str, float]] = {}
    for op in operators:
        suffix = OPERATOR_CFG_SUFFIX[op]
        model_dir = hetero_root / f"{dataset}_{suffix}"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing hetero dir: {model_dir}")
        pickle_path = _find_pickle(model_dir, op)
        acc = _load_avg_accuracy(pickle_path)
        appearances_csv = model_dir / "test_appearances.csv"
        if not appearances_csv.is_file():
            matches = sorted(model_dir.glob("*_test_appearances.csv"))
            appearances_csv = matches[0] if matches else appearances_csv
        if appearances_csv.is_file() and min_appearances > 1:
            allowed: set[int] = set()
            with appearances_csv.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if int(row["n_test_appearances"]) >= min_appearances:
                        allowed.add(int(row["graph_idx"]))
            acc = {g: v for g, v in acc.items() if g in allowed}
        for gidx, val in acc.items():
            merged.setdefault(gidx, {})[op] = val
    return merged


def _preferred_operator(acc: Mapping[str, float]) -> Tuple[str, float]:
    """Return argmax operator and margin over the runner-up."""
    if not acc:
        raise ValueError("empty accuracy map")
    ordered = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    best_op, best_val = ordered[0]
    second_val = ordered[1][1] if len(ordered) > 1 else 0.0
    return best_op, float(best_val - second_val)


def _subset_global_indices(subset: object) -> List[int]:
    """Return global dataset indices from a GraphGym train/val/test subset."""
    raw: object | None = None
    if hasattr(subset, "indices"):
        raw = subset.indices
        if callable(raw):
            raw = raw()
    elif hasattr(subset, "_indices"):
        raw = subset._indices
        if callable(raw):
            raw = raw()
    if raw is None:
        raise RuntimeError(f"Cannot recover indices from subset {type(subset)!r}")
    return [int(i) for i in raw]  # type: ignore[arg-type]


def _load_test_global_indices(
    run_dir: Path,
    dataset_dir: Optional[str],
) -> List[int]:
    """Recreate the test-split global graph indices for a SiGMA TU run."""
    import GNNPlus  # noqa: F401 — register custom modules
    from torch_geometric import seed_everything
    from torch_geometric.graphgym.cmd_args import parse_args as gg_parse_args
    from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
    from torch_geometric.graphgym.loader import create_loader

    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path} (needed to align graph indices)")
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--cfg", str(cfg_path)]
    try:
        args = gg_parse_args()
        set_cfg(cfg)
        load_cfg(cfg, args)
    finally:
        sys.argv = old_argv
    if dataset_dir:
        cfg.dataset.dir = dataset_dir
    seed_everything(int(cfg.seed))
    loaders = create_loader()
    if len(loaders) < 3:
        raise RuntimeError(f"Expected train/val/test loaders, got {len(loaders)}")
    return _subset_global_indices(loaders[2].dataset)


def _load_sigma_gates(
    gate_pt: Path,
    layer_idx: int,
    *,
    run_dir: Path,
    dataset_dir: Optional[str],
) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
    """Load per-graph mean MP gates on the test split (keyed by global graph_idx)."""
    payload = torch.load(gate_pt, map_location="cpu", weights_only=False)
    gnn: torch.Tensor = payload["gnn"]
    split: torch.Tensor = payload["split"]
    meta = payload.get("meta", {})
    gnn_types_raw = str(meta.get("gnn_types", "GCN,GIN,SAGE,GAT"))
    head_names = [s.strip().upper() for s in gnn_types_raw.split(",") if s.strip()]
    if not head_names:
        head_names = list(SIGMA_MP_HEADS)

    test_global = _load_test_global_indices(run_dir, dataset_dir)
    test_rows = np.where(split.numpy() == 2)[0]
    if len(test_rows) != len(test_global):
        raise RuntimeError(
            f"Test row count {len(test_rows)} != test graph count {len(test_global)} "
            f"for {run_dir}"
        )

    n_layers = int(gnn.shape[1])
    layer = layer_idx if layer_idx >= 0 else n_layers + layer_idx
    if layer < 0 or layer >= n_layers:
        raise IndexError(f"gate layer {layer_idx} out of range for L={n_layers}")

    gates_by_graph: Dict[int, Dict[str, float]] = {}
    for row, gidx in zip(test_rows.tolist(), test_global):
        vec: Dict[str, float] = {}
        for h, name in enumerate(head_names):
            if h >= int(gnn.shape[2]):
                break
            vec[name] = float(gnn[row, layer, h].item())
        gates_by_graph[int(gidx)] = vec
    return gates_by_graph, head_names


def _build_records(
    profiles: Dict[int, Dict[str, float]],
    gates: Dict[int, Dict[str, float]],
    operators: Sequence[str],
) -> List[GraphOperatorRecord]:
    """Align graphs present in both hetero profiles and gate dumps."""
    records: List[GraphOperatorRecord] = []
    n_ops = len(operators)
    common = sorted(set(profiles.keys()) & set(gates.keys()))
    for gidx in common:
        acc = profiles[gidx]
        if len(acc) < n_ops:
            continue
        pref, margin = _preferred_operator(acc)
        records.append(
            GraphOperatorRecord(
                graph_idx=gidx,
                accuracies=dict(acc),
                preferred=pref,
                margin=margin,
                sigma_gates=gates[gidx],
            ),
        )
    return records


def _write_csv(
    records: Sequence[GraphOperatorRecord],
    out_path: Path,
    operators: Sequence[str],
) -> None:
    """Write per-graph join table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["graph_idx", "preferred", "margin"]
        + [f"acc_{op.lower()}" for op in operators]
        + [f"gate_{h.lower()}" for h in SIGMA_MP_HEADS]
    )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row: Dict[str, object] = {
                "graph_idx": rec.graph_idx,
                "preferred": rec.preferred,
                "margin": f"{rec.margin:.4f}",
            }
            for op in operators:
                row[f"acc_{op.lower()}"] = f"{rec.accuracies[op]:.4f}"
            for head in SIGMA_MP_HEADS:
                row[f"gate_{head.lower()}"] = (
                    f"{rec.sigma_gates[head]:.4f}" if head in rec.sigma_gates else ""
                )
            writer.writerow(row)


def _plot_preferred_vs_gate(
    records: Sequence[GraphOperatorRecord],
    head_names: Sequence[str],
    out_path: Path,
    dataset: str,
    dpi: int,
) -> None:
    """Boxplot: SiGMA gate for head H when operator H is preferred vs not."""
    fig, axes = plt.subplots(1, len(head_names), figsize=(3.2 * len(head_names), 4.0))
    if len(head_names) == 1:
        axes = [axes]
    for ax, head in zip(axes, head_names):
        when_pref = [
            rec.sigma_gates[head]
            for rec in records
            if rec.preferred == head and head in rec.sigma_gates
        ]
        when_not = [
            rec.sigma_gates[head]
            for rec in records
            if rec.preferred != head and head in rec.sigma_gates
        ]
        data = [when_pref, when_not]
        ax.boxplot(data, tick_labels=[f"pref {head}", "other"], showfliers=False)
        ax.set_title(head)
        ax.set_ylabel(r"$\gamma$ (readout layer)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"{dataset.upper()}: SiGMA MP gate when operator is preferred vs not",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_acc_vs_gate(
    records: Sequence[GraphOperatorRecord],
    out_path: Path,
    dataset: str,
    dpi: int,
) -> None:
    """Scatter accuracy margin for operator H vs SiGMA gate on head H."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    pairs = [("GCN", "GCN"), ("GIN", "GIN"), ("SAGE", "SAGE"), ("GATEDGCN", "GCN")]
    for ax, (op, head) in zip(axes.ravel(), pairs):
        xs: List[float] = []
        ys: List[float] = []
        for rec in records:
            if head not in rec.sigma_gates:
                continue
            xs.append(rec.accuracies[op] - np.mean(list(rec.accuracies.values())))
            ys.append(rec.sigma_gates[head])
        if xs:
            ax.scatter(xs, ys, s=12, alpha=0.55)
            corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
            ax.set_title(f"{op} acc margin vs gate {head} (r={corr:.2f})")
        ax.set_xlabel(f"{op} acc − mean acc")
        ax.set_ylabel(rf"$\gamma_{{\mathrm{{{head}}}}}$")
        ax.grid(alpha=0.22)
    fig.suptitle(f"{dataset.upper()}: operator accuracy vs SiGMA gate", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    ds = args.dataset.lower().strip()
    hetero_root = Path(args.hetero_root)
    gate_pt = Path(args.gate_pt)
    gate_run_dir = Path(args.gate_run_dir) if args.gate_run_dir else gate_pt.parent
    dataset_dir = args.dataset_dir.strip() or None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    operators = tuple(
        op.strip().upper() for op in str(args.operators).split(",") if op.strip()
    )
    for op in operators:
        if op not in OPERATOR_CFG_SUFFIX:
            raise ValueError(f"Unknown operator {op!r}; choose from {list(OPERATOR_CFG_SUFFIX)}")

    profiles = _load_operator_profiles(hetero_root, ds, args.min_appearances, operators)
    gates, head_names = _load_sigma_gates(
        gate_pt,
        args.gate_layer,
        run_dir=gate_run_dir,
        dataset_dir=dataset_dir,
    )
    records = _build_records(profiles, gates, operators)
    if not records:
        raise RuntimeError("No overlapping graphs between hetero profiles and gate dump.")

    csv_path = out_dir / f"{ds}_operator_gate_join.csv"
    _write_csv(records, csv_path, operators)
    logging.info("Wrote %s (%d graphs)", csv_path, len(records))

    # Preference fractions (paper Table 1 precursor).
    prefs = [rec.preferred for rec in records]
    for op in operators:
        n = sum(1 for p in prefs if p == op)
        logging.info(
            "preferred %s: %d / %d (%.1f%%)",
            op,
            n,
            len(prefs),
            100.0 * n / max(len(prefs), 1),
        )

    pref_plot = out_dir / f"{ds}_preferred_operator_gate_boxplot.png"
    _plot_preferred_vs_gate(records, head_names, pref_plot, ds, args.dpi)
    logging.info("Wrote %s", pref_plot)

    scatter_plot = out_dir / f"{ds}_operator_acc_vs_sigma_gate_scatter.png"
    _plot_scatter_acc_vs_gate(records, scatter_plot, ds, args.dpi)
    logging.info("Wrote %s", scatter_plot)

    # Console summary for paper tables.
    for head in ("GCN", "GIN", "SAGE"):
        pref_vals = [
            rec.sigma_gates[head]
            for rec in records
            if rec.preferred == head and head in rec.sigma_gates
        ]
        other_vals = [
            rec.sigma_gates[head]
            for rec in records
            if rec.preferred != head and head in rec.sigma_gates
        ]
        if pref_vals and other_vals:
            logging.info(
                "%s preferred n=%d mean_gate=%.3f | other n=%d mean_gate=%.3f",
                head,
                len(pref_vals),
                float(np.mean(pref_vals)),
                len(other_vals),
                float(np.mean(other_vals)),
            )


if __name__ == "__main__":
    main()
