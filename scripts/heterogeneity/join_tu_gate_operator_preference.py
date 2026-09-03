#!/usr/bin/env python3
"""Join TU specialist preference with SiGMA hetero gate dumps.

Operator preference comes from standalone heterogeneity pickles (per-graph mean
test accuracy). Gates come from ``gate_values_per_graph.pt``. Dump rows are
aligned to global TUDataset indices by reconstructing GraphGym's
``ShuffleSplit`` (val/test keep loader order; train is shuffled and is only
kept when a topology fingerprint is unique).

Example — local GCN/GIN profiles already on disk::

    python scripts/heterogeneity/join_tu_gate_operator_preference.py \\
      --datasets mutag,enzymes \\
      --hetero-root results/heterogeneity \\
      --gate-root results/tu_sigma_homo_hetero \\
      --operators GCN,GIN \\
      --out-dir results/heterogeneity/tu_gate_bridge_analysis

After pulling ``tu_gate_bridge`` SAGE pickles, add ``SAGE`` to ``--operators``
and point ``--hetero-root`` at ``results/heterogeneity/powerful_gnns/tu_gate_bridge``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OPERATOR_CFG_SUFFIX: Mapping[str, str] = {
    "GCN": "gcn",
    "GIN": "gin",
    "SAGE": "sage",
    "GAT": "gat",
    "GATEDGCN": "gatedgcn",
}
SIGMA_MP_HEADS: Tuple[str, ...] = ("GCN", "GIN", "SAGE", "GAT")
TU_NAME: Mapping[str, str] = {"mutag": "MUTAG", "enzymes": "ENZYMES"}
SPLIT_IDS: Mapping[str, int] = {"train": 0, "val": 1, "test": 2}
DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.5, 0.25, 0.25)
PALETTE: Mapping[str, str] = {
    "GIN": "#55A868",
    "GCN": "#4C72B0",
    "SAGE": "#DD8452",
    "GATEDGCN": "#8172B3",
    "GAT": "#937860",
    "TIE": "#8C8C8C",
}
HEADER_COLOR = "#4C78A8"
ROW_ALT = "#F5F7FA"
ROW_BASE = "#FFFFFF"
EDGE_COLOR = "#D0D7DE"
TABLE_FONT_SIZE = 9.5
DegFingerprint = Tuple[int, int, int, Tuple[int, ...]]


@dataclass(frozen=True)
class GraphOperatorRecord:
    """Per-graph operator accuracies and SiGMA gate vector."""

    graph_idx: int
    split: str
    y: int
    accuracies: Dict[str, float]
    preferred: str
    margin: float
    sigma_gates: Dict[str, float]
    n_gate_seeds: int = 1


@dataclass
class DatasetJoin:
    """Join outputs for one dataset."""

    dataset: str
    records: List[GraphOperatorRecord]
    head_names: List[str]
    operators: Tuple[str, ...]
    n_mapped: int
    n_train_unique: int
    n_train_ambiguous: int
    seed_ids: Tuple[int, ...] = ()
    per_seed_records: List[List[GraphOperatorRecord]] = field(default_factory=list)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Single dataset tag (mutag, enzymes). Prefer --datasets.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated datasets (default: --dataset or mutag,enzymes).",
    )
    parser.add_argument(
        "--hetero-root",
        type=str,
        required=True,
        help="Root with <ds>_<model>/ heterogeneity pickles.",
    )
    parser.add_argument(
        "--gate-pt",
        type=str,
        default="",
        help="Single-dataset gate_values_per_graph.pt (overrides --gate-root).",
    )
    parser.add_argument(
        "--gate-root",
        type=str,
        default="results/tu_sigma_homo_hetero",
        help="Root of <ds>_SiGMA_hetero_<lr>_seed<s>/ dumps.",
    )
    parser.add_argument(
        "--gate-run-tag",
        type=str,
        default="",
        help="Single-run suffix after <ds>_ (overrides --lr-tag/--seeds).",
    )
    parser.add_argument(
        "--lr-tag",
        type=str,
        default="lr001",
        help="SiGMA LR tag used in dump dir names (default: lr001).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated SiGMA seeds to remap then average (default: 0-4).",
    )
    parser.add_argument(
        "--tu-root",
        type=str,
        default="",
        help="PyG TUDataset cache root (default: ~/.cache/pyg_tu_gate_join).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory for CSV + paper_figures.",
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
        default="GCN,GIN",
        help="Comma-separated specialist operators (default: GCN,GIN).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Dump splits to join (default: val,test). Add train for unique-fingerprint matches.",
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=100,
        help="Minimum specialist test appearances per graph (default: 100).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Drop graphs whose specialist accuracy margin is below this.",
    )
    parser.add_argument(
        "--scan-layers",
        action="store_true",
        default=True,
        help="Also compute Δγ at every MP layer (default: on).",
    )
    parser.add_argument(
        "--no-scan-layers",
        action="store_false",
        dest="scan_layers",
        help="Skip the all-layer Δγ sweep.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI (default: 200).",
    )
    return parser.parse_args(argv)


def _parse_csv_list(raw: str) -> Tuple[str, ...]:
    """Split a comma-separated CLI list."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    """Write PNG and PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r, or NaN if undefined."""
    if len(xs) < 3:
        return float("nan")
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)
    if float(arr_x.std()) < 1e-12 or float(arr_y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rho via rank transform."""
    if len(xs) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64)))
    return _pearson(rx.tolist(), ry.tolist())


def reconstruct_graphgym_random_split(
    labels: np.ndarray,
    *,
    seed: int,
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce GraphGym ``setup_random_split`` index arrays."""
    y = np.asarray(labels).reshape(-1)
    train_index, val_test_index = next(
        ShuffleSplit(train_size=split_ratios[0], random_state=seed).split(y, y)
    )
    val_test_ratio = split_ratios[1] / (1.0 - split_ratios[0])
    val_rel, test_rel = next(
        ShuffleSplit(train_size=val_test_ratio, random_state=seed).split(
            y[val_test_index],
            y[val_test_index],
        )
    )
    val_index = val_test_index[val_rel]
    test_index = val_test_index[test_rel]
    return train_index, val_index, test_index


def _degree_fingerprint(edge_index: torch.Tensor, n_nodes: int) -> Tuple[int, ...]:
    """Sorted degree sequence from a (possibly local) edge_index."""
    if n_nodes <= 0:
        return ()
    src = edge_index[0].detach().cpu().numpy().astype(np.int64)
    deg = np.bincount(src, minlength=n_nodes)
    return tuple(int(v) for v in sorted(deg.tolist()))


def _graph_fingerprint(
    y: int,
    n_nodes: int,
    n_edges: int,
    edge_index: torch.Tensor,
) -> DegFingerprint:
    """Label + size + sorted-degree fingerprint."""
    return (int(y), int(n_nodes), int(n_edges), _degree_fingerprint(edge_index, n_nodes))


def _load_tu_dataset(dataset: str, tu_root: Path) -> object:
    """Load the PyG TUDataset used by GraphGym (graph order = global index)."""
    from torch_geometric.datasets import TUDataset

    name = TU_NAME.get(dataset.lower(), dataset.upper())
    tu_root.mkdir(parents=True, exist_ok=True)
    return TUDataset(root=str(tu_root), name=name)


def _tu_labels(dataset_obj: object) -> np.ndarray:
    """Per-graph integer labels in dataset index order."""
    y_attr = getattr(dataset_obj, "y", None)
    if y_attr is not None:
        return np.asarray(y_attr.view(-1).cpu().numpy(), dtype=np.int64)
    n_graphs = int(len(dataset_obj))  # type: ignore[arg-type]
    labels = np.zeros(n_graphs, dtype=np.int64)
    for i in range(n_graphs):
        graph = dataset_obj[i]  # type: ignore[index]
        labels[i] = int(graph.y.view(-1)[0].item())
    return labels


def _tu_fingerprint(dataset_obj: object, graph_idx: int) -> DegFingerprint:
    """Fingerprint of TUDataset graph ``graph_idx``."""
    graph = dataset_obj[graph_idx]  # type: ignore[index]
    n_nodes = int(graph.num_nodes)
    n_edges = int(graph.edge_index.size(1))
    y = int(graph.y.view(-1)[0].item())
    return _graph_fingerprint(y, n_nodes, n_edges, graph.edge_index)


def _dump_row_fingerprint(
    *,
    y: int,
    ptr: torch.Tensor,
    edge_ptr: torch.Tensor,
    edge_index: torch.Tensor,
    row: int,
) -> DegFingerprint:
    """Fingerprint of dump row ``row`` from the per-node payload."""
    lo = int(ptr[row].item())
    hi = int(ptr[row + 1].item())
    n_nodes = hi - lo
    elo = int(edge_ptr[row].item())
    ehi = int(edge_ptr[row + 1].item())
    n_edges = ehi - elo
    local_edges = edge_index[:, elo:ehi] - lo
    return _graph_fingerprint(y, n_nodes, n_edges, local_edges)


def _load_avg_accuracy(pickle_path: Path) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Load per-graph mean test accuracy and appearance counts."""
    with pickle_path.open("rb") as fh:
        payload = pickle.load(fh)
    graph_dict: Dict[int, List[int]] = payload["graph_dict"]
    raw_apps = payload.get("test_appearances", {})
    acc: Dict[int, float] = {}
    apps: Dict[int, int] = {}
    for gidx, vals in graph_dict.items():
        if not vals:
            continue
        key = int(gidx)
        acc[key] = float(np.mean(vals))
        if isinstance(raw_apps, dict) and gidx in raw_apps:
            apps[key] = int(raw_apps[gidx])
        else:
            apps[key] = len(vals)
    return acc, apps


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
        acc, apps = _load_avg_accuracy(pickle_path)
        if min_appearances > 1:
            acc = {g: v for g, v in acc.items() if apps.get(g, 0) >= min_appearances}
        logging.info(
            "%s %s: %d graphs with ≥%d appearances (%s)",
            dataset,
            op,
            len(acc),
            min_appearances,
            pickle_path.name,
        )
        for gidx, val in acc.items():
            merged.setdefault(gidx, {})[op] = val
    return merged


def _preferred_operator(acc: Mapping[str, float]) -> Tuple[str, float]:
    """Return argmax operator and margin over the runner-up (TIE if equal)."""
    if not acc:
        raise ValueError("empty accuracy map")
    ordered = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    best_op, best_val = ordered[0]
    second_val = ordered[1][1] if len(ordered) > 1 else best_val
    margin = float(best_val - second_val)
    if margin <= 0.0:
        return "TIE", 0.0
    return best_op, margin


def _gate_vector(
    gnn: torch.Tensor,
    row: int,
    layer: int,
    head_names: Sequence[str],
) -> Dict[str, float]:
    """MP-gate dict for one dump row."""
    vec: Dict[str, float] = {}
    n_heads = int(gnn.shape[2])
    for h, name in enumerate(head_names):
        if h >= n_heads:
            break
        vec[name] = float(gnn[row, layer, h].item())
    return vec


def _load_sigma_gates(
    gate_pt: Path,
    layer_idx: int,
    *,
    dataset: str,
    tu_root: Path,
    splits: Sequence[str],
) -> Tuple[Dict[int, Tuple[str, int, Dict[str, float]]], List[str], Tuple[int, int]]:
    """Load MP gates keyed by global graph_idx.

    Returns
    -------
    gates
        graph_idx → (split_name, y, gate_dict)
    head_names
        SiGMA MP head names from dump meta.
    train_match_counts
        ``(n_unique_train, n_ambiguous_train)``.
    """
    payload = torch.load(gate_pt, map_location="cpu", weights_only=False)
    gnn: torch.Tensor = payload["gnn"]
    split: torch.Tensor = payload["split"]
    y_dump = payload["y"].view(-1).long()
    meta = payload.get("meta", {})
    gnn_types_raw = str(meta.get("gnn_types", "GCN,GIN,SAGE,GAT"))
    head_names = [s.strip().upper() for s in gnn_types_raw.split(",") if s.strip()]
    if not head_names:
        head_names = list(SIGMA_MP_HEADS)
    seed = int(meta.get("seed", 2))

    n_layers = int(gnn.shape[1])
    layer = layer_idx if layer_idx >= 0 else n_layers + layer_idx
    if layer < 0 or layer >= n_layers:
        raise IndexError(f"gate layer {layer_idx} out of range for L={n_layers}")

    ds_obj = _load_tu_dataset(dataset, tu_root)
    labels = _tu_labels(ds_obj)
    if int(gnn.shape[0]) != int(labels.shape[0]):
        raise RuntimeError(
            f"{dataset}: dump has {int(gnn.shape[0])} graphs, TUDataset has {int(labels.shape[0])}"
        )
    train_index, val_index, test_index = reconstruct_graphgym_random_split(
        labels, seed=seed
    )
    ordered = {"train": train_index, "val": val_index, "test": test_index}
    split_np = split.cpu().numpy()

    mapped: Dict[int, Tuple[str, int, Dict[str, float]]] = {}
    n_unique_train = 0
    n_ambiguous_train = 0

    for split_name in splits:
        sid = SPLIT_IDS[split_name]
        rows = np.where(split_np == sid)[0]
        if split_name in ("val", "test"):
            indices = ordered[split_name]
            if len(rows) != len(indices):
                raise RuntimeError(
                    f"{dataset} {split_name}: dump rows {len(rows)} != split {len(indices)}"
                )
            dump_y = y_dump[rows].cpu().numpy()
            expect_y = labels[np.asarray(indices, dtype=np.int64)]
            if not np.array_equal(dump_y, expect_y):
                raise RuntimeError(
                    f"{dataset} {split_name}: reconstructed ShuffleSplit y does not match dump"
                )
            for row, gidx in zip(rows.tolist(), indices.tolist()):
                mapped[int(gidx)] = (
                    split_name,
                    int(y_dump[row].item()),
                    _gate_vector(gnn, int(row), layer, head_names),
                    int(row),
                )
            continue

        node_pt = gate_pt.parent / "gate_values_per_node.pt"
        if not node_pt.is_file():
            logging.warning(
                "%s: skipping train match (missing %s)", dataset, node_pt
            )
            continue
        node = torch.load(node_pt, map_location="cpu", weights_only=False)
        ptr = node["ptr"].long()
        edge_ptr = node["edge_ptr"].long()
        edge_index = node["edge_index"]
        fps: Dict[DegFingerprint, List[int]] = defaultdict(list)
        for gidx in train_index.tolist():
            fps[_tu_fingerprint(ds_obj, int(gidx))].append(int(gidx))
        for row in rows.tolist():
            fp = _dump_row_fingerprint(
                y=int(y_dump[row].item()),
                ptr=ptr,
                edge_ptr=edge_ptr,
                edge_index=edge_index,
                row=int(row),
            )
            cands = fps.get(fp, [])
            if len(cands) != 1:
                n_ambiguous_train += 1
                continue
            gidx = cands[0]
            if gidx in mapped:
                n_ambiguous_train += 1
                continue
            n_unique_train += 1
            mapped[gidx] = (
                "train",
                int(y_dump[row].item()),
                _gate_vector(gnn, int(row), layer, head_names),
                int(row),
            )

    return mapped, head_names, (n_unique_train, n_ambiguous_train)


def _gates_from_gnn_rows(
    row_map: Mapping[int, Tuple[str, int, int]],
    gnn: torch.Tensor,
    layer: int,
    head_names: Sequence[str],
) -> Dict[int, Tuple[str, int, Dict[str, float]]]:
    """Slice one MP layer out of an already-aligned dump."""
    out: Dict[int, Tuple[str, int, Dict[str, float]]] = {}
    for gidx, (split_name, y, row) in row_map.items():
        out[int(gidx)] = (
            split_name,
            int(y),
            _gate_vector(gnn, int(row), layer, head_names),
        )
    return out


def _row_map_from_layer_payload(
    mapped: Mapping[int, Tuple[str, int, Dict[str, float], int]],
) -> Dict[int, Tuple[str, int, int]]:
    """Drop the layer-specific gate vector, keep dump-row alignment."""
    out: Dict[int, Tuple[str, int, int]] = {}
    for gidx, payload in mapped.items():
        split_name, y, _vec, row = payload
        out[int(gidx)] = (split_name, int(y), int(row))
    return out


def _build_records(
    profiles: Dict[int, Dict[str, float]],
    gates: Dict[int, Tuple[str, int, Dict[str, float]]],
    operators: Sequence[str],
    min_margin: float,
) -> List[GraphOperatorRecord]:
    """Align graphs present in both hetero profiles and gate dumps."""
    records: List[GraphOperatorRecord] = []
    n_ops = len(operators)
    for gidx in sorted(set(profiles.keys()) & set(gates.keys())):
        acc = profiles[gidx]
        if len(acc) < n_ops:
            continue
        pref, margin = _preferred_operator(acc)
        if pref != "TIE" and margin < min_margin:
            continue
        split_name, y, gate_vec = gates[gidx][:3]
        records.append(
            GraphOperatorRecord(
                graph_idx=gidx,
                split=split_name,
                y=y,
                accuracies=dict(acc),
                preferred=pref,
                margin=margin,
                sigma_gates=gate_vec,
            )
        )
    return records


def _write_csv(
    records: Sequence[GraphOperatorRecord],
    out_path: Path,
    operators: Sequence[str],
    head_names: Sequence[str],
) -> None:
    """Write per-graph join table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["graph_idx", "split", "y", "preferred", "margin", "n_gate_seeds"]
        + [f"acc_{op.lower()}" for op in operators]
        + [f"gate_{h.lower()}" for h in head_names]
    )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row: Dict[str, object] = {
                "graph_idx": rec.graph_idx,
                "split": rec.split,
                "y": rec.y,
                "preferred": rec.preferred,
                "margin": f"{rec.margin:.4f}",
                "n_gate_seeds": rec.n_gate_seeds,
            }
            for op in operators:
                row[f"acc_{op.lower()}"] = f"{rec.accuracies[op]:.4f}"
            for head in head_names:
                row[f"gate_{head.lower()}"] = (
                    f"{rec.sigma_gates[head]:.4f}" if head in rec.sigma_gates else ""
                )
            writer.writerow(row)


def _untied(records: Sequence[GraphOperatorRecord]) -> List[GraphOperatorRecord]:
    """Drop specialist ties."""
    return [rec for rec in records if rec.preferred != "TIE"]


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample std (0 if n<2)."""
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def _seed_title_tag(join: DatasetJoin) -> str:
    """Short seed annotation for figure titles."""
    seeds = join.seed_ids
    if len(seeds) <= 1:
        seed = seeds[0] if seeds else 2
        return f"seed {seed}"
    return f"{len(seeds)}-seed mean ± std"


def _mean_gate_by_pref(
    records: Sequence[GraphOperatorRecord],
    *,
    preferred: str,
    head: str,
) -> float:
    """Mean last-layer γ on ``head`` among graphs with the given specialist."""
    vals = [
        rec.sigma_gates[head]
        for rec in records
        if rec.preferred == preferred and head in rec.sigma_gates
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _series_over_seeds(
    join: DatasetJoin,
    *,
    preferred: str,
    head: str,
) -> List[float]:
    """One mean-γ per seed among graphs whose specialist is ``preferred``."""
    groups = join.per_seed_records or [join.records]
    out: List[float] = []
    for recs in groups:
        recs_u = _untied(recs)
        val = _mean_gate_by_pref(recs_u, preferred=preferred, head=head)
        if not np.isnan(val):
            out.append(val)
    return out


def _series_over_seeds_other(
    join: DatasetJoin,
    *,
    preferred: str,
    head: str,
) -> List[float]:
    """One mean-γ per seed among untied graphs that prefer a different operator."""
    groups = join.per_seed_records or [join.records]
    out: List[float] = []
    for recs in groups:
        recs_u = [r for r in _untied(recs) if r.preferred != preferred]
        vals = [
            rec.sigma_gates[head]
            for rec in recs_u
            if head in rec.sigma_gates
        ]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def _delta_gamma_over_seeds(join: DatasetJoin, operator: str) -> List[float]:
    """Per-seed Δγ_H = mean γ_H | pref H minus mean γ_H | pref ≠ H."""
    pref = _series_over_seeds(join, preferred=operator, head=operator)
    other = _series_over_seeds_other(join, preferred=operator, head=operator)
    n = min(len(pref), len(other))
    return [pref[i] - other[i] for i in range(n)]


def _delta_from_records(records: Sequence[GraphOperatorRecord], operator: str) -> float:
    """Δγ_H on one seed: mean γ_H | pref H minus mean γ_H | pref ≠ H (untied)."""
    recs = _untied(records)
    pref_vals = [
        rec.sigma_gates[operator]
        for rec in recs
        if rec.preferred == operator and operator in rec.sigma_gates
    ]
    other_vals = [
        rec.sigma_gates[operator]
        for rec in recs
        if rec.preferred != operator and operator in rec.sigma_gates
    ]
    if not pref_vals or not other_vals:
        return float("nan")
    return float(np.mean(pref_vals) - np.mean(other_vals))


def _pearson_from_records(
    records: Sequence[GraphOperatorRecord],
    op_a: str,
    op_b: str,
) -> float:
    """Pearson r between Δacc and Δγ for one operator pair."""
    xs: List[float] = []
    ys: List[float] = []
    for rec in records:
        if op_a not in rec.accuracies or op_b not in rec.accuracies:
            continue
        if op_a not in rec.sigma_gates or op_b not in rec.sigma_gates:
            continue
        xs.append(rec.accuracies[op_a] - rec.accuracies[op_b])
        ys.append(rec.sigma_gates[op_a] - rec.sigma_gates[op_b])
    return _pearson(xs, ys)


def plot_preference_fractions(
    joins: Sequence[DatasetJoin],
    out_path: Path,
    dpi: int,
) -> None:
    """Bar chart of specialist argmax fractions (routing fig01 analogue)."""
    fig, axes = plt.subplots(1, len(joins), figsize=(4.6 * len(joins), 4.4), squeeze=False)
    for ax, join in zip(axes[0], joins):
        recs = _untied(join.records)
        ops = [op for op in join.operators]
        counts = [sum(1 for r in recs if r.preferred == op) for op in ops]
        n = max(len(recs), 1)
        fracs = [c / n for c in counts]
        colors = [PALETTE.get(op, "#888888") for op in ops]
        bars = ax.bar(
            ops,
            fracs,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Fraction of graphs")
        ax.set_title(join.dataset.upper())
        ax.grid(axis="y", alpha=0.22)
        for bar, count, frac in zip(bars, counts, fracs):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                frac + 0.03,
                f"{count}\n({100.0 * frac:.0f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        n_tie = sum(1 for r in join.records if r.preferred == "TIE")
        ax.text(
            0.02,
            0.98,
            f"n={len(recs)}" + (f", ties={n_tie}" if n_tie else ""),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
        )
    fig.suptitle(
        "Specialist preference (Xu-style ≥100-appearance profiles)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_gate_by_preference(
    joins: Sequence[DatasetJoin],
    out_path: Path,
    dpi: int,
) -> None:
    """Mean γ by specialist preference for GCN / GIN / SAGE heads."""
    heads = ("GCN", "GIN", "SAGE")
    prefs = ("GCN", "GIN", "SAGE")
    fig, axes = plt.subplots(1, len(joins), figsize=(7.2 * max(len(joins), 1), 5.0), squeeze=False)
    err_kw = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"}
    bar_w = 0.24
    for ax, join in zip(axes[0], joins):
        recs = _untied(join.records)
        xs = np.arange(len(prefs))
        for hi, head in enumerate(heads):
            means: List[float] = []
            stds: List[float] = []
            for pref in prefs:
                vals = _series_over_seeds(join, preferred=pref, head=head)
                mean, std = _mean_std(vals)
                means.append(0.0 if np.isnan(mean) else mean)
                stds.append(0.0 if np.isnan(std) else std)
            offset = (hi - 1) * bar_w
            ax.bar(
                xs + offset,
                means,
                width=bar_w,
                yerr=stds,
                capsize=3,
                color=PALETTE[head],
                edgecolor="black",
                linewidth=0.5,
                error_kw=err_kw,
                label=rf"$\gamma_{{\mathrm{{{head}}}}}$",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([rf"pref {p}" for p in prefs])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(r"SiGMA MP gate $\gamma$ (last layer)")
        counts = [sum(1 for r in recs if r.preferred == p) for p in prefs]
        ax.set_title(
            rf"{join.dataset.upper()}  "
            rf"($n_{{\mathrm{{GCN}}}}={counts[0]}$, "
            rf"$n_{{\mathrm{{GIN}}}}={counts[1]}$, "
            rf"$n_{{\mathrm{{SAGE}}}}={counts[2]}$)"
        )
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        rf"Mean MP gate by specialist preference (last layer, {_seed_title_tag(joins[0])})",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_gate_routing_delta(
    joins: Sequence[DatasetJoin],
    out_path: Path,
    dpi: int,
) -> None:
    """Per-head Δγ: mean γ_H on graphs that prefer H minus those that do not."""
    operators = ("GIN", "GCN", "SAGE")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    xs = np.arange(len(joins))
    bar_w = 0.24
    err_kw = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"}
    labels = {
        "GIN": r"$\Delta\gamma_{\mathrm{GIN}}=\bar\gamma_{\mathrm{pref\,GIN}}-\bar\gamma_{\mathrm{other}}$",
        "GCN": r"$\Delta\gamma_{\mathrm{GCN}}=\bar\gamma_{\mathrm{pref\,GCN}}-\bar\gamma_{\mathrm{other}}$",
        "SAGE": r"$\Delta\gamma_{\mathrm{SAGE}}=\bar\gamma_{\mathrm{pref\,SAGE}}-\bar\gamma_{\mathrm{other}}$",
    }
    for hi, op in enumerate(operators):
        means: List[float] = []
        stds: List[float] = []
        for join in joins:
            series = _delta_gamma_over_seeds(join, op)
            mean, std = _mean_std(series)
            means.append(0.0 if np.isnan(mean) else mean)
            stds.append(0.0 if np.isnan(std) else std)
        offset = (hi - 1) * bar_w
        ax.bar(
            xs + offset,
            means,
            width=bar_w,
            yerr=stds,
            capsize=3,
            color=PALETTE[op],
            edgecolor="black",
            linewidth=0.4,
            error_kw=err_kw,
            label=labels[op],
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([j.dataset.upper() for j in joins])
    ax.set_ylabel("Gate contrast")
    ax.set_title(f"Cross-preference head contrast (last layer, {_seed_title_tag(joins[0])})")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8.0, frameon=True)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    _save_fig(fig, out_path, dpi)


def plot_scatter_margin_vs_gate(
    joins: Sequence[DatasetJoin],
    out_path: Path,
    dpi: int,
) -> None:
    """Δacc vs Δγ for GCN−GIN (top) and SAGE−GIN (bottom)."""
    pairs = (("GCN", "GIN"), ("SAGE", "GIN"))
    fig, axes = plt.subplots(
        len(pairs),
        len(joins),
        figsize=(4.8 * len(joins), 4.4 * len(pairs)),
        squeeze=False,
    )
    for row, (op_a, op_b) in enumerate(pairs):
        for col, join in enumerate(joins):
            ax = axes[row, col]
            xs: List[float] = []
            ys: List[float] = []
            colors: List[str] = []
            for rec in join.records:
                if op_a not in rec.accuracies or op_b not in rec.accuracies:
                    continue
                if op_a not in rec.sigma_gates or op_b not in rec.sigma_gates:
                    continue
                xs.append(rec.accuracies[op_a] - rec.accuracies[op_b])
                ys.append(rec.sigma_gates[op_a] - rec.sigma_gates[op_b])
                colors.append(PALETTE.get(rec.preferred, PALETTE["TIE"]))
            ax.scatter(xs, ys, c=colors, s=18, alpha=0.7, edgecolors="none")
            ax.axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
            ax.axvline(0.0, color="gray", linewidth=0.7, linestyle=":")
            r_p = _pearson(xs, ys)
            r_s = _spearman(xs, ys)
            ax.set_xlabel(rf"acc$_{{\mathrm{{{op_a}}}}}$ − acc$_{{\mathrm{{{op_b}}}}}$")
            ax.set_ylabel(rf"$\gamma_{{\mathrm{{{op_a}}}}}$ − $\gamma_{{\mathrm{{{op_b}}}}}$")
            ax.set_title(rf"{join.dataset.upper()}  ($r={r_p:.2f}$, $\rho={r_s:.2f}$)")
            ax.grid(alpha=0.22)
    fig.suptitle(
        "Specialist accuracy margin vs seed-averaged SiGMA gate margin",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_ranked_gates(
    join: DatasetJoin,
    out_path: Path,
    dpi: int,
) -> None:
    """Ranked last-layer MP gates colored by specialist preference."""
    recs = list(join.records)
    if not recs:
        return
    heads = [h for h in join.head_names if any(h in r.sigma_gates for r in recs)]
    n_heads = len(heads)
    fig, axes = plt.subplots(n_heads, 1, figsize=(7.2, 2.15 * n_heads), sharex=True, squeeze=False)
    gcn_vals = np.array([r.sigma_gates.get("GCN", 0.0) for r in recs], dtype=np.float64)
    order = np.argsort(-gcn_vals)
    ranks = np.arange(len(recs))
    pref_order = [op for op in list(join.operators) + ["TIE"] if any(r.preferred == op for r in recs)]
    for panel, head in enumerate(heads):
        ax = axes[panel, 0]
        vals = np.array([r.sigma_gates.get(head, float("nan")) for r in recs], dtype=np.float64)
        y_ord = vals[order]
        pref_ord = np.array([recs[i].preferred for i in order], dtype=object)
        for op in pref_order:
            mask = pref_ord == op
            ax.scatter(
                ranks[mask],
                y_ord[mask],
                s=14,
                alpha=0.8,
                c=PALETTE.get(op, "#888888"),
                edgecolors="none",
                label=op if panel == 0 else None,
            )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(rf"$\gamma_{{\mathrm{{{head}}}}}$")
        ax.set_title(head, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        if panel == 0:
            for spine in ax.spines.values():
                spine.set_color("#C44E52")
                spine.set_linewidth(1.4)
        if panel == n_heads - 1:
            ax.set_xlabel(r"Rank ($\gamma_{\mathrm{GCN}}$ ↓)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=8,
            framealpha=0.95,
            title="preferred",
        )
    fig.suptitle(
        f"{join.dataset.upper()}: last-layer MP gates ranked by GCN γ",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_pairwise_agreement(
    joins: Sequence[DatasetJoin],
    out_path: Path,
    dpi: int,
) -> None:
    """GCN vs GIN specialist vs gate sign agreement (routing fig05 analogue)."""
    fig, axes = plt.subplots(1, len(joins), figsize=(5.2 * len(joins), 4.6), squeeze=False)
    outcome_order = ("both_gcn", "agree_split", "both_gin", "disagree")
    outcome_colors = {
        "both_gcn": PALETTE["GCN"],
        "both_gin": PALETTE["GIN"],
        "agree_split": "#55A868",
        "disagree": "#E45756",
    }
    outcome_labels = {
        "both_gcn": r"acc & $\gamma$ prefer GCN",
        "both_gin": r"acc & $\gamma$ prefer GIN",
        "agree_split": "same sign (mixed)",
        "disagree": "opposite sign",
    }
    for ax, join in zip(axes[0], joins):
        counts = {k: 0 for k in outcome_order}
        n = 0
        for rec in join.records:
            if "GCN" not in rec.accuracies or "GIN" not in rec.accuracies:
                continue
            d_acc = rec.accuracies["GCN"] - rec.accuracies["GIN"]
            d_gate = rec.sigma_gates.get("GCN", 0.0) - rec.sigma_gates.get("GIN", 0.0)
            if abs(d_acc) < 1e-12 or abs(d_gate) < 1e-12:
                continue
            n += 1
            acc_gcn = d_acc > 0
            gate_gcn = d_gate > 0
            if acc_gcn and gate_gcn:
                counts["both_gcn"] += 1
            elif (not acc_gcn) and (not gate_gcn):
                counts["both_gin"] += 1
            else:
                counts["disagree"] += 1
        fracs = [counts[k] / max(n, 1) for k in outcome_order]
        ax.bar(
            [outcome_labels[k] for k in outcome_order],
            fracs,
            color=[outcome_colors[k] for k in outcome_order],
            edgecolor="black",
            linewidth=0.4,
        )
        agree = (counts["both_gcn"] + counts["both_gin"]) / max(n, 1)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Fraction")
        ax.set_title(rf"{join.dataset.upper()}  (agree {100.0 * agree:.0f}%, n={n})")
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Does the GCN/GIN gate sign match specialist preference?", y=1.02, fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_summary_table(
    joins: Sequence[DatasetJoin],
    out_path: Path,
    dpi: int,
) -> None:
    """One-page numeric summary (routing table-figure style)."""
    col_labels = [
        "Dataset",
        "n",
        "pref GCN",
        "pref GIN",
        "pref SAGE",
        r"$\Delta\gamma_{\mathrm{GIN}}$",
        r"$\Delta\gamma_{\mathrm{GCN}}$",
        r"$\Delta\gamma_{\mathrm{SAGE}}$",
        r"$r_{\mathrm{GCN/GIN}}$",
        r"$r_{\mathrm{SAGE/GIN}}$",
    ]
    rows: List[List[str]] = []
    for join in joins:
        recs = _untied(join.records)
        n_gcn = sum(1 for r in recs if r.preferred == "GCN")
        n_gin = sum(1 for r in recs if r.preferred == "GIN")
        n_sage = sum(1 for r in recs if r.preferred == "SAGE")
        d_gin, _ = _mean_std(_delta_gamma_over_seeds(join, "GIN"))
        d_gcn, _ = _mean_std(_delta_gamma_over_seeds(join, "GCN"))
        d_sage, _ = _mean_std(_delta_gamma_over_seeds(join, "SAGE"))
        xs_gcn = [r.accuracies["GCN"] - r.accuracies["GIN"] for r in join.records]
        ys_gcn = [
            r.sigma_gates.get("GCN", 0.0) - r.sigma_gates.get("GIN", 0.0) for r in join.records
        ]
        xs_sage = [
            r.accuracies["SAGE"] - r.accuracies["GIN"]
            for r in join.records
            if "SAGE" in r.accuracies
        ]
        ys_sage = [
            r.sigma_gates.get("SAGE", 0.0) - r.sigma_gates.get("GIN", 0.0)
            for r in join.records
            if "SAGE" in r.accuracies
        ]
        r_gcn = _pearson(xs_gcn, ys_gcn)
        r_sage = _pearson(xs_sage, ys_sage)
        n_untied = max(len(recs), 1)
        rows.append(
            [
                join.dataset.upper(),
                str(len(join.records)),
                f"{n_gcn} ({100.0 * n_gcn / n_untied:.0f}%)",
                f"{n_gin} ({100.0 * n_gin / n_untied:.0f}%)",
                f"{n_sage} ({100.0 * n_sage / n_untied:.0f}%)",
                f"{d_gin:+.3f}",
                f"{d_gcn:+.3f}",
                f"{d_sage:+.3f}",
                f"{r_gcn:.2f}",
                f"{r_sage:.2f}",
            ]
        )
    n_rows = len(rows)
    fig_h = 1.55 + 0.42 * max(n_rows, 1)
    fig, ax = plt.subplots(figsize=(13.2, fig_h))
    ax.axis("off")
    fig.suptitle("TU specialist preference vs SiGMA gates", fontsize=12, fontweight="bold", y=0.98)
    ax.text(
        0.5,
        0.90,
        f"Xu-style GCN/GIN/SAGE profiles (≥100 apps) · SiGMA a2g4 "
        f"{_seed_title_tag(joins[0])}, val+test · protocols differ",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.0,
    )
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.08, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", weight="bold", fontsize=TABLE_FONT_SIZE)
            cell.set_height(0.38)
        else:
            cell.set_facecolor(ROW_ALT if row % 2 == 0 else ROW_BASE)
            cell.set_text_props(fontsize=TABLE_FONT_SIZE)
            cell.set_height(0.34)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_delta_gamma_by_layer(
    layer_table: Sequence[Mapping[str, object]],
    out_path: Path,
    dpi: int,
) -> None:
    """Line plot of Δγ vs MP layer (one panel per dataset)."""
    datasets = sorted({str(row["dataset"]) for row in layer_table})
    operators = ("GIN", "GCN", "SAGE")
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.4 * max(len(datasets), 1), 4.6),
        squeeze=False,
    )
    for ax, ds in zip(axes[0], datasets):
        layers = sorted(
            {int(row["layer"]) for row in layer_table if str(row["dataset"]) == ds}
        )
        for op in operators:
            means: List[float] = []
            stds: List[float] = []
            for layer in layers:
                match = [
                    row
                    for row in layer_table
                    if str(row["dataset"]) == ds
                    and int(row["layer"]) == layer
                    and str(row["operator"]) == op
                    and str(row["delta_mean"]) != ""
                ]
                if not match:
                    means.append(0.0)
                    stds.append(0.0)
                    continue
                means.append(float(match[0]["delta_mean"]))
                stds.append(float(match[0]["delta_std"] or 0.0))
            ax.errorbar(
                layers,
                means,
                yerr=stds,
                marker="o",
                markersize=4,
                capsize=2,
                color=PALETTE[op],
                label=rf"$\Delta\gamma_{{\mathrm{{{op}}}}}$",
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("MP layer")
        ax.set_ylabel(r"$\Delta\gamma$ (pref H − other)")
        ax.set_title(ds.upper())
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle(
        r"Gate–preference contrast vs depth (5-seed mean $\pm$ std, val+test)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def _write_layer_scan_csv(rows: Sequence[Mapping[str, object]], out_path: Path) -> None:
    """Write per-dataset per-layer Δγ table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "layer",
        "operator",
        "delta_mean",
        "delta_std",
        "n_seeds",
        "pearson_vs_gin",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _scan_layers_for_dataset(
    dataset: str,
    *,
    profiles: Dict[int, Dict[str, float]],
    gate_pts: Sequence[Tuple[int, Path]],
    tu_root: Path,
    operators: Tuple[str, ...],
    splits: Tuple[str, ...],
    min_margin: float,
) -> List[Dict[str, object]]:
    """Compute per-layer Δγ over seeds for one dataset."""
    per_layer_seed: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    per_layer_r: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    n_layers = 0
    for seed, gate_pt in gate_pts:
        mapped, head_names, _counts = _load_sigma_gates(
            gate_pt,
            -1,
            dataset=dataset,
            tu_root=tu_root,
            splits=splits,
        )
        row_map = _row_map_from_layer_payload(mapped)  # type: ignore[arg-type]
        payload = torch.load(gate_pt, map_location="cpu", weights_only=False)
        gnn = payload["gnn"]
        n_layers = int(gnn.shape[1])
        for layer in range(n_layers):
            gates = _gates_from_gnn_rows(row_map, gnn, layer, head_names)
            records = _build_records(profiles, gates, operators, min_margin)
            for op in ("GCN", "GIN", "SAGE"):
                if op not in operators:
                    continue
                per_layer_seed[(layer, op)].append(_delta_from_records(records, op))
            if "GIN" in operators:
                for op in ("GCN", "SAGE"):
                    if op in operators:
                        per_layer_r[(layer, op)].append(
                            _pearson_from_records(records, op, "GIN")
                        )
        logging.info("%s seed %d: scanned %d MP layers", dataset, seed, n_layers)
    rows: List[Dict[str, object]] = []
    for layer in range(n_layers):
        for op in ("GIN", "GCN", "SAGE"):
            if op not in operators:
                continue
            vals = [v for v in per_layer_seed[(layer, op)] if not np.isnan(v)]
            mean, std = _mean_std(vals)
            r_vals = [v for v in per_layer_r[(layer, op)] if not np.isnan(v)]
            r_mean, _rstd = (
                _mean_std(r_vals) if r_vals else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "dataset": dataset,
                    "layer": layer,
                    "operator": op,
                    "delta_mean": f"{mean:.6f}" if not np.isnan(mean) else "",
                    "delta_std": f"{std:.6f}" if not np.isnan(std) else "",
                    "n_seeds": len(vals),
                    "pearson_vs_gin": f"{r_mean:.4f}" if not np.isnan(r_mean) else "",
                }
            )
            logging.info(
                "%s L%02d %s Δγ=%.3f±%.3f  r_vs_GIN=%.2f",
                dataset,
                layer,
                op,
                mean if not np.isnan(mean) else float("nan"),
                std if not np.isnan(std) else float("nan"),
                r_mean if not np.isnan(r_mean) else float("nan"),
            )
    return rows


def _write_paper_figures(
    joins: Sequence[DatasetJoin],
    fig_dir: Path,
    dpi: int,
) -> None:
    """Write routing-style paper figures for one or more datasets."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_preference_fractions(joins, fig_dir / "fig01_preference_fractions.png", dpi)
    plot_gate_by_preference(joins, fig_dir / "fig_gate_by_preference.png", dpi)
    plot_gate_routing_delta(joins, fig_dir / "fig02_gate_routing_delta.png", dpi)
    plot_scatter_margin_vs_gate(joins, fig_dir / "fig_scatter_acc_vs_gate.png", dpi)
    plot_pairwise_agreement(joins, fig_dir / "fig05_pairwise_gate_agreement.png", dpi)
    plot_summary_table(joins, fig_dir / "fig_summary_table.png", dpi)
    for join in joins:
        plot_ranked_gates(
            join,
            fig_dir / f"fig_ranked_gates_{join.dataset}.png",
            dpi,
        )


def _average_gate_maps(
    seed_maps: Sequence[Dict[int, Tuple[str, int, Dict[str, float]]]],
) -> Dict[int, Tuple[str, int, Dict[str, float], int]]:
    """Average per-head γ over seeds that mapped the same ``graph_idx``."""
    by_graph: Dict[int, List[Tuple[str, int, Dict[str, float]]]] = defaultdict(list)
    for seed_map in seed_maps:
        for gidx, payload in seed_map.items():
            by_graph[int(gidx)].append(payload[:3])
    out: Dict[int, Tuple[str, int, Dict[str, float], int]] = {}
    for gidx, items in by_graph.items():
        heads = sorted({h for _s, _y, vec in items for h in vec})
        mean_vec: Dict[str, float] = {}
        for head in heads:
            vals = [vec[head] for _s, _y, vec in items if head in vec]
            mean_vec[head] = float(np.mean(vals))
        split_name = items[0][0]
        y = items[0][1]
        out[gidx] = (split_name, y, mean_vec, len(items))
    return out


def _discover_gate_pts(
    *,
    dataset: str,
    gate_root: Path,
    gate_pt: str,
    gate_run_tag: str,
    lr_tag: str,
    seeds: Sequence[int],
) -> List[Tuple[int, Path]]:
    """Resolve dump paths and seed ids."""
    if gate_pt:
        pt = Path(gate_pt)
        if not pt.is_file():
            raise FileNotFoundError(f"Missing gate dump: {pt}")
        return [(2, pt)]
    if gate_run_tag:
        pt = gate_root / f"{dataset}_{gate_run_tag}" / "gate_values_per_graph.pt"
        if not pt.is_file():
            raise FileNotFoundError(f"Missing gate dump: {pt}")
        seed = 2
        if "seed" in gate_run_tag:
            seed = int(gate_run_tag.rsplit("seed", 1)[-1])
        return [(seed, pt)]
    found: List[Tuple[int, Path]] = []
    missing: List[int] = []
    for seed in seeds:
        pt = gate_root / f"{dataset}_SiGMA_hetero_{lr_tag}_seed{seed}" / "gate_values_per_graph.pt"
        if pt.is_file():
            found.append((int(seed), pt))
        else:
            missing.append(int(seed))
    if missing:
        logging.warning("%s: missing dumps for seeds %s", dataset, missing)
    if not found:
        raise FileNotFoundError(
            f"No SiGMA dumps for {dataset} under {gate_root} ({lr_tag}, seeds={list(seeds)})"
        )
    return found


def _join_one_dataset(
    dataset: str,
    *,
    hetero_root: Path,
    gate_pts: Sequence[Tuple[int, Path]],
    tu_root: Path,
    operators: Tuple[str, ...],
    splits: Tuple[str, ...],
    min_appearances: int,
    min_margin: float,
    gate_layer: int,
) -> DatasetJoin:
    """Run the preference–gate join for one dataset, averaging remapped seeds."""
    profiles = _load_operator_profiles(hetero_root, dataset, min_appearances, operators)
    seed_maps: List[Dict[int, Tuple[str, int, Dict[str, float]]]] = []
    per_seed_records: List[List[GraphOperatorRecord]] = []
    head_names: List[str] = []
    n_unique_total = 0
    n_amb_total = 0
    seed_ids = tuple(seed for seed, _pt in gate_pts)
    for seed, gate_pt in gate_pts:
        gates, head_names, (n_unique, n_amb) = _load_sigma_gates(
            gate_pt,
            gate_layer,
            dataset=dataset,
            tu_root=tu_root,
            splits=splits,
        )
        n_unique_total += n_unique
        n_amb_total += n_amb
        seed_maps.append(gates)
        seed_records = _build_records(profiles, gates, operators, min_margin)
        per_seed_records.append(seed_records)
        logging.info(
            "%s seed %d: mapped %d dump graphs → joined %d",
            dataset,
            seed,
            len(gates),
            len(seed_records),
        )
    averaged = _average_gate_maps(seed_maps)
    gates_mean: Dict[int, Tuple[str, int, Dict[str, float]]] = {
        gidx: (split_name, y, vec) for gidx, (split_name, y, vec, _n) in averaged.items()
    }
    records = _build_records(profiles, gates_mean, operators, min_margin)
    n_seeds_by_graph = {gidx: n for gidx, (_s, _y, _v, n) in averaged.items()}
    records = [
        GraphOperatorRecord(
            graph_idx=rec.graph_idx,
            split=rec.split,
            y=rec.y,
            accuracies=rec.accuracies,
            preferred=rec.preferred,
            margin=rec.margin,
            sigma_gates=rec.sigma_gates,
            n_gate_seeds=int(n_seeds_by_graph.get(rec.graph_idx, 1)),
        )
        for rec in records
    ]
    if not records:
        raise RuntimeError(f"{dataset}: no overlapping graphs after join filters.")
    logging.info(
        "%s: joined %d graphs over seeds %s (mean n_seeds=%.2f)",
        dataset,
        len(records),
        ",".join(str(s) for s in seed_ids),
        float(np.mean([r.n_gate_seeds for r in records])),
    )
    prefs = [rec.preferred for rec in records]
    for op in list(operators) + ["TIE"]:
        n = sum(1 for p in prefs if p == op)
        if n == 0:
            continue
        logging.info(
            "%s preferred %s: %d / %d (%.1f%%)",
            dataset,
            op,
            n,
            len(prefs),
            100.0 * n / max(len(prefs), 1),
        )
    for head in ("GCN", "GIN", "SAGE"):
        pref_vals = [
            rec.sigma_gates[head]
            for rec in records
            if rec.preferred == head and head in rec.sigma_gates
        ]
        other_vals = [
            rec.sigma_gates[head]
            for rec in records
            if rec.preferred != head and rec.preferred != "TIE" and head in rec.sigma_gates
        ]
        if pref_vals and other_vals:
            logging.info(
                "%s %s preferred n=%d mean_gate=%.3f | other n=%d mean_gate=%.3f",
                dataset,
                head,
                len(pref_vals),
                float(np.mean(pref_vals)),
                len(other_vals),
                float(np.mean(other_vals)),
            )
    return DatasetJoin(
        dataset=dataset,
        records=records,
        head_names=head_names,
        operators=operators,
        n_mapped=len(records),
        n_train_unique=n_unique_total,
        n_train_ambiguous=n_amb_total,
        seed_ids=seed_ids,
        per_seed_records=per_seed_records,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    datasets_raw = args.datasets.strip() or args.dataset.strip() or "mutag,enzymes"
    datasets = tuple(ds.lower() for ds in _parse_csv_list(datasets_raw))
    operators = tuple(op.strip().upper() for op in _parse_csv_list(args.operators))
    splits = tuple(s.strip().lower() for s in _parse_csv_list(args.splits))
    for op in operators:
        if op not in OPERATOR_CFG_SUFFIX:
            raise ValueError(f"Unknown operator {op!r}; choose from {list(OPERATOR_CFG_SUFFIX)}")
    for split_name in splits:
        if split_name not in SPLIT_IDS:
            raise ValueError(f"Unknown split {split_name!r}")

    hetero_root = Path(args.hetero_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tu_root = Path(args.tu_root) if args.tu_root else Path.home() / ".cache" / "pyg_tu_gate_join"

    joins: List[DatasetJoin] = []
    layer_rows: List[Dict[str, object]] = []
    seeds = tuple(int(s) for s in _parse_csv_list(args.seeds))
    for dataset in datasets:
        gate_pts = _discover_gate_pts(
            dataset=dataset,
            gate_root=Path(args.gate_root),
            gate_pt=args.gate_pt if len(datasets) == 1 else "",
            gate_run_tag=str(args.gate_run_tag).strip(),
            lr_tag=str(args.lr_tag).strip(),
            seeds=seeds,
        )
        join = _join_one_dataset(
            dataset,
            hetero_root=hetero_root,
            gate_pts=gate_pts,
            tu_root=tu_root,
            operators=operators,
            splits=splits,
            min_appearances=int(args.min_appearances),
            min_margin=float(args.min_margin),
            gate_layer=int(args.gate_layer),
        )
        ds_dir = out_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ds_dir / f"{dataset}_operator_gate_join.csv"
        _write_csv(join.records, csv_path, operators, join.head_names)
        logging.info("Wrote %s (%d graphs)", csv_path, len(join.records))
        joins.append(join)

        if bool(args.scan_layers):
            layer_rows.extend(
                _scan_layers_for_dataset(
                    dataset,
                    profiles=_load_operator_profiles(
                        hetero_root,
                        dataset,
                        int(args.min_appearances),
                        operators,
                    ),
                    gate_pts=gate_pts,
                    tu_root=tu_root,
                    operators=operators,
                    splits=splits,
                    min_margin=float(args.min_margin),
                )
            )

    fig_dir = out_dir / "paper_figures"
    _write_paper_figures(joins, fig_dir, int(args.dpi))
    logging.info("Wrote paper figures under %s", fig_dir)
    if bool(args.scan_layers) and layer_rows:
        csv_layers = out_dir / "layer_delta_gamma.csv"
        _write_layer_scan_csv(layer_rows, csv_layers)
        plot_delta_gamma_by_layer(
            layer_rows,
            fig_dir / "fig_delta_gamma_by_layer.png",
            int(args.dpi),
        )
        logging.info("Wrote layer sweep %s", csv_layers)


if __name__ == "__main__":
    main()
