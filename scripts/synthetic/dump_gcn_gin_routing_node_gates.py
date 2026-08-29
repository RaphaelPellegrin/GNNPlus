#!/usr/bin/env python3
"""Dump per-node MP gate γ from best checkpoints (GCN/GIN routing, gated runs).

Loads ``a0g2_gated`` checkpoints under ``<results_root>/{toy,sigma}/``, runs
``collect_per_graph_gates`` on train/val/test, and writes packed tensors plus a
per-graph CSV summary (root γ and node-mean γ, split by ``tau``).

Outputs per run directory (default paths):
  - ``gate_values_per_node.pt`` — packed node gates + ``tau`` / ``y`` / topology
  - ``gate_graph_summary.csv`` — one row per graph (all requested splits)

Example (single run, cluster):
  python scripts/synthetic/dump_gcn_gin_routing_node_gates.py \\
    --run-dir /n/netscratch/.../gcn_gin_routing/toy/a0g2_gated_lr001_seed0 \\
    --dataset-dir /n/netscratch/.../gnnplus_datasets

Example (all gated runs on both tracks):
  python scripts/synthetic/dump_gcn_gin_routing_node_gates.py \\
    --results-root /n/netscratch/.../gcn_gin_routing \\
    --dataset-dir /n/netscratch/.../gnnplus_datasets \\
    --tracks toy;sigma
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import torch
from torch_geometric.graphgym.checkpoint import get_ckpt_epochs, load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401 — register modules
from GNNPlus.gcn_gin_routing_gate_tracking import hybrid_head_indices
from GNNPlus.hybrid_gate_tracking import _unwrap_model

RUN_NAME_RE = re.compile(
    r"^(?P<model>.+)_lr(?P<lr_tag>\d+)_seed(?P<seed>\d+)$",
)
GATED_MODEL = "a0g2_gated"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class RunRef:
    """One training run directory."""

    track: str
    run_dir: Path
    model: str
    lr_tag: str
    seed: int


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Parent of toy/ and sigma/ (batch mode).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Single run directory (overrides --results-root batch mode).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get(
            "GNNPLUS_DATASET_DIR",
            "results/gcn_gin_routing/data",
        ),
        help="Parent of GcnGinRouting/ for PyG loader.",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="toy,sigma",
        help="Tracks in batch mode (comma or semicolon separated).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to dump.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Hybrid MP layer index for gates (default 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have gate_values_per_node.pt.",
    )
    return parser.parse_args(argv)


def _resolve_run_config(run_dir: Path) -> Optional[Path]:
    """Return config yaml path for a run, if present."""
    for name in ("config_used.yaml", "config.yaml"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _parse_run_ref(track: str, run_dir: Path) -> Optional[RunRef]:
    """Parse ``model_lrXXX_seedY`` from a run directory name."""
    match = RUN_NAME_RE.match(run_dir.name)
    if match is None:
        return None
    return RunRef(
        track=track,
        run_dir=run_dir,
        model=match.group("model"),
        lr_tag=f"lr{match.group('lr_tag')}",
        seed=int(match.group("seed")),
    )


def iter_gated_run_refs(
    results_root: Path,
    tracks: Sequence[str],
) -> Iterator[RunRef]:
    """Yield gated run directories with config + checkpoint."""
    for track in tracks:
        track_dir = results_root / track
        if not track_dir.is_dir():
            logging.warning("Missing track directory: %s", track_dir)
            continue
        for run_dir in sorted(track_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not run_dir.name.startswith(f"{GATED_MODEL}_"):
                continue
            if _resolve_run_config(run_dir) is None:
                logging.warning("Skipping %s (no config yaml)", run_dir)
                continue
            ckpt_dir = run_dir / "ckpt"
            if not ckpt_dir.is_dir() or not any(ckpt_dir.glob("*.ckpt")):
                logging.warning("Skipping %s (no checkpoint)", run_dir)
                continue
            ref = _parse_run_ref(track, run_dir)
            if ref is None:
                logging.warning("Skipping unrecognized name: %s", run_dir.name)
                continue
            yield ref


def _load_cfg_for_run(run_ref: RunRef, dataset_dir: str) -> None:
    """Load GraphGym cfg from the run's saved config yaml."""
    cfg_path_obj = _resolve_run_config(run_ref.run_dir)
    if cfg_path_obj is None:
        raise FileNotFoundError(f"No config yaml in {run_ref.run_dir}")
    old_argv = sys.argv
    sys.argv = [
        old_argv[0],
        "--cfg",
        str(cfg_path_obj),
        "dataset.dir",
        dataset_dir,
        "seed",
        str(run_ref.seed),
    ]
    try:
        args = parse_args()
        set_cfg(cfg)
        load_cfg(cfg, args)
    finally:
        sys.argv = old_argv
    cfg.run_dir = str(run_ref.run_dir)
    cfg.out_dir = str(run_ref.run_dir.parent.parent)


def _pick_best_epoch(run_dir: Path) -> int:
    """Return latest checkpoint epoch under ``run_dir/ckpt``."""
    cfg.run_dir = str(run_dir)
    epochs = list(get_ckpt_epochs())
    if not epochs:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/ckpt")
    return int(max(epochs))


def _ptr_from_batch(batch_ids: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Build CSR ``ptr`` of length ``num_graphs + 1``."""
    counts = torch.bincount(batch_ids, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def _root_local_indices(ptr: torch.Tensor) -> torch.Tensor:
    """Local node index of the root (first node) per graph."""
    return ptr[:-1].long()


@torch.no_grad()
def _collect_split_gates(
    model: torch.nn.Module,
    loader: Any,
    split_id: int,
    device: torch.device,
    gin_idx: int,
    gcn_idx: int,
    layer_idx: int,
) -> dict[str, Any]:
    """Collect per-node gates and graph metadata for one split."""
    core = _unwrap_model(model)
    if not hasattr(core, "collect_per_graph_gates"):
        raise TypeError(f"{type(core).__name__} lacks collect_per_graph_gates")

    attn_n_parts: list[torch.Tensor] = []
    gnn_n_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    edge_batch_parts: list[torch.Tensor] = []
    tau_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    split_parts: list[torch.Tensor] = []
    graph_offset = 0
    node_offset = 0

    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, "tau") or batch.tau is None:
                raise AttributeError("Batch missing graph-level tau.")
            tau_g = batch.tau.view(-1).long().detach().cpu()
            ei_local = batch.edge_index.detach().cpu().long()
            n_nodes_batch = int(batch.num_nodes)

            gate_out = core.collect_per_graph_gates(batch.clone())
            n_graphs = int(gate_out["num_graphs"])
            split_parts.append(
                torch.full((n_graphs,), split_id, dtype=torch.long),
            )
            tau_parts.append(tau_g)
            if gate_out["y"] is not None:
                y_parts.append(gate_out["y"].detach().cpu().long().view(-1))
            else:
                y_parts.append(torch.full((n_graphs,), -1, dtype=torch.long))

            attn_n_parts.append(gate_out["attn_node"].detach().cpu())
            gnn_n_parts.append(gate_out["gnn_node"].detach().cpu())
            batch_local = gate_out["batch"].long().cpu()
            batch_parts.append(batch_local + graph_offset)

            edge_parts.append(ei_local + int(node_offset))
            edge_batch_parts.append(batch_local[ei_local[0]].long() + graph_offset)
            node_offset += n_nodes_batch
            graph_offset += n_graphs
    finally:
        if was_training:
            model.train()

    if not batch_parts:
        empty3 = torch.zeros(0, 0, 0)
        return {
            "attn_node": empty3,
            "gnn_node": empty3,
            "batch": torch.zeros(0, dtype=torch.long),
            "ptr": torch.zeros(1, dtype=torch.long),
            "tau": torch.zeros(0, dtype=torch.long),
            "y": torch.zeros(0, dtype=torch.long),
            "split": torch.zeros(0, dtype=torch.long),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_batch": torch.zeros(0, dtype=torch.long),
            "edge_ptr": torch.zeros(1, dtype=torch.long),
            "num_graphs": 0,
            "num_nodes": 0,
            "num_edges": 0,
            "gin_idx": gin_idx,
            "gcn_idx": gcn_idx,
            "layer_idx": layer_idx,
        }

    batch_ids = torch.cat(batch_parts, dim=0)
    num_graphs = graph_offset
    ptr = _ptr_from_batch(batch_ids, num_graphs)
    edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.zeros(2, 0)
    edge_batch = (
        torch.cat(edge_batch_parts, dim=0)
        if edge_batch_parts
        else torch.zeros(0, dtype=torch.long)
    )
    edge_ptr = _ptr_from_batch(edge_batch, num_graphs)

    return {
        "attn_node": torch.cat(attn_n_parts, dim=0),
        "gnn_node": torch.cat(gnn_n_parts, dim=0),
        "batch": batch_ids,
        "ptr": ptr,
        "tau": torch.cat(tau_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
        "split": torch.cat(split_parts, dim=0),
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "edge_ptr": edge_ptr,
        "num_graphs": num_graphs,
        "num_nodes": int(batch_ids.numel()),
        "num_edges": int(edge_index.size(1)),
        "gin_idx": gin_idx,
        "gcn_idx": gcn_idx,
        "layer_idx": layer_idx,
    }


def _merge_split_payloads(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate per-split gate payloads into one packed dict."""
    valid = [p for p in parts if int(p["num_graphs"]) > 0]
    if not valid:
        return parts[0] if parts else {}

    graph_offset = 0
    node_offset = 0
    merged: dict[str, Any] = {
        "attn_node": [],
        "gnn_node": [],
        "batch": [],
        "tau": [],
        "y": [],
        "split": [],
        "edge_index": [],
        "edge_batch": [],
    }
    for part in valid:
        merged["attn_node"].append(part["attn_node"])
        merged["gnn_node"].append(part["gnn_node"])
        merged["batch"].append(part["batch"] + graph_offset)
        merged["tau"].append(part["tau"])
        merged["y"].append(part["y"])
        merged["split"].append(part["split"])
        if int(part["num_edges"]) > 0:
            merged["edge_index"].append(part["edge_index"] + node_offset)
            merged["edge_batch"].append(part["edge_batch"] + graph_offset)
        graph_offset += int(part["num_graphs"])
        node_offset += int(part["num_nodes"])

    batch_ids = torch.cat(merged["batch"], dim=0)
    num_graphs = graph_offset
    ptr = _ptr_from_batch(batch_ids, num_graphs)
    if merged["edge_index"]:
        edge_index = torch.cat(merged["edge_index"], dim=1)
        edge_batch = torch.cat(merged["edge_batch"], dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_batch = torch.zeros(0, dtype=torch.long)
    edge_ptr = _ptr_from_batch(edge_batch, num_graphs)
    roots = _root_local_indices(ptr)

    out = {
        "attn_node": torch.cat(merged["attn_node"], dim=0),
        "gnn_node": torch.cat(merged["gnn_node"], dim=0),
        "batch": batch_ids,
        "ptr": ptr,
        "root_local_idx": roots,
        "tau": torch.cat(merged["tau"], dim=0),
        "y": torch.cat(merged["y"], dim=0),
        "split": torch.cat(merged["split"], dim=0),
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "edge_ptr": edge_ptr,
        "num_graphs": num_graphs,
        "num_nodes": int(batch_ids.numel()),
        "num_edges": int(edge_index.size(1)),
        "gin_head_idx": int(valid[0]["gin_idx"]),
        "gcn_head_idx": int(valid[0]["gcn_idx"]),
        "layer_idx": int(valid[0]["layer_idx"]),
    }
    return out


def _write_graph_summary_csv(
    payload: dict[str, Any],
    csv_path: Path,
    run_ref: RunRef,
    epoch: int,
) -> None:
    """Write per-graph gate summary (root + node-mean γ_GIN / γ_GCN)."""
    gnn_node = payload["gnn_node"]
    ptr: torch.Tensor = payload["ptr"]
    layer_idx = int(payload["layer_idx"])
    gin_idx = int(payload["gin_head_idx"])
    gcn_idx = int(payload["gcn_head_idx"])
    tau = payload["tau"].long()
    y = payload["y"].long()
    split = payload["split"].long()
    num_graphs = int(payload["num_graphs"])

    fieldnames = [
        "graph_idx",
        "split",
        "tau",
        "y",
        "gin_gamma_root",
        "gcn_gamma_root",
        "gin_gamma_mean_nodes",
        "gcn_gamma_mean_nodes",
        "num_nodes",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for g in range(num_graphs):
            lo = int(ptr[g].item())
            hi = int(ptr[g + 1].item())
            nodes = gnn_node[lo:hi, layer_idx, :]
            gin_n = nodes[:, gin_idx].float()
            gcn_n = nodes[:, gcn_idx].float()
            writer.writerow(
                {
                    "graph_idx": g,
                    "split": SPLIT_NAMES[int(split[g].item())],
                    "tau": int(tau[g].item()),
                    "y": int(y[g].item()),
                    "gin_gamma_root": float(gin_n[0].item()),
                    "gcn_gamma_root": float(gcn_n[0].item()),
                    "gin_gamma_mean_nodes": float(gin_n.mean().item()),
                    "gcn_gamma_mean_nodes": float(gcn_n.mean().item()),
                    "num_nodes": hi - lo,
                },
            )


def dump_run_node_gates(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
    splits: Sequence[str],
    layer_idx: int,
    skip_existing: bool,
) -> Path:
    """Dump node-level gates for one gated run; return path to ``.pt``."""
    out_pt = run_ref.run_dir / "gate_values_per_node.pt"
    out_csv = run_ref.run_dir / "gate_graph_summary.csv"
    if skip_existing and out_pt.is_file():
        logging.info("Skipping existing %s", out_pt)
        return out_pt

    _load_cfg_for_run(run_ref, dataset_dir)
    seed_everything(int(cfg.seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    hybrid = getattr(cfg.gnn, "hybrid", None)
    gnn_types = str(getattr(hybrid, "gnn_types", "")) if hybrid is not None else ""
    gin_idx, gcn_idx, two_head = hybrid_head_indices(gnn_types)
    if not two_head:
        raise RuntimeError(f"Run {run_ref.run_dir} is not a two-head hybrid model.")

    loaders = create_loader()
    name_to_loader = {
        "train": loaders[0] if len(loaders) > 0 else None,
        "val": loaders[1] if len(loaders) > 1 else None,
        "test": loaders[2] if len(loaders) > 2 else None,
    }

    model = create_model()
    epoch = _pick_best_epoch(run_ref.run_dir)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    model.to(device)

    split_parts: list[dict[str, Any]] = []
    for split_name in splits:
        loader = name_to_loader.get(split_name)
        if loader is None:
            logging.warning("Split %s missing; skipping.", split_name)
            continue
        split_id = SPLIT_NAMES.index(split_name)
        logging.info(
            "Collecting %s gates for %s / %s",
            split_name,
            run_ref.track,
            run_ref.run_dir.name,
        )
        split_parts.append(
            _collect_split_gates(
                model,
                loader,
                split_id,
                device,
                gin_idx,
                gcn_idx,
                layer_idx,
            ),
        )

    payload = _merge_split_payloads(split_parts)
    meta = {
        "run_dir": str(run_ref.run_dir),
        "track": run_ref.track,
        "model": run_ref.model,
        "lr_tag": run_ref.lr_tag,
        "seed": run_ref.seed,
        "epoch": epoch,
        "dataset": str(cfg.dataset.name),
        "gnn_types": gnn_types,
        "splits": list(splits),
        "split_names": list(SPLIT_NAMES),
        "aggregation_graph": "mean_over_nodes",
    }
    payload["meta"] = meta

    torch.save(payload, out_pt)
    _write_graph_summary_csv(payload, out_csv, run_ref, epoch)
    logging.info(
        "Wrote %s (gnn_node %s, graphs=%d, nodes=%d)",
        out_pt,
        tuple(payload["gnn_node"].shape),
        int(payload["num_graphs"]),
        int(payload["num_nodes"]),
    )
    logging.info("Wrote %s", out_csv)
    return out_pt


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dump node-level gates for one or all gated routing runs."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    tracks = [t.strip() for t in re.split(r"[,;]+", args.tracks) if t.strip()]

    if args.run_dir:
        run_dir = Path(args.run_dir)
        match = RUN_NAME_RE.match(run_dir.name)
        if match is None:
            raise SystemExit(f"Unrecognized run directory name: {run_dir.name}")
        track = "toy"
        if "sigma" in run_dir.parts:
            track = "sigma"
        elif "toy" in run_dir.parts:
            track = "toy"
        run_refs = [
            RunRef(
                track=track,
                run_dir=run_dir,
                model=match.group("model"),
                lr_tag=f"lr{match.group('lr_tag')}",
                seed=int(match.group("seed")),
            ),
        ]
    elif args.results_root:
        run_refs = list(iter_gated_run_refs(Path(args.results_root), tracks))
    else:
        raise SystemExit("Provide --run-dir or --results-root.")

    if not run_refs:
        raise SystemExit("No gated runs found.")

    logging.info("Dumping %d gated run(s) on device=%s", len(run_refs), device)
    failed: list[str] = []
    for ref in run_refs:
        try:
            dump_run_node_gates(
                ref,
                args.dataset_dir,
                device,
                splits,
                int(args.layer_idx),
                bool(args.skip_existing),
            )
        except Exception:
            logging.exception("Failed on %s", ref.run_dir)
            failed.append(str(ref.run_dir))

    if failed:
        logging.warning("Failed %d / %d runs", len(failed), len(run_refs))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
