#!/usr/bin/env python3
"""Dump per-graph root MP gates (all layers) for GIN depth-routing gated runs.

Writes into each gated run directory:
  - ``gate_values_per_node.pt`` — packed ``gnn_node`` [N, L, Ng] + tau/y/split
  - ``gate_graph_summary.csv`` — root γ at layer 0 and layer 1 per graph

Example::

  python scripts/synthetic/dump_gin_depth_routing_node_gates.py \\
    --results-root $GNNPLUS_OUT_DIR/gin_routing_depth \\
    --dataset-dir $GNNPLUS_DATASET_DIR \\
    --tracks toy
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

import GNNPlus  # noqa: F401
from GNNPlus.hybrid_gate_tracking import _unwrap_model

RUN_NAME_RE = re.compile(
    r"^(?P<model>.+)_lr(?P<lr_tag>\d+)_seed(?P<seed>\d+)$",
)
GATED_MODEL = "l2_a0g1_gated"
SPLIT_NAMES = ("train", "val", "test")
HEAD_IDX = 0


@dataclass(frozen=True)
class RunRef:
    """One training run directory."""

    track: str
    run_dir: Path
    model: str
    lr_tag: str
    seed: int


def _default_results_root() -> str:
    """Resolve default results root from env or local path."""
    if "GNNPLUS_OUT_DIR" in os.environ:
        return f"{os.environ['GNNPLUS_OUT_DIR'].rstrip('/')}/gin_routing_depth"
    return "results/gin_routing_depth"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get(
            "GNNPLUS_DATASET_DIR",
            "results/gin_routing_depth/data",
        ),
    )
    parser.add_argument("--tracks", type=str, default="toy")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
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
                continue
            ckpt_dir = run_dir / "ckpt"
            if not ckpt_dir.is_dir() or not any(ckpt_dir.glob("*.ckpt")):
                continue
            ref = _parse_run_ref(track, run_dir)
            if ref is not None:
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
    # run_dir is not a YACS key — set after load_cfg.
    cfg.run_dir = str(run_ref.run_dir)
    cfg.out_dir = str(run_ref.run_dir.parent.parent)


def _pick_best_epoch(run_dir: Path) -> int:
    """Return latest checkpoint epoch under ``run_dir/ckpt``."""
    cfg.run_dir = str(run_dir)
    epochs = list(get_ckpt_epochs())
    if not epochs:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/ckpt")
    return int(max(epochs))


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ptr_from_batch(batch_ids: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Build CSR ``ptr`` of length ``num_graphs + 1``."""
    counts = torch.bincount(batch_ids, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


@torch.no_grad()
def _collect_split_gates(
    model: torch.nn.Module,
    loader: Any,
    split_id: int,
    device: torch.device,
) -> dict[str, Any]:
    """Collect per-node gates and graph metadata for one split."""
    core = _unwrap_model(model)
    if not hasattr(core, "collect_per_graph_gates"):
        raise TypeError(f"{type(core).__name__} lacks collect_per_graph_gates")

    gnn_n_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    role_parts: list[torch.Tensor] = []
    tau_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    split_parts: list[torch.Tensor] = []
    graph_offset = 0

    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, "tau") or batch.tau is None:
                raise AttributeError("Batch missing graph-level tau.")
            if not hasattr(batch, "node_role") or batch.node_role is None:
                raise AttributeError("Batch missing node_role.")
            tau_g = batch.tau.view(-1).long().detach().cpu()
            role_n = batch.node_role.view(-1).long().detach().cpu()
            gate_out = core.collect_per_graph_gates(batch.clone())
            n_graphs = int(gate_out["num_graphs"])
            split_parts.append(torch.full((n_graphs,), split_id, dtype=torch.long))
            tau_parts.append(tau_g)
            role_parts.append(role_n)
            if gate_out["y"] is not None:
                y_parts.append(gate_out["y"].detach().cpu().long().view(-1))
            else:
                y_parts.append(torch.full((n_graphs,), -1, dtype=torch.long))
            gnn_n_parts.append(gate_out["gnn_node"].detach().cpu())
            batch_local = gate_out["batch"].long().cpu()
            batch_parts.append(batch_local + graph_offset)
            graph_offset += n_graphs
    finally:
        if was_training:
            model.train()

    if not batch_parts:
        return {
            "gnn_node": torch.zeros(0, 0, 0),
            "batch": torch.zeros(0, dtype=torch.long),
            "ptr": torch.zeros(1, dtype=torch.long),
            "tau": torch.zeros(0, dtype=torch.long),
            "y": torch.zeros(0, dtype=torch.long),
            "split": torch.zeros(0, dtype=torch.long),
            "node_role": torch.zeros(0, dtype=torch.long),
            "num_graphs": 0,
            "num_nodes": 0,
        }

    batch_ids = torch.cat(batch_parts, dim=0)
    num_graphs = graph_offset
    ptr = _ptr_from_batch(batch_ids, num_graphs)
    return {
        "gnn_node": torch.cat(gnn_n_parts, dim=0),
        "batch": batch_ids,
        "ptr": ptr,
        "tau": torch.cat(tau_parts, dim=0),
        "y": torch.cat(y_parts, dim=0),
        "split": torch.cat(split_parts, dim=0),
        "node_role": torch.cat(role_parts, dim=0),
        "num_graphs": num_graphs,
        "num_nodes": int(batch_ids.numel()),
    }


def _merge_split_payloads(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate per-split gate payloads into one packed dict."""
    valid = [p for p in parts if int(p["num_graphs"]) > 0]
    if not valid:
        return parts[0] if parts else {}

    graph_offset = 0
    merged: dict[str, Any] = {
        "gnn_node": [],
        "batch": [],
        "tau": [],
        "y": [],
        "split": [],
        "node_role": [],
    }
    for part in valid:
        merged["gnn_node"].append(part["gnn_node"])
        merged["batch"].append(part["batch"] + graph_offset)
        merged["tau"].append(part["tau"])
        merged["y"].append(part["y"])
        merged["split"].append(part["split"])
        merged["node_role"].append(part["node_role"])
        graph_offset += int(part["num_graphs"])

    batch_ids = torch.cat(merged["batch"], dim=0)
    num_graphs = graph_offset
    ptr = _ptr_from_batch(batch_ids, num_graphs)
    return {
        "gnn_node": torch.cat(merged["gnn_node"], dim=0),
        "batch": batch_ids,
        "ptr": ptr,
        "tau": torch.cat(merged["tau"], dim=0),
        "y": torch.cat(merged["y"], dim=0),
        "split": torch.cat(merged["split"], dim=0),
        "node_role": torch.cat(merged["node_role"], dim=0),
        "num_graphs": num_graphs,
        "num_nodes": int(batch_ids.numel()),
        "head_idx": HEAD_IDX,
    }


def _write_graph_summary_csv(
    payload: dict[str, Any],
    csv_path: Path,
    run_ref: RunRef,
    epoch: int,
) -> None:
    """Write per-graph root γ at layer 0 and layer 1."""
    gnn_node = payload["gnn_node"]
    ptr: torch.Tensor = payload["ptr"]
    tau = payload["tau"].long()
    y = payload["y"].long()
    split = payload["split"].long()
    num_graphs = int(payload["num_graphs"])
    num_layers = int(gnn_node.shape[1]) if gnn_node.ndim == 3 else 0

    fieldnames = [
        "graph_idx",
        "split",
        "tau",
        "y",
        "layer0_gamma_root",
        "layer1_gamma_root",
        "layer0_gamma_mid_mean",
        "layer1_gamma_mid_mean",
        "delta_l1_minus_l0_root",
        "num_nodes",
        "track",
        "lr_tag",
        "seed",
        "epoch",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    roles = payload.get("node_role")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for g in range(num_graphs):
            lo = int(ptr[g].item())
            hi = int(ptr[g + 1].item())
            # Root is the first node of each graph.
            l0 = float(gnn_node[lo, 0, HEAD_IDX].item()) if num_layers > 0 else float("nan")
            l1 = float(gnn_node[lo, 1, HEAD_IDX].item()) if num_layers > 1 else float("nan")
            mid_l0 = float("nan")
            mid_l1 = float("nan")
            if roles is not None and num_layers > 0:
                role_slice = roles[lo:hi].long()
                mid_mask = role_slice == 1  # ROLE_MID
                if bool(mid_mask.any()):
                    mid_l0 = float(gnn_node[lo:hi, 0, HEAD_IDX][mid_mask].mean().item())
                    if num_layers > 1:
                        mid_l1 = float(gnn_node[lo:hi, 1, HEAD_IDX][mid_mask].mean().item())
            writer.writerow(
                {
                    "graph_idx": g,
                    "split": SPLIT_NAMES[int(split[g].item())],
                    "tau": int(tau[g].item()),
                    "y": int(y[g].item()),
                    "layer0_gamma_root": l0,
                    "layer1_gamma_root": l1,
                    "layer0_gamma_mid_mean": mid_l0,
                    "layer1_gamma_mid_mean": mid_l1,
                    "delta_l1_minus_l0_root": l1 - l0,
                    "num_nodes": hi - lo,
                    "track": run_ref.track,
                    "lr_tag": run_ref.lr_tag,
                    "seed": run_ref.seed,
                    "epoch": epoch,
                },
            )


def dump_run(
    run_ref: RunRef,
    dataset_dir: str,
    splits: Sequence[str],
    device: torch.device,
) -> None:
    """Dump gates for one gated run."""
    _load_cfg_for_run(run_ref, dataset_dir)
    seed_everything(int(cfg.seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    loaders = create_loader()
    split_name_to_id = {name: i for i, name in enumerate(SPLIT_NAMES)}
    model = create_model()
    epoch = _pick_best_epoch(run_ref.run_dir)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    model.eval()
    model.to(device)

    parts: list[dict[str, Any]] = []
    for split_name in splits:
        if split_name not in split_name_to_id:
            raise ValueError(f"Unknown split {split_name!r}")
        split_id = split_name_to_id[split_name]
        if split_id >= len(loaders):
            raise RuntimeError(f"Loader missing for split {split_name}")
        logging.info("%s · %s", run_ref.run_dir.name, split_name)
        parts.append(
            _collect_split_gates(model, loaders[split_id], split_id, device),
        )

    payload = _merge_split_payloads(parts)
    pt_path = run_ref.run_dir / "gate_values_per_node.pt"
    csv_path = run_ref.run_dir / "gate_graph_summary.csv"
    torch.save(payload, pt_path)
    _write_graph_summary_csv(payload, csv_path, run_ref, epoch)
    logging.info("Wrote %s and %s", pt_path, csv_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    device = _select_device(args.device)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    refs: list[RunRef] = []
    if args.run_dir:
        run_dir = Path(args.run_dir)
        track = run_dir.parent.name
        ref = _parse_run_ref(track, run_dir)
        if ref is None:
            raise SystemExit(f"Unrecognized run dir name: {run_dir.name}")
        refs = [ref]
    else:
        root = Path(args.results_root) if args.results_root else Path(_default_results_root())
        tracks = [t.strip() for t in args.tracks.replace(";", ",").split(",") if t.strip()]
        refs = list(iter_gated_run_refs(root, tracks))

    if not refs:
        raise SystemExit("No gated runs found.")

    for ref in refs:
        pt_path = ref.run_dir / "gate_values_per_node.pt"
        if args.skip_existing and pt_path.is_file():
            logging.info("Skip existing %s", pt_path)
            continue
        dump_run(ref, args.dataset_dir, splits, device)


if __name__ == "__main__":
    main()
