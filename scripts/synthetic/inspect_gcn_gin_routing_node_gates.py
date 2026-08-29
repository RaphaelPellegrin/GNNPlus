#!/usr/bin/env python3
"""Inspect packed node-level gates from ``dump_gcn_gin_routing_node_gates.py``.

Loads ``gate_values_per_node.pt`` (or a path you pass) and prints routing
summaries: mean root γ_GIN / γ_GCN by ``tau`` and split, plus optional per-node
root-vs-neighbor breakdown on the test split.

Example:
  python scripts/synthetic/inspect_gcn_gin_routing_node_gates.py \\
    --pt /path/to/a0g2_gated_lr001_seed0/gate_values_per_node.pt

  python scripts/synthetic/inspect_gcn_gin_routing_node_gates.py \\
    --results-root /n/netscratch/.../gcn_gin_routing/toy \\
    --split test
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPLIT_NAMES = ("train", "val", "test")
RUN_NAME_RE = re.compile(r"^a0g2_gated_lr")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt",
        type=str,
        default="",
        help="Path to gate_values_per_node.pt (single-file mode).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Directory with gated run subdirs (batch inspect).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=SPLIT_NAMES,
        help="Split to summarize (default: test).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="Optional path to write aggregated per-run summary CSV.",
    )
    return parser.parse_args(argv)


def _safe_mean(vals: list[float]) -> float:
    """Mean or NaN when empty."""
    return float(mean(vals)) if vals else float("nan")


def summarize_payload(payload: dict[str, Any], split_name: str) -> dict[str, float]:
    """Compute routing stats for one loaded ``.pt`` payload."""
    split_id = SPLIT_NAMES.index(split_name)
    mask_split = payload["split"].long() == split_id
    if not bool(mask_split.any()):
        raise ValueError(f"No graphs in split={split_name!r}")

    gnn_node: torch.Tensor = payload["gnn_node"]
    ptr: torch.Tensor = payload["ptr"]
    layer_idx = int(payload.get("layer_idx", payload["meta"].get("layer_idx", 0)))
    gin_idx = int(payload["gin_head_idx"])
    gcn_idx = int(payload["gcn_head_idx"])
    tau_all = payload["tau"].long()
    graph_ids = torch.nonzero(mask_split, as_tuple=False).view(-1).tolist()

    gin_root_t0: list[float] = []
    gin_root_t1: list[float] = []
    gcn_root_t0: list[float] = []
    gcn_root_t1: list[float] = []
    gin_nbr_t0: list[float] = []
    gin_nbr_t1: list[float] = []
    gcn_nbr_t0: list[float] = []
    gcn_nbr_t1: list[float] = []

    for g in graph_ids:
        lo = int(ptr[g].item())
        hi = int(ptr[g + 1].item())
        nodes = gnn_node[lo:hi, layer_idx, :]
        gin_n = nodes[:, gin_idx].float()
        gcn_n = nodes[:, gcn_idx].float()
        tau = int(tau_all[g].item())
        gin_root = float(gin_n[0].item())
        gcn_root = float(gcn_n[0].item())
        if hi - lo > 1:
            gin_nbr = float(gin_n[1:].mean().item())
            gcn_nbr = float(gcn_n[1:].mean().item())
        else:
            gin_nbr = float("nan")
            gcn_nbr = float("nan")

        if tau == 0:
            gin_root_t0.append(gin_root)
            gcn_root_t0.append(gcn_root)
            gin_nbr_t0.append(gin_nbr)
            gcn_nbr_t0.append(gcn_nbr)
        else:
            gin_root_t1.append(gin_root)
            gcn_root_t1.append(gcn_root)
            gin_nbr_t1.append(gin_nbr)
            gcn_nbr_t1.append(gcn_nbr)

    mean_gin_t0 = _safe_mean(gin_root_t0)
    mean_gin_t1 = _safe_mean(gin_root_t1)
    mean_gcn_t0 = _safe_mean(gcn_root_t0)
    mean_gcn_t1 = _safe_mean(gcn_root_t1)

    return {
        "n_graphs": float(len(graph_ids)),
        "n_tau0": float(len(gin_root_t0)),
        "n_tau1": float(len(gin_root_t1)),
        "gin_root_tau0": mean_gin_t0,
        "gin_root_tau1": mean_gin_t1,
        "gcn_root_tau0": mean_gcn_t0,
        "gcn_root_tau1": mean_gcn_t1,
        "delta_gcn": mean_gcn_t0 - mean_gcn_t1,
        "delta_gin": mean_gin_t1 - mean_gin_t0,
        "gin_nbr_tau0": _safe_mean([v for v in gin_nbr_t0 if v == v]),
        "gin_nbr_tau1": _safe_mean([v for v in gin_nbr_t1 if v == v]),
        "gcn_nbr_tau0": _safe_mean([v for v in gcn_nbr_t0 if v == v]),
        "gcn_nbr_tau1": _safe_mean([v for v in gcn_nbr_t1 if v == v]),
    }


def _print_summary(label: str, stats: dict[str, float]) -> None:
    """Pretty-print one run's routing summary."""
    print(f"\n=== {label} ===")
    print(f"  graphs: {int(stats['n_graphs'])} (tau0={int(stats['n_tau0'])}, tau1={int(stats['n_tau1'])})")
    print(
        f"  root γ_GIN:  tau0={stats['gin_root_tau0']:.4f}  "
        f"tau1={stats['gin_root_tau1']:.4f}  "
        f"Δ(tau1-tau0)={stats['delta_gin']:.4f}"
    )
    print(
        f"  root γ_GCN:  tau0={stats['gcn_root_tau0']:.4f}  "
        f"tau1={stats['gcn_root_tau1']:.4f}  "
        f"Δ(tau0-tau1)={stats['delta_gcn']:.4f}"
    )
    print(
        f"  nbr  γ_GIN:  tau0={stats['gin_nbr_tau0']:.4f}  "
        f"tau1={stats['gin_nbr_tau1']:.4f}"
    )
    print(
        f"  nbr  γ_GCN:  tau0={stats['gcn_nbr_tau0']:.4f}  "
        f"tau1={stats['gcn_nbr_tau1']:.4f}"
    )


def _iter_pt_files(results_root: Path) -> list[Path]:
    """Find ``gate_values_per_node.pt`` under gated run dirs."""
    paths: list[Path] = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not RUN_NAME_RE.match(run_dir.name):
            continue
        pt = run_dir / "gate_values_per_node.pt"
        if pt.is_file():
            paths.append(pt)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load gate dumps and print routing summaries."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.pt:
        pt_paths = [Path(args.pt)]
    elif args.results_root:
        pt_paths = _iter_pt_files(Path(args.results_root))
    else:
        raise SystemExit("Provide --pt or --results-root.")

    if not pt_paths:
        raise SystemExit("No gate_values_per_node.pt files found.")

    rows: list[dict[str, Any]] = []
    for pt_path in pt_paths:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        label = pt_path.parent.name
        stats = summarize_payload(payload, args.split)
        _print_summary(label, stats)
        rows.append(
            {
                "run_dir": str(pt_path.parent),
                "track": meta.get("track", ""),
                "lr_tag": meta.get("lr_tag", ""),
                "seed": meta.get("seed", ""),
                "split": args.split,
                **stats,
            },
        )

    if args.export_csv and rows:
        import csv

        out = Path(args.export_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote aggregated summary: {out.resolve()}")


if __name__ == "__main__":
    main()
