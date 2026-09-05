#!/usr/bin/env python3
"""Gate γ by node role × layer × τ for GIN depth-routing (gated runs).

Depth is a 2-hop pipeline; root-only plots miss mid-L0 (leaf intake). This
script evaluates gated checkpoints on the test split and reports mean MP gate
γ for roles ``root`` / ``mid`` / ``leaf`` at each layer, split by τ.

Priority readouts:
  - mid @ L0: hop-2 intake (want τ=1 ≥ τ=0)
  - root @ L1: deep readout (want τ=1 > τ=0)
  - root @ L0: shallow readout (want τ=0 > τ=1)

Outputs under ``--out-dir``:
  - ``gates_by_role_per_run.csv``
  - ``gates_by_role_summary.csv``
  - ``fig_gates_by_role_layer_tau.png`` / ``.pdf``
  - ``paper_figures/fig_gates_by_role_layer_tau.png``

Example::

  python scripts/synthetic/analyze_gin_depth_gates_by_role.py \\
    --results-root $GNNPLUS_OUT_DIR/gin_routing_depth \\
    --dataset-dir $GNNPLUS_DATASET_DIR \\
    --out-dir results/gin_routing_depth/analysis \\
    --tracks toy --lr-tag lr001
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401
from GNNPlus.hybrid_gate_tracking import _unwrap_model
from GNNPlus.loader.dataset.gin_depth_routing import (
    ROLE_LEAF,
    ROLE_MID,
    ROLE_ROOT,
)
from scripts.synthetic.analyze_gin_depth_routing_results import (
    RunRef,
    _load_cfg_for_run,
    _pick_best_epoch,
    discover_run_refs,
)
from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

ROLE_NAMES: dict[int, str] = {
    ROLE_ROOT: "root",
    ROLE_MID: "mid",
    ROLE_LEAF: "leaf",
}
ROLE_ORDER: tuple[str, ...] = ("root", "mid", "leaf")
HEAD_IDX = 0
GATED_MODEL = "l2_a0g1_gated"


@dataclass(frozen=True)
class RoleGateRow:
    """Mean γ for one (run, layer, role, τ) cell."""

    track: str
    model: str
    lr_tag: str
    seed: int
    layer: int
    role: str
    tau: int
    mean_gamma: float
    n_nodes: int


def _default_results_root() -> str:
    """Resolve default results root from env or local path."""
    if "GNNPLUS_OUT_DIR" in os.environ:
        return f"{os.environ['GNNPLUS_OUT_DIR'].rstrip('/')}/gin_routing_depth"
    return "results/gin_routing_depth"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default=_default_results_root())
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get(
            "GNNPLUS_DATASET_DIR",
            "results/gin_routing_depth/data",
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gin_routing_depth/analysis",
    )
    parser.add_argument("--tracks", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument("--model", type=str, default=GATED_MODEL)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing gates_by_role_per_run.csv.",
    )
    return parser.parse_args(argv)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _agg_mean_std(vals: Sequence[float]) -> tuple[float, float]:
    """Mean/std over finite values."""
    finite = [v for v in vals if v == v]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return float(finite[0]), 0.0
    return float(mean(finite)), float(pstdev(finite))


@torch.no_grad()
def evaluate_role_gates(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
) -> list[RoleGateRow]:
    """Collect mean γ by (layer, role, τ) on the test split for one run."""
    _load_cfg_for_run(run_ref, dataset_dir)
    seed_everything(int(cfg.seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    loaders = create_loader()
    test_loader = loaders[2] if len(loaders) > 2 else None
    if test_loader is None:
        raise RuntimeError("Test loader missing.")

    model = create_model()
    epoch = _pick_best_epoch(run_ref.run_dir)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    model.eval()
    model.to(device)
    core = _unwrap_model(model)
    if not hasattr(core, "collect_per_graph_gates"):
        raise TypeError(f"{type(core).__name__} lacks collect_per_graph_gates")

    # (layer, role_name, tau) -> list of per-node γ
    buckets: dict[tuple[int, str, int], list[float]] = defaultdict(list)

    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing tau.")
        if not hasattr(batch, "node_role") or batch.node_role is None:
            raise AttributeError("Batch missing node_role.")

        tau_g = batch.tau.view(-1).long()
        roles = batch.node_role.view(-1).long()
        gate_out = core.collect_per_graph_gates(batch.clone())
        gnn_node = gate_out["gnn_node"]  # [N, L, Ng]
        batch_ids = gate_out["batch"].long()
        if gnn_node.ndim != 3:
            raise ValueError(f"expected gnn_node [N,L,Ng], got {tuple(gnn_node.shape)}")
        num_layers = int(gnn_node.shape[1])
        if int(roles.numel()) != int(gnn_node.shape[0]):
            raise ValueError("node_role length mismatch vs gnn_node")

        # Expand graph-level τ to nodes.
        tau_node = tau_g[batch_ids]
        for layer_idx in range(num_layers):
            gamma = gnn_node[:, layer_idx, HEAD_IDX].detach().float().cpu()
            role_cpu = roles.detach().cpu()
            tau_cpu = tau_node.detach().cpu()
            for role_id, role_name in ROLE_NAMES.items():
                for tau_val in (0, 1):
                    mask = (role_cpu == role_id) & (tau_cpu == tau_val)
                    if bool(mask.any()):
                        buckets[(layer_idx, role_name, tau_val)].extend(
                            gamma[mask].tolist(),
                        )

    rows: list[RoleGateRow] = []
    for (layer_idx, role_name, tau_val), vals in sorted(buckets.items()):
        rows.append(
            RoleGateRow(
                track=run_ref.track,
                model=run_ref.model,
                lr_tag=run_ref.lr_tag,
                seed=run_ref.seed,
                layer=layer_idx,
                role=role_name,
                tau=tau_val,
                mean_gamma=float(mean(vals)) if vals else float("nan"),
                n_nodes=len(vals),
            ),
        )
    return rows


def _write_per_run_csv(rows: Sequence[RoleGateRow], path: Path) -> None:
    """Write per-run role×layer×τ CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)


def _load_per_run_csv(path: Path) -> list[RoleGateRow]:
    """Load prior per-run CSV."""
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            RoleGateRow(
                track=raw["track"],
                model=raw["model"],
                lr_tag=raw["lr_tag"],
                seed=int(raw["seed"]),
                layer=int(raw["layer"]),
                role=raw["role"],
                tau=int(raw["tau"]),
                mean_gamma=float(raw["mean_gamma"]),
                n_nodes=int(raw["n_nodes"]),
            )
            for raw in csv.DictReader(fh)
        ]


def _summarize(rows: Sequence[RoleGateRow]) -> list[dict[str, Any]]:
    """Mean/std over seeds by track/lr/layer/role/τ."""
    groups: dict[tuple[str, str, str, int, str, int], list[RoleGateRow]] = {}
    for row in rows:
        key = (row.track, row.model, row.lr_tag, row.layer, row.role, row.tau)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (track, model, lr_tag, layer, role, tau), items in sorted(groups.items()):
        m, s = _agg_mean_std([it.mean_gamma for it in items])
        out.append(
            {
                "track": track,
                "model": model,
                "lr_tag": lr_tag,
                "layer": layer,
                "role": role,
                "tau": tau,
                "n_seeds": len(items),
                "mean_gamma_mean": m,
                "mean_gamma_std": s,
                "n_nodes_mean": float(mean(it.n_nodes for it in items)),
            },
        )
    return out


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write summary CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_by_role(
    summary: Sequence[dict[str, Any]],
    out_path: Path,
    dpi: int,
    *,
    track: str,
    lr_tag: str,
) -> None:
    """Grouped bars: γ by role×layer, τ=0 vs τ=1."""
    subset = [
        r
        for r in summary
        if str(r["track"]) == track and str(r["lr_tag"]) == lr_tag
    ]
    if not subset:
        logging.warning("No summary rows for track=%s lr=%s", track, lr_tag)
        return

    layers = sorted({int(r["layer"]) for r in subset})
    # Panel order: highlight mid-L0 and root-L1 via annotation.
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    lookup = {
        (int(r["layer"]), str(r["role"]), int(r["tau"])): r for r in subset
    }
    x = list(range(len(ROLE_ORDER)))
    bar_w = 0.36
    for ax, layer in zip(axes, layers):
        means0 = [
            float(lookup[(layer, role, 0)]["mean_gamma_mean"])
            if (layer, role, 0) in lookup
            else float("nan")
            for role in ROLE_ORDER
        ]
        stds0 = [
            float(lookup[(layer, role, 0)]["mean_gamma_std"])
            if (layer, role, 0) in lookup
            else 0.0
            for role in ROLE_ORDER
        ]
        means1 = [
            float(lookup[(layer, role, 1)]["mean_gamma_mean"])
            if (layer, role, 1) in lookup
            else float("nan")
            for role in ROLE_ORDER
        ]
        stds1 = [
            float(lookup[(layer, role, 1)]["mean_gamma_std"])
            if (layer, role, 1) in lookup
            else 0.0
            for role in ROLE_ORDER
        ]
        ax.bar(
            [xi - bar_w / 2 for xi in x],
            means0,
            width=bar_w,
            yerr=stds0,
            capsize=3,
            color="#4C72B0",
            label=r"$\tau=0$ (shallow)" if layer == layers[0] else None,
        )
        ax.bar(
            [xi + bar_w / 2 for xi in x],
            means1,
            width=bar_w,
            yerr=stds1,
            capsize=3,
            color="#DD8452",
            label=r"$\tau=1$ (deep)" if layer == layers[0] else None,
        )
        # Δ annotation for key cells
        for role in ROLE_ORDER:
            if (layer, role, 0) in lookup and (layer, role, 1) in lookup:
                d = (
                    float(lookup[(layer, role, 1)]["mean_gamma_mean"])
                    - float(lookup[(layer, role, 0)]["mean_gamma_mean"])
                )
                xi = ROLE_ORDER.index(role)
                ax.text(
                    xi,
                    1.02,
                    rf"$\Delta$={d:+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )
        title = f"Layer {layer}"
        if layer == 0:
            title += r" · watch mid (hop-2 intake)"
        if layer == 1:
            title += r" · watch root (deep readout)"
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ROLE_ORDER)
        ax.set_ylim(0.0, 1.18)
        ax.grid(axis="y", alpha=0.25)
        if layer == layers[0]:
            ax.set_ylabel(r"Mean MP gate $\gamma$")
            ax.legend(loc="lower left", fontsize=8)

    n_seeds = int(subset[0]["n_seeds"])
    fig.suptitle(
        rf"Depth routing · gates by role$\times$layer$\times\tau$ "
        rf"({track}, {lr_tag}, $n={n_seeds}$ seeds)",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = out_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    per_run_path = out_dir / "gates_by_role_per_run.csv"
    summary_path = out_dir / "gates_by_role_summary.csv"

    if args.plots_only:
        rows = _load_per_run_csv(per_run_path)
    else:
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
        refs = [
            r
            for r in discover_run_refs(Path(args.results_root), tracks)
            if r.model == args.model and (not args.lr_tag or r.lr_tag == args.lr_tag)
        ]
        if not refs:
            raise SystemExit(
                f"No gated runs matching model={args.model} lr={args.lr_tag} "
                f"under {args.results_root}",
            )
        device = _select_device(args.device)
        rows = []
        for ref in refs:
            logging.info("Role gates: %s", ref.run_dir)
            rows.extend(evaluate_role_gates(ref, args.dataset_dir, device))
        _write_per_run_csv(rows, per_run_path)

    summary = _summarize(rows)
    _write_summary_csv(summary, summary_path)

    tracks = sorted({str(r["track"]) for r in summary})
    lr_tags = sorted({str(r["lr_tag"]) for r in summary})
    preferred_lr = args.lr_tag if args.lr_tag in lr_tags else (
        "lr001" if "lr001" in lr_tags else lr_tags[0]
    )
    for track in tracks:
        fig_path = out_dir / f"fig_gates_by_role_layer_tau_{track}.png"
        _plot_by_role(
            summary,
            fig_path,
            args.dpi,
            track=track,
            lr_tag=preferred_lr,
        )
        _plot_by_role(
            summary,
            paper / f"fig_gates_by_role_layer_tau_{track}.png",
            args.dpi,
            track=track,
            lr_tag=preferred_lr,
        )
        print(f"Wrote {fig_path}")

    # Convenience alias for toy.
    if "toy" in tracks:
        _plot_by_role(
            summary,
            out_dir / "fig_gates_by_role_layer_tau.png",
            args.dpi,
            track="toy",
            lr_tag=preferred_lr,
        )
        _plot_by_role(
            summary,
            paper / "fig_gates_by_role_layer_tau.png",
            args.dpi,
            track="toy",
            lr_tag=preferred_lr,
        )

    print(f"Wrote {per_run_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
