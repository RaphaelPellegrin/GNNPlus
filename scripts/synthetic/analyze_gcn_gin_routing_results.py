#!/usr/bin/env python3
"""Aggregate per-type test accuracy and root gates for GCN/GIN routing runs.

Loads best checkpoints under ``<results_root>/{toy,sigma}/``, evaluates on the
test split, and reports accuracy for ``tau=0`` (GCN-type) vs ``tau=1`` (GIN-type).
For two-head gated models, also collects root-node MP gate values.

Outputs (under ``--out-dir``):
  - ``per_run_metrics.csv``
  - ``summary_by_model.csv`` (mean ± std over seeds, per track/model/lr)
  - ``fig_baseline_per_type.png`` / ``.pdf``
  - ``fig_gate_by_type.png`` / ``.pdf`` (gated a0g2 only)

Example (cluster netscratch):
  python scripts/synthetic/analyze_gcn_gin_routing_results.py \\
    --results-root /n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results/gcn_gin_routing \\
    --dataset-dir /n/netscratch/mweber_lab/Lab/gnnplus_datasets \\
    --out-dir results/gcn_gin_routing/analysis
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
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
from GNNPlus.hybrid_gate_tracking import _unwrap_model

RUN_NAME_RE = re.compile(
    r"^(?P<model>.+)_lr(?P<lr_tag>\d+)_seed(?P<seed>\d+)$",
)

MODEL_ORDER: tuple[str, ...] = (
    "a0g2_gated",
    "a0g2_ungated",
    "a0g1_gcn",
    "a0g1_gin",
)
MODEL_LABELS: dict[str, str] = {
    "a0g2_gated": "SiGMA gated",
    "a0g2_ungated": "SiGMA ungated",
    "a0g1_gcn": "GCN-only",
    "a0g1_gin": "GIN-only",
}


@dataclass(frozen=True)
class RunRef:
    """One training run directory."""

    track: str
    run_dir: Path
    model: str
    lr_tag: str
    seed: int


@dataclass
class RunMetrics:
    """Test metrics for one checkpoint."""

    track: str
    model: str
    lr_tag: str
    seed: int
    run_dir: str
    epoch: int
    n_all: int
    n_tau0: int
    n_tau1: int
    acc_all: float
    acc_tau0: float
    acc_tau1: float
    gin_head_idx: int
    gcn_head_idx: int
    gin_gate_tau0: float
    gin_gate_tau1: float
    gcn_gate_tau0: float
    gcn_gate_tau1: float
    has_gates: bool


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/gcn_gin_routing",
        help="Parent of toy/ and sigma/ run folders.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get(
            "GNNPLUS_DATASET_DIR",
            "results/gcn_gin_routing/data",
        ),
        help="Parent of GcnGinRouting/ (PyG loader appends subfolder).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gcn_gin_routing/analysis",
        help="Directory for CSV summaries and figures.",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="toy,sigma",
        help="Comma-separated tracks to analyze (toy,sigma).",
    )
    parser.add_argument(
        "--lr-tag",
        type=str,
        default="",
        help="If set, only runs with this lr tag (e.g. lr001). Default: all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    return parser.parse_args(argv)


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


def iter_run_refs(results_root: Path, tracks: Sequence[str]) -> Iterator[RunRef]:
    """Yield run directories that contain ``config_used.yaml`` and ``ckpt/``."""
    for track in tracks:
        track_dir = results_root / track
        if not track_dir.is_dir():
            logging.warning("Missing track directory: %s", track_dir)
            continue
        for run_dir in sorted(track_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "config_used.yaml").is_file():
                continue
            if not (run_dir / "ckpt").is_dir():
                logging.warning("Skipping %s (no ckpt/)", run_dir)
                continue
            ref = _parse_run_ref(track, run_dir)
            if ref is None:
                logging.warning("Skipping unrecognized run name: %s", run_dir.name)
                continue
            yield ref


def _head_indices(gnn_types: str) -> tuple[int, int, bool]:
    """Return (gin_head_idx, gcn_head_idx, is_two_head)."""
    parts = [p.strip() for p in str(gnn_types).split(",") if p.strip()]
    if len(parts) != 2:
        return 0, 1, False
    gin_idx = 0
    gcn_idx = 1
    for idx, kind in enumerate(parts):
        upper = kind.upper()
        if "GIN" in upper or "ROUTING_SUM" in upper:
            gin_idx = idx
        if "GCN" in upper or "ROUTING_NORM" in upper or "NORMGCN" in upper:
            gcn_idx = idx
    return gin_idx, gcn_idx, True


def _load_cfg_for_run(
    run_ref: RunRef,
    dataset_dir: str,
) -> None:
    """Load GraphGym cfg from ``config_used.yaml`` for one run."""
    cfg_path = str(run_ref.run_dir / "config_used.yaml")
    old_argv = sys.argv
    sys.argv = [
        old_argv[0],
        "--cfg",
        cfg_path,
        "dataset.dir",
        dataset_dir,
        "seed",
        str(run_ref.seed),
        "run_dir",
        str(run_ref.run_dir),
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


def _root_indices(batch: Any) -> torch.Tensor:
    """Global node indices of the root (first node) per graph in a batch."""
    if hasattr(batch, "ptr") and batch.ptr is not None:
        return batch.ptr[:-1].long()
    batch_ids = batch.batch
    num_graphs = int(batch_ids.max().item()) + 1 if batch_ids.numel() else 0
    roots = torch.zeros(num_graphs, dtype=torch.long, device=batch_ids.device)
    for g in range(num_graphs):
        mask = batch_ids == g
        roots[g] = int(torch.nonzero(mask, as_tuple=False)[0].item())
    return roots


@torch.no_grad()
def evaluate_run(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
) -> RunMetrics:
    """Load checkpoint and compute per-type test metrics (+ root gates)."""
    _load_cfg_for_run(run_ref, dataset_dir)
    seed_everything(int(cfg.seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    loaders = create_loader()
    test_loader = loaders[2] if len(loaders) > 2 else None
    if test_loader is None:
        raise RuntimeError("Test loader missing — check dataset.dir and splits.")

    model = create_model()
    epoch = _pick_best_epoch(run_ref.run_dir)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=epoch)
    model.eval()
    model.to(device)

    hybrid = getattr(cfg.gnn, "hybrid", None)
    gnn_types = str(getattr(hybrid, "gnn_types", "")) if hybrid is not None else ""
    gin_idx, gcn_idx, two_head = _head_indices(gnn_types)
    gate_mode = str(getattr(hybrid, "gate", "none")).lower() if hybrid is not None else "none"
    collect_gates = two_head and gate_mode not in ("none", "off")

    core = _unwrap_model(model)
    has_collect = hasattr(core, "collect_per_graph_gates")

    correct_all = 0
    correct_t0 = 0
    correct_t1 = 0
    n_all = 0
    n_t0 = 0
    n_t1 = 0

    gin_gates_t0: list[float] = []
    gin_gates_t1: list[float] = []
    gcn_gates_t0: list[float] = []
    gcn_gates_t1: list[float] = []

    for batch in test_loader:
        batch = batch.to(device)
        pred, true = model(batch)
        _loss, pred_score = compute_loss(pred, true)
        pred_label = pred_score.argmax(dim=-1).view(-1)
        true_label = true.view(-1).long()

        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing graph-level tau — regenerate dataset.")
        tau = batch.tau.view(-1).long()

        correct = pred_label == true_label
        correct_all += int(correct.sum().item())
        n_all += int(correct.numel())

        mask0 = tau == 0
        mask1 = tau == 1
        n_t0 += int(mask0.sum().item())
        n_t1 += int(mask1.sum().item())
        if mask0.any():
            correct_t0 += int(correct[mask0].sum().item())
        if mask1.any():
            correct_t1 += int(correct[mask1].sum().item())

        if collect_gates and has_collect:
            gate_out = core.collect_per_graph_gates(batch)
            gnn_node = gate_out["gnn_node"]  # [N, L, Ng]
            roots = _root_indices(batch).to(gnn_node.device)
            layer_idx = 0
            gin_root = gnn_node[roots, layer_idx, gin_idx].detach().cpu()
            gcn_root = gnn_node[roots, layer_idx, gcn_idx].detach().cpu()
            tau_cpu = tau.detach().cpu()
            gin_gates_t0.extend(gin_root[tau_cpu == 0].tolist())
            gin_gates_t1.extend(gin_root[tau_cpu == 1].tolist())
            gcn_gates_t0.extend(gcn_root[tau_cpu == 0].tolist())
            gcn_gates_t1.extend(gcn_root[tau_cpu == 1].tolist())

    def _safe_acc(num: int, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    def _safe_mean(vals: list[float]) -> float:
        return float(mean(vals)) if vals else float("nan")

    return RunMetrics(
        track=run_ref.track,
        model=run_ref.model,
        lr_tag=run_ref.lr_tag,
        seed=run_ref.seed,
        run_dir=str(run_ref.run_dir),
        epoch=epoch,
        n_all=n_all,
        n_tau0=n_t0,
        n_tau1=n_t1,
        acc_all=_safe_acc(correct_all, n_all),
        acc_tau0=_safe_acc(correct_t0, n_t0),
        acc_tau1=_safe_acc(correct_t1, n_t1),
        gin_head_idx=gin_idx,
        gcn_head_idx=gcn_idx,
        gin_gate_tau0=_safe_mean(gin_gates_t0),
        gin_gate_tau1=_safe_mean(gin_gates_t1),
        gcn_gate_tau0=_safe_mean(gcn_gates_t0),
        gcn_gate_tau1=_safe_mean(gcn_gates_t1),
        has_gates=collect_gates and bool(gin_gates_t0 or gin_gates_t1),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_runs(rows: Sequence[RunMetrics]) -> list[dict[str, Any]]:
    """Mean/std over seeds grouped by track, model, lr."""
    groups: dict[tuple[str, str, str], list[RunMetrics]] = {}
    for row in rows:
        key = (row.track, row.model, row.lr_tag)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (track, model, lr_tag), items in sorted(groups.items()):
        def _agg_attr(items: list[RunMetrics], attr: str) -> tuple[float, float]:
            vals = [float(getattr(it, attr)) for it in items]
            if len(vals) == 1:
                return vals[0], 0.0
            return float(mean(vals)), float(pstdev(vals))

        acc_all_m, acc_all_s = _agg_attr(items, "acc_all")
        acc_t0_m, acc_t0_s = _agg_attr(items, "acc_tau0")
        acc_t1_m, acc_t1_s = _agg_attr(items, "acc_tau1")
        summary.append(
            {
                "track": track,
                "model": model,
                "lr_tag": lr_tag,
                "n_seeds": len(items),
                "acc_all_mean": acc_all_m,
                "acc_all_std": acc_all_s,
                "acc_tau0_mean": acc_t0_m,
                "acc_tau0_std": acc_t0_s,
                "acc_tau1_mean": acc_t1_m,
                "acc_tau1_std": acc_t1_s,
            }
        )
    return summary


def _plot_baseline_per_type(
    summary: Sequence[dict[str, Any]],
    out_path: Path,
) -> None:
    """Grouped bar chart: per-type test accuracy by model (toy vs sigma panels)."""
    tracks = sorted({str(row["track"]) for row in summary})
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.5 * len(tracks), 4.5), squeeze=False)
    bar_w = 0.36
    tau_colors = {"tau0": "#4C72B0", "tau1": "#DD8452"}

    for ax, track in zip(axes[0], tracks, strict=True):
        subset = [r for r in summary if r["track"] == track]
        by_model = {str(r["model"]): r for r in subset}
        models = [m for m in MODEL_ORDER if m in by_model]
        x = list(range(len(models)))
        t0_vals = [float(by_model[m]["acc_tau0_mean"]) for m in models]
        t1_vals = [float(by_model[m]["acc_tau1_mean"]) for m in models]
        t0_err = [float(by_model[m]["acc_tau0_std"]) for m in models]
        t1_err = [float(by_model[m]["acc_tau1_std"]) for m in models]

        ax.bar(
            [xi - bar_w / 2 for xi in x],
            t0_vals,
            width=bar_w,
            yerr=t0_err,
            capsize=3,
            label=r"$\tau=0$ (GCN-type)",
            color=tau_colors["tau0"],
        )
        ax.bar(
            [xi + bar_w / 2 for xi in x],
            t1_vals,
            width=bar_w,
            yerr=t1_err,
            capsize=3,
            label=r"$\tau=1$ (GIN-type)",
            color=tau_colors["tau1"],
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=15, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Track {track}")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)

    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Per-type test accuracy (mean over seeds)", y=1.02, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_gates_by_type(rows: Sequence[RunMetrics], out_path: Path) -> None:
    """Box-style bar plot of root gates for gated models (mean per run, grouped)."""
    gated = [r for r in rows if r.model == "a0g2_gated" and r.has_gates]
    if not gated:
        logging.warning("No gated runs with gate stats — skipping gate figure.")
        return

    tracks = sorted({r.track for r in gated})
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.0 * len(tracks), 4.2), squeeze=False)

    for ax, track in zip(axes[0], tracks, strict=True):
        subset = [r for r in gated if r.track == track]
        # Mean over seeds for each (tau, head) combination.
        series = {
            (0, "GIN"): [r.gin_gate_tau0 for r in subset],
            (1, "GIN"): [r.gin_gate_tau1 for r in subset],
            (0, "GCN"): [r.gcn_gate_tau0 for r in subset],
            (1, "GCN"): [r.gcn_gate_tau1 for r in subset],
        }
        labels = [r"$\tau=0$ GIN $\gamma$", r"$\tau=1$ GIN $\gamma$", r"$\tau=0$ GCN $\gamma$", r"$\tau=1$ GCN $\gamma$"]
        keys = [(0, "GIN"), (1, "GIN"), (0, "GCN"), (1, "GCN")]
        colors = ["#55A868", "#55A868", "#4C72B0", "#4C72B0"]
        alphas = [1.0, 0.55, 1.0, 0.55]
        x = list(range(len(keys)))
        means = [float(mean(series[k])) if series[k] else float("nan") for k in keys]
        stds = [
            float(pstdev(series[k])) if len(series[k]) > 1 else 0.0 for k in keys
        ]
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            alpha=alphas,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(r"Root gate $\gamma$ (layer 0)")
        ax.set_title(f"Track {track} — SiGMA gated")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Mean root MP gate by graph type (seed mean ± std)", y=1.02, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Analyze all runs and write CSV + figures."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info("Results root: %s", results_root)
    logging.info("Dataset dir: %s", args.dataset_dir)
    logging.info("Device: %s", device)

    run_refs = list(iter_run_refs(results_root, tracks))
    if args.lr_tag:
        run_refs = [r for r in run_refs if r.lr_tag == args.lr_tag]
    if not run_refs:
        raise SystemExit(f"No runs found under {results_root}")

    metrics: list[RunMetrics] = []
    for ref in run_refs:
        logging.info("Evaluating %s / %s", ref.track, ref.run_dir.name)
        try:
            metrics.append(evaluate_run(ref, args.dataset_dir, device))
        except Exception:
            logging.exception("Failed on %s", ref.run_dir)

    if not metrics:
        raise SystemExit("No runs evaluated successfully.")

    per_run_fields = list(asdict(metrics[0]).keys())
    _write_csv(
        out_dir / "per_run_metrics.csv",
        [asdict(m) for m in metrics],
        per_run_fields,
    )
    summary = _summarize_runs(metrics)
    summary_fields = list(summary[0].keys()) if summary else []
    if summary:
        _write_csv(out_dir / "summary_by_model.csv", summary, summary_fields)

    _plot_baseline_per_type(summary, out_dir / "fig_baseline_per_type.png")
    _plot_gates_by_type(metrics, out_dir / "fig_gate_by_type.png")

    logging.info("Wrote analysis to %s", out_dir)
    print(f"\nAnalysis saved to: {out_dir.resolve()}")
    print(f"  - {out_dir / 'per_run_metrics.csv'}")
    print(f"  - {out_dir / 'summary_by_model.csv'}")
    print(f"  - {out_dir / 'fig_baseline_per_type.png'}")
    print(f"  - {out_dir / 'fig_gate_by_type.png'}")


if __name__ == "__main__":
    main()
