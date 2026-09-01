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
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
from GNNPlus.gcn_gin_routing_gate_tracking import (
    RootGateAccumulator,
    accumulate_root_gates_from_batch,
    hybrid_head_indices,
)
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
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing per_run_metrics.csv (skip eval).",
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


@dataclass(frozen=True)
class RunDiscoveryStats:
    """Counts from scanning a results tree for analyzable runs."""

    track: str
    n_dirs: int
    n_accepted: int
    n_no_config: int
    n_no_ckpt_dir: int
    n_empty_ckpt: int
    n_bad_name: int


def _resolve_run_config(run_dir: Path) -> Optional[Path]:
    """Return config yaml for a run dir, or None if missing."""
    for name in ("config_used.yaml", "config.yaml"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def discover_run_refs(
    results_root: Path,
    tracks: Sequence[str],
) -> tuple[list[RunRef], list[RunDiscoveryStats]]:
    """Find run directories with config + checkpoint files."""
    refs: list[RunRef] = []
    stats_out: list[RunDiscoveryStats] = []

    for track in tracks:
        track_dir = results_root / track
        n_dirs = 0
        n_accepted = 0
        n_no_config = 0
        n_no_ckpt_dir = 0
        n_empty_ckpt = 0
        n_bad_name = 0

        if not track_dir.is_dir():
            logging.warning("Missing track directory: %s", track_dir)
            stats_out.append(
                RunDiscoveryStats(track, 0, 0, 0, 0, 0, 0),
            )
            continue

        for run_dir in sorted(track_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            n_dirs += 1
            if _resolve_run_config(run_dir) is None:
                n_no_config += 1
                logging.warning("Skipping %s (no config_used.yaml / config.yaml)", run_dir)
                continue
            ckpt_dir = run_dir / "ckpt"
            if not ckpt_dir.is_dir():
                n_no_ckpt_dir += 1
                logging.warning("Skipping %s (no ckpt/)", run_dir)
                continue
            if not any(ckpt_dir.glob("*.ckpt")):
                n_empty_ckpt += 1
                logging.warning("Skipping %s (empty ckpt/)", run_dir)
                continue
            ref = _parse_run_ref(track, run_dir)
            if ref is None:
                n_bad_name += 1
                logging.warning("Skipping unrecognized run name: %s", run_dir.name)
                continue
            refs.append(ref)
            n_accepted += 1

        stats_out.append(
            RunDiscoveryStats(
                track=track,
                n_dirs=n_dirs,
                n_accepted=n_accepted,
                n_no_config=n_no_config,
                n_no_ckpt_dir=n_no_ckpt_dir,
                n_empty_ckpt=n_empty_ckpt,
                n_bad_name=n_bad_name,
            )
        )

    return refs, stats_out


def iter_run_refs(results_root: Path, tracks: Sequence[str]) -> Iterator[RunRef]:
    """Yield run directories that contain config + ``ckpt/*.ckpt``."""
    refs, _ = discover_run_refs(results_root, tracks)
    yield from refs


def _load_cfg_for_run(
    run_ref: RunRef,
    dataset_dir: str,
) -> None:
    """Load GraphGym cfg from the run's saved config yaml."""
    cfg_path_obj = _resolve_run_config(run_ref.run_dir)
    if cfg_path_obj is None:
        raise FileNotFoundError(f"No config yaml in {run_ref.run_dir}")
    cfg_path = str(cfg_path_obj)
    old_argv = sys.argv
    sys.argv = [
        old_argv[0],
        "--cfg",
        cfg_path,
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
    # run_dir is not a YACS key — set after load_cfg (see dump_attention_maps.py).
    cfg.run_dir = str(run_ref.run_dir)
    cfg.out_dir = str(run_ref.run_dir.parent.parent)


def _pick_best_epoch(run_dir: Path) -> int:
    """Return latest checkpoint epoch under ``run_dir/ckpt``."""
    cfg.run_dir = str(run_dir)
    epochs = list(get_ckpt_epochs())
    if not epochs:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/ckpt")
    return int(max(epochs))


def _pred_labels_from_score(pred_score: torch.Tensor) -> torch.Tensor:
    """Convert model scores to integer labels (matches GraphGym ``Logger``)."""
    if pred_score.ndim == 1 or (pred_score.ndim == 2 and pred_score.shape[1] == 1):
        thresh = float(getattr(cfg.model, "thresh", 0.5))
        return (pred_score.view(-1) > thresh).long()
    return pred_score.argmax(dim=-1).view(-1)


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
    gin_idx, gcn_idx, two_head = hybrid_head_indices(gnn_types)
    gate_mode = str(getattr(hybrid, "gate", "none")).lower() if hybrid is not None else "none"
    collect_gates = two_head and gate_mode not in ("none", "off")

    core = _unwrap_model(model)
    gate_accum = RootGateAccumulator()

    correct_all = 0
    correct_t0 = 0
    correct_t1 = 0
    n_all = 0
    n_t0 = 0
    n_t1 = 0

    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing graph-level tau — regenerate dataset.")
        tau = batch.tau.view(-1).long()

        # Collect gates on a clone — both collect_per_graph_gates and model(batch)
        # run the encoder and overwrite node features.
        if collect_gates:
            accumulate_root_gates_from_batch(
                core,
                batch,
                tau,
                gin_idx,
                gcn_idx,
                gate_accum,
                layer_idx=0,
            )

        pred, true = model(batch)
        _loss, pred_score = compute_loss(pred, true)
        pred_label = _pred_labels_from_score(pred_score)
        true_label = true.view(-1).long()

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

    def _safe_acc(num: int, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    gate_stats = gate_accum.means()
    if collect_gates and not gate_accum.has_samples:
        logging.warning(
            "No root gates collected for %s (%d batch failures)",
            run_ref.run_dir.name,
            gate_accum.n_fail_batches,
        )

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
        gin_gate_tau0=gate_stats["gin_gate_tau0"],
        gin_gate_tau1=gate_stats["gin_gate_tau1"],
        gcn_gate_tau0=gate_stats["gcn_gate_tau0"],
        gcn_gate_tau1=gate_stats["gcn_gate_tau1"],
        has_gates=collect_gates and gate_accum.has_samples,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_metrics_from_csv(path: Path) -> list[RunMetrics]:
    """Load ``RunMetrics`` rows written by a prior analyze pass."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[RunMetrics] = []
        for row in reader:
            rows.append(
                RunMetrics(
                    track=str(row["track"]),
                    model=str(row["model"]),
                    lr_tag=str(row["lr_tag"]),
                    seed=int(row["seed"]),
                    run_dir=str(row["run_dir"]),
                    epoch=int(row["epoch"]),
                    n_all=int(row["n_all"]),
                    n_tau0=int(row["n_tau0"]),
                    n_tau1=int(row["n_tau1"]),
                    acc_all=float(row["acc_all"]),
                    acc_tau0=float(row["acc_tau0"]),
                    acc_tau1=float(row["acc_tau1"]),
                    gin_head_idx=int(row["gin_head_idx"]),
                    gcn_head_idx=int(row["gcn_head_idx"]),
                    gin_gate_tau0=float(row["gin_gate_tau0"]),
                    gin_gate_tau1=float(row["gin_gate_tau1"]),
                    gcn_gate_tau0=float(row["gcn_gate_tau0"]),
                    gcn_gate_tau1=float(row["gcn_gate_tau1"]),
                    has_gates=row["has_gates"].strip().lower() in ("true", "1", "yes"),
                )
            )
    return rows


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


TRACK_ORDER: tuple[str, ...] = ("toy", "sigma")
GATE_BY_TYPE_TRACK_LABELS: dict[str, str] = {
    "toy": r"Track A (Toy, $d_h{=}1$)",
    "sigma": r"Track B (SiGMA, PyG GIN/GCN, $d_h{=}4$)",
}
# GIN / GCN head colors (τ=0 full, τ=1 lighter) — unchanged from original figure.
GATE_BAR_COLORS: tuple[str, ...] = ("#55A868", "#55A868", "#4C72B0", "#4C72B0")
GATE_BAR_ALPHAS: tuple[float, ...] = (1.0, 0.55, 1.0, 0.55)


def _format_lr_tag(lr_tag: str) -> str:
    """Format ``lr001`` as ``$\\mathrm{LR}=10^{-3}$`` for figure titles."""
    if lr_tag.startswith("lr") and lr_tag[2:].isdigit():
        exponent = -len(lr_tag[2:])
        return rf"$\mathrm{{LR}}=10^{{{exponent}}}$"
    return rf"$\mathrm{{LR}}={lr_tag}$"


def _plot_gates_by_type(
    rows: Sequence[RunMetrics],
    out_path: Path,
    *,
    lr_tag: str = "lr001",
    dpi: int = 160,
) -> None:
    """Bar plot of root gates for gated models (mean ± std over seeds)."""
    gated = [
        r
        for r in rows
        if r.model == "a0g2_gated" and r.has_gates and r.lr_tag == lr_tag
    ]
    if not gated:
        logging.warning("No gated runs with gate stats — skipping gate figure.")
        return

    available = {r.track for r in gated}
    tracks = [t for t in TRACK_ORDER if t in available]
    tracks.extend(sorted(available - set(tracks)))

    fig, axes = plt.subplots(1, len(tracks), figsize=(6.8 * len(tracks), 5.0), squeeze=False)
    labels = [
        r"$\tau{=}0$ GIN $\gamma$",
        r"$\tau{=}1$ GIN $\gamma$",
        r"$\tau{=}0$ GCN $\gamma$",
        r"$\tau{=}1$ GCN $\gamma$",
    ]
    keys = [(0, "GIN"), (1, "GIN"), (0, "GCN"), (1, "GCN")]
    bar_colors = [
        to_rgba(c, a)
        for c, a in zip(GATE_BAR_COLORS, GATE_BAR_ALPHAS, strict=True)
    ]

    for ax, track in zip(axes[0], tracks, strict=True):
        subset = [r for r in gated if r.track == track]
        series = {
            (0, "GIN"): [r.gin_gate_tau0 for r in subset],
            (1, "GIN"): [r.gin_gate_tau1 for r in subset],
            (0, "GCN"): [r.gcn_gate_tau0 for r in subset],
            (1, "GCN"): [r.gcn_gate_tau1 for r in subset],
        }
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
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(r"Root gate $\gamma$ (layer 0)")
        ax.set_title(GATE_BY_TYPE_TRACK_LABELS.get(track, track))
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"Mean root MP gate by graph type (5-seed mean ± std, {_format_lr_tag(lr_tag)})",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Analyze all runs and write CSV + figures."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    tracks = [t.strip() for t in re.split(r"[,;]+", args.tracks) if t.strip()]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info("Results root: %s", results_root)
    logging.info("Dataset dir: %s", args.dataset_dir)
    logging.info("Tracks: %s", tracks)
    logging.info("Device: %s", device)

    per_run_csv = out_dir / "per_run_metrics.csv"
    if args.plots_only:
        if not per_run_csv.is_file():
            raise SystemExit(f"--plots-only requires {per_run_csv}")
        metrics = _load_metrics_from_csv(per_run_csv)
        summary = _summarize_runs(metrics)
        logging.info("Loaded %d runs from %s (plots only)", len(metrics), per_run_csv)
        _plot_baseline_per_type(summary, out_dir / "fig_baseline_per_type.png")
        _plot_gates_by_type(
            metrics,
            out_dir / "fig_gate_by_type.png",
            lr_tag=args.lr_tag or "lr001",
        )
        logging.info("Wrote figures to %s", out_dir)
        return

    run_refs, discovery = discover_run_refs(results_root, tracks)
    for row in discovery:
        logging.info(
            "Discovery track=%s dirs=%d accepted=%d "
            "(no_config=%d no_ckpt=%d empty_ckpt=%d bad_name=%d)",
            row.track,
            row.n_dirs,
            row.n_accepted,
            row.n_no_config,
            row.n_no_ckpt_dir,
            row.n_empty_ckpt,
            row.n_bad_name,
        )
    if args.lr_tag:
        run_refs = [r for r in run_refs if r.lr_tag == args.lr_tag]
    if not run_refs:
        raise SystemExit(f"No runs found under {results_root}")

    track_counts = Counter(ref.track for ref in run_refs)
    logging.info("Run directories with checkpoints: %s", dict(track_counts))

    metrics: list[RunMetrics] = []
    failed: list[str] = []
    for ref in run_refs:
        logging.info("Evaluating %s / %s", ref.track, ref.run_dir.name)
        try:
            metrics.append(evaluate_run(ref, args.dataset_dir, device))
        except Exception:
            logging.exception("Failed on %s", ref.run_dir)
            failed.append(f"{ref.track}/{ref.run_dir.name}")

    if failed:
        logging.warning(
            "Skipped %d / %d runs: %s",
            len(failed),
            len(run_refs),
            ", ".join(failed[:8]) + (" ..." if len(failed) > 8 else ""),
        )

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
    _plot_gates_by_type(
        metrics,
        out_dir / "fig_gate_by_type.png",
        lr_tag=args.lr_tag or "lr001",
    )

    logging.info("Wrote analysis to %s", out_dir)
    print(f"\nAnalysis saved to: {out_dir.resolve()}")
    print(f"  - {out_dir / 'per_run_metrics.csv'}")
    print(f"  - {out_dir / 'summary_by_model.csv'}")
    print(f"  - {out_dir / 'fig_baseline_per_type.png'}")
    print(f"  - {out_dir / 'fig_gate_by_type.png'}")


if __name__ == "__main__":
    main()
