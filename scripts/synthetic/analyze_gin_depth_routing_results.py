#!/usr/bin/env python3
"""Aggregate per-τ test accuracy and per-layer root gates for GIN depth-routing.

Loads best checkpoints under ``<results_root>/toy/``, evaluates on the test
split, and reports accuracy for ``tau=0`` (1-GIN / shallow) vs ``tau=1``
(2-GIN / deep). For gated models, also collects root MP gates at each layer.

Outputs (under ``--out-dir``):
  - ``per_run_metrics.csv``
  - ``summary_by_model.csv``
  - ``fig_baseline_per_type.png`` / ``.pdf``
  - ``fig_gate_by_layer_tau.png`` / ``.pdf`` (gated only)

Example::

  python scripts/synthetic/analyze_gin_depth_routing_results.py \\
    --results-root $GNNPLUS_OUT_DIR/gin_routing_depth \\
    --dataset-dir $GNNPLUS_DATASET_DIR \\
    --out-dir results/gin_routing_depth/analysis
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

import GNNPlus  # noqa: F401
from GNNPlus.gin_depth_routing_gate_tracking import (
    DepthGateAccumulator,
    accumulate_depth_root_gates_from_batch,
)
from GNNPlus.hybrid_gate_tracking import _unwrap_model

RUN_NAME_RE = re.compile(
    r"^(?P<model>.+)_lr(?P<lr_tag>\d+)_seed(?P<seed>\d+)$",
)

MODEL_ORDER: tuple[str, ...] = (
    "l2_a0g1_gated",
    "l2_a0g1_ungated",
)
MODEL_LABELS: dict[str, str] = {
    "l2_a0g1_gated": "SiGMA gated (L=2)",
    "l2_a0g1_ungated": "SiGMA ungated (L=2)",
}


def _default_results_root() -> str:
    """Resolve default results root from env or local path."""
    if "GNNPLUS_OUT_DIR" in os.environ:
        return f"{os.environ['GNNPLUS_OUT_DIR'].rstrip('/')}/gin_routing_depth"
    return "results/gin_routing_depth"


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
    has_gates: bool
    # Flattened layer gates: layer0_tau0, layer0_tau1, layer1_tau0, ...
    gate_layer0_tau0: float = float("nan")
    gate_layer0_tau1: float = float("nan")
    gate_layer1_tau0: float = float("nan")
    gate_layer1_tau1: float = float("nan")
    delta_layer0: float = float("nan")
    delta_layer1: float = float("nan")


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default=_default_results_root(),
        help="Parent of toy/ run folders.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get(
            "GNNPLUS_DATASET_DIR",
            "results/gin_routing_depth/data",
        ),
        help="Parent of GinDepthRouting/.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gin_routing_depth/analysis",
    )
    parser.add_argument("--tracks", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing per_run_metrics.csv.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def _resolve_run_config(run_dir: Path) -> Optional[Path]:
    """Return config yaml for a run dir."""
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


def discover_run_refs(
    results_root: Path,
    tracks: Sequence[str],
) -> list[RunRef]:
    """Find run directories with config + checkpoint files."""
    refs: list[RunRef] = []
    for track in tracks:
        track_dir = results_root / track
        if not track_dir.is_dir():
            logging.warning("Missing track directory: %s", track_dir)
            continue
        for run_dir in sorted(track_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if _resolve_run_config(run_dir) is None:
                continue
            ckpt_dir = run_dir / "ckpt"
            if not ckpt_dir.is_dir() or not any(ckpt_dir.glob("*.ckpt")):
                continue
            ref = _parse_run_ref(track, run_dir)
            if ref is not None:
                refs.append(ref)
    return refs


def iter_run_refs(results_root: Path, tracks: Sequence[str]) -> Iterator[RunRef]:
    """Yield run directories that contain config + ``ckpt/*.ckpt``."""
    yield from discover_run_refs(results_root, tracks)


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
    """Convert model scores to integer labels."""
    if pred_score.ndim == 1 or (pred_score.ndim == 2 and pred_score.shape[1] == 1):
        thresh = float(getattr(cfg.model, "thresh", 0.5))
        return (pred_score.view(-1) > thresh).long()
    return pred_score.argmax(dim=-1).view(-1)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_run(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
) -> RunMetrics:
    """Load checkpoint and compute per-τ test metrics (+ per-layer root gates)."""
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

    hybrid = getattr(cfg.gnn, "hybrid", None)
    gate_mode = str(getattr(hybrid, "gate", "none")).lower() if hybrid else "none"
    collect_gates = gate_mode not in ("none", "off")
    core = _unwrap_model(model)
    gate_accum = DepthGateAccumulator()

    correct_all = correct_t0 = correct_t1 = 0
    n_all = n_t0 = n_t1 = 0

    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing tau.")
        tau = batch.tau.view(-1).long()
        if collect_gates:
            accumulate_depth_root_gates_from_batch(
                core, batch, tau, gate_accum, head_idx=0
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

    def _acc(num: int, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    def _layer_means(layer_idx: int) -> tuple[float, float, float]:
        from statistics import mean as _mean

        t0 = gate_accum.by_layer_tau0.get(layer_idx, [])
        t1 = gate_accum.by_layer_tau1.get(layer_idx, [])
        m0 = float(_mean(t0)) if t0 else float("nan")
        m1 = float(_mean(t1)) if t1 else float("nan")
        return m0, m1, m1 - m0

    l0_t0, l0_t1, d0 = _layer_means(0)
    l1_t0, l1_t1, d1 = _layer_means(1)

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
        acc_all=_acc(correct_all, n_all),
        acc_tau0=_acc(correct_t0, n_t0),
        acc_tau1=_acc(correct_t1, n_t1),
        has_gates=collect_gates and gate_accum.has_samples,
        gate_layer0_tau0=l0_t0,
        gate_layer0_tau1=l0_t1,
        gate_layer1_tau0=l1_t0,
        gate_layer1_tau1=l1_t1,
        delta_layer0=d0,
        delta_layer1=d1,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: Sequence[RunMetrics]) -> list[dict[str, Any]]:
    """Mean/std over seeds by track/model/lr."""
    groups: dict[tuple[str, str, str], list[RunMetrics]] = {}
    for row in rows:
        groups.setdefault((row.track, row.model, row.lr_tag), []).append(row)
    out: list[dict[str, Any]] = []
    for (track, model, lr_tag), items in sorted(groups.items()):
        def agg(attr: str) -> tuple[float, float]:
            vals = [float(getattr(it, attr)) for it in items]
            if len(vals) == 1:
                return vals[0], 0.0
            return float(mean(vals)), float(pstdev(vals))

        row: dict[str, Any] = {
            "track": track,
            "model": model,
            "lr_tag": lr_tag,
            "n_seeds": len(items),
        }
        for key in (
            "acc_all",
            "acc_tau0",
            "acc_tau1",
            "gate_layer0_tau0",
            "gate_layer0_tau1",
            "gate_layer1_tau0",
            "gate_layer1_tau1",
            "delta_layer0",
            "delta_layer1",
        ):
            m, s = agg(key)
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = s
        out.append(row)
    return out


def _plot_acc(summary: Sequence[dict[str, Any]], out_path: Path, dpi: int) -> None:
    """Bar chart: test accuracy by τ for each model."""
    # Prefer lr001 if present
    lr_tags = sorted({str(r["lr_tag"]) for r in summary})
    preferred = "lr001" if "lr001" in lr_tags else lr_tags[0]
    subset = [r for r in summary if r["lr_tag"] == preferred]
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in subset)]
    by_model = {str(r["model"]): r for r in subset}
    x = list(range(len(models)))
    bar_w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(
        [xi - bar_w / 2 for xi in x],
        [float(by_model[m]["acc_tau0_mean"]) for m in models],
        width=bar_w,
        yerr=[float(by_model[m]["acc_tau0_std"]) for m in models],
        capsize=3,
        label=r"$\tau=0$ (1-GIN / shallow)",
        color="#4C72B0",
    )
    ax.bar(
        [xi + bar_w / 2 for xi in x],
        [float(by_model[m]["acc_tau1_mean"]) for m in models],
        width=bar_w,
        yerr=[float(by_model[m]["acc_tau1_std"]) for m in models],
        capsize=3,
        label=r"$\tau=1$ (2-GIN / deep)",
        color="#DD8452",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=10, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"GIN depth-routing · per-τ accuracy ({preferred})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_gates(rows: Sequence[RunMetrics], out_path: Path, dpi: int, lr_tag: str) -> None:
    """Root gate γ by layer × τ for gated runs."""
    gated = [
        r
        for r in rows
        if r.model == "l2_a0g1_gated" and r.has_gates and (not lr_tag or r.lr_tag == lr_tag)
    ]
    if not gated:
        logging.warning("No gated gate stats — skipping gate figure.")
        return
    tags = sorted({r.lr_tag for r in gated})
    use_lr = "lr001" if "lr001" in tags else tags[0]
    gated = [r for r in gated if r.lr_tag == use_lr]

    labels = [
        r"L0 $\tau{=}0$",
        r"L0 $\tau{=}1$",
        r"L1 $\tau{=}0$",
        r"L1 $\tau{=}1$",
    ]
    attrs = [
        "gate_layer0_tau0",
        "gate_layer0_tau1",
        "gate_layer1_tau0",
        "gate_layer1_tau1",
    ]
    means = [float(mean([getattr(r, a) for r in gated])) for a in attrs]
    stds = [
        float(pstdev([getattr(r, a) for r in gated])) if len(gated) > 1 else 0.0
        for a in attrs
    ]
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#55A868"]
    alphas = [1.0, 0.55, 1.0, 0.55]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = list(range(len(labels)))
    for i, (lab, m, s, c, a) in enumerate(zip(labels, means, stds, colors, alphas)):
        ax.bar(i, m, yerr=s, capsize=3, color=c, alpha=a, label=lab if i in (0, 2) else None)
    # legend only L0/L1 colors
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#4C72B0", label="Layer 0"),
            Patch(facecolor="#55A868", label="Layer 1"),
        ],
        loc="best",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"Root MP gate $\gamma$")
    ax.set_title(
        rf"Depth routing · root gates by layer$\times\tau$ "
        rf"(gated, $n={len(gated)}$, {use_lr})"
    )
    ax.grid(axis="y", alpha=0.25)
    # annotate deltas
    d0 = means[1] - means[0]
    d1 = means[3] - means[2]
    ax.text(
        0.02,
        0.98,
        rf"$\Delta$L0$=\gamma_{{\tau1}}-\gamma_{{\tau0}}={d0:+.3f}$"
        "\n"
        rf"$\Delta$L1$={d1:+.3f}$ (want $>0$ for depth routing)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_metrics_csv(path: Path) -> list[RunMetrics]:
    """Load prior per_run_metrics.csv."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[RunMetrics] = []
        for row in reader:
            rows.append(
                RunMetrics(
                    track=row["track"],
                    model=row["model"],
                    lr_tag=row["lr_tag"],
                    seed=int(row["seed"]),
                    run_dir=row["run_dir"],
                    epoch=int(row["epoch"]),
                    n_all=int(row["n_all"]),
                    n_tau0=int(row["n_tau0"]),
                    n_tau1=int(row["n_tau1"]),
                    acc_all=float(row["acc_all"]),
                    acc_tau0=float(row["acc_tau0"]),
                    acc_tau1=float(row["acc_tau1"]),
                    has_gates=row["has_gates"].strip().lower() in ("1", "true", "yes"),
                    gate_layer0_tau0=float(row.get("gate_layer0_tau0", "nan")),
                    gate_layer0_tau1=float(row.get("gate_layer0_tau1", "nan")),
                    gate_layer1_tau0=float(row.get("gate_layer1_tau0", "nan")),
                    gate_layer1_tau1=float(row.get("gate_layer1_tau1", "nan")),
                    delta_layer0=float(row.get("delta_layer0", "nan")),
                    delta_layer1=float(row.get("delta_layer1", "nan")),
                )
            )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_cli(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "per_run_metrics.csv"

    if args.plots_only:
        rows = _load_metrics_csv(metrics_path)
    else:
        results_root = Path(args.results_root)
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
        refs = discover_run_refs(results_root, tracks)
        if args.lr_tag:
            refs = [r for r in refs if r.lr_tag == args.lr_tag]
        if not refs:
            raise SystemExit(f"No runs found under {results_root}/{tracks}")
        device = _select_device(args.device)
        rows = []
        for ref in refs:
            logging.info("Evaluating %s", ref.run_dir)
            rows.append(evaluate_run(ref, args.dataset_dir, device))

        fieldnames = list(asdict(rows[0]).keys())
        _write_csv(metrics_path, [asdict(r) for r in rows], fieldnames)

    summary = _summarize(rows)
    summary_path = out_dir / "summary_by_model.csv"
    _write_csv(summary_path, summary, list(summary[0].keys()) if summary else [])

    _plot_acc(summary, out_dir / "fig_baseline_per_type.png", args.dpi)
    paper = out_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    _plot_acc(summary, paper / "fig_acc_by_tau.png", args.dpi)
    _plot_gates(rows, out_dir / "fig_gate_by_layer_tau.png", args.dpi, args.lr_tag)
    _plot_gates(rows, paper / "fig_gate_by_layer_tau.png", args.dpi, args.lr_tag)

    print(f"Wrote {metrics_path} ({len(rows)} runs)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {out_dir / 'fig_baseline_per_type.png'}")
    print(f"Wrote {out_dir / 'fig_gate_by_layer_tau.png'}")


if __name__ == "__main__":
    main()
