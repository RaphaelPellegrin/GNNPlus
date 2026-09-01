#!/usr/bin/env python3
"""Head-masking ablation for gated SiGMA on GCN/GIN routing (eval only).

At test time, zero out one MP head's contribution before fusion:
  - ``none``: both heads active (baseline)
  - ``mask_gin``: force GIN / ROUTING_SUM head off
  - ``mask_gcn``: force GCN / ROUTING_NORMGCN head off

Expectation (routing specialization):
  - On τ=0 graphs, masking GCN hurts more than masking GIN
  - On τ=1 graphs, masking GIN hurts more than masking GCN

Outputs (under ``--out-dir``):
  - ``mask_ablation_per_run.csv``
  - ``mask_ablation_summary.csv``
  - ``fig_mask_ablation.png`` / ``.pdf``
  - ``paper_figures/fig06_mask_ablation.png`` (copy for paper set)
  - ``paper_figures/fig06_mask_ablation_table.png`` / ``.pdf``

Example (cluster):
  python scripts/synthetic/eval_gcn_gin_routing_masks.py \\
    --results-root /n/netscratch/.../gcn_gin_routing \\
    --dataset-dir /n/netscratch/.../gnnplus_datasets \\
    --out-dir results/gcn_gin_routing/analysis
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PLOT_ONLY = "--plot-only" in sys.argv

if not _PLOT_ONLY:
    from torch_geometric.data import Batch
    from torch_geometric.graphgym.checkpoint import load_ckpt
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.loss import compute_loss
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.device import auto_select_device
    from torch_geometric import seed_everything

    import GNNPlus  # noqa: F401

    from GNNPlus.gcn_gin_routing_gate_tracking import hybrid_head_indices
    from GNNPlus.hybrid_gate_tracking import _unwrap_model
    from scripts.synthetic.analyze_gcn_gin_routing_results import (  # noqa: E402
        RunRef,
        _load_cfg_for_run,
        _pick_best_epoch,
        _pred_labels_from_score,
        discover_run_refs,
        iter_run_refs,
    )

MaskMode = Literal["none", "mask_gin", "mask_gcn"]
MASK_MODES: tuple[MaskMode, ...] = ("none", "mask_gin", "mask_gcn")
MASK_LABELS: dict[MaskMode, str] = {
    "none": "Both heads",
    "mask_gin": "Mask GIN head",
    "mask_gcn": "Mask GCN head",
}
# Zoom y-axis so masked-head drops (e.g. τ=1 mask GIN → ~0.44) are visible.
MASK_ABLATION_YMIN: float = 0.3
MASK_ABLATION_YMAX: float = 1.02
TRACK_ORDER: tuple[str, ...] = ("toy", "sigma")
TRACK_LABELS: dict[str, str] = {
    "toy": r"Track A (Toy, $d_h{=}1$)",
    "sigma": r"Track B (SiGMA, PyG GIN/GCN, $d_h{=}4$)",
}
METRIC_PALETTE: dict[str, str] = {
    "acc_all": "#E45756",
    "acc_tau0": "#4C72B0",
    "acc_tau1": "#DD8452",
}


@dataclass(frozen=True)
class MaskEvalRow:
    """Test accuracy under one mask setting for one run."""

    track: str
    model: str
    lr_tag: str
    seed: int
    mask_mode: MaskMode
    run_dir: str
    epoch: int
    n_all: int
    n_tau0: int
    n_tau1: int
    acc_all: float
    acc_tau0: float
    acc_tau1: float


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
        default=os.environ.get("GNNPLUS_DATASET_DIR", "results/gcn_gin_routing/data"),
        help="Parent of GcnGinRouting/.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gcn_gin_routing/analysis",
        help="Directory for CSV + figures.",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="toy,sigma",
        help="Comma-separated tracks.",
    )
    parser.add_argument(
        "--lr-tag",
        type=str,
        default="lr001",
        help="Only evaluate runs with this lr tag.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="a0g2_gated",
        help="Model slug to evaluate (default: gated SiGMA).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing mask_ablation_summary.csv (no GPU eval).",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=MASK_ABLATION_YMIN,
        help="Lower y-axis limit for mask ablation bar chart.",
    )
    return parser.parse_args(argv)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mp_head_mask(
    mask_mode: MaskMode,
    *,
    gin_idx: int,
    gcn_idx: int,
    num_heads: int,
) -> list[bool]:
    """Build per-head active mask for eval ablation."""
    active = [True] * num_heads
    if mask_mode == "mask_gin":
        active[gin_idx] = False
    elif mask_mode == "mask_gcn":
        active[gcn_idx] = False
    if len(active) != num_heads:
        raise ValueError(f"Expected {num_heads} MP heads, got mask len {len(active)}")
    return active


@torch.no_grad()
def _forward_masked(
    core: torch.nn.Module,
    batch: Batch,
    mp_head_mask: Optional[Sequence[bool]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run hybrid core through layers with optional MP head masking."""
    (
        x,
        batch_enc,
        edge_index_attn,
        edge_attr_attn,
        edge_index_mp,
        edge_attr_mp,
        _ei,
        _ea,
    ) = core._encode_batch(batch)

    for layer_i, layer in enumerate(core.layers):
        mask = mp_head_mask if layer_i == 0 else None
        layer_out = layer(
            x,
            edge_index_mp,
            batch_enc.batch,
            edge_attr_mp,
            edge_index_attn=edge_index_attn,
            edge_attr_attn=edge_attr_attn,
            edge_index_mp=edge_index_mp,
            edge_attr_mp=edge_attr_mp,
            mp_head_mask=mask,
        )
        x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        if core.ffn_blocks is not None:
            x = core.ffn_blocks[layer_i](x)
        batch_enc.x = x

    pred, true = core.post_mp(batch_enc)
    return pred, true


@torch.no_grad()
def evaluate_masked_run(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
    mask_mode: MaskMode,
) -> MaskEvalRow:
    """Load checkpoint and evaluate test accuracy under one mask mode."""
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

    hybrid = getattr(cfg.gnn, "hybrid", None)
    gnn_types = str(getattr(hybrid, "gnn_types", "")) if hybrid is not None else ""
    gin_idx, gcn_idx, two_head = hybrid_head_indices(gnn_types)
    if not two_head:
        raise ValueError(f"Mask ablation requires two MP heads; got {gnn_types!r}")
    num_heads = int(getattr(hybrid, "num_gnn_heads", 2))
    head_mask = _mp_head_mask(
        mask_mode,
        gin_idx=gin_idx,
        gcn_idx=gcn_idx,
        num_heads=num_heads,
    )

    correct_all = 0
    correct_t0 = 0
    correct_t1 = 0
    n_all = 0
    n_t0 = 0
    n_t1 = 0

    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing tau.")
        tau = batch.tau.view(-1).long()

        pred, true = _forward_masked(core, batch, head_mask)
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

    return MaskEvalRow(
        track=run_ref.track,
        model=run_ref.model,
        lr_tag=run_ref.lr_tag,
        seed=run_ref.seed,
        mask_mode=mask_mode,
        run_dir=str(run_ref.run_dir),
        epoch=epoch,
        n_all=n_all,
        n_tau0=n_t0,
        n_tau1=n_t1,
        acc_all=_acc(correct_all, n_all),
        acc_tau0=_acc(correct_t0, n_t0),
        acc_tau1=_acc(correct_t1, n_t1),
    )


def _write_per_run_csv(rows: Sequence[MaskEvalRow], path: Path) -> None:
    """Write per-run mask evaluation CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "model",
        "lr_tag",
        "seed",
        "mask_mode",
        "run_dir",
        "epoch",
        "n_all",
        "n_tau0",
        "n_tau1",
        "acc_all",
        "acc_tau0",
        "acc_tau1",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "track": r.track,
                    "model": r.model,
                    "lr_tag": r.lr_tag,
                    "seed": r.seed,
                    "mask_mode": r.mask_mode,
                    "run_dir": r.run_dir,
                    "epoch": r.epoch,
                    "n_all": r.n_all,
                    "n_tau0": r.n_tau0,
                    "n_tau1": r.n_tau1,
                    "acc_all": r.acc_all,
                    "acc_tau0": r.acc_tau0,
                    "acc_tau1": r.acc_tau1,
                },
            )


def _summarize(rows: Sequence[MaskEvalRow]) -> list[dict[str, object]]:
    """Mean ± std per (track, mask_mode)."""
    keys = sorted({(r.track, r.mask_mode) for r in rows})
    summary: list[dict[str, object]] = []
    for track, mask_mode in keys:
        subset = [r for r in rows if r.track == track and r.mask_mode == mask_mode]
        if not subset:
            continue
        for metric in ("acc_all", "acc_tau0", "acc_tau1"):
            vals = [getattr(r, metric) for r in subset]
            summary.append(
                {
                    "track": track,
                    "mask_mode": mask_mode,
                    "metric": metric,
                    "n_seeds": len(subset),
                    "mean": float(mean(vals)),
                    "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
                },
            )
    return summary


def _write_summary_csv(summary: Sequence[dict[str, object]], path: Path) -> None:
    """Write aggregated summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track", "mask_mode", "metric", "n_seeds", "mean", "std"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def _summary_lookup(
    summary: Sequence[dict[str, object]],
) -> dict[tuple[str, MaskMode, str], float]:
    """Map (track, mask, metric) → mean."""
    means, _ = _summary_mean_std_lookups(summary)
    return means


def _summary_mean_std_lookups(
    summary: Sequence[dict[str, object]],
) -> tuple[
    dict[tuple[str, MaskMode, str], float],
    dict[tuple[str, MaskMode, str], float],
]:
    """Map (track, mask, metric) → (mean, seed std)."""
    means: dict[tuple[str, MaskMode, str], float] = {}
    stds: dict[tuple[str, MaskMode, str], float] = {}
    for row in summary:
        key = (str(row["track"]), row["mask_mode"], str(row["metric"]))
        means[key] = float(row["mean"])  # type: ignore[index]
        stds[key] = float(row["std"])  # type: ignore[index]
    return means, stds


def _minmax_lookup_from_per_run(
    rows: Sequence[MaskEvalRow],
) -> tuple[
    dict[tuple[str, MaskMode, str], float],
    dict[tuple[str, MaskMode, str], float],
    dict[tuple[str, MaskMode, str], float],
]:
    """Map (track, mask, metric) → (mean, lower_err, upper_err) over seeds."""
    grouped: dict[tuple[str, MaskMode, str], list[float]] = {}
    for row in rows:
        for metric in ("acc_all", "acc_tau0", "acc_tau1"):
            key = (row.track, row.mask_mode, metric)
            grouped.setdefault(key, []).append(float(getattr(row, metric)))
    means: dict[tuple[str, MaskMode, str], float] = {}
    lowers: dict[tuple[str, MaskMode, str], float] = {}
    uppers: dict[tuple[str, MaskMode, str], float] = {}
    for key, vals in grouped.items():
        m = float(mean(vals))
        lo = float(min(vals))
        hi = float(max(vals))
        means[key] = m
        lowers[key] = m - lo
        uppers[key] = hi - m
    return means, lowers, uppers


def _load_mask_per_run_csv(path: Path) -> list[MaskEvalRow]:
    """Load ``mask_ablation_per_run.csv``."""
    rows: list[MaskEvalRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                MaskEvalRow(
                    track=str(raw["track"]),
                    model=str(raw["model"]),
                    lr_tag=str(raw["lr_tag"]),
                    seed=int(raw["seed"]),
                    mask_mode=raw["mask_mode"],  # type: ignore[arg-type]
                    run_dir=str(raw["run_dir"]),
                    epoch=int(raw["epoch"]),
                    n_all=int(raw["n_all"]),
                    n_tau0=int(raw["n_tau0"]),
                    n_tau1=int(raw["n_tau1"]),
                    acc_all=float(raw["acc_all"]),
                    acc_tau0=float(raw["acc_tau0"]),
                    acc_tau1=float(raw["acc_tau1"]),
                ),
            )
    return rows


def _format_lr_tag(lr_tag: str) -> str:
    """Format ``lr001`` as ``$\\mathrm{LR}=10^{-3}$`` for figure titles."""
    if lr_tag.startswith("lr") and lr_tag[2:].isdigit():
        exponent = -len(lr_tag[2:])
        return rf"$\mathrm{{LR}}=10^{{{exponent}}}$"
    return rf"$\mathrm{{LR}}={lr_tag}$"


def _plot_mask_ablation(
    summary: Sequence[dict[str, object]],
    out_path: Path,
    *,
    per_run: Sequence[MaskEvalRow] | None = None,
    dpi: int,
    ymin: float = MASK_ABLATION_YMIN,
    ymax: float = MASK_ABLATION_YMAX,
    lr_tag: str = "lr001",
) -> None:
    """Grouped bars: all + per-type accuracy under each mask (fig01 style)."""
    import numpy as np

    if per_run:
        lookup, lower_lookup, upper_lookup = _minmax_lookup_from_per_run(per_run)
    else:
        lookup, std_lookup = _summary_mean_std_lookups(summary)
        lower_lookup = std_lookup
        upper_lookup = std_lookup

    summary_tracks = {str(r["track"]) for r in summary}
    tracks = [t for t in TRACK_ORDER if t in summary_tracks]
    tracks.extend(sorted(summary_tracks - set(tracks)))

    metric_specs = (
        ("acc_all", "All graphs"),
        ("acc_tau0", r"$\tau{=}0$ (GCN-type)"),
        ("acc_tau1", r"$\tau{=}1$ (GIN-type)"),
    )
    bar_w = 0.24
    offsets = (-bar_w, 0.0, bar_w)

    fig, axes = plt.subplots(1, len(tracks), figsize=(6.8 * len(tracks), 5.0), squeeze=False)
    legend_handles: list = []
    legend_labels: list[str] = []

    for ax_idx, (ax, track) in enumerate(zip(axes[0], tracks, strict=True)):
        x = np.arange(len(MASK_MODES))
        for offset, (metric, label) in zip(offsets, metric_specs, strict=True):
            vals = [
                lookup.get((track, mode, metric), float("nan"))
                for mode in MASK_MODES
            ]
            lo_errs = [
                lower_lookup.get((track, mode, metric), 0.0)
                for mode in MASK_MODES
            ]
            hi_errs = [
                upper_lookup.get((track, mode, metric), 0.0)
                for mode in MASK_MODES
            ]
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_w,
                yerr=[lo_errs, hi_errs],
                capsize=3,
                label=label,
                color=METRIC_PALETTE[metric],
                edgecolor="white",
                linewidth=0.6,
                error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
            )
            if ax_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)

        ax.set_xticks(x)
        ax.set_xticklabels([MASK_LABELS[m] for m in MASK_MODES], rotation=12, ha="right")
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Test accuracy")
        ax.set_title(TRACK_LABELS.get(track, track))
        ax.grid(axis="y", alpha=0.22)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        f"MP head masking at eval (5-seed mean, min–max whiskers, {_format_lr_tag(lr_tag)})",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _print_asymmetry_report(summary: Sequence[dict[str, object]]) -> None:
    """Print mask-asymmetry metrics from summary means."""
    lookup = _summary_lookup(summary)
    print("\n=== Mask asymmetry (mean acc drop from baseline) ===\n")
    for track in sorted({str(r["track"]) for r in summary}):
        print(f"Track: {track}")
        for tau, metric in ((0, "acc_tau0"), (1, "acc_tau1")):
            base = lookup.get((track, "none", metric), float("nan"))
            drop_gcn = base - lookup.get((track, "mask_gcn", metric), float("nan"))
            drop_gin = base - lookup.get((track, "mask_gin", metric), float("nan"))
            if tau == 0:
                asym = drop_gcn - drop_gin
                expect = "mask GCN should hurt more on τ=0"
            else:
                asym = drop_gin - drop_gcn
                expect = "mask GIN should hurt more on τ=1"
            print(
                f"  τ={tau}: baseline={100*base:.1f}% | "
                f"Δ(mask GCN)={100*drop_gcn:.1f}pp | "
                f"Δ(mask GIN)={100*drop_gin:.1f}pp | "
                f"asymmetry={100*asym:+.1f}pp ({expect})",
            )
        print()


def _load_mask_summary_csv(path: Path) -> list[dict[str, object]]:
    """Load ``mask_ablation_summary.csv``."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                {
                    "track": raw["track"],
                    "mask_mode": raw["mask_mode"],
                    "metric": raw["metric"],
                    "n_seeds": int(raw["n_seeds"]),
                    "mean": float(raw["mean"]),
                    "std": float(raw["std"]),
                },
            )
    return rows


def plot_mask_ablation_from_summary_csv(
    summary_path: Path,
    out_dir: Path,
    *,
    dpi: int,
    ymin: float = MASK_ABLATION_YMIN,
    model: str = "a0g2_gated",
    lr_tag: str = "lr001",
    tracks: Sequence[str] = ("toy", "sigma"),
) -> None:
    """Regenerate mask ablation bar chart + table from an existing summary CSV."""
    summary = _load_mask_summary_csv(summary_path)
    per_run_path = summary_path.parent / "mask_ablation_per_run.csv"
    per_run = _load_mask_per_run_csv(per_run_path) if per_run_path.is_file() else None
    fig_path = out_dir / "fig_mask_ablation.png"
    paper_fig_path = out_dir / "paper_figures" / "fig06_mask_ablation.png"

    _plot_mask_ablation(summary, fig_path, per_run=per_run, dpi=dpi, ymin=ymin, lr_tag=lr_tag)
    _plot_mask_ablation(summary, paper_fig_path, per_run=per_run, dpi=dpi, ymin=ymin, lr_tag=lr_tag)
    _print_asymmetry_report(summary)

    from scripts.synthetic.gcn_gin_routing_table_figures import (  # noqa: WPS433
        plot_mask_ablation_table,
    )

    table_path = out_dir / "paper_figures" / "fig06_mask_ablation_table.png"
    track_list = [t.strip() for t in tracks if t.strip()]
    for track in track_list:
        track_table = table_path if len(track_list) == 1 else table_path.with_name(
            f"fig06_mask_ablation_table_{track}.png",
        )
        plot_mask_ablation_table(
            summary_path,
            track_table,
            track=track,
            model_label=model,
            lr_tag=lr_tag,
            dpi=dpi,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run mask ablation for all gated runs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    if args.plot_only:
        summary_path = out_dir / "mask_ablation_summary.csv"
        if not summary_path.is_file():
            raise SystemExit(f"Missing {summary_path} (run full eval first)")
        plot_mask_ablation_from_summary_csv(
            summary_path,
            out_dir,
            dpi=args.dpi,
            ymin=args.ymin,
            model=args.model,
            lr_tag=args.lr_tag,
            tracks=tracks,
        )
        print(f"Wrote {out_dir / 'fig_mask_ablation.png'}")
        print(f"Wrote {out_dir / 'paper_figures/fig06_mask_ablation.png'}")
        return

    results_root = Path(args.results_root)
    device = _select_device(args.device)

    refs = [
        r
        for r in iter_run_refs(results_root, tracks)
        if r.model == args.model and r.lr_tag == args.lr_tag
    ]
    if not refs:
        _, stats = discover_run_refs(results_root, tracks)
        logging.error("No runs for model=%s lr=%s", args.model, args.lr_tag)
        for st in stats:
            logging.error(
                "  track=%s dirs=%d accepted=%d",
                st.track,
                st.n_dirs,
                st.n_accepted,
            )
        raise SystemExit(1)

    rows: list[MaskEvalRow] = []
    for run_ref in refs:
        for mask_mode in MASK_MODES:
            logging.info(
                "Eval %s seed=%d mask=%s",
                run_ref.run_dir.name,
                run_ref.seed,
                mask_mode,
            )
            rows.append(
                evaluate_masked_run(run_ref, args.dataset_dir, device, mask_mode),
            )

    summary = _summarize(rows)
    per_run_path = out_dir / "mask_ablation_per_run.csv"
    summary_path = out_dir / "mask_ablation_summary.csv"
    fig_path = out_dir / "fig_mask_ablation.png"
    paper_fig_path = out_dir / "paper_figures" / "fig06_mask_ablation.png"

    _write_per_run_csv(rows, per_run_path)
    _write_summary_csv(summary, summary_path)
    _plot_mask_ablation(summary, fig_path, per_run=rows, dpi=args.dpi, ymin=args.ymin, lr_tag=args.lr_tag)
    _plot_mask_ablation(summary, paper_fig_path, per_run=rows, dpi=args.dpi, ymin=args.ymin, lr_tag=args.lr_tag)
    _print_asymmetry_report(summary)

    from scripts.synthetic.gcn_gin_routing_table_figures import (  # noqa: WPS433
        plot_mask_ablation_table,
    )

    table_path = out_dir / "paper_figures" / "fig06_mask_ablation_table.png"
    for track in tracks:
        track_table = table_path if len(tracks) == 1 else table_path.with_name(
            f"fig06_mask_ablation_table_{track}.png",
        )
        plot_mask_ablation_table(
            summary_path,
            track_table,
            track=track,
            model_label=args.model,
            lr_tag=args.lr_tag,
            dpi=args.dpi,
        )
        print(f"Wrote {track_table}")

    print(f"Wrote {per_run_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {paper_fig_path}")


if __name__ == "__main__":
    main()
