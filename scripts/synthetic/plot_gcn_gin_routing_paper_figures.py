#!/usr/bin/env python3
"""Publication-style figures for GCN/GIN routing benchmark (from analysis CSVs).

Reads outputs of ``analyze_gcn_gin_routing_results.py`` and optional
``inspect_gcn_gin_routing_node_gates.py`` exports, then writes figures under
``<out-dir>/paper_figures/``.

Example (after scp from cluster):
  python scripts/synthetic/plot_gcn_gin_routing_paper_figures.py \\
    --analysis-dir results/gcn_gin_routing/analysis
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
TRACK_LABELS: dict[str, str] = {
    "toy": "Toy (routing convs, $d_h{=}1$)",
    "sigma": "Sigma (PyG GIN/GCN, $d_h{=}4$)",
}
PALETTE = {
    "tau0": "#4C72B0",
    "tau1": "#DD8452",
    "gin": "#55A868",
    "gcn": "#4C72B0",
}


@dataclass(frozen=True)
class SummaryRow:
    """One row from summary_by_model.csv."""

    track: str
    model: str
    lr_tag: str
    n_seeds: int
    acc_all_mean: float
    acc_all_std: float
    acc_tau0_mean: float
    acc_tau0_std: float
    acc_tau1_mean: float
    acc_tau1_std: float


@dataclass(frozen=True)
class PerRunRow:
    """One row from per_run_metrics.csv."""

    track: str
    model: str
    lr_tag: str
    seed: int
    acc_all: float
    acc_tau0: float
    acc_tau1: float
    gin_gate_tau0: float
    gin_gate_tau1: float
    gcn_gate_tau0: float
    gcn_gate_tau1: float
    has_gates: bool


@dataclass(frozen=True)
class GateNodeRow:
    """One row from gate_node_summary_*_test.csv."""

    run_dir: str
    track: str
    lr_tag: str
    seed: int
    delta_gin: float
    delta_gcn: float
    gin_root_tau0: float
    gin_root_tau1: float
    gcn_root_tau0: float
    gcn_root_tau1: float
    gin_nbr_tau0: float
    gin_nbr_tau1: float
    gcn_nbr_tau0: float
    gcn_nbr_tau1: float


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="results/gcn_gin_routing/analysis",
        help="Directory with per_run_metrics.csv and summary_by_model.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: <analysis-dir>/paper_figures).",
    )
    parser.add_argument(
        "--lr-tag",
        type=str,
        default="lr001",
        help="Learning-rate tag for per-seed gate plots (default: lr001).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI (default: 200).",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV as list of row dicts."""
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_summary(analysis_dir: Path) -> list[SummaryRow]:
    """Load summary_by_model.csv."""
    rows = _read_csv(analysis_dir / "summary_by_model.csv")
    return [
        SummaryRow(
            track=str(r["track"]),
            model=str(r["model"]),
            lr_tag=str(r["lr_tag"]),
            n_seeds=int(r["n_seeds"]),
            acc_all_mean=float(r["acc_all_mean"]),
            acc_all_std=float(r["acc_all_std"]),
            acc_tau0_mean=float(r["acc_tau0_mean"]),
            acc_tau0_std=float(r["acc_tau0_std"]),
            acc_tau1_mean=float(r["acc_tau1_mean"]),
            acc_tau1_std=float(r["acc_tau1_std"]),
        )
        for r in rows
    ]


def _load_per_run(analysis_dir: Path) -> list[PerRunRow]:
    """Load per_run_metrics.csv."""
    rows = _read_csv(analysis_dir / "per_run_metrics.csv")
    out: list[PerRunRow] = []
    for r in rows:
        out.append(
            PerRunRow(
                track=str(r["track"]),
                model=str(r["model"]),
                lr_tag=str(r["lr_tag"]),
                seed=int(r["seed"]),
                acc_all=float(r["acc_all"]),
                acc_tau0=float(r["acc_tau0"]),
                acc_tau1=float(r["acc_tau1"]),
                gin_gate_tau0=float(r["gin_gate_tau0"]),
                gin_gate_tau1=float(r["gin_gate_tau1"]),
                gcn_gate_tau0=float(r["gcn_gate_tau0"]),
                gcn_gate_tau1=float(r["gcn_gate_tau1"]),
                has_gates=str(r["has_gates"]).lower() in {"true", "1", "yes"},
            ),
        )
    return out


def _load_gate_node_csv(path: Path) -> list[GateNodeRow]:
    """Load gate_node_summary_*_test.csv if present."""
    if not path.is_file():
        return []
    rows = _read_csv(path)
    return [
        GateNodeRow(
            run_dir=str(r["run_dir"]),
            track=str(r.get("track", path.stem.split("_")[2] if "_" in path.stem else "")),
            lr_tag=str(r["lr_tag"]),
            seed=int(float(r["seed"])),
            delta_gin=float(r["delta_gin"]),
            delta_gcn=float(r["delta_gcn"]),
            gin_root_tau0=float(r["gin_root_tau0"]),
            gin_root_tau1=float(r["gin_root_tau1"]),
            gcn_root_tau0=float(r["gcn_root_tau0"]),
            gcn_root_tau1=float(r["gcn_root_tau1"]),
            gin_nbr_tau0=float(r["gin_nbr_tau0"]),
            gin_nbr_tau1=float(r["gin_nbr_tau1"]),
            gcn_nbr_tau0=float(r["gcn_nbr_tau0"]),
            gcn_nbr_tau1=float(r["gcn_nbr_tau1"]),
        )
        for r in rows
    ]


def _apply_style() -> None:
    """Set matplotlib rcParams for paper figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    """Save PNG and PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_per_type(
    summary: Sequence[SummaryRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
) -> None:
    """Grouped bar chart: per-type test accuracy (toy vs sigma)."""
    tracks = sorted({r.track for r in summary})
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.8 * len(tracks), 4.8), squeeze=False)
    bar_w = 0.36

    for ax, track in zip(axes[0], tracks, strict=True):
        subset = [r for r in summary if r.track == track and r.lr_tag == lr_tag]
        by_model = {r.model: r for r in subset}
        models = [m for m in MODEL_ORDER if m in by_model]
        x = np.arange(len(models))

        t0_vals = [by_model[m].acc_tau0_mean for m in models]
        t1_vals = [by_model[m].acc_tau1_mean for m in models]
        t0_err = [by_model[m].acc_tau0_std for m in models]
        t1_err = [by_model[m].acc_tau1_std for m in models]

        ax.bar(
            x - bar_w / 2,
            t0_vals,
            width=bar_w,
            yerr=t0_err,
            capsize=3,
            label=r"$\tau{=}0$ (GCN-type)",
            color=PALETTE["tau0"],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            x + bar_w / 2,
            t1_vals,
            width=bar_w,
            yerr=t1_err,
            capsize=3,
            label=r"$\tau{=}1$ (GIN-type)",
            color=PALETTE["tau1"],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=12, ha="right")
        ax.set_ylim(0.65, 1.02)
        ax.set_ylabel("Test accuracy")
        ax.set_title(TRACK_LABELS.get(track, track))
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y", alpha=0.22)

    axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.suptitle(
        f"Per-type test accuracy (5-seed mean ± std, {lr_tag})",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_gate_routing_delta(
    per_run: Sequence[PerRunRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
) -> None:
    """Per-seed routing contrast: Δγ at root for gated model."""
    gated = [
        r
        for r in per_run
        if r.model == "a0g2_gated" and r.has_gates and r.lr_tag == lr_tag
    ]
    if not gated:
        return

    tracks = sorted({r.track for r in gated})
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.2 * len(tracks), 4.6), squeeze=False)

    for ax, track in zip(axes[0], tracks, strict=True):
        subset = sorted([r for r in gated if r.track == track], key=lambda r: r.seed)
        seeds = [r.seed for r in subset]
        delta_gin = [r.gin_gate_tau1 - r.gin_gate_tau0 for r in subset]
        delta_gcn = [r.gcn_gate_tau0 - r.gcn_gate_tau1 for r in subset]

        x = np.arange(len(seeds))
        w = 0.35
        ax.bar(
            x - w / 2,
            delta_gin,
            width=w,
            label=r"$\Delta\gamma_{\mathrm{GIN}} = \bar{\gamma}_{\tau1} - \bar{\gamma}_{\tau0}$",
            color=PALETTE["gin"],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + w / 2,
            delta_gcn,
            width=w,
            label=r"$\Delta\gamma_{\mathrm{GCN}} = \bar{\gamma}_{\tau0} - \bar{\gamma}_{\tau1}$",
            color=PALETTE["gcn"],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(r"Root gate contrast")
        ax.set_title(TRACK_LABELS.get(track, track))
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="y", alpha=0.22)

    axes[0, 0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        f"SiGMA gated — per-seed root routing contrast ({lr_tag}, test)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_root_vs_neighbor_gates(
    gate_rows: Sequence[GateNodeRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
) -> None:
    """Root vs neighbor mean gates (gated, one LR)."""
    rows = [r for r in gate_rows if r.lr_tag == lr_tag]
    if not rows:
        return

    tracks = sorted({r.track for r in rows})
    fig, axes = plt.subplots(2, len(tracks), figsize=(5.0 * len(tracks), 7.0), squeeze=False)

    for col, track in enumerate(tracks):
        subset = [r for r in rows if r.track == track]
        gin_root_t0 = mean(r.gin_root_tau0 for r in subset)
        gin_root_t1 = mean(r.gin_root_tau1 for r in subset)
        gcn_root_t0 = mean(r.gcn_root_tau0 for r in subset)
        gcn_root_t1 = mean(r.gcn_root_tau1 for r in subset)
        gin_nbr_t0 = mean(r.gin_nbr_tau0 for r in subset)
        gin_nbr_t1 = mean(r.gin_nbr_tau1 for r in subset)
        gcn_nbr_t0 = mean(r.gcn_nbr_tau0 for r in subset)
        gcn_nbr_t1 = mean(r.gcn_nbr_tau1 for r in subset)

        for row_idx, (title, t0_gin, t1_gin, t0_gcn, t1_gcn) in enumerate(
            [
                ("Root node", gin_root_t0, gin_root_t1, gcn_root_t0, gcn_root_t1),
                ("Neighbors (mean)", gin_nbr_t0, gin_nbr_t1, gcn_nbr_t0, gcn_nbr_t1),
            ],
        ):
            ax = axes[row_idx, col]
            labels = [r"$\tau{=}0$ GIN", r"$\tau{=}1$ GIN", r"$\tau{=}0$ GCN", r"$\tau{=}1$ GCN"]
            vals = [t0_gin, t1_gin, t0_gcn, t1_gcn]
            colors = [
                to_rgba(PALETTE["gin"], 1.0),
                to_rgba(PALETTE["gin"], 0.55),
                to_rgba(PALETTE["gcn"], 1.0),
                to_rgba(PALETTE["gcn"], 0.55),
            ]
            ax.bar(range(4), vals, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel(r"Mean $\gamma$")
            if row_idx == 0:
                ax.set_title(TRACK_LABELS.get(track, track))
            ax.text(-0.12, 1.04, title, transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"Root vs neighbor gates (5-seed mean, {lr_tag}, test)",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def plot_gated_accuracy_scatter(
    per_run: Sequence[PerRunRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
) -> None:
    """Scatter: τ=0 vs τ=1 accuracy per seed (gated only)."""
    gated = [
        r
        for r in per_run
        if r.model == "a0g2_gated" and r.lr_tag == lr_tag
    ]
    if not gated:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), squeeze=False)
    markers = {"toy": "o", "sigma": "s"}

    for track in ("toy", "sigma"):
        ax = axes[0, 0 if track == "toy" else 1]
        subset = [r for r in gated if r.track == track]
        xs = [r.acc_tau0 for r in subset]
        ys = [r.acc_tau1 for r in subset]
        ax.scatter(xs, ys, s=70, marker=markers[track], label=track, edgecolors="black", linewidths=0.5)
        for r in subset:
            ax.annotate(str(r.seed), (r.acc_tau0, r.acc_tau1), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.plot([0.9, 1.0], [0.9, 1.0], "k--", alpha=0.35, linewidth=0.8)
        ax.set_xlim(0.96, 1.005)
        ax.set_ylim(0.96, 1.005)
        ax.set_xlabel(r"Acc@$\tau{=}0$")
        ax.set_ylabel(r"Acc@$\tau{=}1$")
        ax.set_title(TRACK_LABELS.get(track, track))
        ax.grid(alpha=0.22)

    fig.suptitle(f"SiGMA gated per-seed accuracy ({lr_tag}, test)", y=1.02, fontsize=12)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate paper figures from analysis CSVs."""
    args = _parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "paper_figures"

    per_run_path = analysis_dir / "per_run_metrics.csv"
    summary_path = analysis_dir / "summary_by_model.csv"
    if not per_run_path.is_file() or not summary_path.is_file():
        raise SystemExit(
            f"Missing CSVs in {analysis_dir}. "
            "scp from cluster first — see results/gcn_gin_routing/TEAM_BRIEF.md.",
        )

    _apply_style()
    summary = _load_summary(analysis_dir)
    per_run = _load_per_run(analysis_dir)
    gate_toy = _load_gate_node_csv(analysis_dir / "gate_node_summary_toy_test.csv")
    gate_sigma = _load_gate_node_csv(analysis_dir / "gate_node_summary_sigma_test.csv")
    gate_rows = gate_toy + gate_sigma

    plot_baseline_per_type(
        summary,
        out_dir / "fig01_baseline_per_type.png",
        dpi=args.dpi,
        lr_tag=args.lr_tag,
    )
    plot_gate_routing_delta(
        per_run,
        out_dir / "fig02_gate_routing_delta_per_seed.png",
        dpi=args.dpi,
        lr_tag=args.lr_tag,
    )
    plot_root_vs_neighbor_gates(
        gate_rows,
        out_dir / "fig03_root_vs_neighbor_gates.png",
        dpi=args.dpi,
        lr_tag=args.lr_tag,
    )
    plot_gated_accuracy_scatter(
        per_run,
        out_dir / "fig04_gated_accuracy_per_seed.png",
        dpi=args.dpi,
        lr_tag=args.lr_tag,
    )

    pairwise_summary = analysis_dir / "pairwise_baseline_summary.csv"
    if pairwise_summary.is_file():
        from scripts.synthetic.compare_gcn_gin_baselines_per_graph import (  # noqa: WPS433
            plot_from_summary_csv,
        )

        plot_from_summary_csv(
            pairwise_summary,
            out_dir / "fig05_pairwise_baseline_comparison.png",
            lr_tag=args.lr_tag,
            dpi=args.dpi,
        )

    from scripts.synthetic.gcn_gin_routing_table_figures import (  # noqa: WPS433
        plot_all_table_figures,
    )

    plot_all_table_figures(analysis_dir, lr_tag=args.lr_tag, dpi=args.dpi)

    print(f"\nPaper figures written to: {out_dir.resolve()}")
    for name in (
        "fig01_baseline_per_type",
        "fig02_gate_routing_delta_per_seed",
        "fig03_root_vs_neighbor_gates",
        "fig04_gated_accuracy_per_seed",
        "fig05_pairwise_baseline_comparison",
        "fig05_pairwise_baseline_table",
        "fig06_mask_ablation_table",
    ):
        png = out_dir / f"{name}.png"
        if png.is_file():
            print(f"  - {png}")


if __name__ == "__main__":
    main()
