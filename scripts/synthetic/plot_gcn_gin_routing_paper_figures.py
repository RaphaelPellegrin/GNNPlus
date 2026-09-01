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
    "a0g2_gated": "SiGMA",
    "a0g2_ungated": "SiGMA ungated",
    "a0g1_gcn": "GCN-only",
    "a0g1_gin": "GIN-only",
}
TRACK_SHORT_LABELS: dict[str, str] = {
    "toy": "Toy (routing convs)",
    "sigma": "Sigma (PyG GIN/GCN)",
}
TRACK_LABELS: dict[str, str] = {
    "toy": r"Track A (Toy, $d_h{=}1$)",
    "sigma": r"Track B (SiGMA, PyG GIN/GCN, $d_h{=}4$)",
}
TRACK_ORDER: tuple[str, ...] = ("toy", "sigma")
PALETTE = {
    "all": "#E45756",
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


def _format_lr_tag(lr_tag: str) -> str:
    """Format ``lr001`` as ``$\\mathrm{LR}=10^{-3}$`` for figure titles."""
    if lr_tag.startswith("lr") and lr_tag[2:].isdigit():
        exponent = -len(lr_tag[2:])
        return rf"$\mathrm{{LR}}=10^{{{exponent}}}$"
    return rf"$\mathrm{{LR}}={lr_tag}$"


def _save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    """Save PNG and PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _mean_minmax_yerr(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean, lower_err, upper_err) for min–max whiskers over seeds."""
    if not values:
        return float("nan"), 0.0, 0.0
    m = float(mean(values))
    lo = float(min(values))
    hi = float(max(values))
    return m, m - lo, hi - m


def _minmax_yerr_batch(
    value_groups: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], list[float]]:
    """Batch helper: means plus asymmetric lower/upper errors."""
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for values in value_groups:
        m, lo_err, hi_err = _mean_minmax_yerr(values)
        means.append(m)
        lowers.append(lo_err)
        uppers.append(hi_err)
    return means, lowers, uppers


def plot_baseline_per_type(
    summary: Sequence[SummaryRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
    *,
    per_run: Sequence[PerRunRow] | None = None,
    ymin: float = 0.65,
    title_suffix: str = "",
) -> None:
    """Grouped bar chart: all + per-type test accuracy (Track A left, Track B right)."""
    tracks = [t for t in TRACK_ORDER if any(r.track == t for r in summary)]
    tracks.extend(sorted({r.track for r in summary} - set(tracks)))
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.8 * len(tracks), 5.0), squeeze=False)
    bar_w = 0.24
    offsets = (-bar_w, 0.0, bar_w)
    series_specs = (
        ("all", "acc_all", "acc_all_mean", "acc_all_std", "All graphs"),
        ("tau0", "acc_tau0", "acc_tau0_mean", "acc_tau0_std", r"$\tau{=}0$ (GCN-type)"),
        ("tau1", "acc_tau1", "acc_tau1_mean", "acc_tau1_std", r"$\tau{=}1$ (GIN-type)"),
    )
    err_kw = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"}
    legend_handles: list = []
    legend_labels: list[str] = []

    for ax_idx, (ax, track) in enumerate(zip(axes[0], tracks, strict=True)):
        subset = [r for r in summary if r.track == track and r.lr_tag == lr_tag]
        by_model = {r.model: r for r in subset}
        models = [m for m in MODEL_ORDER if m in by_model]
        x = np.arange(len(models))

        for offset, (key, run_attr, mean_attr, std_attr, label) in zip(
            offsets,
            series_specs,
            strict=True,
        ):
            if per_run is not None:
                groups = [
                    [getattr(r, run_attr) for r in per_run if r.track == track and r.model == m and r.lr_tag == lr_tag]
                    for m in models
                ]
                vals, lo, hi = _minmax_yerr_batch(groups)
            else:
                vals = [getattr(by_model[m], mean_attr) for m in models]
                lo = [getattr(by_model[m], std_attr) for m in models]
                hi = lo

            bars = ax.bar(
                x + offset,
                vals,
                width=bar_w,
                yerr=[lo, hi],
                capsize=3,
                label=label,
                color=PALETTE[key],
                edgecolor="white",
                linewidth=0.6,
                error_kw=err_kw,
            )
            if ax_idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=12, ha="right")
        ax.set_ylim(ymin, 1.02)
        ax.set_ylabel("Test accuracy")
        ax.set_title(TRACK_LABELS.get(track, track))
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
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
        f"Test accuracy (5-seed mean, min–max whiskers, {_format_lr_tag(lr_tag)}){title_suffix}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.98))
    _save_fig(fig, out_path, dpi)


def plot_gate_routing_delta(
    per_run: Sequence[PerRunRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
    *,
    title_suffix: str = "",
) -> None:
    """Mean root routing contrast (gated model) with min–max whiskers over seeds."""
    gated = [
        r
        for r in per_run
        if r.model == "a0g2_gated" and r.has_gates and r.lr_tag == lr_tag
    ]
    if not gated:
        return

    track_order = [t for t in TRACK_ORDER if any(r.track == t for r in gated)]
    track_order.extend(
        sorted({r.track for r in gated} - set(track_order)),
    )

    gin_means: list[float] = []
    gin_lo: list[float] = []
    gin_hi: list[float] = []
    gcn_means: list[float] = []
    gcn_lo: list[float] = []
    gcn_hi: list[float] = []
    for track in track_order:
        subset = [r for r in gated if r.track == track]
        delta_gin = [r.gin_gate_tau1 - r.gin_gate_tau0 for r in subset]
        delta_gcn = [r.gcn_gate_tau0 - r.gcn_gate_tau1 for r in subset]
        m_gin, lo_gin, hi_gin = _mean_minmax_yerr(delta_gin)
        m_gcn, lo_gcn, hi_gcn = _mean_minmax_yerr(delta_gcn)
        gin_means.append(m_gin)
        gin_lo.append(lo_gin)
        gin_hi.append(hi_gin)
        gcn_means.append(m_gcn)
        gcn_lo.append(lo_gcn)
        gcn_hi.append(hi_gcn)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    x = np.arange(len(track_order))
    bar_w = 0.34
    err_kw = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"}

    ax.bar(
        x - bar_w / 2,
        gin_means,
        width=bar_w,
        yerr=[gin_lo, gin_hi],
        capsize=3,
        label=r"$\Delta\gamma_{\mathrm{GIN}} = \bar{\gamma}_{\tau{=}1} - \bar{\gamma}_{\tau{=}0}$",
        color=PALETTE["gin"],
        edgecolor="black",
        linewidth=0.4,
        error_kw=err_kw,
    )
    ax.bar(
        x + bar_w / 2,
        gcn_means,
        width=bar_w,
        yerr=[gcn_lo, gcn_hi],
        capsize=3,
        label=r"$\Delta\gamma_{\mathrm{GCN}} = \bar{\gamma}_{\tau{=}0} - \bar{\gamma}_{\tau{=}1}$",
        color=PALETTE["gcn"],
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        error_kw=err_kw,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TRACK_LABELS.get(t, t) for t in track_order],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"Root gate contrast")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.22)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        f"Root routing contrast (5-seed mean, min–max whiskers, "
        f"{_format_lr_tag(lr_tag)}, test){title_suffix}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.98))
    _save_fig(fig, out_path, dpi)


def plot_root_vs_neighbor_gates(
    gate_rows: Sequence[GateNodeRow],
    out_path: Path,
    dpi: int,
    lr_tag: str = "lr001",
) -> None:
    """Root vs neighbor mean gates (gated, one LR), with min–max whiskers over seeds."""
    rows = [r for r in gate_rows if r.lr_tag == lr_tag]
    if not rows:
        return

    tracks = sorted({r.track for r in rows})
    fig, axes = plt.subplots(2, len(tracks), figsize=(5.0 * len(tracks), 7.0), squeeze=False)
    err_kw = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"}

    for col, track in enumerate(tracks):
        subset = [r for r in rows if r.track == track]
        row_specs = (
            (
                "Root node",
                ("gin_root_tau0", "gin_root_tau1", "gcn_root_tau0", "gcn_root_tau1"),
            ),
            (
                "Neighbors (mean)",
                ("gin_nbr_tau0", "gin_nbr_tau1", "gcn_nbr_tau0", "gcn_nbr_tau1"),
            ),
        )

        for row_idx, (title, attrs) in enumerate(row_specs):
            vals: list[float] = []
            lo_errs: list[float] = []
            hi_errs: list[float] = []
            for attr in attrs:
                m, lo, hi = _mean_minmax_yerr([getattr(r, attr) for r in subset])
                vals.append(m)
                lo_errs.append(lo)
                hi_errs.append(hi)

            ax = axes[row_idx, col]
            labels = [r"$\tau{=}0$ GIN", r"$\tau{=}1$ GIN", r"$\tau{=}0$ GCN", r"$\tau{=}1$ GCN"]
            colors = [
                to_rgba(PALETTE["gin"], 1.0),
                to_rgba(PALETTE["gin"], 0.55),
                to_rgba(PALETTE["gcn"], 1.0),
                to_rgba(PALETTE["gcn"], 0.55),
            ]
            ax.bar(
                range(4),
                vals,
                yerr=[lo_errs, hi_errs],
                capsize=3,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
                error_kw=err_kw,
            )
            ax.set_xticks(range(4))
            ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel(r"Mean $\gamma$")
            if row_idx == 0:
                ax.set_title(TRACK_LABELS.get(track, track))
            ax.text(-0.12, 1.04, title, transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"Root vs neighbor gates (5-seed mean, min–max whiskers, {lr_tag}, test)",
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
        per_run=per_run,
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
