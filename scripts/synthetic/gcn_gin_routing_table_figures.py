#!/usr/bin/env python3
"""Render paper-ready summary tables from GCN/GIN routing analysis CSVs.

Reads ``pairwise_baseline_summary.csv`` and ``mask_ablation_summary.csv`` and
writes matplotlib table figures (PNG + PDF) under ``paper_figures/``.

Example:
  python scripts/synthetic/gcn_gin_routing_table_figures.py \\
    --analysis-dir results/gcn_gin_routing/analysis
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HEADER_COLOR = "#4C78A8"
ROW_ALT = "#F5F7FA"
ROW_BASE = "#FFFFFF"
EDGE_COLOR = "#D0D7DE"
TABLE_FONT_SIZE = 9.5


@dataclass(frozen=True)
class PairwiseSummaryRow:
    """One row of ``pairwise_baseline_summary.csv``."""

    track: str
    lr_tag: str
    seed: int
    tau: int
    n_graphs: int
    both_correct: int
    gcn_only: int
    gin_only: int
    both_wrong: int


@dataclass(frozen=True)
class MaskAsymmetryRow:
    """One row of the mask-ablation asymmetry table."""

    track: str
    tau: int
    baseline_pct: float
    drop_gcn_pp: float
    drop_gin_pp: float
    asymmetry_label: str


def _style_table(table: plt.Table) -> None:
    """Apply consistent paper styling to a matplotlib table."""
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", weight="bold", fontsize=TABLE_FONT_SIZE)
            cell.set_height(0.38)
        else:
            cell.set_facecolor(ROW_ALT if row % 2 == 0 else ROW_BASE)
            cell.set_text_props(fontsize=TABLE_FONT_SIZE)
            cell.set_height(0.34)


def _save_table_figure(
    *,
    title: str,
    subtitle: str,
    col_labels: list[str],
    rows: list[list[str]],
    out_path: Path,
    dpi: int,
    col_widths: list[float] | None = None,
) -> None:
    """Save a single matplotlib table figure."""
    n_rows = len(rows)
    fig_h = 1.35 + 0.42 * max(n_rows, 1)
    fig_w = max(7.5, 1.15 * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    ax.text(0.5, 0.92, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.02, 0.08, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    _style_table(table)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_pairwise_summary(path: Path) -> list[PairwiseSummaryRow]:
    """Load ``pairwise_baseline_summary.csv``."""
    rows: list[PairwiseSummaryRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                PairwiseSummaryRow(
                    track=raw["track"],
                    lr_tag=raw["lr_tag"],
                    seed=int(raw["seed"]),
                    tau=int(raw["tau"]),
                    n_graphs=int(raw["n_graphs"]),
                    both_correct=int(raw["both_correct"]),
                    gcn_only=int(raw["gcn_only"]),
                    gin_only=int(raw["gin_only"]),
                    both_wrong=int(raw["both_wrong"]),
                ),
            )
    return rows


def _mean_pairwise_fractions(
    summary: Sequence[PairwiseSummaryRow],
) -> dict[tuple[str, int], dict[str, float]]:
    """Mean fractions across seeds for each (track, tau)."""
    grouped: dict[tuple[str, int], list[PairwiseSummaryRow]] = {}
    for row in summary:
        grouped.setdefault((row.track, row.tau), []).append(row)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        out[key] = {
            "both_correct": sum(r.both_correct / r.n_graphs for r in rows) / len(rows),
            "gcn_only": sum(r.gcn_only / r.n_graphs for r in rows) / len(rows),
            "gin_only": sum(r.gin_only / r.n_graphs for r in rows) / len(rows),
            "both_wrong": sum(r.both_wrong / r.n_graphs for r in rows) / len(rows),
        }
    return out


def pairwise_table_rows(
    summary: Sequence[PairwiseSummaryRow],
    *,
    lr_tag: str = "lr001",
) -> list[list[str]]:
    """Build table rows from pairwise summary (5-seed mean fractions)."""
    filtered = [r for r in summary if r.lr_tag == lr_tag]
    if not filtered:
        raise ValueError(f"No pairwise rows for lr_tag={lr_tag}")

    means = _mean_pairwise_fractions(filtered)
    track_order = sorted({r.track for r in filtered}, key=lambda t: (t != "sigma", t))
    rows: list[list[str]] = []
    for track in track_order:
        for tau in (0, 1):
            m = means.get((track, tau))
            if m is None:
                continue
            rows.append(
                [
                    track,
                    str(tau),
                    f"{100 * m['both_correct']:.1f}%",
                    f"{100 * m['gcn_only']:.1f}%",
                    f"{100 * m['gin_only']:.1f}%",
                    f"{100 * m['both_wrong']:.1f}%",
                ],
            )
    return rows


def plot_pairwise_baseline_table(
    summary_path: Path,
    out_path: Path,
    *,
    lr_tag: str = "lr001",
    dpi: int = 200,
) -> None:
    """Render fig05 table from ``pairwise_baseline_summary.csv``."""
    summary = _load_pairwise_summary(summary_path)
    rows = pairwise_table_rows(summary, lr_tag=lr_tag)
    _save_table_figure(
        title="GCN-only vs GIN-only per-graph outcomes",
        subtitle=f"5-seed mean fractions on test set ({lr_tag})",
        col_labels=["Track", "τ", "Both correct", "Only GCN", "Only GIN", "Both wrong"],
        rows=rows,
        out_path=out_path,
        dpi=dpi,
        col_widths=[0.14, 0.08, 0.18, 0.16, 0.16, 0.16],
    )


def _load_mask_summary(path: Path) -> list[dict[str, object]]:
    """Load ``mask_ablation_summary.csv``."""
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                {
                    "track": raw["track"],
                    "mask_mode": raw["mask_mode"],
                    "metric": raw["metric"],
                    "mean": float(raw["mean"]),
                },
            )
    return rows


def mask_asymmetry_rows(
    summary: Sequence[dict[str, object]],
    *,
    track: str = "toy",
) -> list[MaskAsymmetryRow]:
    """Compute asymmetry table rows for one track."""
    lookup: dict[tuple[str, str, str], float] = {}
    for row in summary:
        lookup[(str(row["track"]), str(row["mask_mode"]), str(row["metric"]))] = float(row["mean"])

    out: list[MaskAsymmetryRow] = []
    for tau, metric in ((0, "acc_tau0"), (1, "acc_tau1")):
        base = lookup.get((track, "none", metric), float("nan"))
        drop_gcn = base - lookup.get((track, "mask_gcn", metric), float("nan"))
        drop_gin = base - lookup.get((track, "mask_gin", metric), float("nan"))
        if tau == 0:
            label = "GCN essential" if drop_gcn > drop_gin + 1e-6 else "GIN essential"
        else:
            label = "GIN essential" if drop_gin > drop_gcn + 1e-6 else "GCN essential"
        out.append(
            MaskAsymmetryRow(
                track=track,
                tau=tau,
                baseline_pct=100.0 * base,
                drop_gcn_pp=100.0 * drop_gcn,
                drop_gin_pp=100.0 * drop_gin,
                asymmetry_label=label,
            ),
        )
    return out


def plot_mask_ablation_table(
    summary_path: Path,
    out_path: Path,
    *,
    track: str = "toy",
    model_label: str = "a0g2_gated",
    lr_tag: str = "lr001",
    dpi: int = 200,
) -> None:
    """Render fig06 table from ``mask_ablation_summary.csv``."""
    summary = _load_mask_summary(summary_path)
    rows_data = mask_asymmetry_rows(summary, track=track)
    rows = [
        [
            str(r.tau),
            f"{r.baseline_pct:.1f}%",
            f"{-r.drop_gcn_pp:+.1f} pp" if r.drop_gcn_pp != 0 else "−0.0 pp",
            f"{-r.drop_gin_pp:+.1f} pp" if r.drop_gin_pp != 0 else "−0.0 pp",
            r.asymmetry_label,
        ]
        for r in rows_data
    ]
    track_title = "Toy (routing convs)" if track == "toy" else "Sigma (PyG GIN/GCN)"
    _save_table_figure(
        title="SiGMA gated — MP head masking at eval",
        subtitle=f"{track_title} · {model_label} · {lr_tag} · 5-seed mean",
        col_labels=["τ", "Baseline", "Δ(mask GCN)", "Δ(mask GIN)", "Asymmetry"],
        rows=rows,
        out_path=out_path,
        dpi=dpi,
        col_widths=[0.10, 0.18, 0.20, 0.20, 0.22],
    )


@dataclass(frozen=True)
class OppositeSignPairSummaryRow:
    """One row of ``opposite_sign_pair_summary.csv``."""

    track: str
    lr_tag: str
    seed: int
    model: str
    n_pairs: int
    both_correct: int
    only_tau0: int
    only_tau1: int
    both_wrong: int


def _load_opposite_sign_pair_summary(path: Path) -> list[OppositeSignPairSummaryRow]:
    """Load ``opposite_sign_pair_summary.csv``."""
    rows: list[OppositeSignPairSummaryRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(
                OppositeSignPairSummaryRow(
                    track=raw["track"],
                    lr_tag=raw["lr_tag"],
                    seed=int(raw["seed"]),
                    model=raw["model"],
                    n_pairs=int(raw["n_pairs"]),
                    both_correct=int(raw["both_correct"]),
                    only_tau0=int(raw["only_tau0"]),
                    only_tau1=int(raw["only_tau1"]),
                    both_wrong=int(raw["both_wrong"]),
                ),
            )
    return rows


_OPPOSITE_SIGN_MODEL_LABELS: dict[str, str] = {
    "oracle_gcn_rule": "Oracle GCN rule",
    "oracle_gin_rule": "Oracle GIN rule",
    "gcn_only": "GCN-only",
    "gin_only": "GIN-only",
    "gated": "SiGMA gated",
}

_OPPOSITE_SIGN_MODEL_ORDER: tuple[str, ...] = (
    "oracle_gcn_rule",
    "oracle_gin_rule",
    "gcn_only",
    "gin_only",
    "gated",
)


def opposite_sign_pair_table_rows(
    summary: Sequence[OppositeSignPairSummaryRow],
    *,
    lr_tag: str = "lr001",
) -> list[list[str]]:
    """Build table rows for opposite-sign pair outcomes (5-seed mean)."""
    filtered = [r for r in summary if r.lr_tag == lr_tag]
    if not filtered:
        raise ValueError(f"No opposite-sign rows for lr_tag={lr_tag}")

    grouped: dict[tuple[str, str], list[OppositeSignPairSummaryRow]] = {}
    for row in filtered:
        grouped.setdefault((row.track, row.model), []).append(row)

    track_order = sorted({r.track for r in filtered}, key=lambda t: (t != "sigma", t))
    rows: list[list[str]] = []
    for track in track_order:
        for model in _OPPOSITE_SIGN_MODEL_ORDER:
            subset = grouped.get((track, model))
            if not subset:
                continue
            n_pairs = subset[0].n_pairs
            means = {
                key: sum(getattr(r, key) / r.n_pairs for r in subset) / len(subset)
                for key in ("both_correct", "only_tau0", "only_tau1", "both_wrong")
            }
            rows.append(
                [
                    track,
                    _OPPOSITE_SIGN_MODEL_LABELS.get(model, model),
                    str(n_pairs),
                    f"{100 * means['both_correct']:.1f}%",
                    f"{100 * means['only_tau0']:.1f}%",
                    f"{100 * means['only_tau1']:.1f}%",
                    f"{100 * means['both_wrong']:.1f}%",
                ],
            )
    return rows


def plot_opposite_sign_pair_table(
    summary_path: Path,
    out_path: Path,
    *,
    lr_tag: str = "lr001",
    dpi: int = 200,
) -> None:
    """Render fig07 table from ``opposite_sign_pair_summary.csv``."""
    summary = _load_opposite_sign_pair_summary(summary_path)
    rows = opposite_sign_pair_table_rows(summary, lr_tag=lr_tag)
    _save_table_figure(
        title="Opposite-sign τ pairs — per-pair specialist outcomes",
        subtitle=(
            f"Matched twins (shared topology, τ∈{{0,1}}); 5-seed mean on test ({lr_tag})"
        ),
        col_labels=[
            "Track",
            "Model",
            "Pairs",
            "Both correct",
            "Only τ=0",
            "Only τ=1",
            "Both wrong",
        ],
        rows=rows,
        out_path=out_path,
        dpi=dpi,
        col_widths=[0.12, 0.22, 0.10, 0.14, 0.12, 0.12, 0.12],
    )


def plot_all_table_figures(
    analysis_dir: Path,
    *,
    lr_tag: str = "lr001",
    mask_track: str = "toy",
    dpi: int = 200,
) -> list[Path]:
    """Generate table figures; return paths written."""
    paper_dir = analysis_dir / "paper_figures"
    written: list[Path] = []

    pairwise_csv = analysis_dir / "pairwise_baseline_summary.csv"
    if pairwise_csv.is_file():
        out = paper_dir / "fig05_pairwise_baseline_table.png"
        plot_pairwise_baseline_table(pairwise_csv, out, lr_tag=lr_tag, dpi=dpi)
        written.append(out)

    mask_csv = analysis_dir / "mask_ablation_summary.csv"
    if mask_csv.is_file():
        out = paper_dir / "fig06_mask_ablation_table.png"
        plot_mask_ablation_table(mask_csv, out, track=mask_track, lr_tag=lr_tag, dpi=dpi)
        written.append(out)

    opposite_csv = analysis_dir / "opposite_sign_pair_summary.csv"
    if opposite_csv.is_file():
        out = paper_dir / "fig07_opposite_sign_pair_table.png"
        plot_opposite_sign_pair_table(opposite_csv, out, lr_tag=lr_tag, dpi=dpi)
        written.append(out)

    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=_REPO_ROOT / "results/gcn_gin_routing/analysis",
        help="Directory containing summary CSVs",
    )
    parser.add_argument("--lr-tag", default="lr001")
    parser.add_argument("--mask-track", default="toy", choices=["toy", "sigma"])
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    written = plot_all_table_figures(
        args.analysis_dir,
        lr_tag=args.lr_tag,
        mask_track=args.mask_track,
        dpi=args.dpi,
    )
    if not written:
        raise SystemExit(f"No summary CSVs found under {args.analysis_dir}")
    for path in written:
        print(f"Wrote {path}")
        pdf = path.with_suffix(".pdf")
        if pdf.is_file():
            print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
