#!/usr/bin/env python3
"""Plot analytic score-sign relation vs pairwise baseline outcomes (fig. 05 supplement).

Cross-tabulates test-graph analytic scores ``s_GCN``, ``s_GIN`` with trained
GCN-only vs GIN-only pairwise outcomes from ``pairwise_baseline_per_graph.csv``.

Example::

  python scripts/synthetic/plot_gcn_gin_routing_score_sign_figure.py \\
    --analysis-dir results/gcn_gin_routing/analysis \\
    --dataset-dir results/gcn_gin_routing/data/GcnGinRouting \\
    --track toy --lr-tag lr001 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OutcomeKind = Literal["both_correct", "gcn_only", "gin_only", "both_wrong"]
SignKind = Literal["same_sign", "opposite", "has_zero"]

SIGN_LABELS: dict[SignKind, str] = {
    "same_sign": "Same sign",
    "opposite": "Opposite sign",
    "has_zero": r"Near zero ($s{=}0$)",
}
SIGN_COLORS: dict[SignKind, str] = {
    "same_sign": "#B8B8B8",
    "opposite": "#DD8452",
    "has_zero": "#C44E52",
}
OUTCOME_GROUPS: tuple[tuple[str, str], ...] = (
    ("both_correct", "Both models correct"),
    ("specialist_only", "Only matching specialist correct"),
)
SIGN_ORDER: tuple[SignKind, ...] = ("same_sign", "opposite", "has_zero")


@dataclass(frozen=True)
class PairwiseRow:
    """One row from ``pairwise_baseline_per_graph.csv``."""

    track: str
    lr_tag: str
    seed: int
    graph_idx: int
    tau: int
    outcome: OutcomeKind


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="results/gcn_gin_routing/analysis",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/gcn_gin_routing/data/GcnGinRouting",
    )
    parser.add_argument("--track", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG (default: analysis/paper_figures/fig08_score_sign_vs_pairwise.png).",
    )
    return parser.parse_args(argv)


def _load_gcn_gin_routing_module() -> object:
    """Import dataset module without pulling full ``GNNPlus`` package."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "GNNPlus/loader/dataset/gcn_gin_routing.py"
    spec = importlib.util.spec_from_file_location("gcn_gin_routing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gcn_gin_routing"] = mod
    spec.loader.exec_module(mod)
    return mod


def _classify_sign(gcn_s: float, gin_s: float) -> SignKind:
    """Mutually exclusive sign bucket for analytic scores."""
    if gcn_s == 0.0 or gin_s == 0.0:
        return "has_zero"
    if (gcn_s > 0.0) != (gin_s > 0.0):
        return "opposite"
    return "same_sign"


def _specialist_outcome(tau: int, outcome: OutcomeKind) -> bool:
    """True when only the τ-matching trained baseline is correct."""
    if tau == 0 and outcome == "gcn_only":
        return True
    if tau == 1 and outcome == "gin_only":
        return True
    return False


def _load_pairwise_rows(
    csv_path: Path,
    *,
    track: str,
    lr_tag: str,
    seed: int,
) -> list[PairwiseRow]:
    """Load filtered pairwise per-graph rows."""
    rows: list[PairwiseRow] = []
    with csv_path.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if raw["track"] != track or raw["lr_tag"] != lr_tag:
                continue
            if int(raw["seed"]) != seed:
                continue
            rows.append(
                PairwiseRow(
                    track=str(raw["track"]),
                    lr_tag=str(raw["lr_tag"]),
                    seed=int(raw["seed"]),
                    graph_idx=int(raw["graph_idx"]),
                    tau=int(raw["tau"]),
                    outcome=str(raw["outcome"]),  # type: ignore[arg-type]
                ),
            )
    return rows


def _build_counts(
    pairwise_rows: Sequence[PairwiseRow],
    scores_by_idx: dict[int, tuple[float, float]],
) -> dict[str, Counter[SignKind]]:
    """Count sign buckets per outcome group."""
    counts: dict[str, Counter[SignKind]] = {
        "both_correct": Counter(),
        "specialist_only": Counter(),
    }
    for row in pairwise_rows:
        gcn_s, gin_s = scores_by_idx[row.graph_idx]
        sign = _classify_sign(gcn_s, gin_s)
        if row.outcome == "both_correct":
            counts["both_correct"][sign] += 1
        elif _specialist_outcome(row.tau, row.outcome):
            counts["specialist_only"][sign] += 1
    return counts


def _fractions(counter: Counter[SignKind]) -> dict[SignKind, float]:
    """Normalize counts to fractions summing to 1."""
    total = sum(counter.values())
    if total == 0:
        return {k: 0.0 for k in SIGN_ORDER}
    return {k: counter[k] / total for k in SIGN_ORDER}


def plot_score_sign_vs_pairwise(
    counts: dict[str, Counter[SignKind]],
    *,
    track: str,
    lr_tag: str,
    seed: int,
    out_path: Path,
    dpi: int,
) -> None:
    """Write stacked bar chart (PNG + PDF)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x_labels = [label for _, label in OUTCOME_GROUPS]
    x_pos = np.arange(len(x_labels))
    width = 0.55

    bottoms = np.zeros(len(x_labels))
    for sign in SIGN_ORDER:
        heights = np.array(
            [_fractions(counts[key])[sign] for key, _ in OUTCOME_GROUPS],
            dtype=np.float64,
        )
        ns = [sum(counts[key].values()) for key, _ in OUTCOME_GROUPS]
        bars = ax.bar(
            x_pos,
            heights,
            width,
            bottom=bottoms,
            color=SIGN_COLORS[sign],
            edgecolor="white",
            linewidth=0.8,
            label=SIGN_LABELS[sign],
        )
        for bar, frac, n in zip(bars, heights, ns, strict=True):
            if frac < 0.08:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_y() + bar.get_height() / 2.0,
                f"{100.0 * frac:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if sign != "same_sign" else "#333333",
            )
        bottoms += heights

    for i, (key, _) in enumerate(OUTCOME_GROUPS):
        n = sum(counts[key].values())
        ax.text(
            x_pos[i],
            1.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Fraction of test graphs", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_title(
        f"Analytic score signs vs pairwise baseline outcomes\n"
        f"({track}, {lr_tag}, seed {seed}; test set)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load data, cross-tabulate, and write figure."""
    args = _parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    dataset_dir = Path(args.dataset_dir)
    pairwise_csv = analysis_dir / "pairwise_baseline_per_graph.csv"
    if not pairwise_csv.is_file():
        raise FileNotFoundError(pairwise_csv)

    mod = _load_gcn_gin_routing_module()
    dataset_cls = mod.GcnGinRoutingDataset
    ds = dataset_cls(str(dataset_dir), split="test")
    scores_by_idx: dict[int, tuple[float, float]] = {
        i: (float(ds[i].gcn_score), float(ds[i].gin_score)) for i in range(len(ds))
    }

    pairwise_rows = _load_pairwise_rows(
        pairwise_csv,
        track=str(args.track),
        lr_tag=str(args.lr_tag),
        seed=int(args.seed),
    )
    counts = _build_counts(pairwise_rows, scores_by_idx)

    out_path = (
        Path(args.out)
        if args.out
        else analysis_dir / "paper_figures" / "fig08_score_sign_vs_pairwise.png"
    )
    plot_score_sign_vs_pairwise(
        counts,
        track=str(args.track),
        lr_tag=str(args.lr_tag),
        seed=int(args.seed),
        out_path=out_path,
        dpi=int(args.dpi),
    )
    print(f"Wrote {out_path}")
    for key, label in OUTCOME_GROUPS:
        n = sum(counts[key].values())
        fr = _fractions(counts[key])
        print(f"  {label} (n={n}):")
        for sign in SIGN_ORDER:
            print(f"    {SIGN_LABELS[sign]}: {100.0 * fr[sign]:.1f}%")


if __name__ == "__main__":
    main()
