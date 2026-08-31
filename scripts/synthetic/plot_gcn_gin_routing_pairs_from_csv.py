#!/usr/bin/env python3
"""Build pairs-only fig01/fig05 from existing per-graph CSV (no GPU / no GNNPlus).

Filters ``pairwise_baseline_per_graph.csv`` to opposite-sign twin graphs only
(456 graphs = 228 pairs on test) and writes figures matching the full-test style.

For gated SiGMA + root gates on pairs (fig01 all models, fig02), run on cluster:
  bash bash_interface/cluster/run_analyze_gcn_gin_routing_pairs_only.sh

Example:
  python scripts/synthetic/plot_gcn_gin_routing_pairs_from_csv.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.synthetic.analyze_opposite_sign_pairs import (  # noqa: E402
    build_opposite_sign_pairs,
    load_test_graph_meta,
)
from scripts.synthetic.plot_gcn_gin_routing_paper_figures import (  # noqa: E402
    SummaryRow,
    _apply_style,
    plot_baseline_per_type,
)


OutcomeKind = Literal["both_correct", "gcn_only", "gin_only", "both_wrong"]


@dataclass(frozen=True)
class GraphRow:
    """One filtered per-graph row."""

    track: str
    lr_tag: str
    seed: int
    graph_idx: int
    tau: int
    label: int
    correct_gcn: bool
    correct_gin: bool

    @property
    def outcome(self) -> OutcomeKind:
        """Pairwise outcome for this graph."""
        if self.correct_gcn and self.correct_gin:
            return "both_correct"
        if self.correct_gcn and not self.correct_gin:
            return "gcn_only"
        if not self.correct_gcn and self.correct_gin:
            return "gin_only"
        return "both_wrong"


@dataclass(frozen=True)
class PairwiseSummaryRow:
    """Aggregated 4-way counts per (track, tau, lr, seed)."""

    track: str
    lr_tag: str
    seed: int
    tau: int
    n_graphs: int
    both_correct: int
    gcn_only: int
    gin_only: int
    both_wrong: int


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairwise-csv",
        type=str,
        default="results/gcn_gin_routing/analysis/pairwise_baseline_per_graph.csv",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/gcn_gin_routing/data",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gcn_gin_routing/analysis/pairs_only",
    )
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_pair_indices(dataset_dir: str) -> frozenset[int]:
    """Test graph indices in opposite-sign twin pairs."""
    meta = load_test_graph_meta(dataset_dir)
    pairs = build_opposite_sign_pairs(meta)
    idx: set[int] = set()
    for pair in pairs:
        idx.add(pair.tau0_idx)
        idx.add(pair.tau1_idx)
    if not idx:
        raise RuntimeError("No opposite-sign pairs in test split.")
    return frozenset(idx)


def load_filtered_rows(pairwise_csv: Path, pair_indices: frozenset[int], lr_tag: str) -> list[GraphRow]:
    """Load per-graph rows restricted to twin-pair graphs."""
    rows: list[GraphRow] = []
    with pairwise_csv.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            if raw["lr_tag"] != lr_tag:
                continue
            graph_idx = int(raw["graph_idx"])
            if graph_idx not in pair_indices:
                continue
            rows.append(
                GraphRow(
                    track=raw["track"],
                    lr_tag=raw["lr_tag"],
                    seed=int(raw["seed"]),
                    graph_idx=graph_idx,
                    tau=int(raw["tau"]),
                    label=int(raw["label"]),
                    correct_gcn=bool(int(raw["correct_gcn"])),
                    correct_gin=bool(int(raw["correct_gin"])),
                ),
            )
    if not rows:
        raise RuntimeError(f"No rows after filtering {pairwise_csv}")
    return rows


def summarize_pairwise(rows: Sequence[GraphRow]) -> list[PairwiseSummaryRow]:
    """Aggregate 4-way outcome counts per (track, lr, seed, tau)."""
    keys = {(r.track, r.lr_tag, r.seed, r.tau) for r in rows}
    out: list[PairwiseSummaryRow] = []
    for track, lr_tag, seed, tau in sorted(keys):
        subset = [r for r in rows if (r.track, r.lr_tag, r.seed, r.tau) == (track, lr_tag, seed, tau)]
        out.append(
            PairwiseSummaryRow(
                track=track,
                lr_tag=lr_tag,
                seed=seed,
                tau=tau,
                n_graphs=len(subset),
                both_correct=sum(1 for r in subset if r.outcome == "both_correct"),
                gcn_only=sum(1 for r in subset if r.outcome == "gcn_only"),
                gin_only=sum(1 for r in subset if r.outcome == "gin_only"),
                both_wrong=sum(1 for r in subset if r.outcome == "both_wrong"),
            ),
        )
    return out


def build_baseline_summary(rows: Sequence[GraphRow], lr_tag: str) -> list[SummaryRow]:
    """Per-model τ accuracy summary for GCN-only and GIN-only on pairs."""
    grouped: dict[tuple[str, str, int], list[GraphRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.track, row.seed)].append(row)

    per_seed: list[dict[str, float | int | str]] = []
    for (track, seed), subset in sorted(grouped.items()):
        n_t0 = sum(1 for r in subset if r.tau == 0)
        n_t1 = sum(1 for r in subset if r.tau == 1)
        for model, use_gcn in (("a0g1_gcn", True), ("a0g1_gin", False)):
            c0 = sum(1 for r in subset if r.tau == 0 and (r.correct_gcn if use_gcn else r.correct_gin))
            c1 = sum(1 for r in subset if r.tau == 1 and (r.correct_gcn if use_gcn else r.correct_gin))
            per_seed.append(
                {
                    "track": track,
                    "model": model,
                    "lr_tag": lr_tag,
                    "seed": seed,
                    "acc_all": (c0 + c1) / len(subset),
                    "acc_tau0": c0 / n_t0 if n_t0 else float("nan"),
                    "acc_tau1": c1 / n_t1 if n_t1 else float("nan"),
                },
            )

    summary: list[SummaryRow] = []
    keys = {(str(r["track"]), str(r["model"])) for r in per_seed}
    for track, model in sorted(keys):
        items = [r for r in per_seed if r["track"] == track and r["model"] == model]

        def agg(key: str) -> tuple[float, float]:
            vals = [float(r[key]) for r in items]
            if len(vals) == 1:
                return vals[0], 0.0
            return float(mean(vals)), float(pstdev(vals))

        acc_all_m, acc_all_s = agg("acc_all")
        acc_t0_m, acc_t0_s = agg("acc_tau0")
        acc_t1_m, acc_t1_s = agg("acc_tau1")
        summary.append(
            SummaryRow(
                track=track,
                model=model,
                lr_tag=lr_tag,
                n_seeds=len(items),
                acc_all_mean=acc_all_m,
                acc_all_std=acc_all_s,
                acc_tau0_mean=acc_t0_m,
                acc_tau0_std=acc_t0_s,
                acc_tau1_mean=acc_t1_m,
                acc_tau1_std=acc_t1_s,
            ),
        )
    return summary


def mean_summary_across_seeds(
    summary: Sequence[PairwiseSummaryRow],
) -> dict[tuple[str, int], dict[str, float]]:
    """Mean fractions across seeds for each (track, tau)."""
    grouped: dict[tuple[str, int], list[PairwiseSummaryRow]] = defaultdict(list)
    for row in summary:
        grouped[(row.track, row.tau)].append(row)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        out[key] = {
            "both_correct": float(mean(r.both_correct / r.n_graphs for r in rows)),
            "gcn_only": float(mean(r.gcn_only / r.n_graphs for r in rows)),
            "gin_only": float(mean(r.gin_only / r.n_graphs for r in rows)),
            "both_wrong": float(mean(r.both_wrong / r.n_graphs for r in rows)),
        }
    return out


def plot_pairwise_pairs(
    summary: Sequence[PairwiseSummaryRow],
    out_path: Path,
    *,
    lr_tag: str,
    dpi: int,
) -> None:
    """Fig05-style stacked bars for pairs-only subset."""
    import matplotlib.pyplot as plt
    import numpy as np

    means = mean_summary_across_seeds(summary)
    tracks = sorted({s.track for s in summary})
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.5 * len(tracks), 4.8), squeeze=False)
    colors = {
        "both_correct": "#BAB0AC",
        "gcn_only": "#4C72B0",
        "gin_only": "#DD8452",
        "both_wrong": "#E45756",
    }
    labels = {
        "both_correct": "Both correct",
        "gcn_only": "GCN-only correct",
        "gin_only": "GIN-only correct",
        "both_wrong": "Both wrong",
    }
    kinds = ("both_correct", "gcn_only", "gin_only", "both_wrong")
    tau_labels = {0: r"$\tau{=}0$ (GCN-type)", 1: r"$\tau{=}1$ (GIN-type)"}

    for ax, track in zip(axes[0], tracks, strict=True):
        x = np.arange(2)
        for bar_i, tau in enumerate((0, 1)):
            bottom = 0.0
            for kind in kinds:
                val = means.get((track, tau), {}).get(kind, 0.0)
                ax.bar(
                    x[bar_i],
                    val,
                    bottom=bottom,
                    width=0.55,
                    color=colors[kind],
                    label=labels[kind] if bar_i == 0 else None,
                    edgecolor="white",
                    linewidth=0.5,
                )
                if val > 0.04:
                    ax.text(
                        x[bar_i],
                        bottom + val / 2,
                        f"{100 * val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if kind != "both_correct" else "black",
                        fontweight="bold",
                    )
                bottom += val
        ax.set_xticks(x)
        ax.set_xticklabels([tau_labels[t] for t in (0, 1)])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction of twin-pair graphs")
        ax.set_title("Toy (routing convs)" if track == "toy" else "Sigma (PyG GIN/GCN)")
        ax.grid(axis="y", alpha=0.22)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        f"GCN-only vs GIN-only on opposite-sign twins (5-seed mean, {lr_tag}, n=228 pairs)",
        y=1.08,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Filter CSV and write pairs-only figures."""
    args = _parse_args()
    out_dir = Path(args.out_dir)
    paper_dir = out_dir / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    pair_indices = load_pair_indices(args.dataset_dir)
    n_pairs = len(pair_indices) // 2
    print(f"Pairs subset: {len(pair_indices)} graphs ({n_pairs} twin pairs)")

    rows = load_filtered_rows(Path(args.pairwise_csv), pair_indices, args.lr_tag)
    pairwise_summary = summarize_pairwise(rows)
    baseline_summary = build_baseline_summary(rows, args.lr_tag)

    _apply_style()
    plot_baseline_per_type(
        baseline_summary,
        paper_dir / "fig01_baseline_per_type_pairs.png",
        dpi=args.dpi,
        lr_tag=args.lr_tag,
        ymin=0.0,
        title_suffix=" — opposite-sign twin pairs only (GCN/GIN baselines)",
    )
    plot_pairwise_pairs(
        pairwise_summary,
        paper_dir / "fig05_pairwise_baseline_comparison_pairs.png",
        lr_tag=args.lr_tag,
        dpi=args.dpi,
    )

    # Write summary CSVs
    with (out_dir / "pairwise_baseline_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "track",
                "lr_tag",
                "seed",
                "tau",
                "n_graphs",
                "both_correct",
                "gcn_only",
                "gin_only",
                "both_wrong",
            ],
        )
        writer.writeheader()
        for row in pairwise_summary:
            writer.writerow(
                {
                    "track": row.track,
                    "lr_tag": row.lr_tag,
                    "seed": row.seed,
                    "tau": row.tau,
                    "n_graphs": row.n_graphs,
                    "both_correct": row.both_correct,
                    "gcn_only": row.gcn_only,
                    "gin_only": row.gin_only,
                    "both_wrong": row.both_wrong,
                },
            )

    print(f"Wrote {paper_dir / 'fig01_baseline_per_type_pairs.png'}")
    print(f"Wrote {paper_dir / 'fig05_pairwise_baseline_comparison_pairs.png'}")
    print(
        "Note: fig02 (root gates) and SiGMA gated/ungated on pairs require "
        "bash_interface/cluster/run_analyze_gcn_gin_routing_pairs_only.sh on cluster.",
    )


if __name__ == "__main__":
    main()
