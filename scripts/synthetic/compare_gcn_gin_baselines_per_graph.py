#!/usr/bin/env python3
"""Pairwise per-graph comparison of GCN-only vs GIN-only baselines.

For each test graph, records whether GCN-only and GIN-only predict correctly,
then summarizes routing-specialization:

- On τ=0 (GCN-type): is GCN-only correct more often than GIN-only (per graph)?
- On τ=1 (GIN-type): is GIN-only correct more often than GCN-only?

Outputs:
  - ``pairwise_baseline_per_graph.csv`` (one row per graph × seed)
  - ``pairwise_baseline_summary.csv`` (aggregated 4-way counts)
  - ``fig05_pairwise_baseline_comparison.png`` / ``.pdf``

Example:
  python scripts/synthetic/compare_gcn_gin_baselines_per_graph.py \\
    --results-root results/gcn_gin_routing/gates \\
    --dataset-dir results/gcn_gin_routing/data \\
    --out-dir results/gcn_gin_routing/analysis \\
    --lr-tag lr001
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401

from scripts.synthetic.analyze_gcn_gin_routing_results import (  # noqa: E402
    RunRef,
    _load_cfg_for_run,
    _pick_best_epoch,
    _pred_labels_from_score,
)

OutcomeKind = Literal["both_correct", "gcn_only", "gin_only", "both_wrong"]


@dataclass(frozen=True)
class GraphPairOutcome:
    """Per-graph correctness for GCN-only vs GIN-only on the same test graph."""

    track: str
    lr_tag: str
    seed: int
    graph_idx: int
    tau: int
    label: int
    pred_gcn: int
    pred_gin: int
    correct_gcn: bool
    correct_gin: bool

    @property
    def outcome(self) -> OutcomeKind:
        """Classify pairwise outcome."""
        if self.correct_gcn and self.correct_gin:
            return "both_correct"
        if self.correct_gcn and not self.correct_gin:
            return "gcn_only"
        if not self.correct_gcn and self.correct_gin:
            return "gin_only"
        return "both_wrong"


@dataclass(frozen=True)
class PairwiseSummaryRow:
    """Aggregated 4-way counts for one (track, tau, lr, seed)."""

    track: str
    lr_tag: str
    seed: int
    tau: int
    n_graphs: int
    both_correct: int
    gcn_only: int
    gin_only: int
    both_wrong: int

    @property
    def frac_gcn_wins(self) -> float:
        """Fraction where only GCN is correct."""
        return self.gcn_only / self.n_graphs if self.n_graphs else float("nan")

    @property
    def frac_gin_wins(self) -> float:
        """Fraction where only GIN is correct."""
        return self.gin_only / self.n_graphs if self.n_graphs else float("nan")

    @property
    def frac_specialist_wins(self) -> float:
        """τ=0 → gcn_only; τ=1 → gin_only."""
        wins = self.gcn_only if self.tau == 0 else self.gin_only
        return wins / self.n_graphs if self.n_graphs else float("nan")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/gcn_gin_routing/gates",
        help="Parent of toy/ and sigma/ (e.g. gates/ or netscratch gcn_gin_routing/).",
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
        help="Output directory for CSVs and figure.",
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
        help="Learning-rate tag filter (e.g. lr001).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds to evaluate.",
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
    return parser.parse_args(argv)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_dir(results_root: Path, track: str, model: str, lr_tag: str, seed: int) -> Path:
    """Path to one training run directory."""
    lr_num = lr_tag.removeprefix("lr")
    return results_root / track / f"{model}_{lr_tag}_seed{seed}"


@torch.no_grad()
def _collect_test_predictions(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
) -> list[tuple[int, int, int]]:
    """Return ``(tau, label, pred)`` per test graph in loader order."""
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

    rows: list[tuple[int, int, int]] = []
    graph_idx = 0
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing tau.")
        tau = batch.tau.view(-1).long()
        pred, true = model(batch)
        _loss, pred_score = compute_loss(pred, true)
        pred_label = _pred_labels_from_score(pred_score).view(-1).long()
        true_label = true.view(-1).long()
        for g in range(pred_label.numel()):
            rows.append(
                (
                    int(tau[g].item()),
                    int(true_label[g].item()),
                    int(pred_label[g].item()),
                ),
            )
            graph_idx += 1
    return rows


def _compare_pair(
    track: str,
    lr_tag: str,
    seed: int,
    results_root: Path,
    dataset_dir: str,
    device: torch.device,
) -> list[GraphPairOutcome]:
    """Compare GCN-only vs GIN-only on the same test graphs."""
    gcn_ref = RunRef(
        track=track,
        run_dir=_run_dir(results_root, track, "a0g1_gcn", lr_tag, seed),
        model="a0g1_gcn",
        lr_tag=lr_tag,
        seed=seed,
    )
    gin_ref = RunRef(
        track=track,
        run_dir=_run_dir(results_root, track, "a0g1_gin", lr_tag, seed),
        model="a0g1_gin",
        lr_tag=lr_tag,
        seed=seed,
    )
    for ref in (gcn_ref, gin_ref):
        if not ref.run_dir.is_dir():
            raise FileNotFoundError(f"Missing run dir: {ref.run_dir}")

    gcn_preds = _collect_test_predictions(gcn_ref, dataset_dir, device)
    gin_preds = _collect_test_predictions(gin_ref, dataset_dir, device)
    if len(gcn_preds) != len(gin_preds):
        raise RuntimeError(
            f"Test set size mismatch track={track} seed={seed}: "
            f"gcn={len(gcn_preds)} gin={len(gin_preds)}",
        )

    out: list[GraphPairOutcome] = []
    for idx, ((tau_g, y_g, p_g), (tau_i, y_i, p_i)) in enumerate(
        zip(gcn_preds, gin_preds, strict=True),
    ):
        if tau_g != tau_i or y_g != y_i:
            raise RuntimeError(f"Graph {idx} label/tau mismatch between models.")
        out.append(
            GraphPairOutcome(
                track=track,
                lr_tag=lr_tag,
                seed=seed,
                graph_idx=idx,
                tau=tau_g,
                label=y_g,
                pred_gcn=p_g,
                pred_gin=p_i,
                correct_gcn=p_g == y_g,
                correct_gin=p_i == y_i,
            ),
        )
    return out


def _summarize(rows: Sequence[GraphPairOutcome]) -> list[PairwiseSummaryRow]:
    """Aggregate 4-way outcome counts per (track, lr, seed, tau)."""
    keys: set[tuple[str, str, int, int]] = {
        (r.track, r.lr_tag, r.seed, r.tau) for r in rows
    }
    summary: list[PairwiseSummaryRow] = []
    for track, lr_tag, seed, tau in sorted(keys):
        subset = [r for r in rows if (r.track, r.lr_tag, r.seed, r.tau) == (track, lr_tag, seed, tau)]
        counts = {
            "both_correct": sum(1 for r in subset if r.outcome == "both_correct"),
            "gcn_only": sum(1 for r in subset if r.outcome == "gcn_only"),
            "gin_only": sum(1 for r in subset if r.outcome == "gin_only"),
            "both_wrong": sum(1 for r in subset if r.outcome == "both_wrong"),
        }
        summary.append(
            PairwiseSummaryRow(
                track=track,
                lr_tag=lr_tag,
                seed=seed,
                tau=tau,
                n_graphs=len(subset),
                both_correct=counts["both_correct"],
                gcn_only=counts["gcn_only"],
                gin_only=counts["gin_only"],
                both_wrong=counts["both_wrong"],
            ),
        )
    return summary


def _write_per_graph_csv(rows: Sequence[GraphPairOutcome], path: Path) -> None:
    """Write per-graph pairwise CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "lr_tag",
        "seed",
        "graph_idx",
        "tau",
        "label",
        "pred_gcn",
        "pred_gin",
        "correct_gcn",
        "correct_gin",
        "outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "track": r.track,
                    "lr_tag": r.lr_tag,
                    "seed": r.seed,
                    "graph_idx": r.graph_idx,
                    "tau": r.tau,
                    "label": r.label,
                    "pred_gcn": r.pred_gcn,
                    "pred_gin": r.pred_gin,
                    "correct_gcn": int(r.correct_gcn),
                    "correct_gin": int(r.correct_gin),
                    "outcome": r.outcome,
                },
            )


def _write_summary_csv(summary: Sequence[PairwiseSummaryRow], path: Path) -> None:
    """Write aggregated summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "lr_tag",
        "seed",
        "tau",
        "n_graphs",
        "both_correct",
        "gcn_only",
        "gin_only",
        "both_wrong",
        "frac_gcn_only_correct",
        "frac_gin_only_correct",
        "frac_specialist_wins",
        "all_graphs_specialist_wins",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary:
            specialist = s.gcn_only if s.tau == 0 else s.gin_only
            counter = s.gin_only if s.tau == 0 else s.gcn_only
            writer.writerow(
                {
                    "track": s.track,
                    "lr_tag": s.lr_tag,
                    "seed": s.seed,
                    "tau": s.tau,
                    "n_graphs": s.n_graphs,
                    "both_correct": s.both_correct,
                    "gcn_only": s.gcn_only,
                    "gin_only": s.gin_only,
                    "both_wrong": s.both_wrong,
                    "frac_gcn_only_correct": s.gcn_only / s.n_graphs,
                    "frac_gin_only_correct": s.gin_only / s.n_graphs,
                    "frac_specialist_wins": specialist / s.n_graphs,
                    "all_graphs_specialist_wins": int(
                        specialist == s.n_graphs and counter == 0 and s.both_wrong == 0,
                    ),
                },
            )


def _mean_summary_across_seeds(
    summary: Sequence[PairwiseSummaryRow],
) -> dict[tuple[str, int], dict[str, float]]:
    """Mean fractions across seeds for each (track, tau)."""
    grouped: dict[tuple[str, int], list[PairwiseSummaryRow]] = {}
    for s in summary:
        grouped.setdefault((s.track, s.tau), []).append(s)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        n = len(rows)
        out[key] = {
            "both_correct": float(np.mean([r.both_correct / r.n_graphs for r in rows])),
            "gcn_only": float(np.mean([r.gcn_only / r.n_graphs for r in rows])),
            "gin_only": float(np.mean([r.gin_only / r.n_graphs for r in rows])),
            "both_wrong": float(np.mean([r.both_wrong / r.n_graphs for r in rows])),
        }
    return out


def _plot_pairwise_comparison(
    summary: Sequence[PairwiseSummaryRow],
    out_path: Path,
    *,
    lr_tag: str,
    dpi: int,
) -> None:
    """Stacked bar chart of 4-way pairwise outcomes by track and τ."""
    means = _mean_summary_across_seeds(summary)
    tracks = sorted({s.track for s in summary})
    tau_labels = {0: r"$\tau{=}0$ (GCN-type)", 1: r"$\tau{=}1$ (GIN-type)"}
    colors = {
        "both_correct": "#B8B8B8",
        "gcn_only": "#4C72B0",
        "gin_only": "#55A868",
        "both_wrong": "#C44E52",
    }
    labels = {
        "both_correct": "Both correct",
        "gcn_only": "Only GCN correct",
        "gin_only": "Only GIN correct",
        "both_wrong": "Both wrong",
    }
    order: tuple[OutcomeKind, ...] = ("both_correct", "gcn_only", "gin_only", "both_wrong")

    fig, axes = plt.subplots(1, len(tracks), figsize=(5.5 * len(tracks), 5.0), squeeze=False)
    x = np.array([0, 1])
    bar_w = 0.55

    for ax, track in zip(axes[0], tracks, strict=True):
        bottom = np.zeros(2)
        for kind in order:
            vals = np.array([means.get((track, tau), {}).get(kind, 0.0) for tau in (0, 1)])
            ax.bar(
                x,
                vals,
                bar_w,
                bottom=bottom,
                label=labels[kind],
                color=colors[kind],
                edgecolor="white",
                linewidth=0.5,
            )
            for i, v in enumerate(vals):
                if v >= 0.04:
                    ax.text(
                        x[i],
                        bottom[i] + v / 2,
                        f"{100 * v:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if kind != "both_correct" else "black",
                        fontweight="bold",
                    )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([tau_labels[t] for t in (0, 1)])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction of test graphs")
        ax.set_title("Toy (routing convs)" if track == "toy" else "Sigma (PyG GIN/GCN)")
        ax.grid(axis="y", alpha=0.22)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        f"GCN-only vs GIN-only per-graph outcomes (5-seed mean, {lr_tag}, test)",
        y=1.08,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_summary_csv(path: Path) -> list[PairwiseSummaryRow]:
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


def plot_from_summary_csv(
    summary_path: Path,
    out_path: Path,
    *,
    lr_tag: str = "lr001",
    dpi: int = 160,
) -> None:
    """Regenerate fig05 from an existing summary CSV."""
    summary = [r for r in _load_summary_csv(summary_path) if r.lr_tag == lr_tag]
    if not summary:
        raise FileNotFoundError(f"No rows for lr_tag={lr_tag} in {summary_path}")
    _plot_pairwise_comparison(summary, out_path, lr_tag=lr_tag, dpi=dpi)


def _print_report(summary: Sequence[PairwiseSummaryRow]) -> None:
    """Print human-readable answers to specialist-dominance questions."""
    means = _mean_summary_across_seeds(summary)
    print("\n=== Pairwise GCN-only vs GIN-only (5-seed mean fractions) ===\n")
    for track in sorted({s.track for s in summary}):
        print(f"Track: {track}")
        for tau in (0, 1):
            m = means.get((track, tau), {})
            specialist = "GCN" if tau == 0 else "GIN"
            counter = "GIN" if tau == 0 else "GCN"
            print(
                f"  τ={tau}: both correct {100*m.get('both_correct', 0):.1f}% | "
                f"only {specialist} {100*m.get('gcn_only' if tau == 0 else 'gin_only', 0):.1f}% | "
                f"only {counter} {100*m.get('gin_only' if tau == 0 else 'gcn_only', 0):.1f}% | "
                f"both wrong {100*m.get('both_wrong', 0):.1f}%",
            )
            all_specialist = all(
                (s.gcn_only if tau == 0 else s.gin_only) == s.n_graphs
                and (s.gin_only if tau == 0 else s.gcn_only) == 0
                and s.both_wrong == 0
                for s in summary
                if s.track == track and s.tau == tau
            )
            print(
                f"         → ALL graphs favor {specialist}-only? {'YES' if all_specialist else 'NO'}",
            )
        print()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run pairwise comparison and write outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    device = _select_device(args.device)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_rows: list[GraphPairOutcome] = []
    for track in tracks:
        for seed in seeds:
            logging.info("Comparing track=%s seed=%d lr=%s", track, seed, args.lr_tag)
            all_rows.extend(
                _compare_pair(
                    track,
                    args.lr_tag,
                    seed,
                    results_root,
                    args.dataset_dir,
                    device,
                ),
            )

    summary = _summarize(all_rows)
    per_graph_path = out_dir / "pairwise_baseline_per_graph.csv"
    summary_path = out_dir / "pairwise_baseline_summary.csv"
    fig_path = out_dir / "paper_figures" / "fig05_pairwise_baseline_comparison.png"

    _write_per_graph_csv(all_rows, per_graph_path)
    _write_summary_csv(summary, summary_path)
    _plot_pairwise_comparison(summary, fig_path, lr_tag=args.lr_tag, dpi=args.dpi)
    _print_report(summary)

    print(f"Wrote {per_graph_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
