#!/usr/bin/env python3
"""Pair-level analysis of opposite-sign τ twins in the GCN/GIN routing benchmark.

Opposite-sign pairs share identical neighbor topology and features; only the root
type bit τ (and thus the labeling rule) differs. For each matched pair we report
whether a fixed model is correct on **both** members, only the τ=0 member, only the
τ=1 member, or neither — the direct test of the rebuttal claim that no single
fixed operator succeeds on both graphs in a pair.

Inputs (one of):
  - ``--from-per-graph-csv`` + existing ``pairwise_baseline_per_graph.csv`` (fast)
  - ``--results-root`` + checkpoints (full eval, optional gated hybrid)

Always loads test-split metadata from ``--dataset-dir/GcnGinRouting``.

Outputs (under ``--out-dir``):
  - ``opposite_sign_pair_per_pair.csv``
  - ``opposite_sign_pair_summary.csv``
  - ``paper_figures/fig07_opposite_sign_pair_outcomes.png`` / ``.pdf``
  - ``paper_figures/fig07_opposite_sign_pair_table.png`` / ``.pdf``

Example:
  python scripts/synthetic/analyze_opposite_sign_pairs.py \\
    --dataset-dir results/gcn_gin_routing/data \\
    --from-per-graph-csv results/gcn_gin_routing/analysis/pairwise_baseline_per_graph.csv \\
    --out-dir results/gcn_gin_routing/analysis
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PairOutcome = Literal["both_correct", "only_tau0", "only_tau1", "both_wrong"]
ModelKind = Literal[
    "oracle_gcn_rule",
    "oracle_gin_rule",
    "gcn_only",
    "gin_only",
    "gated",
    "ungated",
]

_SIGMA_MODEL_SLUGS: dict[ModelKind, str] = {
    "gated": "a0g2_gated",
    "ungated": "a0g2_ungated",
}


@dataclass(frozen=True)
class TestGraphMeta:
    """Metadata for one test graph (loader order)."""

    graph_idx: int
    tau: int
    label: int
    gcn_score: float
    gin_score: float
    difficulty: str
    pair_id: Optional[int]


@dataclass(frozen=True)
class OppositeSignPair:
    """Matched τ=0 / τ=1 twin with shared neighbor structure."""

    pair_key: str
    pair_id: Optional[int]
    tau0_idx: int
    tau1_idx: int
    label_tau0: int
    label_tau1: int
    gcn_score: float
    gin_score: float


@dataclass(frozen=True)
class PairEvalRow:
    """Per-pair correctness for one model and seed."""

    track: str
    lr_tag: str
    seed: int
    pair_key: str
    pair_id: Optional[int]
    model: ModelKind
    correct_tau0: bool
    correct_tau1: bool
    outcome: PairOutcome


@dataclass(frozen=True)
class PairSummaryRow:
    """Aggregated pair-outcome counts."""

    track: str
    lr_tag: str
    seed: int
    model: ModelKind
    n_pairs: int
    both_correct: int
    only_tau0: int
    only_tau1: int
    both_wrong: int


def _load_dataset_class() -> type:
    """Load ``GcnGinRoutingDataset`` without importing full ``GNNPlus``."""
    module_path = _REPO_ROOT / "GNNPlus" / "loader" / "dataset" / "gcn_gin_routing.py"
    spec = importlib.util.spec_from_file_location("gcn_gin_routing_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GcnGinRoutingDataset  # type: ignore[no-any-return]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="results/gcn_gin_routing/data",
        help="Parent of GcnGinRouting/.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gcn_gin_routing/analysis",
        help="Directory for CSVs and figures.",
    )
    parser.add_argument(
        "--from-per-graph-csv",
        type=str,
        default=None,
        help="Use existing pairwise_baseline_per_graph.csv (no GPU).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Parent of toy/ and sigma/ for checkpoint eval (optional gated).",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Evaluate SiGMA gated + ungated (a0g2_*); requires --results-root.",
    )
    parser.add_argument(
        "--include-ungated",
        action="store_true",
        help="Evaluate SiGMA ungated only (with --include-gated for both).",
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
        help="Learning-rate tag filter.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for oracle / GCN / GIN (from pairwise CSV).",
    )
    parser.add_argument(
        "--sigma-seeds",
        type=str,
        default="0,2,3,4,5",
        help="Comma-separated seeds for SiGMA gated/ungated checkpoint eval.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device when loading checkpoints.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args(argv)


def _classify_pair_outcome(correct_tau0: bool, correct_tau1: bool) -> PairOutcome:
    """Map member-level correctness to a pair-level outcome."""
    if correct_tau0 and correct_tau1:
        return "both_correct"
    if correct_tau0 and not correct_tau1:
        return "only_tau0"
    if not correct_tau0 and correct_tau1:
        return "only_tau1"
    return "both_wrong"


def _oracle_rule_correct(pair: OppositeSignPair, *, rule: Literal["gcn", "gin"], tau_member: Literal[0, 1]) -> bool:
    """Whether a fixed labeling rule matches the ground truth on one pair member."""
    if rule == "gcn":
        pred = int(pair.gcn_score > 0.0)
    else:
        pred = int(pair.gin_score > 0.0)
    label = pair.label_tau0 if tau_member == 0 else pair.label_tau1
    return pred == label


def load_test_graph_meta(dataset_dir: str) -> list[TestGraphMeta]:
    """Load per-graph metadata for the test split in loader order."""
    dataset_root = Path(dataset_dir) / "GcnGinRouting"
    dataset_cls = _load_dataset_class()
    dataset = dataset_cls(str(dataset_root), split="test")  # type: ignore[arg-type]

    records: list[TestGraphMeta] = []
    for graph_idx in range(len(dataset)):
        data = dataset[graph_idx]
        pair_id: Optional[int] = None
        if hasattr(data, "pair_id") and data.pair_id is not None:
            raw_pair_id = int(data.pair_id.view(-1)[0].item())
            pair_id = raw_pair_id if raw_pair_id >= 0 else None
        records.append(
            TestGraphMeta(
                graph_idx=graph_idx,
                tau=int(data.tau.view(-1)[0].item()),
                label=int(data.y.view(-1)[0].item()),
                gcn_score=float(data.gcn_score.view(-1)[0].item()),
                gin_score=float(data.gin_score.view(-1)[0].item()),
                difficulty=str(data.difficulty),
                pair_id=pair_id,
            ),
        )
    return records


def build_opposite_sign_pairs(meta: Sequence[TestGraphMeta]) -> list[OppositeSignPair]:
    """Group opposite-sign test graphs into τ=0 / τ=1 pairs."""
    by_pair_id: dict[int, list[TestGraphMeta]] = defaultdict(list)
    by_score: dict[tuple[float, float], list[TestGraphMeta]] = defaultdict(list)

    for record in meta:
        if record.difficulty != "opposite_sign":
            continue
        if record.pair_id is not None:
            by_pair_id[record.pair_id].append(record)
        else:
            key = (round(record.gcn_score, 8), round(record.gin_score, 8))
            by_score[key].append(record)

    pairs: list[OppositeSignPair] = []
    if by_pair_id:
        for pair_id, members in sorted(by_pair_id.items()):
            pairs.append(_pair_from_members(f"pair_id={pair_id}", pair_id, members))
    else:
        for score_key, members in sorted(by_score.items(), key=lambda kv: kv[0]):
            pairs.append(
                _pair_from_members(
                    f"scores={score_key[0]:.6f},{score_key[1]:.6f}",
                    None,
                    members,
                ),
            )
    return pairs


def _pair_from_members(
    pair_key: str,
    pair_id: Optional[int],
    members: Sequence[TestGraphMeta],
) -> OppositeSignPair:
    """Validate and build one opposite-sign pair."""
    if len(members) != 2:
        raise ValueError(f"Expected 2 graphs for {pair_key}, got {len(members)}")
    tau0 = next((m for m in members if m.tau == 0), None)
    tau1 = next((m for m in members if m.tau == 1), None)
    if tau0 is None or tau1 is None:
        raise ValueError(f"Pair {pair_key} missing tau=0 or tau=1 member")
    if tau0.label == tau1.label:
        raise ValueError(
            f"Pair {pair_key} has same label on both members "
            f"({tau0.label}); expected opposite labels.",
        )
    return OppositeSignPair(
        pair_key=pair_key,
        pair_id=pair_id,
        tau0_idx=tau0.graph_idx,
        tau1_idx=tau1.graph_idx,
        label_tau0=tau0.label,
        label_tau1=tau1.label,
        gcn_score=tau0.gcn_score,
        gin_score=tau0.gin_score,
    )


def _load_per_graph_csv(path: Path) -> dict[tuple[str, str, int, int], dict[str, int]]:
    """Load pairwise per-graph predictions keyed by (track, lr, seed, graph_idx)."""
    lookup: dict[tuple[str, str, int, int], dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            key = (
                raw["track"],
                raw["lr_tag"],
                int(raw["seed"]),
                int(raw["graph_idx"]),
            )
            lookup[key] = {
                "pred_gcn": int(raw["pred_gcn"]),
                "pred_gin": int(raw["pred_gin"]),
                "label": int(raw["label"]),
                "tau": int(raw["tau"]),
            }
    return lookup


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_sigma_predictions(
    track: str,
    lr_tag: str,
    seed: int,
    results_root: Path,
    dataset_dir: str,
    device: torch.device,
    *,
    model: ModelKind,
) -> Optional[list[int]]:
    """Return SiGMA hybrid predictions per test graph (loader order)."""
    if model not in _SIGMA_MODEL_SLUGS:
        raise ValueError(f"Not a SiGMA model kind: {model}")
    import GNNPlus  # noqa: F401
    from scripts.synthetic.analyze_gcn_gin_routing_results import (  # noqa: E402
        RunRef,
        _load_cfg_for_run,
        _pick_best_epoch,
        _pred_labels_from_score,
    )
    from torch_geometric.graphgym.checkpoint import load_ckpt
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.loss import compute_loss
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.device import auto_select_device
    from torch_geometric import seed_everything

    model_slug = _SIGMA_MODEL_SLUGS[model]
    run_dir = results_root / track / f"{model_slug}_{lr_tag}_seed{seed}"
    run_ref = RunRef(
        track=track,
        run_dir=run_dir,
        model=model_slug,
        lr_tag=lr_tag,
        seed=seed,
    )
    if not run_dir.is_dir():
        logging.warning("Missing SiGMA run dir: %s", run_dir)
        return None

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

    preds: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred, true = model(batch)
            _loss, pred_score = compute_loss(pred, true)
            pred_label = _pred_labels_from_score(pred_score).view(-1).long()
            preds.extend(int(x) for x in pred_label.tolist())
    return preds


def _evaluate_pairs_for_model(
    *,
    track: str,
    lr_tag: str,
    seed: int,
    pairs: Sequence[OppositeSignPair],
    model: ModelKind,
    per_graph_lookup: Optional[dict[tuple[str, str, int, int], dict[str, int]]],
    sigma_preds: Optional[Sequence[int]],
) -> list[PairEvalRow]:
    """Evaluate one model on all opposite-sign pairs."""
    rows: list[PairEvalRow] = []
    for pair in pairs:
        if model in ("oracle_gcn_rule", "oracle_gin_rule"):
            rule = "gcn" if model == "oracle_gcn_rule" else "gin"
            correct_tau0 = _oracle_rule_correct(pair, rule=rule, tau_member=0)
            correct_tau1 = _oracle_rule_correct(pair, rule=rule, tau_member=1)
        elif model in ("gated", "ungated"):
            if sigma_preds is None:
                raise ValueError(f"sigma_preds required for {model}")
            pred_tau0 = sigma_preds[pair.tau0_idx]
            pred_tau1 = sigma_preds[pair.tau1_idx]
            correct_tau0 = pred_tau0 == pair.label_tau0
            correct_tau1 = pred_tau1 == pair.label_tau1
        else:
            if per_graph_lookup is None:
                raise ValueError("per_graph_lookup required for baseline models")
            g0 = per_graph_lookup[(track, lr_tag, seed, pair.tau0_idx)]
            g1 = per_graph_lookup[(track, lr_tag, seed, pair.tau1_idx)]
            pred_key = "pred_gcn" if model == "gcn_only" else "pred_gin"
            correct_tau0 = g0[pred_key] == pair.label_tau0
            correct_tau1 = g1[pred_key] == pair.label_tau1

        rows.append(
            PairEvalRow(
                track=track,
                lr_tag=lr_tag,
                seed=seed,
                pair_key=pair.pair_key,
                pair_id=pair.pair_id,
                model=model,
                correct_tau0=correct_tau0,
                correct_tau1=correct_tau1,
                outcome=_classify_pair_outcome(correct_tau0, correct_tau1),
            ),
        )
    return rows


def _summarize_pair_rows(rows: Sequence[PairEvalRow]) -> list[PairSummaryRow]:
    """Aggregate pair-level rows into counts."""
    keys = {(r.track, r.lr_tag, r.seed, r.model) for r in rows}
    summary: list[PairSummaryRow] = []
    for track, lr_tag, seed, model in sorted(keys):
        subset = [
            r
            for r in rows
            if (r.track, r.lr_tag, r.seed, r.model) == (track, lr_tag, seed, model)
        ]
        counts = {
            "both_correct": sum(1 for r in subset if r.outcome == "both_correct"),
            "only_tau0": sum(1 for r in subset if r.outcome == "only_tau0"),
            "only_tau1": sum(1 for r in subset if r.outcome == "only_tau1"),
            "both_wrong": sum(1 for r in subset if r.outcome == "both_wrong"),
        }
        summary.append(
            PairSummaryRow(
                track=track,
                lr_tag=lr_tag,
                seed=seed,
                model=model,  # type: ignore[arg-type]
                n_pairs=len(subset),
                both_correct=counts["both_correct"],
                only_tau0=counts["only_tau0"],
                only_tau1=counts["only_tau1"],
                both_wrong=counts["both_wrong"],
            ),
        )
    return summary


def _write_per_pair_csv(rows: Sequence[PairEvalRow], path: Path) -> None:
    """Write per-pair evaluation CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "lr_tag",
        "seed",
        "pair_key",
        "pair_id",
        "model",
        "correct_tau0",
        "correct_tau1",
        "outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "track": row.track,
                    "lr_tag": row.lr_tag,
                    "seed": row.seed,
                    "pair_key": row.pair_key,
                    "pair_id": "" if row.pair_id is None else row.pair_id,
                    "model": row.model,
                    "correct_tau0": int(row.correct_tau0),
                    "correct_tau1": int(row.correct_tau1),
                    "outcome": row.outcome,
                },
            )


def _write_summary_csv(summary: Sequence[PairSummaryRow], path: Path) -> None:
    """Write aggregated summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "lr_tag",
        "seed",
        "model",
        "n_pairs",
        "both_correct",
        "only_tau0",
        "only_tau1",
        "both_wrong",
        "frac_both_correct",
        "frac_only_tau0",
        "frac_only_tau1",
        "frac_both_wrong",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            n = row.n_pairs
            writer.writerow(
                {
                    "track": row.track,
                    "lr_tag": row.lr_tag,
                    "seed": row.seed,
                    "model": row.model,
                    "n_pairs": n,
                    "both_correct": row.both_correct,
                    "only_tau0": row.only_tau0,
                    "only_tau1": row.only_tau1,
                    "both_wrong": row.both_wrong,
                    "frac_both_correct": row.both_correct / n,
                    "frac_only_tau0": row.only_tau0 / n,
                    "frac_only_tau1": row.only_tau1 / n,
                    "frac_both_wrong": row.both_wrong / n,
                },
            )


def _mean_summary_fractions(
    summary: Sequence[PairSummaryRow],
) -> dict[tuple[str, ModelKind], dict[str, float]]:
    """Mean pair-outcome fractions across seeds."""
    grouped: dict[tuple[str, ModelKind], list[PairSummaryRow]] = defaultdict(list)
    for row in summary:
        grouped[(row.track, row.model)].append(row)

    out: dict[tuple[str, ModelKind], dict[str, float]] = {}
    for key, rows in grouped.items():
        out[key] = {
            "both_correct": mean(r.both_correct / r.n_pairs for r in rows),
            "only_tau0": mean(r.only_tau0 / r.n_pairs for r in rows),
            "only_tau1": mean(r.only_tau1 / r.n_pairs for r in rows),
            "both_wrong": mean(r.both_wrong / r.n_pairs for r in rows),
        }
    return out


_MODEL_LABELS: dict[ModelKind, str] = {
    "oracle_gcn_rule": "Oracle GCN rule",
    "oracle_gin_rule": "Oracle GIN rule",
    "gcn_only": "GCN-only (trained)",
    "gin_only": "GIN-only (trained)",
    "gated": "SiGMA",
    "ungated": "SiGMA ungated",
}

_MODEL_ORDER: tuple[ModelKind, ...] = (
    "oracle_gcn_rule",
    "oracle_gin_rule",
    "gcn_only",
    "gin_only",
    "gated",
    "ungated",
)

TRACK_ORDER: tuple[str, ...] = ("toy", "sigma")
TRACK_LABELS: dict[str, str] = {
    "toy": r"Track A (Toy, $d_h{=}1$)",
    "sigma": r"Track B (SiGMA, PyG GIN/GCN, $d_h{=}4$)",
}

_OUTCOME_COLORS = {
    "both_correct": "#55A868",
    "only_tau0": "#4C72B0",
    "only_tau1": "#8172B2",
    "both_wrong": "#C44E52",
}

_OUTCOME_LABELS = {
    "both_correct": "Both members correct",
    "only_tau0": "Only τ=0 member",
    "only_tau1": "Only τ=1 member",
    "both_wrong": "Both wrong",
}


def _plot_pair_outcomes(
    summary: Sequence[PairSummaryRow],
    out_path: Path,
    *,
    lr_tag: str,
    dpi: int,
) -> None:
    """Stacked bar chart of pair-level outcomes by track and model."""
    means = _mean_summary_fractions(summary)
    summary_tracks = {s.track for s in summary}
    tracks = [t for t in TRACK_ORDER if t in summary_tracks]
    tracks.extend(sorted(summary_tracks - set(tracks)))
    models = [m for m in _MODEL_ORDER if any((t, m) in means for t in tracks)]
    if not models:
        raise ValueError("No summary rows to plot.")

    fig_w = max(9.0, 1.35 * len(models))
    fig, axes = plt.subplots(len(tracks), 1, figsize=(fig_w, 5.0 * len(tracks)), squeeze=False)
    x = np.arange(len(models))
    bar_w = 0.62
    order: tuple[PairOutcome, ...] = ("both_correct", "only_tau0", "only_tau1", "both_wrong")

    for ax, track in zip(axes[:, 0], tracks, strict=True):
        bottom = np.zeros(len(models))
        for outcome in order:
            vals = np.array([means.get((track, model), {}).get(outcome, 0.0) for model in models])
            ax.bar(
                x,
                vals,
                bar_w,
                bottom=bottom,
                label=_OUTCOME_LABELS[outcome],
                color=_OUTCOME_COLORS[outcome],
                edgecolor="white",
                linewidth=0.5,
            )
            for i, val in enumerate(vals):
                if val >= 0.05:
                    ax.text(
                        x[i],
                        bottom[i] + val / 2,
                        f"{100 * val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if outcome != "both_correct" else "black",
                        fontweight="bold",
                    )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([_MODEL_LABELS[m] for m in models], rotation=28, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Fraction of opposite-sign pairs")
        ax.set_title(TRACK_LABELS.get(track, track))
        ax.grid(axis="y", alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        f"Opposite-sign τ pairs: per-pair outcomes (5-seed mean, {lr_tag}, n≈228 pairs/test)",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _print_report(
    pairs: Sequence[OppositeSignPair],
    summary: Sequence[PairSummaryRow],
) -> None:
    """Print human-readable verification summary."""
    means = _mean_summary_fractions(summary)
    print(f"\n=== Opposite-sign pairs (n={len(pairs)} per track, test split) ===\n")
    for track in [t for t in TRACK_ORDER if t in {s.track for s in summary}]:
        print(f"Track: {track}")
        for model in _MODEL_ORDER:
            m = means.get((track, model))
            if m is None:
                continue
            print(
                f"  {_MODEL_LABELS[model]:22s}  "
                f"both={100 * m['both_correct']:.1f}%  "
                f"τ0 only={100 * m['only_tau0']:.1f}%  "
                f"τ1 only={100 * m['only_tau1']:.1f}%  "
                f"both wrong={100 * m['both_wrong']:.1f}%",
            )
        print()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run opposite-sign pair analysis."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    sigma_seeds = [int(s.strip()) for s in args.sigma_seeds.split(",") if s.strip()]
    eval_seeds = sorted(set(seeds) | set(sigma_seeds))
    device = _select_device(args.device)

    meta = load_test_graph_meta(args.dataset_dir)
    pairs = build_opposite_sign_pairs(meta)
    logging.info("Found %d opposite-sign pairs in test split", len(pairs))

    per_graph_lookup: Optional[dict[tuple[str, str, int, int], dict[str, int]]] = None
    if args.from_per_graph_csv:
        per_graph_path = Path(args.from_per_graph_csv)
        if not per_graph_path.is_file():
            raise FileNotFoundError(per_graph_path)
        per_graph_lookup = _load_per_graph_csv(per_graph_path)
        logging.info("Loaded per-graph predictions from %s", per_graph_path)
    elif args.results_root is None:
        logging.warning(
            "No --from-per-graph-csv or --results-root: skipping trained GCN/GIN baselines.",
        )

    baseline_models: tuple[ModelKind, ...] = ("gcn_only", "gin_only")
    if per_graph_lookup is None:
        baseline_models = ()

    sigma_models: tuple[ModelKind, ...] = ()
    if args.include_gated or args.include_ungated:
        if args.results_root is None:
            raise ValueError("SiGMA eval requires --results-root")
        if args.include_gated:
            sigma_models = ("gated", "ungated")
        elif args.include_ungated:
            sigma_models = ("ungated",)

    all_rows: list[PairEvalRow] = []
    for track in tracks:
        for seed in eval_seeds:
            sigma_preds_by_model: dict[ModelKind, list[int]] = {}
            if seed in sigma_seeds:
                for sigma_model in sigma_models:
                    logging.info(
                        "Evaluating SiGMA %s track=%s seed=%d",
                        sigma_model,
                        track,
                        seed,
                    )
                    preds = _collect_sigma_predictions(
                        track,
                        args.lr_tag,
                        seed,
                        Path(args.results_root),  # type: ignore[arg-type]
                        args.dataset_dir,
                        device,
                        model=sigma_model,
                    )
                    if preds is not None:
                        sigma_preds_by_model[sigma_model] = preds

            if seed in seeds:
                for model in ("oracle_gcn_rule", "oracle_gin_rule", *baseline_models):
                    all_rows.extend(
                        _evaluate_pairs_for_model(
                            track=track,
                            lr_tag=args.lr_tag,
                            seed=seed,
                            pairs=pairs,
                            model=model,  # type: ignore[arg-type]
                            per_graph_lookup=per_graph_lookup,
                            sigma_preds=None,
                        ),
                    )

            for sigma_model, preds in sigma_preds_by_model.items():
                all_rows.extend(
                    _evaluate_pairs_for_model(
                        track=track,
                        lr_tag=args.lr_tag,
                        seed=seed,
                        pairs=pairs,
                        model=sigma_model,
                        per_graph_lookup=per_graph_lookup,
                        sigma_preds=preds,
                    ),
                )

    summary = _summarize_pair_rows(all_rows)
    per_pair_path = out_dir / "opposite_sign_pair_per_pair.csv"
    summary_path = out_dir / "opposite_sign_pair_summary.csv"
    fig_path = out_dir / "paper_figures" / "fig07_opposite_sign_pair_outcomes.png"

    _write_per_pair_csv(all_rows, per_pair_path)
    _write_summary_csv(summary, summary_path)
    _plot_pair_outcomes(summary, fig_path, lr_tag=args.lr_tag, dpi=args.dpi)
    _print_report(pairs, summary)

    from scripts.synthetic.gcn_gin_routing_table_figures import (  # noqa: WPS433
        plot_opposite_sign_pair_table,
    )

    table_path = out_dir / "paper_figures" / "fig07_opposite_sign_pair_table.png"
    plot_opposite_sign_pair_table(summary_path, table_path, lr_tag=args.lr_tag, dpi=args.dpi)

    print(f"Wrote {per_pair_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
