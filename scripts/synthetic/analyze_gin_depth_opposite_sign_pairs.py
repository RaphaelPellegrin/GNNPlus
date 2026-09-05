#!/usr/bin/env python3
"""Opposite-sign τ-twin analysis for GIN depth-routing.

Opposite-sign pairs share identical trees and features; only τ (and thus the
labeling rule) differs. Oracles:

  - ``oracle_s1_rule``: always predict ``1[S1 > 0]`` (shallow / 1-GIN)
  - ``oracle_s2_rule``: always predict ``1[S2 > 0]`` (deep / 2-GIN)

A fixed depth rule cannot be correct on both members of an opposite-sign pair.
Trained SiGMA gated / ungated should approach ``both_correct`` if they route by τ.

Outputs under ``--out-dir``:
  - ``opposite_sign_pair_per_pair.csv``
  - ``opposite_sign_pair_summary.csv``
  - ``paper_figures/fig_opposite_sign_pair_outcomes.png`` / ``.pdf``
  - ``paper_figures/fig_opposite_sign_pair_table.png`` / ``.pdf``

Example::

  python scripts/synthetic/analyze_gin_depth_opposite_sign_pairs.py \\
    --dataset-dir $GNNPLUS_DATASET_DIR \\
    --results-root $GNNPLUS_OUT_DIR/gin_routing_depth \\
    --out-dir results/gin_routing_depth/analysis
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
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
ModelKind = Literal["oracle_s1_rule", "oracle_s2_rule", "gated", "ungated"]

_MODEL_SLUGS: dict[ModelKind, str] = {
    "gated": "l2_a0g1_gated",
    "ungated": "l2_a0g1_ungated",
}
_MODEL_LABELS: dict[ModelKind, str] = {
    "oracle_s1_rule": r"Oracle $S_1$ (1-GIN)",
    "oracle_s2_rule": r"Oracle $S_2$ (2-GIN)",
    "gated": "SiGMA gated",
    "ungated": "SiGMA ungated",
}
_OUTCOME_ORDER: tuple[PairOutcome, ...] = (
    "both_correct",
    "only_tau0",
    "only_tau1",
    "both_wrong",
)
_OUTCOME_COLORS: dict[PairOutcome, str] = {
    "both_correct": "#55A868",
    "only_tau0": "#4C72B0",
    "only_tau1": "#DD8452",
    "both_wrong": "#C44E52",
}
_OUTCOME_LABELS: dict[PairOutcome, str] = {
    "both_correct": "Both correct",
    "only_tau0": r"Only $\tau{=}0$",
    "only_tau1": r"Only $\tau{=}1$",
    "both_wrong": "Both wrong",
}


@dataclass(frozen=True)
class TestGraphMeta:
    """Metadata for one test graph (loader order)."""

    graph_idx: int
    tau: int
    label: int
    s1_score: float
    s2_score: float
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
    s1_score: float
    s2_score: float


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
    """Load ``GinDepthRoutingDataset`` without importing full ``GNNPlus``."""
    module_path = (
        _REPO_ROOT / "GNNPlus" / "loader" / "dataset" / "gin_depth_routing.py"
    )
    spec = importlib.util.spec_from_file_location("gin_depth_routing_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GinDepthRoutingDataset  # type: ignore[no-any-return]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Parent of toy/ for gated/ungated checkpoint eval.",
    )
    parser.add_argument("--tracks", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Seeds for oracle rows (and gated/ungated when present).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--dpi", type=int, default=160)
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


def _oracle_rule_correct(
    pair: OppositeSignPair,
    *,
    rule: Literal["s1", "s2"],
    tau_member: Literal[0, 1],
) -> bool:
    """Whether a fixed depth rule matches the ground truth on one pair member."""
    pred = int(pair.s1_score > 0.0) if rule == "s1" else int(pair.s2_score > 0.0)
    label = pair.label_tau0 if tau_member == 0 else pair.label_tau1
    return pred == label


def load_test_graph_meta(dataset_dir: str) -> list[TestGraphMeta]:
    """Load per-graph metadata for the test split in loader order."""
    dataset_root = Path(dataset_dir) / "GinDepthRouting"
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
                s1_score=float(data.s1_score.view(-1)[0].item()),
                s2_score=float(data.s2_score.view(-1)[0].item()),
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
            key = (round(record.s1_score, 8), round(record.s2_score, 8))
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
        s1_score=tau0.s1_score,
        s2_score=tau0.s2_score,
    )


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_model_predictions(
    track: str,
    lr_tag: str,
    seed: int,
    results_root: Path,
    dataset_dir: str,
    device: torch.device,
    *,
    model: ModelKind,
) -> Optional[list[int]]:
    """Return SiGMA predictions per test graph (loader order)."""
    if model not in _MODEL_SLUGS:
        raise ValueError(f"Not a trainable model kind: {model}")
    import GNNPlus  # noqa: F401
    from torch_geometric.graphgym.checkpoint import load_ckpt
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.loss import compute_loss
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.device import auto_select_device
    from torch_geometric import seed_everything

    from scripts.synthetic.analyze_gin_depth_routing_results import (  # noqa: E402
        RunRef,
        _load_cfg_for_run,
        _pick_best_epoch,
        _pred_labels_from_score,
    )

    slug = _MODEL_SLUGS[model]
    run_dir = results_root / track / f"{slug}_{lr_tag}_seed{seed}"
    if not run_dir.is_dir():
        logging.warning("Missing run dir: %s", run_dir)
        return None
    run_ref = RunRef(
        track=track,
        run_dir=run_dir,
        model=slug,
        lr_tag=lr_tag,
        seed=seed,
    )
    _load_cfg_for_run(run_ref, dataset_dir)
    seed_everything(int(cfg.seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    loaders = create_loader()
    test_loader = loaders[2] if len(loaders) > 2 else None
    if test_loader is None:
        raise RuntimeError("Test loader missing.")

    net = create_model()
    epoch = _pick_best_epoch(run_ref.run_dir)
    load_ckpt(net, optimizer=None, scheduler=None, epoch=epoch)
    net.eval()
    net.to(device)

    preds: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred, true = net(batch)
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
    preds: Optional[Sequence[int]],
) -> list[PairEvalRow]:
    """Evaluate one model on all opposite-sign pairs."""
    rows: list[PairEvalRow] = []
    for pair in pairs:
        if model in ("oracle_s1_rule", "oracle_s2_rule"):
            rule: Literal["s1", "s2"] = "s1" if model == "oracle_s1_rule" else "s2"
            correct_tau0 = _oracle_rule_correct(pair, rule=rule, tau_member=0)
            correct_tau1 = _oracle_rule_correct(pair, rule=rule, tau_member=1)
        else:
            if preds is None:
                raise ValueError(f"preds required for {model}")
            pred_tau0 = preds[pair.tau0_idx]
            pred_tau1 = preds[pair.tau1_idx]
            correct_tau0 = pred_tau0 == pair.label_tau0
            correct_tau1 = pred_tau1 == pair.label_tau1

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
                    "frac_both_correct": row.both_correct / n if n else float("nan"),
                    "frac_only_tau0": row.only_tau0 / n if n else float("nan"),
                    "frac_only_tau1": row.only_tau1 / n if n else float("nan"),
                    "frac_both_wrong": row.both_wrong / n if n else float("nan"),
                },
            )


def _mean_summary_fractions(
    summary: Sequence[PairSummaryRow],
) -> dict[ModelKind, dict[str, float]]:
    """Mean pair-outcome fractions across seeds (single track)."""
    grouped: dict[ModelKind, list[PairSummaryRow]] = defaultdict(list)
    for row in summary:
        grouped[row.model].append(row)
    out: dict[ModelKind, dict[str, float]] = {}
    for model, rows in grouped.items():
        out[model] = {
            "both_correct": mean(r.both_correct / r.n_pairs for r in rows),
            "only_tau0": mean(r.only_tau0 / r.n_pairs for r in rows),
            "only_tau1": mean(r.only_tau1 / r.n_pairs for r in rows),
            "both_wrong": mean(r.both_wrong / r.n_pairs for r in rows),
        }
    return out


def _plot_outcomes(
    summary: Sequence[PairSummaryRow],
    out_path: Path,
    dpi: int,
) -> None:
    """Stacked bar chart of pair outcomes by model."""
    means = _mean_summary_fractions(summary)
    models = [m for m in _MODEL_LABELS if m in means]
    if not models:
        logging.warning("No summary rows for outcome plot.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    for outcome in _OUTCOME_ORDER:
        heights = np.array([means[m][outcome] for m in models])
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=_OUTCOME_COLORS[outcome],
            label=_OUTCOME_LABELS[outcome],
            width=0.65,
        )
        bottoms += heights
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABELS[m] for m in models], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of opposite-sign pairs")
    ax.set_title("GIN depth-routing · opposite-sign pair outcomes")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_table(
    summary: Sequence[PairSummaryRow],
    out_path: Path,
    dpi: int,
) -> None:
    """Numeric table figure of mean outcome fractions."""
    means = _mean_summary_fractions(summary)
    models = [m for m in _MODEL_LABELS if m in means]
    if not models:
        return
    col_labels = [_OUTCOME_LABELS[o] for o in _OUTCOME_ORDER]
    cell_text = [
        [f"{means[m][o]:.3f}" for o in _OUTCOME_ORDER]
        for m in models
    ]
    row_labels = [_MODEL_LABELS[m] for m in models]
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.45 * len(models)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    ax.set_title("Opposite-sign pair outcome fractions (mean over seeds)", pad=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    paper = out_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)

    meta = load_test_graph_meta(args.dataset_dir)
    pairs = build_opposite_sign_pairs(meta)
    if not pairs:
        raise SystemExit("No opposite-sign pairs found in test split.")
    logging.info("Loaded %d opposite-sign pairs from test split", len(pairs))

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = _select_device(args.device)
    all_rows: list[PairEvalRow] = []

    # Oracles are deterministic — one row set per seed (seed used only for grouping).
    for track in tracks:
        for seed in seeds:
            for model in ("oracle_s1_rule", "oracle_s2_rule"):
                all_rows.extend(
                    _evaluate_pairs_for_model(
                        track=track,
                        lr_tag=args.lr_tag,
                        seed=seed,
                        pairs=pairs,
                        model=model,  # type: ignore[arg-type]
                        preds=None,
                    ),
                )

    if args.results_root:
        results_root = Path(args.results_root)
        for track in tracks:
            for seed in seeds:
                for model in ("gated", "ungated"):
                    preds = _collect_model_predictions(
                        track,
                        args.lr_tag,
                        seed,
                        results_root,
                        args.dataset_dir,
                        device,
                        model=model,  # type: ignore[arg-type]
                    )
                    if preds is None:
                        continue
                    all_rows.extend(
                        _evaluate_pairs_for_model(
                            track=track,
                            lr_tag=args.lr_tag,
                            seed=seed,
                            pairs=pairs,
                            model=model,  # type: ignore[arg-type]
                            preds=preds,
                        ),
                    )
    else:
        logging.warning("No --results-root: writing oracle-only analysis.")

    summary = _summarize_pair_rows(all_rows)
    per_pair_path = out_dir / "opposite_sign_pair_per_pair.csv"
    summary_path = out_dir / "opposite_sign_pair_summary.csv"
    _write_per_pair_csv(all_rows, per_pair_path)
    _write_summary_csv(summary, summary_path)
    _plot_outcomes(summary, paper / "fig_opposite_sign_pair_outcomes.png", args.dpi)
    _plot_table(summary, paper / "fig_opposite_sign_pair_table.png", args.dpi)
    # also top-level copies
    _plot_outcomes(summary, out_dir / "fig_opposite_sign_pair_outcomes.png", args.dpi)
    _plot_table(summary, out_dir / "fig_opposite_sign_pair_table.png", args.dpi)

    print(f"Wrote {per_pair_path} ({len(all_rows)} rows)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {paper / 'fig_opposite_sign_pair_outcomes.png'}")


if __name__ == "__main__":
    main()
