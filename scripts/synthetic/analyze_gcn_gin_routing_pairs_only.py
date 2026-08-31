#!/usr/bin/env python3
"""Evaluate GCN/GIN routing models on opposite-sign twin pairs only (test split).

Twin pairs are graphs with ``difficulty == "opposite_sign"``: same neighbor
pattern emitted as both ``tau=0`` and ``tau=1`` with opposite labels. This is
the subset where GCN and GIN rules disagree and routing is required.

Writes under ``<out-dir>/`` (default ``analysis/pairs_only/``):
  - ``per_run_metrics.csv`` / ``summary_by_model.csv``
  - ``pairwise_baseline_summary.csv`` (GCN-only vs GIN-only, pairs only)
  - ``paper_figures/fig01_baseline_per_type_pairs.png``
  - ``paper_figures/fig02_gate_routing_delta_per_seed_pairs.png``
  - ``paper_figures/fig05_pairwise_baseline_comparison_pairs.png``

Example (local gates, lr001):
  python scripts/synthetic/analyze_gcn_gin_routing_pairs_only.py \\
    --results-root results/gcn_gin_routing/gates \\
    --dataset-dir results/gcn_gin_routing/data \\
    --lr-tag lr001
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

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

from scripts.synthetic.analyze_opposite_sign_pairs import (  # noqa: E402
    build_opposite_sign_pairs,
    load_test_graph_meta,
)
from scripts.synthetic.compare_gcn_gin_baselines_per_graph import (  # noqa: E402
    GraphPairOutcome,
    PairwiseSummaryRow,
    _plot_pairwise_comparison,
    _summarize as summarize_pairwise_outcomes,
)
from scripts.synthetic.plot_gcn_gin_routing_paper_figures import (  # noqa: E402
    PerRunRow,
    SummaryRow,
    _apply_style,
    plot_baseline_per_type,
    plot_gate_routing_delta,
)


def _analysis_imports() -> tuple[Any, ...]:
    """Import heavy GNNPlus / eval helpers only when running checkpoint eval."""
    import GNNPlus  # noqa: F401

    from GNNPlus.gcn_gin_routing_gate_tracking import hybrid_head_indices
    from GNNPlus.hybrid_gate_tracking import _unwrap_model
    from scripts.synthetic.analyze_gcn_gin_routing_results import (
        RunMetrics,
        RunRef,
        _load_cfg_for_run,
        _pick_best_epoch,
        _pred_labels_from_score,
        _summarize_runs,
        _write_csv,
        discover_run_refs,
    )

    return (
        RunMetrics,
        RunRef,
        hybrid_head_indices,
        _unwrap_model,
        _load_cfg_for_run,
        _pick_best_epoch,
        _pred_labels_from_score,
        _summarize_runs,
        _write_csv,
        discover_run_refs,
    )


# Type aliases used in annotations before lazy import
RunMetrics = Any
RunRef = Any


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/gcn_gin_routing/gates",
        help="Parent of toy/ and sigma/ run directories.",
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
        default="results/gcn_gin_routing/analysis/pairs_only",
        help="Output directory for CSVs and figures.",
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
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Evaluation device.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from existing pairs_only CSVs (no GPU eval).",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "Aggregate from analysis/pairwise_baseline_per_graph.csv + test metadata "
            "(no checkpoints; GCN/GIN fig01 + fig05 only)."
        ),
    )
    parser.add_argument(
        "--pairwise-csv",
        type=str,
        default="results/gcn_gin_routing/analysis/pairwise_baseline_per_graph.csv",
        help="Per-graph GCN vs GIN CSV (for --from-csv).",
    )
    return parser.parse_args(argv)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pair_graph_indices(dataset_dir: str) -> frozenset[int]:
    """Return test graph indices that belong to opposite-sign twin pairs."""
    meta = load_test_graph_meta(dataset_dir)
    pairs = build_opposite_sign_pairs(meta)
    indices: set[int] = set()
    for pair in pairs:
        indices.add(pair.tau0_idx)
        indices.add(pair.tau1_idx)
    if not indices:
        raise RuntimeError("No opposite-sign pairs found in test split.")
    return frozenset(indices)


@torch.no_grad()
def evaluate_run_on_pairs(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
    pair_indices: frozenset[int],
) -> RunMetrics:
    """Load checkpoint and compute per-type test metrics on twin-pair graphs only."""
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
    gnn_types = str(getattr(hybrid, "gnn_types", "")) if hybrid is not None else ""
    gin_idx, gcn_idx, two_head = hybrid_head_indices(gnn_types)
    gate_mode = str(getattr(hybrid, "gate", "none")).lower() if hybrid is not None else "none"
    collect_gates = two_head and gate_mode not in ("none", "off")

    core = _unwrap_model(model)

    correct_all = 0
    correct_t0 = 0
    correct_t1 = 0
    n_all = 0
    n_t0 = 0
    n_t1 = 0

    gin_t0: list[float] = []
    gin_t1: list[float] = []
    gcn_t0: list[float] = []
    gcn_t1: list[float] = []
    gate_ok = False

    graph_idx = 0
    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing tau.")
        tau = batch.tau.view(-1).long()

        if collect_gates and hasattr(core, "collect_per_graph_gates"):
            try:
                gate_out = core.collect_per_graph_gates(batch.clone())
                gnn_node = gate_out["gnn_node"]
                if gnn_node.ndim == 3 and int(gnn_node.shape[1]) > 0:
                    roots = batch.ptr[:-1].long() if hasattr(batch, "ptr") else None
                    if roots is None:
                        batch_ids = batch.batch
                        num_graphs = int(batch_ids.max().item()) + 1
                        roots = torch.zeros(
                            num_graphs,
                            dtype=torch.long,
                            device=batch_ids.device,
                        )
                        for graph_i in range(num_graphs):
                            mask = batch_ids == graph_i
                            roots[graph_i] = int(torch.nonzero(mask, as_tuple=False)[0].item())
                    gin_root = gnn_node[roots, 0, gin_idx].detach().float().cpu()
                    gcn_root = gnn_node[roots, 0, gcn_idx].detach().float().cpu()
                    tau_cpu = tau.detach().cpu()
                    for local_g in range(int(tau_cpu.numel())):
                        global_idx = graph_idx + local_g
                        if global_idx not in pair_indices:
                            continue
                        tau_val = int(tau_cpu[local_g].item())
                        gin_val = float(gin_root[local_g].item())
                        gcn_val = float(gcn_root[local_g].item())
                        if tau_val == 0:
                            gin_t0.append(gin_val)
                            gcn_t0.append(gcn_val)
                        else:
                            gin_t1.append(gin_val)
                            gcn_t1.append(gcn_val)
                    gate_ok = True
            except Exception:
                logging.exception("Gate collection failed for %s", run_ref.run_dir.name)

        pred, true = model(batch)
        _loss, pred_score = compute_loss(pred, true)
        pred_label = _pred_labels_from_score(pred_score).view(-1).long()
        true_label = true.view(-1).long()
        correct = pred_label == true_label

        for local_g in range(int(correct.numel())):
            global_idx = graph_idx + local_g
            if global_idx not in pair_indices:
                continue
            is_correct = bool(correct[local_g].item())
            tau_val = int(tau[local_g].item())
            correct_all += int(is_correct)
            n_all += 1
            if tau_val == 0:
                n_t0 += 1
                correct_t0 += int(is_correct)
            else:
                n_t1 += 1
                correct_t1 += int(is_correct)

        graph_idx += int(correct.numel())

    def _safe_acc(num: int, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    def _safe_mean(vals: list[float]) -> float:
        return float(mean(vals)) if vals else float("nan")

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
        acc_all=_safe_acc(correct_all, n_all),
        acc_tau0=_safe_acc(correct_t0, n_t0),
        acc_tau1=_safe_acc(correct_t1, n_t1),
        gin_head_idx=gin_idx,
        gcn_head_idx=gcn_idx,
        gin_gate_tau0=_safe_mean(gin_t0),
        gin_gate_tau1=_safe_mean(gin_t1),
        gcn_gate_tau0=_safe_mean(gcn_t0),
        gcn_gate_tau1=_safe_mean(gcn_t1),
        has_gates=collect_gates and gate_ok and bool(gin_t0 or gin_t1),
    )


@torch.no_grad()
def collect_gcn_gin_pairwise_outcomes(
    track: str,
    lr_tag: str,
    seed: int,
    results_root: Path,
    dataset_dir: str,
    device: torch.device,
    pair_indices: frozenset[int],
) -> list[GraphPairOutcome]:
    """Per-graph GCN-only vs GIN-only outcomes restricted to twin-pair graphs."""
    rows: list[GraphPairOutcome] = []
    preds: dict[str, list[tuple[int, int, int]]] = {}
    for model_key, model_name in (("gcn", "a0g1_gcn"), ("gin", "a0g1_gin")):
        run_dir = results_root / track / f"{model_name}_{lr_tag}_seed{seed}"
        run_ref = RunRef(
            track=track,
            run_dir=run_dir,
            model=model_name,
            lr_tag=lr_tag,
            seed=seed,
        )
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing run dir: {run_dir}")
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
        graph_rows: list[tuple[int, int, int]] = []
        for batch in test_loader:
            batch = batch.to(device)
            tau = batch.tau.view(-1).long()
            pred, true = model(batch)
            _loss, pred_score = compute_loss(pred, true)
            pred_label = _pred_labels_from_score(pred_score).view(-1).long()
            true_label = true.view(-1).long()
            for g in range(pred_label.numel()):
                graph_rows.append(
                    (
                        int(tau[g].item()),
                        int(true_label[g].item()),
                        int(pred_label[g].item()),
                    ),
                )
        preds[model_key] = graph_rows

    for graph_idx, (tau_g, label, pred_gcn) in enumerate(preds["gcn"]):
        if graph_idx not in pair_indices:
            continue
        _tau_gin, _label_gin, pred_gin = preds["gin"][graph_idx]
        if _tau_gin != tau_g or _label_gin != label:
            raise RuntimeError(f"GCN/GIN loader mismatch at graph_idx={graph_idx}")
        rows.append(
            GraphPairOutcome(
                track=track,
                lr_tag=lr_tag,
                seed=seed,
                graph_idx=graph_idx,
                tau=tau_g,
                label=label,
                pred_gcn=pred_gcn,
                pred_gin=pred_gin,
                correct_gcn=pred_gcn == label,
                correct_gin=pred_gin == label,
            ),
        )
    return rows


def _metrics_to_summary_rows(summary: list[dict[str, Any]]) -> list[SummaryRow]:
    """Convert summarize_runs dicts to plot SummaryRow."""
    return [
        SummaryRow(
            track=str(row["track"]),
            model=str(row["model"]),
            lr_tag=str(row["lr_tag"]),
            n_seeds=int(row["n_seeds"]),
            acc_all_mean=float(row["acc_all_mean"]),
            acc_all_std=float(row["acc_all_std"]),
            acc_tau0_mean=float(row["acc_tau0_mean"]),
            acc_tau0_std=float(row["acc_tau0_std"]),
            acc_tau1_mean=float(row["acc_tau1_mean"]),
            acc_tau1_std=float(row["acc_tau1_std"]),
        )
        for row in summary
    ]


def _metrics_to_per_run_rows(metrics: Sequence[RunMetrics]) -> list[PerRunRow]:
    """Convert RunMetrics to plot PerRunRow."""
    return [
        PerRunRow(
            track=m.track,
            model=m.model,
            lr_tag=m.lr_tag,
            seed=m.seed,
            acc_all=m.acc_all,
            acc_tau0=m.acc_tau0,
            acc_tau1=m.acc_tau1,
            gin_gate_tau0=m.gin_gate_tau0,
            gin_gate_tau1=m.gin_gate_tau1,
            gcn_gate_tau0=m.gcn_gate_tau0,
            gcn_gate_tau1=m.gcn_gate_tau1,
            has_gates=m.has_gates,
        )
        for m in metrics
    ]


def _write_pairwise_summary(path: Path, rows: Sequence[PairwiseSummaryRow]) -> None:
    """Write pairwise summary CSV."""
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
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
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
                "frac_gcn_only_correct": row.gcn_only / row.n_graphs if row.n_graphs else "",
                "frac_gin_only_correct": row.gin_only / row.n_graphs if row.n_graphs else "",
                "frac_specialist_wins": row.frac_specialist_wins,
                "all_graphs_specialist_wins": int(
                    (row.gcn_only if row.tau == 0 else row.gin_only) == row.n_graphs
                    and row.both_wrong == 0
                ),
            },
        )
    _write_csv(path, out_rows, fieldnames)


def plot_pairs_figures(
    out_dir: Path,
    *,
    lr_tag: str,
    dpi: int,
) -> None:
    """Generate fig01/fig02/fig05 for the pairs-only subset."""
    summary_path = out_dir / "summary_by_model.csv"
    per_run_path = out_dir / "per_run_metrics.csv"
    pairwise_path = out_dir / "pairwise_baseline_summary.csv"
    if not summary_path.is_file() or not per_run_path.is_file():
        raise FileNotFoundError(f"Missing CSVs in {out_dir}")

    _apply_style()
    paper_dir = out_dir / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open(encoding="utf-8", newline="") as fh:
        summary_dicts = list(csv.DictReader(fh))
    summary = _metrics_to_summary_rows(
        [
            {
                "track": r["track"],
                "model": r["model"],
                "lr_tag": r["lr_tag"],
                "n_seeds": int(r["n_seeds"]),
                "acc_all_mean": float(r["acc_all_mean"]),
                "acc_all_std": float(r["acc_all_std"]),
                "acc_tau0_mean": float(r["acc_tau0_mean"]),
                "acc_tau0_std": float(r["acc_tau0_std"]),
                "acc_tau1_mean": float(r["acc_tau1_mean"]),
                "acc_tau1_std": float(r["acc_tau1_std"]),
            }
            for r in summary_dicts
        ],
    )
    per_run = _metrics_to_per_run_rows(
        [
            RunMetrics(
                track=r["track"],
                model=r["model"],
                lr_tag=r["lr_tag"],
                seed=int(r["seed"]),
                run_dir=r["run_dir"],
                epoch=int(r["epoch"]),
                n_all=int(r["n_all"]),
                n_tau0=int(r["n_tau0"]),
                n_tau1=int(r["n_tau1"]),
                acc_all=float(r["acc_all"]),
                acc_tau0=float(r["acc_tau0"]),
                acc_tau1=float(r["acc_tau1"]),
                gin_head_idx=int(r["gin_head_idx"]),
                gcn_head_idx=int(r["gcn_head_idx"]),
                gin_gate_tau0=float(r["gin_gate_tau0"]) if r["gin_gate_tau0"] else float("nan"),
                gin_gate_tau1=float(r["gin_gate_tau1"]) if r["gin_gate_tau1"] else float("nan"),
                gcn_gate_tau0=float(r["gcn_gate_tau0"]) if r["gcn_gate_tau0"] else float("nan"),
                gcn_gate_tau1=float(r["gcn_gate_tau1"]) if r["gcn_gate_tau1"] else float("nan"),
                has_gates=str(r["has_gates"]).lower() in {"true", "1", "yes"},
            )
            for r in csv.DictReader(per_run_path.open(encoding="utf-8", newline=""))
        ],
    )

    pairs_suffix = " — opposite-sign twin pairs only"
    plot_baseline_per_type(
        summary,
        paper_dir / "fig01_baseline_per_type_pairs.png",
        dpi=dpi,
        lr_tag=lr_tag,
        ymin=0.0,
        title_suffix=pairs_suffix,
    )
    plot_gate_routing_delta(
        per_run,
        paper_dir / "fig02_gate_routing_delta_per_seed_pairs.png",
        dpi=dpi,
        lr_tag=lr_tag,
        title_suffix=pairs_suffix,
    )
    if pairwise_path.is_file():
        _plot_pairwise_comparison(
            [r for r in _load_pairwise_summary(pairwise_path) if r.lr_tag == lr_tag],
            paper_dir / "fig05_pairwise_baseline_comparison_pairs.png",
            lr_tag=lr_tag,
            dpi=dpi,
        )


def _load_pairwise_summary(path: Path) -> list[PairwiseSummaryRow]:
    """Load pairwise summary CSV."""
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


def aggregate_from_pairwise_csv(
    pairwise_csv: Path,
    pair_indices: frozenset[int],
    lr_tag: str,
) -> tuple[list[RunMetrics], list[PairwiseSummaryRow]]:
    """Build GCN/GIN-only metrics and pairwise summary from an existing per-graph CSV."""
    per_graph: list[GraphPairOutcome] = []
    with pairwise_csv.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            graph_idx = int(raw["graph_idx"])
            if graph_idx not in pair_indices:
                continue
            if raw["lr_tag"] != lr_tag:
                continue
            per_graph.append(
                GraphPairOutcome(
                    track=raw["track"],
                    lr_tag=raw["lr_tag"],
                    seed=int(raw["seed"]),
                    graph_idx=graph_idx,
                    tau=int(raw["tau"]),
                    label=int(raw["label"]),
                    pred_gcn=int(raw["pred_gcn"]),
                    pred_gin=int(raw["pred_gin"]),
                    correct_gcn=bool(int(raw["correct_gcn"])),
                    correct_gin=bool(int(raw["correct_gin"])),
                ),
            )

    if not per_graph:
        raise RuntimeError(f"No pair graphs found in {pairwise_csv} for lr_tag={lr_tag}")

    pairwise_summary = summarize_pairwise_outcomes(per_graph)

    metrics: list[RunMetrics] = []
    by_track_seed: dict[tuple[str, int], list[GraphPairOutcome]] = defaultdict(list)
    for row in per_graph:
        by_track_seed[(row.track, row.seed)].append(row)

    for (track, seed), rows in sorted(by_track_seed.items()):
        for model in ("a0g1_gcn", "a0g1_gin"):
            if model == "a0g1_gcn":
                correct_t0 = sum(1 for r in rows if r.tau == 0 and r.correct_gcn)
                correct_t1 = sum(1 for r in rows if r.tau == 1 and r.correct_gcn)
            else:
                correct_t0 = sum(1 for r in rows if r.tau == 0 and r.correct_gin)
                correct_t1 = sum(1 for r in rows if r.tau == 1 and r.correct_gin)
            n_t0 = sum(1 for r in rows if r.tau == 0)
            n_t1 = sum(1 for r in rows if r.tau == 1)
            n_all = n_t0 + n_t1
            metrics.append(
                RunMetrics(
                    track=track,
                    model=model,
                    lr_tag=lr_tag,
                    seed=seed,
                    run_dir="from_csv",
                    epoch=-1,
                    n_all=n_all,
                    n_tau0=n_t0,
                    n_tau1=n_t1,
                    acc_all=(correct_t0 + correct_t1) / n_all,
                    acc_tau0=correct_t0 / n_t0 if n_t0 else float("nan"),
                    acc_tau1=correct_t1 / n_t1 if n_t1 else float("nan"),
                    gin_head_idx=0,
                    gcn_head_idx=1,
                    gin_gate_tau0=float("nan"),
                    gin_gate_tau1=float("nan"),
                    gcn_gate_tau0=float("nan"),
                    gcn_gate_tau1=float("nan"),
                    has_gates=False,
                ),
            )
    return metrics, pairwise_summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run pairs-only evaluation and write figures."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    tracks = [t.strip() for t in re.split(r"[,;]+", args.tracks) if t.strip()]
    device = _select_device(args.device)

    if args.plots_only:
        plot_pairs_figures(out_dir, lr_tag=args.lr_tag, dpi=args.dpi)
        logging.info("Wrote pairs-only figures under %s/paper_figures/", out_dir)
        return

    pair_indices = load_pair_graph_indices(args.dataset_dir)
    n_pairs = len(pair_indices) // 2
    logging.info(
        "Pairs-only subset: %d graphs (%d twin pairs) on test split",
        len(pair_indices),
        n_pairs,
    )

    if args.from_csv:
        metrics, pairwise_summary = aggregate_from_pairwise_csv(
            Path(args.pairwise_csv),
            pair_indices,
            args.lr_tag,
        )
        per_run_eval = out_dir / "per_run_metrics.csv"
        summary = _summarize_runs(metrics)
        _write_csv(per_run_eval, [asdict(m) for m in metrics], list(asdict(metrics[0]).keys()))
        _write_csv(out_dir / "summary_by_model.csv", summary, list(summary[0].keys()) if summary else [])
        _write_pairwise_summary(out_dir / "pairwise_baseline_summary.csv", pairwise_summary)
        plot_pairs_figures(out_dir, lr_tag=args.lr_tag, dpi=args.dpi)
        logging.info(
            "From-CSV pairs analysis complete (%d GCN/GIN run-rows). "
            "For gated/ungated + root gates on pairs, run full eval on cluster.",
            len(metrics),
        )
        return

    results_root = Path(args.results_root)
    run_refs, _ = discover_run_refs(results_root, tracks)
    run_refs = [r for r in run_refs if r.lr_tag == args.lr_tag]
    if not run_refs:
        raise SystemExit(f"No runs with lr_tag={args.lr_tag} under {results_root}")

    metrics: list[RunMetrics] = []
    for ref in run_refs:
        logging.info("Evaluating pairs only: %s / %s", ref.track, ref.run_dir.name)
        metrics.append(evaluate_run_on_pairs(ref, args.dataset_dir, device, pair_indices))

    summary = _summarize_runs(metrics)
    per_run_path = out_dir / "per_run_metrics.csv"
    summary_path = out_dir / "summary_by_model.csv"
    _write_csv(per_run_path, [asdict(m) for m in metrics], list(asdict(metrics[0]).keys()))
    _write_csv(summary_path, summary, list(summary[0].keys()) if summary else [])

    pairwise_rows: list[GraphPairOutcome] = []
    seeds = sorted({m.seed for m in metrics if m.model == "a0g1_gcn"})
    for track in tracks:
        for seed in seeds:
            logging.info("Pairwise GCN vs GIN on pairs: track=%s seed=%d", track, seed)
            pairwise_rows.extend(
                collect_gcn_gin_pairwise_outcomes(
                    track,
                    args.lr_tag,
                    seed,
                    results_root,
                    args.dataset_dir,
                    device,
                    pair_indices,
                ),
            )
    pairwise_summary = summarize_pairwise_outcomes(pairwise_rows)
    _write_pairwise_summary(out_dir / "pairwise_baseline_summary.csv", pairwise_summary)

    plot_pairs_figures(out_dir, lr_tag=args.lr_tag, dpi=args.dpi)
    logging.info("Wrote pairs-only analysis to %s", out_dir)


if __name__ == "__main__":
    main()
