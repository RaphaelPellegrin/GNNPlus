#!/usr/bin/env python3
"""Inference gate-clamp eval on gated SiGMA checkpoints.

Loads a trained *gated* hybrid checkpoint and measures test accuracy under:

  - ``learned``: normal forward (learned γ)
  - ``ones``: force γ = 1 on every head / node (full contribution)
  - ``mean``: replace each node γ by the per-graph mean γ

A drop under ``ones`` / ``mean`` vs ``learned`` means the deployed model
depends on its learned gates (not just that an ungated *retraining* can match).

Checkpoints (TU Tab.17/18 gated SiGMA) live on the cluster::

  $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_SiGMA_hetero_<lr>_seed<s>/ckpt/
  $GNNPLUS_OUT_DIR/tu_sigma_1x_gcn/<ds>_SiGMA_hetero_<lr>_seed<s>/ckpt/

Local ``results/tu_sigma_homo_hetero/*/ckpt/`` dirs are usually empty
(only gate dumps were pulled).

Examples::

  # Single run (cluster path)
  python scripts/gate_viz/eval_gate_clamp.py \\
    --run_dir $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/mutag_SiGMA_hetero_lr001_seed0 \\
    --cfg configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml \\
    --dataset-dir $GNNPLUS_DATASET_DIR

  # Sweep reported Tab.17 hetero LRs (6 ds × 5 seeds)
  python scripts/gate_viz/eval_gate_clamp.py \\
    --results-root $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero \\
    --cfg configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml \\
    --dataset-dir $GNNPLUS_DATASET_DIR \\
    --out-csv results/gate_clamp/tu_hh_hetero_clamp.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import torch
from torch_geometric.graphgym.checkpoint import get_ckpt_epochs, load_ckpt
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import GNNPlus  # noqa: F401
from GNNPlus.hybrid_gate_tracking import _unwrap_model
from GNNPlus.layer.gate_override import GateOverrideMode

ClampMode = Literal["learned", "ones", "mean"]
CLAMP_MODES: tuple[ClampMode, ...] = ("learned", "ones", "mean")

# Reported best-LR gated hetero (Tab.17 dh16) for the default sweep.
TAB17_HETERO_BEST_LR: dict[str, str] = {
    "mutag": "lr001",
    "enzymes": "lr001",
    "proteins": "lr001",
    "collab": "lr01",
    "imdb_binary": "lr001",
    "reddit_binary": "lr001",
}

# Tab.18 (dh4) reported best LR for hetero.
TAB18_HETERO_BEST_LR: dict[str, str] = {
    "mutag": "lr001",
    "enzymes": "lr001",
    "proteins": "lr001",
    "collab": "lr001",
    "imdb_binary": "lr01",
    "reddit_binary": "lr001",
}

RUN_NAME_RE = re.compile(
    r"^(?P<ds>[a-z0-9_]+)_SiGMA_hetero_(?P<lr>lr\d+)_seed(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class ClampRow:
    """One (run × clamp mode) test accuracy."""

    dataset: str
    lr_tag: str
    seed: int
    run_dir: str
    epoch: int
    clamp_mode: ClampMode
    n: int
    accuracy: float


def _default_dataset_dir() -> str:
    """Resolve dataset dir from env or GraphGym default."""
    return os.environ.get("GNNPLUS_DATASET_DIR", "datasets")


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Single GraphGym run directory containing ckpt/.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Sweep root (discovers *_SiGMA_hetero_*_seed* with ckpt/).",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Anchor yaml (or path; overridden by run config_used.yaml if present).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=_default_dataset_dir(),
    )
    parser.add_argument(
        "--table",
        type=str,
        default="17",
        choices=("17", "18", "all"),
        help="When sweeping, restrict to reported best LR for Tab.17 / Tab.18.",
    )
    parser.add_argument(
        "--all-lrs",
        action="store_true",
        help="When sweeping, evaluate every SiGMA_hetero LR (not only reported).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="learned,ones,mean",
        help="Comma-separated clamp modes.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Checkpoint epoch (-1 = latest under ckpt/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/gate_clamp/gate_clamp_results.csv",
    )
    parser.add_argument(
        "--paired-ttest",
        action="store_true",
        help="Print seed-paired t-tests (learned − ones / learned − mean).",
    )
    return parser.parse_args(argv)


def _resolve_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_epoch(run_dir: Path, epoch: int) -> int:
    """Resolve checkpoint epoch under ``run_dir/ckpt``."""
    cfg.run_dir = str(run_dir)
    epochs = list(get_ckpt_epochs())
    if not epochs:
        raise FileNotFoundError(f"No checkpoints under {run_dir}/ckpt")
    if epoch < 0:
        return int(max(epochs))
    if epoch not in epochs:
        raise FileNotFoundError(f"Epoch {epoch} not in {epochs} under {run_dir}/ckpt")
    return int(epoch)


def _run_config_path(run_dir: Path, fallback_cfg: Path) -> Path:
    """Prefer saved config_used.yaml when present."""
    for name in ("config_used.yaml", "config.yaml"):
        path = run_dir / name
        if path.is_file():
            return path
    return fallback_cfg


def _load_cfg(run_dir: Path, cfg_path: Path, dataset_dir: str, seed: int) -> None:
    """Load GraphGym cfg for one run."""
    old_argv = sys.argv
    sys.argv = [
        old_argv[0],
        "--cfg",
        str(cfg_path),
        "dataset.dir",
        dataset_dir,
        "seed",
        str(seed),
        "wandb.use",
        "False",
    ]
    try:
        args = parse_args()
        set_cfg(cfg)
        load_cfg(cfg, args)
    finally:
        sys.argv = old_argv
    cfg.run_dir = str(run_dir)
    cfg.out_dir = str(run_dir)


def _pred_labels(pred_score: torch.Tensor) -> torch.Tensor:
    """Convert scores to class labels."""
    if pred_score.ndim == 1 or (pred_score.ndim == 2 and pred_score.shape[1] == 1):
        thresh = float(getattr(cfg.model, "thresh", 0.5))
        return (pred_score.view(-1) > thresh).long()
    return pred_score.argmax(dim=-1).view(-1)


def _override_arg(mode: ClampMode) -> Optional[GateOverrideMode]:
    """Map clamp mode name to HybridGNN kwarg."""
    if mode == "learned":
        return None
    return mode  # type: ignore[return-value]


@torch.no_grad()
def _eval_accuracy(
    core: torch.nn.Module,
    loader: Any,
    device: torch.device,
    mode: ClampMode,
) -> tuple[float, int]:
    """Return (accuracy, n) on ``loader`` under one clamp mode."""
    override = _override_arg(mode)
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        out = core(batch, gate_override=override)
        if isinstance(out, tuple) and len(out) == 2:
            pred, true = out
        else:
            raise TypeError(
                f"Expected (pred, true) from HybridGNN, got {type(out)}"
            )
        _loss, pred_score = compute_loss(pred, true)
        pred_label = _pred_labels(pred_score)
        true_label = true.view(-1).long()
        correct += int((pred_label == true_label).sum().item())
        total += int(true_label.numel())
    if total == 0:
        return float("nan"), 0
    return correct / total, total


def _discover_runs(
    results_root: Path,
    table: str,
    all_lrs: bool,
) -> list[tuple[Path, str, str, int]]:
    """Discover SiGMA_hetero run dirs with checkpoints.

    Returns:
        List of ``(run_dir, dataset, lr_tag, seed)``.
    """
    best_map: dict[str, str]
    if table == "18":
        best_map = TAB18_HETERO_BEST_LR
    else:
        best_map = TAB17_HETERO_BEST_LR

    found: list[tuple[Path, str, str, int]] = []
    if not results_root.is_dir():
        raise FileNotFoundError(f"results-root not found: {results_root}")

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_NAME_RE.match(child.name)
        if match is None:
            continue
        ds = match.group("ds")
        lr_tag = match.group("lr")
        seed = int(match.group("seed"))
        ckpt_dir = child / "ckpt"
        if not ckpt_dir.is_dir() or not any(ckpt_dir.glob("*.ckpt")):
            logging.warning("Skip (no .ckpt): %s", child)
            continue
        if not all_lrs:
            want = best_map.get(ds)
            if want is None:
                continue
            if lr_tag != want:
                continue
        found.append((child, ds, lr_tag, seed))
    return found


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    """Sample mean and std."""
    if not xs:
        return float("nan"), float("nan")
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(var)


def _paired_ttest(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-tailed paired t-test; returns (t, p)."""
    from scipy import stats

    t_stat, p_value = stats.ttest_rel(list(a), list(b))
    return float(t_stat), float(p_value)


def eval_one_run(
    run_dir: Path,
    dataset: str,
    lr_tag: str,
    seed: int,
    anchor_cfg: Path,
    dataset_dir: str,
    device: torch.device,
    modes: Sequence[ClampMode],
    epoch: int,
) -> list[ClampRow]:
    """Load one ckpt and evaluate all clamp modes."""
    cfg_path = _run_config_path(run_dir, anchor_cfg)
    _load_cfg(run_dir, cfg_path, dataset_dir, seed)
    seed_everything(int(seed))
    auto_select_device()
    if device.type == "cpu":
        cfg.accelerator = "cpu"

    loaders = create_loader()
    if len(loaders) < 3:
        raise RuntimeError(f"Expected train/val/test loaders in {run_dir}")
    test_loader = loaders[2]

    model = create_model()
    ep = _pick_epoch(run_dir, epoch)
    load_ckpt(model, optimizer=None, scheduler=None, epoch=ep)
    model.eval()
    model.to(device)
    core = _unwrap_model(model)

    rows: list[ClampRow] = []
    for mode in modes:
        acc, n = _eval_accuracy(core, test_loader, device, mode)
        rows.append(
            ClampRow(
                dataset=dataset,
                lr_tag=lr_tag,
                seed=seed,
                run_dir=str(run_dir),
                epoch=ep,
                clamp_mode=mode,
                n=n,
                accuracy=acc,
            )
        )
        logging.info(
            "%s %s seed%d %s: acc=%.4f (n=%d, ep=%d)",
            dataset,
            lr_tag,
            seed,
            mode,
            acc,
            n,
            ep,
        )
    return rows


def _write_csv(path: Path, rows: Iterable[ClampRow]) -> None:
    """Write results CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "lr_tag",
        "seed",
        "clamp_mode",
        "accuracy",
        "n",
        "epoch",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "lr_tag": row.lr_tag,
                    "seed": row.seed,
                    "clamp_mode": row.clamp_mode,
                    "accuracy": f"{row.accuracy:.6f}",
                    "n": row.n,
                    "epoch": row.epoch,
                    "run_dir": row.run_dir,
                }
            )


def _summarize(rows: list[ClampRow], do_ttest: bool) -> None:
    """Print mean±std and optional paired t-tests."""
    by_key: dict[tuple[str, str, ClampMode], list[float]] = {}
    for row in rows:
        key = (row.dataset, row.lr_tag, row.clamp_mode)
        by_key.setdefault(key, []).append(100.0 * row.accuracy)

    print("\n=== Gate clamp summary (test acc %) ===")
    datasets = sorted({r.dataset for r in rows})
    for ds in datasets:
        lr_tags = sorted({r.lr_tag for r in rows if r.dataset == ds})
        for lr in lr_tags:
            print(f"\n{ds} / {lr}")
            for mode in CLAMP_MODES:
                xs = by_key.get((ds, lr, mode), [])
                if not xs:
                    continue
                mean, std = _mean_std(xs)
                print(f"  {mode:8s}  {mean:.2f}±{std:.2f}  (n={len(xs)})")
            if do_ttest:
                learned = by_key.get((ds, lr, "learned"), [])
                for alt in ("ones", "mean"):
                    other = by_key.get((ds, lr, alt), [])
                    if len(learned) >= 2 and len(learned) == len(other):
                        # align by sorting — seeds may be mixed; use paired via seed order
                        # rebuild by seed
                        learned_by_seed = {
                            r.seed: 100.0 * r.accuracy
                            for r in rows
                            if r.dataset == ds
                            and r.lr_tag == lr
                            and r.clamp_mode == "learned"
                        }
                        other_by_seed = {
                            r.seed: 100.0 * r.accuracy
                            for r in rows
                            if r.dataset == ds
                            and r.lr_tag == lr
                            and r.clamp_mode == alt
                        }
                        seeds = sorted(set(learned_by_seed) & set(other_by_seed))
                        a = [learned_by_seed[s] for s in seeds]
                        b = [other_by_seed[s] for s in seeds]
                        if len(a) >= 2:
                            t_stat, p_value = _paired_ttest(a, b)
                            delta = sum(x - y for x, y in zip(a, b)) / len(a)
                            sig = "YES" if p_value < 0.05 else "no"
                            print(
                                f"  paired learned−{alt}: "
                                f"Δ={delta:+.2f}  t={t_stat:.3f}  p={p_value:.4g}  sig={sig}"
                            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_cli(argv)
    modes = tuple(
        m.strip() for m in args.modes.split(",") if m.strip()
    )  # type: ignore[assignment]
    for mode in modes:
        if mode not in CLAMP_MODES:
            raise SystemExit(f"Unknown mode {mode!r}; expected {CLAMP_MODES}")

    anchor_cfg = Path(args.cfg)
    if not anchor_cfg.is_file():
        raise SystemExit(f"Missing cfg: {anchor_cfg}")

    device = _resolve_device(args.device)
    jobs: list[tuple[Path, str, str, int]] = []

    if args.run_dir:
        run_dir = Path(args.run_dir)
        match = RUN_NAME_RE.match(run_dir.name)
        if match is None:
            # Allow arbitrary names; parse seed from train_meta or cfg later.
            ds = run_dir.name
            lr_tag = "lr?"
            seed = 0
            meta = run_dir / "train_meta.txt"
            if meta.is_file():
                text = meta.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if line.startswith("seed="):
                        seed = int(line.split("=", 1)[1])
                    if line.startswith("lr_tag="):
                        lr_tag = line.split("=", 1)[1]
                    if line.startswith("ds_tag="):
                        ds = line.split("=", 1)[1]
        else:
            ds = match.group("ds")
            lr_tag = match.group("lr")
            seed = int(match.group("seed"))
        jobs.append((run_dir, ds, lr_tag, seed))
    elif args.results_root:
        jobs = _discover_runs(
            Path(args.results_root),
            table=args.table,
            all_lrs=bool(args.all_lrs),
        )
        if not jobs:
            raise SystemExit(
                f"No SiGMA_hetero runs with .ckpt under {args.results_root}. "
                "On cluster use $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero "
                "(local results/ usually has empty ckpt/)."
            )
    else:
        raise SystemExit("Provide --run_dir or --results-root")

    logging.info("Evaluating %d run(s) on %s", len(jobs), device)
    all_rows: list[ClampRow] = []
    for run_dir, ds, lr_tag, seed in jobs:
        try:
            all_rows.extend(
                eval_one_run(
                    run_dir=run_dir,
                    dataset=ds,
                    lr_tag=lr_tag,
                    seed=seed,
                    anchor_cfg=anchor_cfg,
                    dataset_dir=args.dataset_dir,
                    device=device,
                    modes=modes,  # type: ignore[arg-type]
                    epoch=int(args.epoch),
                )
            )
        except Exception:
            logging.exception("Failed on %s", run_dir)

    if not all_rows:
        raise SystemExit("No successful evaluations.")

    out_csv = Path(args.out_csv)
    _write_csv(out_csv, all_rows)
    logging.info("Wrote %s", out_csv)
    _summarize(all_rows, do_ttest=bool(args.paired_ttest))


if __name__ == "__main__":
    main()
