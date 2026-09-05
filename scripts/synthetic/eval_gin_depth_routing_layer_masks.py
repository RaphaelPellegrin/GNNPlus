#!/usr/bin/env python3
"""Layer-masking ablation for gated SiGMA on GIN depth-routing (eval only).

At test time, zero the single MP head on a chosen layer (``a0g1``):

  - ``none``: both layers active
  - ``mask_layer0``: zero MP on layer 0
  - ``mask_layer1``: zero MP on layer 1

Expectation (depth specialization with residual):
  - Masking layer 1 hurts τ=1 (deep) more than τ=0 (shallow)
  - Masking layer 0 hurts both (layer-1 features depend on layer 0)

Outputs under ``--out-dir``:
  - ``layer_mask_ablation_per_run.csv``
  - ``layer_mask_ablation_summary.csv``
  - ``fig_layer_mask_ablation.png`` / ``.pdf``
  - ``paper_figures/fig_layer_mask_ablation.png``

Example::

  python scripts/synthetic/eval_gin_depth_routing_layer_masks.py \\
    --results-root $GNNPLUS_OUT_DIR/gin_routing_depth \\
    --dataset-dir $GNNPLUS_DATASET_DIR \\
    --out-dir results/gin_routing_depth/analysis
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PLOT_ONLY = "--plot-only" in sys.argv

if not _PLOT_ONLY:
    from torch_geometric.data import Batch
    from torch_geometric.graphgym.checkpoint import load_ckpt
    from torch_geometric.graphgym.config import cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.loss import compute_loss
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.device import auto_select_device
    from torch_geometric import seed_everything

    import GNNPlus  # noqa: F401

    from GNNPlus.hybrid_gate_tracking import _unwrap_model
    from scripts.synthetic.analyze_gin_depth_routing_results import (  # noqa: E402
        RunRef,
        _load_cfg_for_run,
        _pick_best_epoch,
        _pred_labels_from_score,
        discover_run_refs,
    )

MaskMode = Literal["none", "mask_layer0", "mask_layer1"]
MASK_MODES: tuple[MaskMode, ...] = ("none", "mask_layer0", "mask_layer1")
MASK_LABELS: dict[MaskMode, str] = {
    "none": "Both layers",
    "mask_layer0": "Mask layer 0",
    "mask_layer1": "Mask layer 1",
}
METRIC_PALETTE: dict[str, str] = {
    "acc_all": "#E45756",
    "acc_tau0": "#4C72B0",
    "acc_tau1": "#DD8452",
}


@dataclass(frozen=True)
class MaskEvalRow:
    """Test accuracy under one layer-mask setting for one run."""

    track: str
    model: str
    lr_tag: str
    seed: int
    mask_mode: MaskMode
    run_dir: str
    epoch: int
    n_all: int
    n_tau0: int
    n_tau1: int
    acc_all: float
    acc_tau0: float
    acc_tau1: float


def _default_results_root() -> str:
    """Resolve default results root from env or local path."""
    if "GNNPLUS_OUT_DIR" in os.environ:
        return f"{os.environ['GNNPLUS_OUT_DIR'].rstrip('/')}/gin_routing_depth"
    return "results/gin_routing_depth"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default=_default_results_root(),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get(
            "GNNPLUS_DATASET_DIR",
            "results/gin_routing_depth/data",
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gin_routing_depth/analysis",
    )
    parser.add_argument("--tracks", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument(
        "--model",
        type=str,
        default="l2_a0g1_gated",
        help="Model slug (default: gated depth SiGMA).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing CSV (no GPU).",
    )
    parser.add_argument("--ymin", type=float, default=0.3)
    return parser.parse_args(argv)


def _select_device(choice: str) -> torch.device:
    """Resolve torch device."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _layer_mp_mask(mask_mode: MaskMode, layer_i: int, num_heads: int) -> Optional[list[bool]]:
    """Return MP head mask for ``layer_i``, or None when all heads stay active."""
    if mask_mode == "none":
        return None
    if mask_mode == "mask_layer0" and layer_i == 0:
        return [False] * num_heads
    if mask_mode == "mask_layer1" and layer_i == 1:
        return [False] * num_heads
    return None


@torch.no_grad()
def _forward_layer_masked(
    core: torch.nn.Module,
    batch: Batch,
    mask_mode: MaskMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run hybrid core with optional per-layer MP head masking."""
    (
        x,
        batch_enc,
        edge_index_attn,
        edge_attr_attn,
        edge_index_mp,
        edge_attr_mp,
        _ei,
        _ea,
    ) = core._encode_batch(batch)

    # Single-head toy hybrid (a0g1); detect head count when available.
    num_heads = 1
    if hasattr(core, "layers") and len(core.layers) > 0:
        layer0 = core.layers[0]
        if hasattr(layer0, "mp_heads"):
            num_heads = len(layer0.mp_heads)
        elif hasattr(layer0, "gnn_heads"):
            num_heads = len(layer0.gnn_heads)

    for layer_i, layer in enumerate(core.layers):
        mp_mask = _layer_mp_mask(mask_mode, layer_i, num_heads)
        layer_out = layer(
            x,
            edge_index_mp,
            batch_enc.batch,
            edge_attr_mp,
            edge_index_attn=edge_index_attn,
            edge_attr_attn=edge_attr_attn,
            edge_index_mp=edge_index_mp,
            edge_attr_mp=edge_attr_mp,
            mp_head_mask=mp_mask,
        )
        x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        if core.ffn_blocks is not None:
            x = core.ffn_blocks[layer_i](x)
        batch_enc.x = x

    pred, true = core.post_mp(batch_enc)
    return pred, true


@torch.no_grad()
def evaluate_masked_run(
    run_ref: RunRef,
    dataset_dir: str,
    device: torch.device,
    mask_mode: MaskMode,
) -> MaskEvalRow:
    """Load checkpoint and evaluate test accuracy under one layer mask."""
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
    core = _unwrap_model(model)

    correct_all = correct_t0 = correct_t1 = 0
    n_all = n_t0 = n_t1 = 0

    for batch in test_loader:
        batch = batch.to(device)
        if not hasattr(batch, "tau") or batch.tau is None:
            raise AttributeError("Batch missing tau.")
        tau = batch.tau.view(-1).long()

        pred, true = _forward_layer_masked(core, batch, mask_mode)
        _loss, pred_score = compute_loss(pred, true)
        pred_label = _pred_labels_from_score(pred_score)
        true_label = true.view(-1).long()
        correct = pred_label == true_label

        correct_all += int(correct.sum().item())
        n_all += int(correct.numel())
        mask0 = tau == 0
        mask1 = tau == 1
        n_t0 += int(mask0.sum().item())
        n_t1 += int(mask1.sum().item())
        if mask0.any():
            correct_t0 += int(correct[mask0].sum().item())
        if mask1.any():
            correct_t1 += int(correct[mask1].sum().item())

    def _acc(num: int, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    return MaskEvalRow(
        track=run_ref.track,
        model=run_ref.model,
        lr_tag=run_ref.lr_tag,
        seed=run_ref.seed,
        mask_mode=mask_mode,
        run_dir=str(run_ref.run_dir),
        epoch=epoch,
        n_all=n_all,
        n_tau0=n_t0,
        n_tau1=n_t1,
        acc_all=_acc(correct_all, n_all),
        acc_tau0=_acc(correct_t0, n_t0),
        acc_tau1=_acc(correct_t1, n_t1),
    )


def _write_per_run_csv(rows: Sequence[MaskEvalRow], path: Path) -> None:
    """Write per-run layer-mask evaluation CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "model",
        "lr_tag",
        "seed",
        "mask_mode",
        "run_dir",
        "epoch",
        "n_all",
        "n_tau0",
        "n_tau1",
        "acc_all",
        "acc_tau0",
        "acc_tau1",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "track": row.track,
                    "model": row.model,
                    "lr_tag": row.lr_tag,
                    "seed": row.seed,
                    "mask_mode": row.mask_mode,
                    "run_dir": row.run_dir,
                    "epoch": row.epoch,
                    "n_all": row.n_all,
                    "n_tau0": row.n_tau0,
                    "n_tau1": row.n_tau1,
                    "acc_all": row.acc_all,
                    "acc_tau0": row.acc_tau0,
                    "acc_tau1": row.acc_tau1,
                },
            )


def _summarize(rows: Sequence[MaskEvalRow]) -> list[dict[str, float | str | int]]:
    """Mean/std over seeds by mask mode."""
    groups: dict[tuple[str, str, str, MaskMode], list[MaskEvalRow]] = {}
    for row in rows:
        key = (row.track, row.model, row.lr_tag, row.mask_mode)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, float | str | int]] = []
    for (track, model, lr_tag, mask_mode), items in sorted(groups.items()):
        def agg(attr: str) -> tuple[float, float]:
            vals = [float(getattr(it, attr)) for it in items]
            if len(vals) == 1:
                return vals[0], 0.0
            return float(mean(vals)), float(pstdev(vals))

        row_out: dict[str, float | str | int] = {
            "track": track,
            "model": model,
            "lr_tag": lr_tag,
            "mask_mode": mask_mode,
            "n_seeds": len(items),
        }
        for key in ("acc_all", "acc_tau0", "acc_tau1"):
            m, s = agg(key)
            row_out[f"{key}_mean"] = m
            row_out[f"{key}_std"] = s
        out.append(row_out)
    return out


def _write_summary_csv(
    rows: Sequence[dict[str, float | str | int]],
    path: Path,
) -> None:
    """Write summary CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summary_csv(path: Path) -> list[dict[str, float | str | int]]:
    """Load summary CSV for plot-only mode."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, float | str | int]] = []
        for raw in reader:
            row: dict[str, float | str | int] = {
                "track": raw["track"],
                "model": raw["model"],
                "lr_tag": raw["lr_tag"],
                "mask_mode": raw["mask_mode"],
                "n_seeds": int(raw["n_seeds"]),
            }
            for key in ("acc_all", "acc_tau0", "acc_tau1"):
                row[f"{key}_mean"] = float(raw[f"{key}_mean"])
                row[f"{key}_std"] = float(raw[f"{key}_std"])
            rows.append(row)
    return rows


def _plot_ablation(
    summary: Sequence[dict[str, float | str | int]],
    out_path: Path,
    dpi: int,
    ymin: float,
) -> None:
    """Grouped bars: accuracy by τ under each layer mask."""
    modes = [m for m in MASK_MODES if any(r["mask_mode"] == m for r in summary)]
    by_mode = {str(r["mask_mode"]): r for r in summary}
    x = list(range(len(modes)))
    bar_w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (metric, label) in enumerate(
        (
            ("acc_tau0", r"$\tau=0$ (shallow)"),
            ("acc_tau1", r"$\tau=1$ (deep)"),
            ("acc_all", "All"),
        ),
    ):
        offsets = [xi + (i - 1) * bar_w for xi in x]
        means = [float(by_mode[m][f"{metric}_mean"]) for m in modes]
        stds = [float(by_mode[m][f"{metric}_std"]) for m in modes]
        ax.bar(
            offsets,
            means,
            width=bar_w,
            yerr=stds,
            capsize=3,
            label=label,
            color=METRIC_PALETTE[metric],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MASK_LABELS[m] for m in modes])  # type: ignore[index]
    ax.set_ylim(ymin, 1.02)
    ax.set_ylabel("Test accuracy")
    ax.set_title("GIN depth-routing · layer-mask ablation (gated)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.25)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    paper = out_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "layer_mask_ablation_summary.csv"
    per_run_path = out_dir / "layer_mask_ablation_per_run.csv"

    if args.plot_only:
        summary = _load_summary_csv(summary_path)
    else:
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
        refs = [
            r
            for r in discover_run_refs(Path(args.results_root), tracks)
            if r.model == args.model and r.lr_tag == args.lr_tag
        ]
        if not refs:
            raise SystemExit(
                f"No runs matching model={args.model} lr={args.lr_tag} "
                f"under {args.results_root}",
            )
        device = _select_device(args.device)
        rows: list[MaskEvalRow] = []
        for ref in refs:
            for mode in MASK_MODES:
                logging.info("%s · %s", ref.run_dir.name, mode)
                rows.append(evaluate_masked_run(ref, args.dataset_dir, device, mode))
        _write_per_run_csv(rows, per_run_path)
        summary = _summarize(rows)
        _write_summary_csv(summary, summary_path)

    _plot_ablation(summary, out_dir / "fig_layer_mask_ablation.png", args.dpi, args.ymin)
    _plot_ablation(summary, paper / "fig_layer_mask_ablation.png", args.dpi, args.ymin)
    print(f"Wrote {summary_path}")
    print(f"Wrote {out_dir / 'fig_layer_mask_ablation.png'}")


if __name__ == "__main__":
    main()
