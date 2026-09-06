#!/usr/bin/env python3
"""Regenerate GIN depth-routing summary figures from CSVs (no checkpoint load).

Useful locally when torch_scatter / GraphGym imports are unavailable. Reads
``summary_by_model.csv`` and (optionally) ``opposite_sign_pair_summary.csv``
under ``--out-dir`` and rewrites the paper figures with specialists included.

Example::

  python scripts/synthetic/plot_gin_depth_routing_summary_figures.py \\
    --out-dir results/gin_routing_depth/analysis
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER: tuple[str, ...] = (
    "l2_a0g1_gated",
    "l2_a0g1_ungated",
    "l1_a0g1",
    "l2_a0g1_gin",
)
MODEL_LABELS: dict[str, str] = {
    "l2_a0g1_gated": "SiGMA gated (L=2)",
    "l2_a0g1_ungated": "SiGMA ungated (L=2)",
    "l1_a0g1": "1-GIN specialist",
    "l2_a0g1_gin": "2-GIN specialist",
}

# Opposite-sign summary uses short model keys.
OPP_MODEL_ORDER: tuple[str, ...] = (
    "oracle_s1_rule",
    "oracle_s2_rule",
    "gated",
    "ungated",
    "gin1",
    "gin2",
)
OPP_MODEL_LABELS: dict[str, str] = {
    "oracle_s1_rule": r"Oracle $R_1$ (1-layer)",
    "oracle_s2_rule": r"Oracle $R_2$ (2-layer+res)",
    "gated": "SiGMA gated",
    "ungated": "SiGMA ungated",
    "gin1": "1-GIN specialist",
    "gin2": "2-GIN specialist",
}
OPP_OUTCOMES: tuple[str, ...] = (
    "both_correct",
    "only_tau0",
    "only_tau1",
    "both_wrong",
)
OPP_COLORS: dict[str, str] = {
    "both_correct": "#55A868",
    "only_tau0": "#4C72B0",
    "only_tau1": "#DD8452",
    "both_wrong": "#C44E52",
}
OPP_LABELS: dict[str, str] = {
    "both_correct": "Both correct",
    "only_tau0": r"Only $\tau{=}0$",
    "only_tau1": r"Only $\tau{=}1$",
    "both_wrong": "Both wrong",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/gin_routing_depth/analysis",
        help="Directory with summary CSVs and figure outputs.",
    )
    parser.add_argument("--track", type=str, default="toy")
    parser.add_argument("--lr-tag", type=str, default="lr001")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV as a list of string dicts."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_acc(
    summary: Sequence[dict[str, str]],
    out_path: Path,
    *,
    track: str,
    lr_tag: str,
    dpi: int,
) -> None:
    """Bar chart: test accuracy by τ for each model."""
    subset = [
        r
        for r in summary
        if r.get("track") == track and r.get("lr_tag") == lr_tag
    ]
    if not subset:
        raise SystemExit(f"No summary rows for track={track} lr={lr_tag} in CSV")
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in subset)]
    by_model = {str(r["model"]): r for r in subset}
    x = list(range(len(models)))
    bar_w = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(
        [xi - bar_w / 2 for xi in x],
        [float(by_model[m]["acc_tau0_mean"]) for m in models],
        width=bar_w,
        yerr=[float(by_model[m]["acc_tau0_std"]) for m in models],
        capsize=3,
        label=r"$\tau=0$ ($R_1$ / 1-layer)",
        color="#4C72B0",
    )
    ax.bar(
        [xi + bar_w / 2 for xi in x],
        [float(by_model[m]["acc_tau1_mean"]) for m in models],
        width=bar_w,
        yerr=[float(by_model[m]["acc_tau1_std"]) for m in models],
        capsize=3,
        label=r"$\tau=1$ ($R_2$ / 2-layer+res)",
        color="#DD8452",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"GIN depth-routing · per-τ accuracy ({lr_tag})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} ({len(models)} models)")


def _plot_opposite_sign(
    rows: Sequence[dict[str, str]],
    out_outcomes: Path,
    out_table: Path,
    *,
    track: str,
    lr_tag: str,
    dpi: int,
) -> None:
    """Stacked outcome bars + compact table from opposite-sign summary CSV."""
    subset = [
        r
        for r in rows
        if r.get("track") == track and r.get("lr_tag") == lr_tag
    ]
    if not subset:
        print(f"Skip opposite-sign plots: no rows for track={track} lr={lr_tag}")
        return

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in subset:
        grouped[str(row["model"])].append(row)

    means: dict[str, dict[str, float]] = {}
    for model, items in grouped.items():
        means[model] = {
            "both_correct": mean(float(r["frac_both_correct"]) for r in items),
            "only_tau0": mean(float(r["frac_only_tau0"]) for r in items),
            "only_tau1": mean(float(r["frac_only_tau1"]) for r in items),
            "both_wrong": mean(float(r["frac_both_wrong"]) for r in items),
        }

    models = [m for m in OPP_MODEL_ORDER if m in means]
    if not models:
        print("Skip opposite-sign plots: no known models in CSV")
        return

    # Outcomes stacked bar
    x = list(range(len(models)))
    bottoms = [0.0] * len(models)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.3 * len(models)), 4.6))
    for outcome in OPP_OUTCOMES:
        heights = [means[m][outcome] for m in models]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=OPP_COLORS[outcome],
            label=OPP_LABELS[outcome],
            width=0.72,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(x)
    ax.set_xticklabels([OPP_MODEL_LABELS[m] for m in models], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of opposite-sign pairs")
    ax.set_title("GIN depth-routing · opposite-sign pair outcomes")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_outcomes.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_outcomes, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_outcomes.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_outcomes} ({len(models)} models)")

    # Compact table figure
    cell_text: list[list[str]] = []
    for model in models:
        row_vals = [means[model][o] for o in OPP_OUTCOMES]
        cell_text.append([f"{v:.3f}" for v in row_vals])
    fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.45 * len(models)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[OPP_MODEL_LABELS[m] for m in models],
        colLabels=[OPP_LABELS[o] for o in OPP_OUTCOMES],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Opposite-sign pair outcome fractions (mean over seeds)")
    fig.tight_layout()
    fig.savefig(out_table, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_table.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_table}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    paper = out_dir / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary_by_model.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")
    summary = _read_csv(summary_path)
    _plot_acc(
        summary,
        out_dir / "fig_baseline_per_type.png",
        track=args.track,
        lr_tag=args.lr_tag,
        dpi=args.dpi,
    )
    _plot_acc(
        summary,
        paper / "fig_acc_by_tau.png",
        track=args.track,
        lr_tag=args.lr_tag,
        dpi=args.dpi,
    )
    _plot_acc(
        summary,
        out_dir / f"fig_baseline_per_type_{args.track}.png",
        track=args.track,
        lr_tag=args.lr_tag,
        dpi=args.dpi,
    )

    opp_path = out_dir / "opposite_sign_pair_summary.csv"
    if opp_path.is_file():
        opp_rows = _read_csv(opp_path)
        models_present = {r.get("model") for r in opp_rows}
        if "gin1" not in models_present and "gin2" not in models_present:
            print(
                "NOTE: opposite-sign CSV has no gin1/gin2 yet — "
                "re-run analyze_gin_depth_opposite_sign_pairs on the cluster, "
                "then re-pull / re-run this plotter."
            )
        _plot_opposite_sign(
            opp_rows,
            out_dir / "fig_opposite_sign_pair_outcomes.png",
            out_dir / "fig_opposite_sign_pair_table.png",
            track=args.track,
            lr_tag=args.lr_tag,
            dpi=args.dpi,
        )
        _plot_opposite_sign(
            opp_rows,
            paper / "fig_opposite_sign_pair_outcomes.png",
            paper / "fig_opposite_sign_pair_table.png",
            track=args.track,
            lr_tag=args.lr_tag,
            dpi=args.dpi,
        )
    else:
        print(f"Skip opposite-sign: missing {opp_path}")


if __name__ == "__main__":
    main()
