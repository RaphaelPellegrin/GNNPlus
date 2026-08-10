#!/usr/bin/env python3
"""Summarize NOP vs broadcast diagnostics from attention ``.pt`` dumps.

For each graph × (layer, head) computes:

* ``max_alpha`` / ``ratio_vs_uniform`` (existence)
* ``vnorm_ratio = ‖v_sink‖ / mean‖v‖`` (NOP ≪ 1)
* ``av_stable_rank`` of clean ``AV`` (broadcast ≈ 1)
* ``av_row_cosine`` (broadcast ⇒ high shared write)
* heuristic ``mechanism`` ∈ {nop, broadcast, ambiguous}

Example::

  python scripts/attention_sinks/summarize_nop_broadcast.py \\
    --input-dir results/tu_attention_sinks/mutag_GPS_ungated_attn_lr001_seed2/attention_matrices \\
    --out-csv results/tu_attention_sinks/mutag_GPS_ungated_mech.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GNNPlus.attention_sink_tracking import (  # noqa: E402
    classify_sink_mechanism,
    mean_row_cosine,
    stable_rank,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI for NOP/broadcast summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Directory of attention .pt bundles.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=None,
        help="Explicit .pt paths (combined with --input-dir).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.5,
        help="τ for sink present (column sum > τ·μ).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_paths(args: argparse.Namespace) -> List[Path]:
    """Collect input ``.pt`` paths."""
    paths: List[Path] = []
    if args.inputs:
        paths.extend(Path(p) for p in args.inputs)
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"--input-dir not found: {d}")
        paths.extend(sorted(d.glob("*.pt")))
    return [p for p in paths if p.is_file()]


def _graph_slices(batch: torch.Tensor) -> List[tuple[int, int]]:
    """Return ``[start, end)`` for each graph in a PyG batch vector."""
    n_graphs = int(batch.max().item()) + 1
    out: List[tuple[int, int]] = []
    for g in range(n_graphs):
        idx = torch.where(batch == g)[0]
        out.append((int(idx[0].item()), int(idx[-1].item()) + 1))
    return out


def _records_from_bundle(path: Path, tau: float) -> List[Dict[str, Any]]:
    """Build per-graph × head mechanism rows from one ``.pt``."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    attention: Dict[str, torch.Tensor] = obj.get("attention", {})
    value_norms: Dict[str, torch.Tensor] = obj.get("value_norms", {})
    head_outputs: Dict[str, torch.Tensor] = obj.get("head_outputs", {})
    attn_gates: Dict[str, torch.Tensor] = obj.get("attn_gates", {})
    batch = obj["batch"]
    slices = _graph_slices(batch)
    rows: List[Dict[str, Any]] = []
    for key, A_full in attention.items():
        A_np = A_full.detach().cpu().float().numpy()
        vn_np = (
            value_norms[key].detach().cpu().float().numpy()
            if key in value_norms
            else None
        )
        av_np = (
            head_outputs[key].detach().cpu().float().numpy()
            if key in head_outputs
            else None
        )
        g_np = (
            attn_gates[key].detach().cpu().float().numpy()
            if key in attn_gates
            else None
        )
        for gi, (start, end) in enumerate(slices):
            A = A_np[start:end, start:end]
            n = A.shape[0]
            if n == 0:
                continue
            alpha = A.mean(axis=0)
            sink = int(alpha.argmax())
            a_hat = A.sum(axis=0)
            mu = float(a_hat.mean())
            is_tau = bool(a_hat[sink] > tau * mu) if mu > 0 else False
            max_alpha = float(alpha[sink])
            uniform = 1.0 / n
            vnr = float("nan")
            if vn_np is not None:
                vn = vn_np[start:end]
                vnr = float(vn[sink] / (float(vn.mean()) + 1e-8))
            sr = float("nan")
            rc = float("nan")
            if av_np is not None:
                av = av_np[start:end]
                sr = stable_rank(av)
                rc = mean_row_cosine(av)
            gate_s = float("nan")
            if g_np is not None:
                gate_s = float(g_np[start:end][sink])
            mech = classify_sink_mechanism(
                vnorm_ratio=vnr, av_stable_rank=sr, row_cosine=rc
            )
            rows.append(
                {
                    "source": str(path),
                    "epoch": int(obj.get("epoch", -1)),
                    "split": str(obj.get("split", "")),
                    "batch_index": int(obj.get("batch_index", -1)),
                    "layer_head": key,
                    "graph_index": gi,
                    "n_g": n,
                    "max_alpha": max_alpha,
                    "ratio_vs_uniform": max_alpha / uniform,
                    "tau_sink": int(is_tau),
                    "vnorm_ratio": vnr,
                    "av_stable_rank": sr,
                    "av_row_cosine": rc,
                    "sink_gate": gate_s,
                    "mechanism": mech,
                }
            )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry: write mechanism CSV from attention dumps."""
    args = _parse_args(argv)
    paths = _resolve_paths(args)
    if not paths:
        print("No .pt inputs found.")
        return 1
    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        all_rows.extend(_records_from_bundle(p, float(args.tau)))
    if not all_rows:
        print("No records produced.")
        return 1
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Quick console summary over τ-sink rows.
    sink_rows = [r for r in all_rows if r["tau_sink"] == 1]
    def _frac(label: str) -> float:
        if not sink_rows:
            return float("nan")
        return float(np.mean([r["mechanism"] == label for r in sink_rows]))

    print(f"Wrote {len(all_rows)} rows → {out}")
    print(
        f"τ-sink rows={len(sink_rows)}  "
        f"nop={_frac('nop'):.2%}  broadcast={_frac('broadcast'):.2%}  "
        f"ambiguous={_frac('ambiguous'):.2%}"
    )
    if sink_rows:
        print(
            f"mean vnorm_ratio={np.nanmean([r['vnorm_ratio'] for r in sink_rows]):.3f}  "
            f"mean stable_rank={np.nanmean([r['av_stable_rank'] for r in sink_rows]):.3f}  "
            f"mean row_cos={np.nanmean([r['av_row_cosine'] for r in sink_rows]):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
