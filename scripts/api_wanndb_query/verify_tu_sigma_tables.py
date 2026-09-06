#!/usr/bin/env python3
"""Verify TU Table 17/18 gated SiGMA numbers against W&B run IDs.

Recomputes mean±std of ``best_test_perf`` for the reported (best-LR) SiGMA
groups and flags mismatches vs the paper table values.

Example:
    python scripts/api_wanndb_query/verify_tu_sigma_tables.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import wandb


ENTITY = "weber-geoml-harvard-university"
PROJECT = "GNNPlus"


@dataclass(frozen=True)
class ReportedGroup:
    """One reported SiGMA cell (dataset × family × table)."""

    dataset: str
    family: str
    run_ids: tuple[str, ...]
    expected_lr: float
    expected_mean: float
    expected_std: float
    expected_dh: int
    table: str


# Paper Table 17 (d_h=16). REDDIT IDs recovered from W&B groups.
# COLLAB hetero: paper 77.25±0.95 is lr=0.01 (not the lr=0.001 IDs in an
# older footnote — those give 76.83±1.18).
TABLE_17: tuple[ReportedGroup, ...] = (
    ReportedGroup("MUTAG", "homo", ("k4cnokb0", "9haif3pf", "a0hc9voh", "rgeuhat1", "1l6jkyfa"), 1e-3, 73.19, 4.15, 16, "17"),
    ReportedGroup("MUTAG", "hetero", ("qn5ws4of", "xdqlpkqk", "lru5cxel", "grl3gpsf", "tsc8bxsa"), 1e-3, 84.68, 3.50, 16, "17"),
    ReportedGroup("ENZYMES", "homo", ("1973nnhe", "tx89hpyi", "yd0jqo9r", "r5nu2asd", "4fbna3pw"), 1e-3, 47.07, 1.98, 16, "17"),
    ReportedGroup("ENZYMES", "hetero", ("4cokvur9", "h5foqi2r", "fc53zbub", "q3mneaur", "8kr7f99c"), 1e-3, 47.60, 6.99, 16, "17"),
    ReportedGroup("PROTEINS", "homo", ("pbh8kdvv", "n5m426mk", "58hhsmy6", "y9n41nq7", "o4voavix"), 1e-2, 73.41, 1.99, 16, "17"),
    ReportedGroup("PROTEINS", "hetero", ("rswaxury", "u1onemwt", "9c0xgvzd", "sywq392j", "p8i9p5ag"), 1e-3, 74.12, 1.06, 16, "17"),
    ReportedGroup("COLLAB", "homo", ("79dz24ct", "0ezq3el2", "8wkztevr", "qo7b9ics", "74m0capk"), 1e-2, 73.97, 0.88, 16, "17"),
    ReportedGroup(
        "COLLAB",
        "hetero",
        ("h1cu3mag", "6qw8bq4u", "aqd3csla", "481cw852", "fmhrfdl4"),
        1e-2,
        77.25,
        0.95,
        16,
        "17",
    ),
    ReportedGroup("IMDB-BINARY", "homo", ("qe55har6", "i0tusdxm", "9262x7qh", "8mdtzf64", "cfwdnojj"), 1e-2, 66.64, 1.19, 16, "17"),
    ReportedGroup("IMDB-BINARY", "hetero", ("mqjb7k0p", "zg7275ur", "zfd27czf", "7qx2fy1m", "eccttjo2"), 1e-3, 69.92, 2.75, 16, "17"),
    ReportedGroup("REDDIT-BINARY", "homo", ("ik7pskkh", "n693yukr", "cfoxsg24", "ppoum8n0", "lilg63qe"), 1e-3, 87.92, 7.51, 16, "17"),
    ReportedGroup("REDDIT-BINARY", "hetero", ("q0e7zmel", "zlzmlqfw", "t8s5wovy", "tcpigbyo", "ksgiw9mn"), 1e-3, 92.72, 1.01, 16, "17"),
)

TABLE_18: tuple[ReportedGroup, ...] = (
    ReportedGroup("MUTAG", "homo", ("ja6bhghu", "5fvsudcg", "87evxnhw", "148yz872", "9dr6t1o7"), 1e-3, 73.19, 3.23, 4, "18"),
    ReportedGroup("MUTAG", "hetero", ("1m9jwn01", "5g0qc4w7", "z5dkj392", "na42acrn", "trpzah99"), 1e-3, 80.43, 5.90, 4, "18"),
    ReportedGroup("ENZYMES", "homo", ("01ulnvxv", "52bna938", "7efkqsyu", "3vrw1tqu", "th179kq9"), 1e-3, 46.67, 6.00, 4, "18"),
    ReportedGroup("ENZYMES", "hetero", ("ourehdfb", "80w9ic8q", "l38t2fwd", "0evbf38g", "ma73yjiw"), 1e-3, 47.60, 4.46, 4, "18"),
    ReportedGroup("PROTEINS", "homo", ("e2pkmphm", "cq9kodcm", "t5f5yfkv", "yxec3tto", "2gfzfuyi"), 1e-3, 73.05, 1.87, 4, "18"),
    ReportedGroup("PROTEINS", "hetero", ("q2exkp7l", "zyrivhba", "ccaqsbm5", "l10xvaj4", "61xu27j9"), 1e-3, 74.34, 1.20, 4, "18"),
    ReportedGroup("COLLAB", "homo", ("n6vzdb9h", "sai1rfvi", "pu920gjw", "yijg7isl", "z8segc6p"), 1e-3, 73.04, 1.94, 4, "18"),
    ReportedGroup("COLLAB", "hetero", ("f2xixc43", "aomix7b0", "1jccklv1", "0d97vcq9", "rtr89r1a"), 1e-3, 76.74, 1.32, 4, "18"),
    ReportedGroup("IMDB-BINARY", "homo", ("epoatcc4", "jcaqw5gl", "mztk2vdz", "gc011i3x", "1oiv8yvs"), 1e-2, 65.36, 1.78, 4, "18"),
    ReportedGroup("IMDB-BINARY", "hetero", ("idyavgvu", "ohm6gs2h", "kwkqkfc2", "zt97gdce", "to0qj3eu"), 1e-2, 71.04, 1.97, 4, "18"),
    ReportedGroup("REDDIT-BINARY", "homo", ("hj4nio5n", "82uj4swu", "qbtl0dlg", "71ht4ubk", "jd817rrj"), 1e-3, 89.96, 5.08, 4, "18"),
    ReportedGroup("REDDIT-BINARY", "hetero", ("9fwch8xh", "fyi7odz6", "jabcqgzq", "yam7enye", "nvnec5wx"), 1e-3, 90.36, 5.42, 4, "18"),
)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Return sample mean and std (ddof=1)."""
    xs = list(values)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _to_percent(raw: float) -> float:
    """Convert W&B accuracy to percent if logged in [0, 1]."""
    return raw * 100.0 if raw <= 1.5 else raw


def _config_get(run: object, *keys: str) -> object | None:
    """Fetch a nested or flattened config key."""
    cfg = getattr(run, "config", {})
    if not isinstance(cfg, dict):
        return None
    for key in keys:
        if key in cfg:
            return cfg[key]
    # Nested dicts
    gnn = cfg.get("gnn")
    if isinstance(gnn, dict):
        hybrid = gnn.get("hybrid", {})
        if isinstance(hybrid, dict):
            if any(k == "gate" or k.endswith("gate") for k in keys) and "gate" in hybrid:
                return hybrid["gate"]
            if any(k == "d_h" or k.endswith("d_h") for k in keys) and "d_h" in hybrid:
                return hybrid["d_h"]
    optim = cfg.get("optim")
    if isinstance(optim, dict) and "base_lr" in optim:
        if any(k.endswith("base_lr") for k in keys):
            return optim["base_lr"]
    return None


def verify_group(api: wandb.Api, group: ReportedGroup, tol: float) -> bool:
    """Print verification for one reported cell; return True if OK."""
    perfs: list[float] = []
    gates: set[object] = set()
    dhs: set[object] = set()
    lrs: set[object] = set()
    for run_id in group.run_ids:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        raw = run.summary.get("best_test_perf")
        if raw is None:
            print(f"MISS {group.dataset} {group.family} {run_id}: no best_test_perf")
            continue
        perfs.append(_to_percent(float(raw)))
        gates.add(_config_get(run, "gnn.hybrid.gate", "gate") or "?")
        dhs.add(_config_get(run, "gnn.hybrid.d_h", "d_h"))
        lrs.add(_config_get(run, "optim.base_lr", "base_lr"))

    mean, std = _mean_std(perfs)
    ok = (
        len(perfs) == len(group.run_ids)
        and abs(mean - group.expected_mean) < tol
        and abs(std - group.expected_std) < tol
        and group.expected_dh in dhs
        and abs(float(next(iter(lrs))) - group.expected_lr) < 1e-9
        and gates == {"headwise"}
    )
    flag = "OK" if ok else "MISMATCH"
    print(
        f"{flag} T{group.table} {group.dataset:14s} {group.family:6s} "
        f"wandb={mean:.2f}±{std:.2f} paper={group.expected_mean:.2f}±{group.expected_std:.2f} "
        f"gate={gates} dh={dhs} lr={lrs}"
    )
    return ok


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Allowed absolute error on mean/std (percentage points).",
    )
    parser.add_argument(
        "--table",
        choices=("17", "18", "both"),
        default="both",
        help="Which paper table to verify.",
    )
    args = parser.parse_args()

    groups: list[ReportedGroup] = []
    if args.table in ("17", "both"):
        groups.extend(TABLE_17)
    if args.table in ("18", "both"):
        groups.extend(TABLE_18)

    api = wandb.Api(timeout=120)
    ok_count = sum(1 for g in groups if verify_group(api, g, args.tol))
    print(f"\n{ok_count}/{len(groups)} cells OK (tol={args.tol})")


if __name__ == "__main__":
    main()
