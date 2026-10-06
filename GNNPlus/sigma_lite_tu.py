"""Run and aggregate the SiGMA-lite TU protocol of Tables 17 / 18 from Python.

Each run is a ``main.py`` subprocess (GraphGym keeps a global config, so one
process per run keeps runs independent) with
``configs/sigma_lite/tu-sigma-lite-a2g4.yaml``: a2g4 heads (2 attention + GCN,
GIN, SAGE, GAT), L=12, H=64, random 50/25/25 split, up to 1000 epochs,
``reduce_on_plateau``. Table 17 uses ``d_h=16`` and Table 18 ``d_h=4``.

The reported number per run is the test accuracy at the first epoch with the
best validation accuracy (``best/test_accuracy`` on W&B). Per (dataset, d_h,
gate) the learning rate with the highest mean over seeds is reported.

Example (e.g. in a Colab notebook)::

    from GNNPlus.sigma_lite_tu import run_sweep, summarize

    run_sweep(["MUTAG"], tables=(17, 18), out_root="results/sigma_lite_tu")
    print(summarize("results/sigma_lite_tu"))
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG: str = "configs/sigma_lite/tu-sigma-lite-a2g4.yaml"
TABLE_D_H: Dict[int, int] = {17: 16, 18: 4}
TU_DATASETS: Tuple[str, ...] = (
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "COLLAB",
    "IMDB-BINARY",
    "REDDIT-BINARY",
)
BATCH_SIZES: Dict[str, int] = {"COLLAB": 32, "REDDIT-BINARY": 16}
DEFAULT_BATCH_SIZE: int = 64
LEARNING_RATES: Tuple[float, ...] = (0.001, 0.01)
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

TAG_RE = re.compile(
    r"^(?P<dataset>.+)_dh(?P<d_h>\d+)_(?P<gate>[a-z]+)_lr(?P<lr>[0-9.e-]+)_seed(?P<seed>\d+)$"
)

GroupKey = Tuple[str, int, str, str]


@dataclass(frozen=True)
class RunSpec:
    """One SiGMA-lite TU run.

    Attributes:
        dataset: TU dataset name (e.g. ``"MUTAG"``).
        d_h: Per-head width (16 for Table 17, 4 for Table 18).
        lr: Base learning rate.
        seed: Random seed.
        gate_act: SiGMA-lite gate activation, ``"sigmoid"`` or ``"gelu"``.
        max_epoch: Maximum number of epochs.
        batch_size: Batch size; ``None`` uses the paper value for ``dataset``.
    """

    dataset: str
    d_h: int
    lr: float
    seed: int
    gate_act: str = "sigmoid"
    max_epoch: int = 1000
    batch_size: Optional[int] = None

    @property
    def tag(self) -> str:
        """Run directory name, ``<dataset>_dh<d_h>_<gate>_lr<lr>_seed<seed>``."""
        return f"{self.dataset}_dh{self.d_h}_{self.gate_act}_lr{self.lr:g}_seed{self.seed}"

    @property
    def resolved_batch_size(self) -> int:
        """Batch size used for the run."""
        if self.batch_size is not None:
            return self.batch_size
        return BATCH_SIZES.get(self.dataset, DEFAULT_BATCH_SIZE)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one run.

    Attributes:
        spec: The run specification.
        test_accuracy: Test accuracy (fraction) at the best validation epoch,
            or ``None`` if the run produced no stats.
        best_epoch: Epoch of the best validation accuracy, or ``None``.
        num_epochs: Number of finished epochs.
        returncode: Exit code of the ``main.py`` process (0 when skipped).
    """

    spec: RunSpec
    test_accuracy: Optional[float]
    best_epoch: Optional[int]
    num_epochs: int
    returncode: int


def default_dataset_dir() -> Path:
    """Return the default dataset directory (``<repo>/datasets``)."""
    return REPO_ROOT / "datasets"


def build_command(
    spec: RunSpec,
    out_root: PathLike,
    dataset_dir: Optional[PathLike] = None,
    wandb: bool = False,
    wandb_entity: Optional[str] = None,
    wandb_project: str = "GNNPlus",
    python: str = sys.executable,
) -> List[str]:
    """Return the ``main.py`` command line for ``spec``.

    Args:
        spec: Run specification.
        out_root: Directory that receives one sub-directory per run.
        dataset_dir: Dataset cache directory (default ``<repo>/datasets``).
        wandb: Whether to log to Weights & Biases.
        wandb_entity: W&B entity (team or user); ``None`` uses the W&B default.
        wandb_project: W&B project.
        python: Python interpreter used to launch ``main.py``.

    Returns:
        The command as a list of arguments.
    """
    data_dir = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    run_dir = Path(out_root).resolve() / spec.tag
    cmd = [
        python,
        "main.py",
        "--cfg",
        CONFIG,
        "dataset.name",
        spec.dataset,
        "dataset.dir",
        str(data_dir.resolve()),
        "gnn.hybrid.d_h",
        str(spec.d_h),
        "gnn.sigma_lite.gate_act",
        spec.gate_act,
        "optim.base_lr",
        f"{spec.lr:g}",
        "optim.max_epoch",
        str(spec.max_epoch),
        "train.batch_size",
        str(spec.resolved_batch_size),
        "seed",
        str(spec.seed),
        "out_dir",
        str(run_dir),
        "wandb.use",
        str(wandb),
        "wandb.project",
        wandb_project,
        "wandb.name",
        f"sigma_lite_{spec.tag}",
        "wandb.group",
        f"sigma_lite_tu_{spec.dataset}_dh{spec.d_h}_{spec.gate_act}",
    ]
    if wandb_entity:
        cmd += ["wandb.entity", wandb_entity]
    return cmd


def _read_accuracies(stats_path: Path) -> List[float]:
    """Return per-epoch accuracies from a GraphGym ``stats.json`` file."""
    accs: List[float] = []
    with stats_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                accs.append(float(json.loads(line)["accuracy"]))
    return accs


def best_test_accuracy(run_dir: PathLike) -> Optional[Tuple[float, int, int]]:
    """Return ``(test_acc_at_best_val, best_epoch, num_epochs)`` or ``None``.

    Args:
        run_dir: Directory containing ``val/stats.json`` and ``test/stats.json``.

    Returns:
        Test accuracy (fraction) at the first best-validation epoch, that epoch,
        and the number of epochs logged; ``None`` if no stats exist yet.
    """
    run_path = Path(run_dir)
    val_path = run_path / "val" / "stats.json"
    test_path = run_path / "test" / "stats.json"
    if not val_path.is_file() or not test_path.is_file():
        return None
    val = _read_accuracies(val_path)
    test = _read_accuracies(test_path)
    n = min(len(val), len(test))
    if n == 0:
        return None
    best_epoch = max(range(n), key=lambda i: (val[i], -i))
    return test[best_epoch], best_epoch, n


def run_one(
    spec: RunSpec,
    out_root: PathLike = "results/sigma_lite_tu",
    dataset_dir: Optional[PathLike] = None,
    wandb: bool = False,
    wandb_entity: Optional[str] = None,
    wandb_project: str = "GNNPlus",
    skip_finished: bool = True,
    print_every: int = 50,
) -> RunResult:
    """Train one SiGMA-lite run in a ``main.py`` subprocess.

    The full log goes to ``<out_root>/<tag>.log``; every ``print_every``-th
    epoch summary is echoed to stdout.

    Args:
        spec: Run specification.
        out_root: Directory that receives one sub-directory per run.
        dataset_dir: Dataset cache directory (default ``<repo>/datasets``).
        wandb: Whether to log to Weights & Biases.
        wandb_entity: W&B entity; ``None`` uses the W&B default.
        wandb_project: W&B project.
        skip_finished: Skip runs whose stats already cover ``spec.max_epoch``
            epochs (useful to resume after a Colab disconnect).
        print_every: Echo every n-th epoch summary; ``0`` silences the output.

    Returns:
        The run result.
    """
    out_path = Path(out_root).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    run_dir = out_path / spec.tag
    if skip_finished:
        done = best_test_accuracy(run_dir)
        if done is not None and done[2] >= spec.max_epoch:
            print(f"[skip] {spec.tag}: already finished")
            return RunResult(spec, done[0], done[1], done[2], 0)

    cmd = build_command(spec, out_path, dataset_dir, wandb, wandb_entity, wandb_project)
    log_path = out_path / f"{spec.tag}.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    # torch>=2.6 defaults torch.load(weights_only=True), which rejects the
    # pickled PyG Data objects written by older dataset loaders.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    print(f"[start] {spec.tag}")
    with log_path.open("w") as log, subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            match = re.match(r"^> Epoch (\d+):", line)
            if print_every and match and int(match.group(1)) % print_every == 0:
                print(f"  [{spec.tag}] {line.strip()[:160]}")
        returncode = proc.wait()

    result = best_test_accuracy(run_dir)
    status = "done" if returncode == 0 else f"FAIL (exit {returncode}, see {log_path})"
    if result is None:
        print(f"[{status}] {spec.tag}: no stats")
        return RunResult(spec, None, None, 0, returncode)
    test_acc, best_epoch, num_epochs = result
    print(f"[{status}] {spec.tag}: test acc {100 * test_acc:.2f}% (best val epoch {best_epoch})")
    return RunResult(spec, test_acc, best_epoch, num_epochs, returncode)


def make_specs(
    datasets: Iterable[str],
    tables: Iterable[int] = (17, 18),
    lrs: Iterable[float] = LEARNING_RATES,
    seeds: Iterable[int] = SEEDS,
    gate_act: str = "sigmoid",
    max_epoch: int = 1000,
) -> List[RunSpec]:
    """Return the run grid ``tables x datasets x lrs x seeds``.

    Args:
        datasets: TU dataset names.
        tables: Paper tables to reproduce (17 -> ``d_h=16``, 18 -> ``d_h=4``).
        lrs: Learning rates.
        seeds: Seeds.
        gate_act: SiGMA-lite gate activation.
        max_epoch: Maximum number of epochs.

    Returns:
        One :class:`RunSpec` per grid point.
    """
    lr_list = list(lrs)
    seed_list = list(seeds)
    specs: List[RunSpec] = []
    for table in tables:
        if table not in TABLE_D_H:
            raise ValueError(f"table must be one of {sorted(TABLE_D_H)}, got {table}")
        for dataset in datasets:
            for lr in lr_list:
                for seed in seed_list:
                    specs.append(
                        RunSpec(dataset, TABLE_D_H[table], lr, seed, gate_act, max_epoch)
                    )
    return specs


def run_sweep(
    datasets: Sequence[str],
    tables: Iterable[int] = (17, 18),
    lrs: Iterable[float] = LEARNING_RATES,
    seeds: Iterable[int] = SEEDS,
    gate_act: str = "sigmoid",
    max_epoch: int = 1000,
    out_root: PathLike = "results/sigma_lite_tu",
    dataset_dir: Optional[PathLike] = None,
    wandb: bool = False,
    wandb_entity: Optional[str] = None,
    wandb_project: str = "GNNPlus",
    jobs: int = 1,
    skip_finished: bool = True,
    print_every: int = 50,
) -> List[RunResult]:
    """Run the Table 17 / 18 grid and return one result per run.

    Args:
        datasets: TU dataset names (see :data:`TU_DATASETS`).
        tables: Paper tables to reproduce (17 -> ``d_h=16``, 18 -> ``d_h=4``).
        lrs: Learning rates.
        seeds: Seeds.
        gate_act: SiGMA-lite gate activation.
        max_epoch: Maximum number of epochs.
        out_root: Directory that receives one sub-directory per run.
        dataset_dir: Dataset cache directory (default ``<repo>/datasets``).
        wandb: Whether to log to Weights & Biases.
        wandb_entity: W&B entity; ``None`` uses the W&B default.
        wandb_project: W&B project.
        jobs: Number of runs trained concurrently (small TU graphs leave a GPU
            mostly idle, so 2-4 is usually faster on Colab).
        skip_finished: Skip runs that already finished.
        print_every: Echo every n-th epoch summary; ``0`` silences the output.

    Returns:
        Results in grid order.
    """
    specs = make_specs(datasets, tables, lrs, seeds, gate_act, max_epoch)
    if jobs > 1:
        # Download / process each dataset once before parallel runs share it.
        for dataset in dict.fromkeys(s.dataset for s in specs):
            prepare_dataset(dataset, dataset_dir)

    def _run(spec: RunSpec) -> RunResult:
        """Run one spec with the sweep-wide settings."""
        return run_one(
            spec,
            out_root,
            dataset_dir,
            wandb,
            wandb_entity,
            wandb_project,
            skip_finished,
            print_every,
        )

    if jobs <= 1:
        return [_run(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_run, specs))


def prepare_dataset(dataset: str, dataset_dir: Optional[PathLike] = None) -> None:
    """Download and pre-process a TU dataset into the loader's cache directory.

    Args:
        dataset: TU dataset name.
        dataset_dir: Dataset cache directory (default ``<repo>/datasets``).
    """
    from GNNPlus.loader.master_loader import preformat_TUDataset

    data_dir = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    preformat_TUDataset(str(data_dir.resolve() / "TUDataset"), dataset)


def collect_results(out_root: PathLike) -> Dict[GroupKey, List[Tuple[float, int]]]:
    """Group finished runs under ``out_root`` by (dataset, d_h, gate, lr).

    Args:
        out_root: Directory with one sub-directory per run.

    Returns:
        Map from group key to ``(test_accuracy, num_epochs)`` per seed.
    """
    groups: Dict[GroupKey, List[Tuple[float, int]]] = defaultdict(list)
    root = Path(out_root)
    if not root.is_dir():
        return groups
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = TAG_RE.match(run_dir.name)
        result = best_test_accuracy(run_dir) if match else None
        if match is None or result is None:
            continue
        key: GroupKey = (match["dataset"], int(match["d_h"]), match["gate"], match["lr"])
        groups[key].append((result[0], result[2]))
    return groups


def _fmt(values: List[float]) -> str:
    """Format accuracies (fractions) as ``mean±std`` in percent."""
    pct = [100.0 * v for v in values]
    std = statistics.stdev(pct) if len(pct) > 1 else 0.0
    return f"{statistics.mean(pct):.2f}±{std:.2f}"


def summarize(out_root: PathLike) -> str:
    """Return per-LR and best-LR summaries for every run group under ``out_root``.

    Args:
        out_root: Directory with one sub-directory per run.

    Returns:
        A printable table (test accuracy in percent, mean±std over seeds).
    """
    groups = collect_results(out_root)
    if not groups:
        return f"No finished runs found under {out_root}"
    lines = [f"{'dataset':<14}{'d_h':>4}  {'gate':<8}{'lr':<7}{'seeds':>6}{'epochs':>8}  test acc (%)"]
    best: Dict[Tuple[str, int, str], Tuple[float, str, List[float]]] = {}
    for key in sorted(groups):
        dataset, d_h, gate, lr = key
        accs = [acc for acc, _ in groups[key]]
        min_epochs = min(n for _, n in groups[key])
        lines.append(
            f"{dataset:<14}{d_h:>4}  {gate:<8}{lr:<7}{len(accs):>6}{min_epochs:>8}  {_fmt(accs)}"
        )
        mean = statistics.mean(accs)
        family = (dataset, d_h, gate)
        if family not in best or mean > best[family][0]:
            best[family] = (mean, lr, accs)
    lines.append("")
    lines.append("Best LR per (dataset, d_h, gate) — Table 17 (d_h=16) / Table 18 (d_h=4) style:")
    for (dataset, d_h, gate), (_, lr, accs) in sorted(best.items()):
        lines.append(f"  {dataset:<14} d_h={d_h:<3} {gate:<8} {_fmt(accs)}  (lr={lr}, n={len(accs)})")
    return "\n".join(lines)
