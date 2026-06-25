"""Run heterogeneity-profile experiments (multi-trial per-graph test tracking)."""

from __future__ import annotations

import json
import logging
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Batch
from torch_geometric.graphgym.checkpoint import save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.epoch import is_eval_epoch
from torch_geometric import seed_everything

from GNNPlus.experiments.track_avg_accuracy import (
    get_gnnplus_model_slug,
    infer_plot_task_type,
    load_and_plot_average_per_graph,
)
from GNNPlus.logger import create_logger
from GNNPlus.optimizer.extra_optimizers import (
    ExtendedOptimizerConfig,
    ExtendedSchedulerConfig,
)
from GNNPlus.train.custom_train import eval_epoch, train_epoch


def resolve_root_dataset(dataset: Any) -> Any:
    """Unwrap PyG ``Subset`` wrappers to the underlying dataset."""
    root = dataset
    while hasattr(root, "dataset"):
        root = root.dataset
    return root


def get_test_graph_indices(dataset: Any) -> List[int]:
    """Return global test graph indices from a graph-level dataset."""
    root = resolve_root_dataset(dataset)
    test_idx = getattr(root.data, "test_graph_index", None)
    if test_idx is None:
        raise ValueError(
            "Dataset has no test_graph_index; use graph-level task with a test split."
        )
    if isinstance(test_idx, torch.Tensor):
        return [int(x) for x in test_idx.cpu().tolist()]
    return [int(x) for x in list(test_idx)]


from GNNPlus.experiments.heterogeneity_metrics import (
    per_graph_performance_value,
    summarize_trial_metrics,
)
def evaluate_per_graph_on_test(
    model: torch.nn.Module,
    dataset: Any,
    test_indices: List[int],
    dataset_task_type: str,
) -> Dict[int, float]:
    """Evaluate best model on each test graph (batch size 1)."""
    device = torch.device(cfg.accelerator)
    model.eval()
    root = resolve_root_dataset(dataset)
    graph_values: Dict[int, float] = {}

    for graph_idx in test_indices:
        data = root[int(graph_idx)].clone()
        batch = Batch.from_data_list([data])
        batch.split = "test"
        batch.to(device)
        if cfg.gnn.head == "inductive_edge":
            pred, true, _extra = model(batch)
        else:
            pred, true = model(batch)
        graph_values[int(graph_idx)] = per_graph_performance_value(
            pred, true, dataset_task_type
        )
    return graph_values


def _optimizer_config() -> ExtendedOptimizerConfig:
    return ExtendedOptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
        schedulefree_beta1=float(getattr(cfg.optim, "schedulefree_beta1", 0.9)),
        schedulefree_beta2=float(getattr(cfg.optim, "schedulefree_beta2", 0.999)),
        schedulefree_warmup_steps=int(
            getattr(cfg.optim, "schedulefree_warmup_steps", 0)
        ),
    )


def _scheduler_config() -> ExtendedSchedulerConfig:
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def _trial_metric_from_perf(perf_split: Dict[str, float], dataset_task_type: str) -> float:
    """Extract the primary trial-level test metric from logger perf dict."""
    if dataset_task_type == "regression":
        return float(perf_split.get("mae", perf_split.get("loss", 0.0)))
    if dataset_task_type == "classification_multilabel":
        return float(perf_split.get("ap", perf_split.get("accuracy", 0.0)))
    return float(perf_split.get("accuracy", 0.0))


def run_single_heterogeneity_trial(
    trial: int,
    base_seed: int,
) -> Tuple[Dict[int, float], float, List[int]]:
    """
    Train one model with a fresh random split and return per-graph test scores.

    Returns:
        (per_graph_values, trial_test_metric, test_indices)
    """
    cfg.seed = int(base_seed + trial - 1)
    seed_everything(cfg.seed)

    loaders = create_loader()
    loggers = create_logger()
    model = create_model()
    optimizer = create_optimizer(model.parameters(), _optimizer_config())
    scheduler = create_scheduler(optimizer, _scheduler_config())
    cfg.params = params_count(model)

    dataset = loaders[0].dataset
    root = resolve_root_dataset(dataset)
    num_graphs = len(root)
    test_indices = get_test_graph_indices(dataset)

    num_splits = len(loggers)
    split_names = ["val", "test"]
    perf = [[] for _ in range(num_splits)]

    best_epoch = 0
    best_val_score = float("inf") if cfg.dataset.task_type == "regression" else -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for cur_epoch in range(cfg.optim.max_epoch):
        train_epoch(
            loggers[0],
            loaders[0],
            model,
            optimizer,
            scheduler,
            cfg.optim.batch_accumulation,
        )
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(
                    loggers[i],
                    loaders[i],
                    model,
                    split=split_names[i - 1],
                    optimizer=optimizer,
                )
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        if cfg.metric_best != "auto":
            m = cfg.metric_best
            val_score = float(perf[1][-1].get(m, perf[1][-1]["loss"]))
            improved = (
                val_score < best_val_score
                if cfg.dataset.task_type == "regression"
                else val_score > best_val_score
            )
            if improved:
                best_val_score = val_score
                best_epoch = cur_epoch
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
                if cfg.train.enable_ckpt and cfg.train.ckpt_best:
                    save_ckpt(model, optimizer, scheduler, cur_epoch)

    if best_state is not None:
        model.load_state_dict(best_state)

    trial_test_metric = _trial_metric_from_perf(
        perf[2][best_epoch], str(cfg.dataset.task_type)
    )
    per_graph = evaluate_per_graph_on_test(
        model,
        dataset,
        test_indices,
        str(cfg.dataset.task_type),
    )

    for logger in loggers:
        logger.close()

    logging.info(
        "Heterogeneity trial %d: seed=%d test_metric=%.4f graphs=%d",
        trial,
        cfg.seed,
        trial_test_metric,
        num_graphs,
    )
    return per_graph, trial_test_metric, test_indices


def graph_dict_pickle_path(output_dir: str, dataset_name: str, model_slug: str, num_layers: int) -> str:
    """Path for ``results/{N}_layers/{dataset}_{model}_graph_dict.pickle``."""
    layers_dir = os.path.join(output_dir, f"{num_layers}_layers")
    os.makedirs(layers_dir, exist_ok=True)
    return os.path.join(layers_dir, f"{dataset_name}_{model_slug}_graph_dict.pickle")


def save_heterogeneity_artifacts(
    graph_dict: Dict[int, List[Union[int, float]]],
    test_appearances: Dict[int, int],
    trial_metrics: List[float],
    output_dir: str,
) -> Tuple[str, str, str]:
    """Save pickle, trial summary JSON, and heterogeneity plots."""
    dataset_name = str(cfg.dataset.name)
    model_slug = get_gnnplus_model_slug(cfg)
    num_layers = int(cfg.gnn.layers_mp)
    task_type = str(cfg.dataset.task_type)
    plot_task_type = infer_plot_task_type(task_type)
    required = int(cfg.heterogeneity.required_test_appearances)

    pickle_path = graph_dict_pickle_path(output_dir, dataset_name, model_slug, num_layers)
    payload = {
        "graph_dict": graph_dict,
        "test_appearances": test_appearances,
        "required_test_appearances": required,
        "trial_metrics": trial_metrics,
        "dataset_name": dataset_name,
        "model_slug": model_slug,
        "model_type": str(cfg.model.type),
        "num_layers": num_layers,
        "task_type": task_type,
        "base_seed": int(cfg.heterogeneity.base_seed),
        "metric_best": str(cfg.metric_best),
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(payload, f)

    summary = summarize_trial_metrics(trial_metrics, task_type)
    summary_path = pickle_path.replace("_graph_dict.pickle", "_trial_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    original_plot, sorted_plot = ("", "")
    if bool(cfg.heterogeneity.plot):
        original_plot, sorted_plot = load_and_plot_average_per_graph(
            pickle_path,
            dataset_name=dataset_name,
            layer_type=model_slug,
            encoding=None,
            num_layers=num_layers,
            task_type=plot_task_type,
            output_dir=output_dir,
            model_slug=model_slug,
        )
    logging.info("Heterogeneity pickle: %s", pickle_path)
    logging.info("Heterogeneity by_index plot: %s", original_plot)
    logging.info("Heterogeneity by_accuracy plot: %s", sorted_plot)
    return pickle_path, summary_path, sorted_plot


def _apply_heterogeneity_split_overrides() -> Tuple[Any, Any]:
    """Force random splits per trial (Heterogeneity_Profile-style)."""
    original_split_mode = deepcopy(cfg.dataset.split_mode)
    original_split = deepcopy(cfg.dataset.split)
    if bool(cfg.heterogeneity.use_random_split):
        cfg.dataset.split_mode = "random"
        cfg.dataset.split = list(cfg.heterogeneity.split)
    return original_split_mode, original_split


def run_heterogeneity_profile() -> None:
    """
    Multi-trial heterogeneity experiment until each graph has enough test appearances.

    Uses ``train.mode: heterogeneity`` and ``cfg.heterogeneity.*`` settings.
    """
    output_dir = str(cfg.out_dir)
    dataset_name = str(cfg.dataset.name)
    model_slug = get_gnnplus_model_slug(cfg)
    num_layers = int(cfg.gnn.layers_mp)
    required = int(cfg.heterogeneity.required_test_appearances)
    max_trials = int(cfg.heterogeneity.max_trials)
    base_seed = int(cfg.heterogeneity.base_seed if cfg.heterogeneity.base_seed >= 0 else cfg.seed)

    seed_everything(base_seed)
    loaders = create_loader()
    root = resolve_root_dataset(loaders[0].dataset)
    num_graphs = len(root)

    graph_dict: Dict[int, List[Union[int, float]]] = {i: [] for i in range(num_graphs)}
    test_appearances: Dict[int, int] = {i: 0 for i in range(num_graphs)}
    trial_metrics: List[float] = []

    logging.info(
        "Heterogeneity profile: dataset=%s model=%s layers=%d graphs=%d "
        "required_appearances=%d max_trials=%d",
        dataset_name,
        model_slug,
        num_layers,
        num_graphs,
        required,
        max_trials,
    )

    original_split_mode, original_split = _apply_heterogeneity_split_overrides()

    trial = 0
    try:
        while True:
            trial += 1
            min_app = min(test_appearances.values()) if test_appearances else 0
            if min_app >= required:
                logging.info(
                    "All graphs reached >=%d test appearances; stopping before trial %d",
                    required,
                    trial,
                )
                break
            if trial > max_trials:
                logging.warning(
                    "Reached max_trials=%d with min appearances %d/%d",
                    max_trials,
                    min_app,
                    required,
                )
                break

            per_graph, trial_metric, test_indices = run_single_heterogeneity_trial(
                trial=trial,
                base_seed=base_seed,
            )
            trial_metrics.append(trial_metric)

            for graph_idx in test_indices:
                test_appearances[int(graph_idx)] += 1
                if int(graph_idx) in per_graph:
                    graph_dict[int(graph_idx)].append(per_graph[int(graph_idx)])

            logging.info(
                "Trial %d complete: test_metric=%.4f min_appearances=%d",
                trial,
                trial_metric,
                min(test_appearances.values()),
            )
    finally:
        cfg.dataset.split_mode = original_split_mode
        cfg.dataset.split = original_split

    if not bool(cfg.heterogeneity.save_pickle):
        return

    save_heterogeneity_artifacts(
        graph_dict=graph_dict,
        test_appearances=test_appearances,
        trial_metrics=trial_metrics,
        output_dir=output_dir,
    )
