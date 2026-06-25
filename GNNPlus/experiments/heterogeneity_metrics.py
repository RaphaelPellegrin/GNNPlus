"""Lightweight heterogeneity metrics (no training dependencies)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def per_graph_performance_value(
    pred: torch.Tensor,
    true: torch.Tensor,
    dataset_task_type: str,
) -> float:
    """
    Scalar per-graph score stored in graph_dict.

    Classification: 1.0 if correct (or mean label accuracy for multilabel).
    Regression: mean absolute error for that graph.
    """
    if dataset_task_type == "regression":
        return float((pred.detach().squeeze() - true.detach().squeeze()).abs().mean().cpu())

    if dataset_task_type == "classification_multilabel":
        pred_bin = (torch.sigmoid(pred.detach()) > 0.0).float()
        true_f = true.detach().float()
        return float((pred_bin == true_f).float().mean().cpu())

    if pred.dim() > 1 and pred.size(-1) > 1:
        pred_int = pred.detach().argmax(dim=-1)
        true_int = true.detach().view(-1)
        return float((pred_int == true_int).float().mean().cpu())

    pred_int = pred.detach().view(-1).round()
    true_int = true.detach().view(-1)
    return float((pred_int == true_int).float().item())


def summarize_trial_metrics(
    trial_metrics: List[float],
    task_type: str,
) -> Dict[str, float]:
    """Compute mean/std of trial-level test metrics."""
    arr = np.asarray(trial_metrics, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0.0}
    out: Dict[str, float] = {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
    if task_type == "regression":
        out["test_mae_mean"] = out["mean"]
        out["test_mae_std"] = out["std"]
    elif task_type == "classification_multilabel":
        out["test_ap_mean"] = out["mean"]
        out["test_ap_std"] = out["std"]
    else:
        out["test_accuracy_mean"] = out["mean"]
        out["test_accuracy_std"] = out["std"]
    return out
