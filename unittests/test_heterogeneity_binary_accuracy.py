"""Unit tests for heterogeneity-profile binary accuracy (logit vs sigmoid)."""

from __future__ import annotations

import torch


def _accuracy_from_pred_fixed(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mirror of fixed helper in ``run_heterogeneity_profiles`` (logits)."""
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)
    if pred.ndim > 1 and pred.size(-1) > 1:
        pred_cls = pred.argmax(dim=-1)
    else:
        pred_cls = (pred > 0).long().view(-1)
    true_cls = true.view(-1).long()
    return (pred_cls == true_cls).float()


def _accuracy_from_pred_buggy_sigmoid(
    pred_prob: torch.Tensor, true: torch.Tensor
) -> torch.Tensor:
    """Old bug: threshold sigmoid probs with ``> 0`` (almost always class 1)."""
    if pred_prob.ndim > 1:
        pred_prob = pred_prob.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)
    pred_cls = (pred_prob > 0).long().view(-1)
    true_cls = true.view(-1).long()
    return (pred_cls == true_cls).float()


def test_binary_logit_threshold_not_class_prior() -> None:
    """Logit > 0 must track the decision boundary, not always predict 1."""
    # Model predicts class 0, 1, 0, 1 via negative/positive logits.
    logits = torch.tensor([-2.0, 1.5, -0.1, 3.0])
    true = torch.tensor([0, 1, 0, 1])
    acc = float(_accuracy_from_pred_fixed(logits, true).mean())
    assert acc == 1.0

    # Same scores after sigmoid: buggy path always predicts class 1.
    probs = torch.sigmoid(logits)
    buggy = float(_accuracy_from_pred_buggy_sigmoid(probs, true).mean())
    # true has 50% positives → buggy accuracy collapses to 0.5
    assert buggy == 0.5


def test_binary_buggy_path_ignores_model() -> None:
    """With sigmoid+`>0`, accuracy equals positive-class rate (model ignored)."""
    probs = torch.tensor([0.01, 0.99, 0.2, 0.8])  # all > 0
    true = torch.tensor([0, 0, 1, 1])
    buggy = float(_accuracy_from_pred_buggy_sigmoid(probs, true).mean())
    assert buggy == 0.5  # always pred 1 → matches 2/4 labels


def test_multiclass_argmax() -> None:
    """Multiclass path uses argmax over class logits."""
    logits = torch.tensor([[0.1, 2.0, 0.0], [3.0, 0.0, 0.1]])
    true = torch.tensor([1, 0])
    acc = float(_accuracy_from_pred_fixed(logits, true).mean())
    assert acc == 1.0
