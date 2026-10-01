from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

from GNNPlus.preprocessing.graph_augmentations import VIRTUAL_NODE_LABEL_IGNORE


def _valid_target_mask(true: torch.Tensor) -> torch.Tensor:
    """Boolean mask over flattened targets that are not virtual-node ignores."""
    return true.view(-1) != VIRTUAL_NODE_LABEL_IGNORE


@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted cross-entropy for unbalanced classes.

    Virtual-node ignore labels (``VIRTUAL_NODE_LABEL_IGNORE``) are excluded from
    class-weight estimation and from the NLL / BCE terms via ``ignore_index``.
    Returned ``pred_score`` keeps the original batch length so the train logger
    assert ``true.shape[0] == pred.shape[0]`` still holds.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        flat_true = true.view(-1)
        valid = _valid_target_mask(true)
        if not bool(valid.any()):
            zero = pred.sum() * 0.0
            if pred.ndim > 1:
                return zero, F.log_softmax(pred, dim=-1)
            return zero, torch.sigmoid(pred)

        true_valid = flat_true[valid]
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        # Class weights from real nodes only (ignore sentinel must not enter bincount).
        label_count = torch.bincount(true_valid, minlength=n_classes)
        cluster_sizes = label_count.to(device=pred.device)
        v_count = float(true_valid.size(0))
        weight = (v_count - cluster_sizes.float()) / v_count
        weight *= (cluster_sizes > 0).float()

        # multiclass
        if pred.ndim > 1:
            pred_log = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(
                pred_log,
                flat_true,
                weight=weight,
                ignore_index=VIRTUAL_NODE_LABEL_IGNORE,
            )
            return loss, pred_log
        # binary
        pred_valid = pred.view(-1)[valid]
        loss = F.binary_cross_entropy_with_logits(
            pred_valid,
            true_valid.float(),
            weight=weight[true_valid],
        )
        return loss, torch.sigmoid(pred)
