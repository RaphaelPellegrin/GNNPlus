import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

from GNNPlus.preprocessing.graph_augmentations import VIRTUAL_NODE_LABEL_IGNORE


def _mask_virtual_node_targets(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop per-node rows with the virtual-node ignore label."""
    if true.numel() == 0:
        return pred, true
    flat = true.view(-1)
    valid = flat != VIRTUAL_NODE_LABEL_IGNORE
    if valid.all():
        return pred, true
    return pred[valid], flat[valid]


@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        pred, true = _mask_virtual_node_targets(pred, true)
        if true.numel() == 0:
            zero = pred.sum() * 0.0
            return zero, pred
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)
