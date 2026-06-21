"""Configuration for gated hybrid (attention + MP) models in GNNPlus."""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('gated_hybrid')
def gated_hybrid_cfg(cfg: CN) -> None:
    """Register ``cfg.gnn.hybrid`` for :class:`HybridGNN`."""
    cfg.gnn.hybrid = CN(new_allowed=True)
    cfg.gnn.hybrid.num_attn_heads = 2
    cfg.gnn.hybrid.num_gnn_heads = 2
    cfg.gnn.hybrid.d_h = 16
    cfg.gnn.hybrid.attn_mask = 'full'  # full | graph_restricted
    cfg.gnn.hybrid.gate = 'headwise'  # elementwise | headwise
    cfg.gnn.hybrid.norm = 'layernorm'  # layernorm | rmsnorm
    cfg.gnn.hybrid.gnn_types = ''  # e.g. "GCN,GIN,GINE,SAGE"
    cfg.gnn.hybrid.attn_dropout = 0.1
    cfg.gnn.hybrid.mp_dropout = 0.0  # 0 => use cfg.gnn.dropout
    cfg.gnn.hybrid.block_bn = False
    cfg.gnn.hybrid.log_gate_stats = True  # W&B gates/layer*/attn_* (headwise + elementwise)
