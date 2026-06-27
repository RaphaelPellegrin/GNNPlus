"""Configuration for GRIT (Graph Inductive Bias Transformer)."""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_gt")
def set_cfg_gt(cfg: CN) -> None:
    """Register ``cfg.gt`` for standalone :class:`GritTransformer` models."""
    cfg.gt = CN(new_allowed=True)
    cfg.gt.layer_type = "GritTransformer"
    cfg.gt.layers = 10
    cfg.gt.n_heads = 8
    cfg.gt.dim_hidden = 64
    cfg.gt.dropout = 0.0
    cfg.gt.attn_dropout = 0.2
    cfg.gt.layer_norm = False
    cfg.gt.batch_norm = True
    cfg.gt.residual = True
    cfg.gt.update_e = True
    cfg.gt.bn_momentum = 0.1
    cfg.gt.bn_no_runner = False
    cfg.gt.rezero = False

    cfg.gt.attn = CN(new_allowed=True)
    cfg.gt.attn.use = True
    cfg.gt.attn.use_bias = False
    cfg.gt.attn.clamp = 5.0
    cfg.gt.attn.act = "relu"
    cfg.gt.attn.edge_enhance = True
    cfg.gt.attn.deg_scaler = True
    cfg.gt.attn.full_attn = True
    cfg.gt.attn.norm_e = True
    cfg.gt.attn.O_e = True


@register_config("grit_hybrid")
def grit_hybrid_cfg(cfg: CN) -> None:
    """Register ``cfg.gnn.grit`` hyperparameters for hybrid GRIT MP heads."""
    cfg.gnn.grit = CN(new_allowed=True)
    cfg.gnn.grit.n_heads = 8
    cfg.gnn.grit.dropout = 0.0
    cfg.gnn.grit.attn_dropout = 0.2
    cfg.gnn.grit.layer_norm = False
    cfg.gnn.grit.batch_norm = True
    cfg.gnn.grit.residual = True
    cfg.gnn.grit.norm_e = True
    cfg.gnn.grit.update_e = False
    cfg.gnn.grit.act = "relu"
