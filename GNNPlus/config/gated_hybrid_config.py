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
    cfg.gnn.hybrid.gate = 'headwise'  # elementwise | headwise | none (ungated)
    cfg.gnn.hybrid.norm = 'layernorm'  # layernorm | rmsnorm | none
    cfg.gnn.hybrid.gnn_types = ''  # e.g. "GCN,GIN,GCNE,GATEDGCN" — see configs/gated_hybrid/README.md (GATEDGCN semantics)
    cfg.gnn.hybrid.attn_dropout = 0.1
    cfg.gnn.hybrid.mp_dropout = 0.0  # 0 => use cfg.gnn.dropout
    cfg.gnn.hybrid.block_bn = False
    # Block residual after fuse/out_proj (default True). Set False for L2bis ablations.
    cfg.gnn.hybrid.residual = True
    # When True (a0g1 + d_h == d), skip in/out Linear maps so MP runs on full-width x.
    # Gate uses a separate Linear (Level-1 style), not split(W_hg · x).
    cfg.gnn.hybrid.identity_proj = False
    cfg.gnn.hybrid.log_gate_stats = True  # W&B gates/layer*/attn_* (headwise + elementwise)
