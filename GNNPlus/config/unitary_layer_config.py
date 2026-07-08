"""Configuration for unitary (complex-valued Taylor GCN) layers."""

from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('unitary_layer')
def unitary_layer_cfg(cfg: CN) -> None:
    """Register UniGCN / unitary convolution options on ``cfg.gnn``."""
    # Hermitian vs complex orthogonal weight parameterization.
    cfg.gnn.use_hermitian = False
    # Taylor expansion order T (default 16 in Weber-GeoML Unitary_Convolutions).
    cfg.gnn.unitary_taylor_order = 16
    # When True, Taylor output is real-valued (required for stacked layers / heads).
    cfg.gnn.unitary_return_real = True
