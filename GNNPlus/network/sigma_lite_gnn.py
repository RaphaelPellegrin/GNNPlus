"""SiGMA-lite network: :class:`HybridGNN` with :class:`SigmaLiteLayer` blocks."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from GNNPlus.layer.sigma_lite_layer import SigmaLiteLayer
from GNNPlus.network.hybrid_gnn import HybridGNN

_UNUSED_HYBRID_KWARGS = ("gate_mode", "mp_gate_mode", "block_bn", "block_dropout", "identity_proj")


@register_network('sigma_lite')
class SigmaLiteGNN(HybridGNN):
    """SiGMA-lite: ungated MMA heads with one gate on the mixed block output.

    Set ``model.type: sigma_lite``. Heads are configured with the same
    ``gnn.hybrid.*`` keys as SiGMA (``num_attn_heads``, ``num_gnn_heads``,
    ``d_h``, ``gnn_types``, ``attn_type``, ...); ``gnn.hybrid.gate`` /
    ``mp_gate`` are ignored because the heads are ungated. The gate
    nonlinearity is ``gnn.sigma_lite.gate_act`` (``sigmoid`` | ``gelu``).
    """

    def _build_layer(self, layer_kwargs: Dict[str, Any]) -> nn.Module:
        """Build a :class:`SigmaLiteLayer` from the shared hybrid kwargs."""
        if layer_kwargs.get("block_bn") or layer_kwargs.get("identity_proj"):
            raise ValueError("sigma_lite does not support gnn.hybrid.block_bn or identity_proj")
        kwargs = {k: v for k, v in layer_kwargs.items() if k not in _UNUSED_HYBRID_KWARGS}
        return SigmaLiteLayer(**kwargs, gate_act=str(cfg.gnn.sigma_lite.gate_act))
