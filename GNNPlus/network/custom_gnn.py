import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from typing import Any, Dict, List, Tuple

from GNNPlus.layer.gatedgcn_layer import GatedGCNLayer
from GNNPlus.layer.gine_conv_layer import GINEConvLayer

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual,ffn=cfg.gnn.ffn))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcn':
            return GatedGCNLayer
        elif model_type == 'gine':
            return GINEConvLayer
        elif model_type == 'gin':
            from GNNPlus.layer.gin_conv_layer import GINConvLayer
            return GINConvLayer
        elif model_type == 'gcn':
            from GNNPlus.layer.gcn_conv_layer import GCNConvLayer
            return GCNConvLayer
        elif model_type == 'gcne':
            from GNNPlus.layer.gcn_conv_layer_e import GCNConvLayer
            return GCNConvLayer
        elif model_type in ('sage', 'graphsage'):
            from GNNPlus.layer.sage_conv_layer import SAGEConvLayer
            return SAGEConvLayer
        elif model_type in ('gat',):
            from GNNPlus.layer.gat_conv_layer import GATConvLayer
            return GATConvLayer
        elif model_type in ('unitarygcn', 'unigcn', 'unitarygcnconv'):
            from GNNPlus.layer.unitary_conv_layer import UnitaryGCNConvLayer
            return UnitaryGCNConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

    def forward_all_layer_features(
        self, batch: Any
    ) -> Tuple[List[Any], Any]:
        """Return node features after every MP layer (before the head).

        Returns:
            ``(layer_features, batch)`` with one ``[N, F]`` tensor per MP layer.
        """
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        layer_features: List[Any] = []
        for layer in self.gnn_layers:
            batch = layer(batch)
            layer_features.append(batch.x)
        return layer_features, batch

    def forward_node_features(self, batch: Any) -> Tuple[Any, Any]:
        """Return last-layer node features before the prediction head.

        Convenience wrapper around :meth:`forward_all_layer_features`.
        """
        layer_features, batch = self.forward_all_layer_features(batch)
        return layer_features[-1], batch

    def collect_gate_stats(self, batch: Any) -> Dict[str, float]:
        """Run encoder + MP layers; return per-layer GCNE gate means."""
        batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        stats: dict[str, float] = {}
        for layer_idx, layer in enumerate(self.gnn_layers):
            batch = layer(batch)
            gate_mean = getattr(layer, 'last_gate_mean', None)
            if gate_mean is not None:
                stats[f'layer{layer_idx}/gcne_gate_mean'] = float(gate_mean)
        return stats


@register_network('custom_gnn_gated')
class CustomGNNGated(CustomGNN):
    """``custom_gnn`` with per-layer γ gates on GCNE (fairness ladder level 1)."""
