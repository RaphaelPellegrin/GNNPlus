"""Complex-valued activations for UniGCN (Weber-GeoML Unitary_Convolutions)."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.graphgym.register import register_act


class ComplexActivation(nn.Module):
    """Apply a real activation separately to real and imaginary parts."""

    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, input: Tensor) -> Tensor:
        if torch.is_complex(input):
            real_part = self.activation(input.real)
            imag_part = self.activation(input.imag)
            return torch.complex(real_part, imag_part)
        return self.activation(input)


def register_complex_act(name: str, activation: nn.Module) -> None:
    """Register ``ComplexActivation(activation)`` under ``name``."""
    complex_activation = partial(ComplexActivation, activation=activation)
    register_act(name, complex_activation)


register_complex_act("c_relu", nn.ReLU())
register_complex_act("c_tanh", nn.Tanh())
register_complex_act("c_sigmoid", nn.Sigmoid())
register_complex_act("c_gelu", nn.GELU())
