"""``scatter`` with the ``torch_scatter`` signature, without requiring the extension.

``torch_scatter`` ships compiled wheels per torch/CUDA version, which are often
unavailable for the torch preinstalled on Google Colab. When it cannot be
imported, :func:`scatter` falls back to :func:`torch_geometric.utils.scatter`.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor

try:
    from torch_scatter import scatter as _torch_scatter
except ImportError:  # pragma: no cover - depends on the environment
    _torch_scatter = None

HAS_TORCH_SCATTER: bool = _torch_scatter is not None


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """Reduce ``src`` into groups given by ``index`` along ``dim``.

    Args:
        src: Source tensor.
        index: Group index for each entry of ``src`` along ``dim``.
        dim: Dimension to scatter along.
        out: Optional output tensor (only supported with ``torch_scatter``).
        dim_size: Size of the output along ``dim``.
        reduce: One of ``"sum"``, ``"add"``, ``"mean"``, ``"min"``, ``"max"``, ``"mul"``.

    Returns:
        The reduced tensor.
    """
    if _torch_scatter is not None:
        return _torch_scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce)
    if out is not None:
        raise NotImplementedError("scatter(out=...) requires torch_scatter")
    from torch_geometric.utils import scatter as pyg_scatter

    if dim < 0:
        dim = src.dim() + dim
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
