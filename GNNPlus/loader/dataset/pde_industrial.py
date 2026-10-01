"""Industrial PDE meshes (AirfRANS, ShapeNetCar) from Transolver preprocessed caches.

Raw VTK/VTU conversion is out of band (needs ``pyvista`` / ``vtk``). This loader
expects per-sample directories produced by Transolver's preprocessing::

    <root>/<name>/raw/<sample_id>/{x,y,pos}.npy  [optional edge_index.npy, surf.npy]

or a flat list of ``*.pt`` ``torch_geometric.data.Data`` files under ``raw/``.
"""

from __future__ import annotations

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg

from GNNPlus.loader.dataset.pde_common import (
    make_mesh_data,
    train_val_test_index_split,
)

logger = logging.getLogger(__name__)

_SUPPORTED = ("airfrans", "shapenet_car", "shapenetcar")


class IndustrialPDE(InMemoryDataset):
    """PyG dataset for AirfRANS / ShapeNetCar preprocessed meshes."""

    def __init__(
        self,
        root: str,
        name: str,
        *,
        knn_k: int = 8,
        transform: Optional[Callable[..., Any]] = None,
        pre_transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Load industrial PDE meshes.

        Args:
            root: Dataset root.
            name: ``airfrans`` or ``shapenet_car``.
            knn_k: Neighbors when ``edge_index`` is absent.
            transform: Optional transform.
            pre_transform: Optional pre-transform.
        """
        self.name = str(name).strip().lower().replace("-", "_")
        if self.name == "shapenetcar":
            self.name = "shapenet_car"
        if self.name not in ("airfrans", "shapenet_car"):
            raise ValueError(
                f"Unknown industrial PDE name {name!r}; expected airfrans|shapenet_car"
            )
        self.knn_k = int(knn_k)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], map_location="cpu", weights_only=False
        )

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self) -> List[str]:
        return [f"data_knn{self.knn_k}.pt"]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self) -> None:
        """No automatic download."""
        return

    def process(self) -> None:
        """Build InMemoryDataset from preprocessed sample folders or ``.pt`` files."""
        raw = Path(self.raw_dir)
        if not raw.is_dir():
            raise FileNotFoundError(
                f"Missing raw dir {raw}. Preprocess AirfRANS/ShapeNetCar "
                "(see scripts/pde/download_transolver_data.md)."
            )

        data_list: List[Data] = []
        pt_files = sorted(raw.glob("*.pt"))
        if pt_files:
            for p in pt_files:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(obj, Data):
                    data_list.append(self._ensure_edges(obj))
                else:
                    raise TypeError(f"{p} did not contain a PyG Data object")
        else:
            sample_dirs = sorted([d for d in raw.iterdir() if d.is_dir()])
            if not sample_dirs:
                raise FileNotFoundError(
                    f"No sample dirs or .pt files under {raw}"
                )
            for d in sample_dirs:
                data_list.append(self._load_sample_dir(d))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        n_total = len(data_list)
        if self.name == "airfrans":
            n_train, n_test = min(800, n_total - 1), min(200, max(1, n_total // 5))
        else:
            n_train, n_test = min(500, n_total - 1), min(100, max(1, n_total // 5))
        n_train = min(n_train, n_total - 1)
        n_test = min(n_test, n_total - n_train)
        train_idx, val_idx, test_idx = train_val_test_index_split(
            n_total, n_train, n_test
        )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(
            {"train": train_idx, "val": val_idx, "test": test_idx},
            osp.join(self.processed_dir, "splits.pt"),
        )
        logger.info(
            "Processed %s: %d graphs (train=%d val=%d test=%d)",
            self.name,
            n_total,
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

    def get_idx_split(self) -> dict[str, List[int]]:
        """Return train/val/test indices."""
        split_path = osp.join(self.processed_dir, "splits.pt")
        if osp.isfile(split_path):
            return torch.load(split_path, weights_only=False)
        n = len(self)
        tr, va, te = train_val_test_index_split(n, max(1, n * 4 // 5), max(1, n // 5))
        return {"train": tr, "val": va, "test": te}

    def _ensure_edges(self, data: Data) -> Data:
        """Add kNN edges when missing."""
        if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
            return data
        return make_mesh_data(
            x=data.x,
            y=data.y,
            pos=data.pos,
            knn_k=self.knn_k,
        )

    def _load_sample_dir(self, sample_dir: Path) -> Data:
        """Load one Transolver-style preprocessed sample directory."""
        x = np.load(sample_dir / "x.npy")
        y = np.load(sample_dir / "y.npy")
        pos = np.load(sample_dir / "pos.npy")
        edge_index = None
        ei_path = sample_dir / "edge_index.npy"
        if ei_path.is_file():
            edge_index = torch.from_numpy(np.load(ei_path)).long()
            if edge_index.dim() == 2 and edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()
        return make_mesh_data(
            x=torch.from_numpy(np.asarray(x)),
            y=torch.from_numpy(np.asarray(y)),
            pos=torch.from_numpy(np.asarray(pos)),
            knn_k=self.knn_k,
            edge_index=edge_index,
        )


def preformat_industrial_pde(dataset_dir: str, name: str) -> IndustrialPDE:
    """Factory used by ``master_loader``."""
    knn_k = int(getattr(cfg.dataset, "knn_k", 8))
    return IndustrialPDE(root=dataset_dir, name=name, knn_k=knn_k)
