"""Transolver standard PDE benchmarks as PyG InMemoryDatasets.

Supports Elasticity, Plasticity, Airfoil, Pipe, Darcy, Navier–Stokes, plus a
tiny synthetic ``smoke`` set for CI / local wiring tests.

Raw files follow the layouts used by ``thuml/Transolver``
``PDE-Solving-StandardBenchmark`` (FNO / GeoFNO releases). Place them under
``$GNNPLUS_DATASET_DIR/TransolverPDE/<name>/`` (or the format-specific root
passed by GraphGym).
"""

from __future__ import annotations

import logging
import os.path as osp
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg

from GNNPlus.loader.dataset.pde_common import (
    make_mesh_data,
    train_val_test_index_split,
)

logger = logging.getLogger(__name__)

_SUPPORTED = (
    "smoke",
    "elasticity",
    "plasticity",
    "airfoil",
    "pipe",
    "darcy",
    "navier_stokes",
    "ns",
)


class TransolverPDE(InMemoryDataset):
    """PyG wrapper around Transolver standard PDE meshes."""

    def __init__(
        self,
        root: str,
        name: str,
        *,
        knn_k: int = 8,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        transform: Optional[Callable[..., Any]] = None,
        pre_transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Load or process a Transolver PDE dataset.

        Args:
            root: Dataset root (GraphGym joins format id already).
            name: Benchmark name (see ``_SUPPORTED``).
            knn_k: Neighbors for kNN edges.
            n_train: Optional override for train count.
            n_test: Optional override for test count.
            transform: Optional PyG transform.
            pre_transform: Optional PyG pre-transform.
        """
        self.name = str(name).strip().lower().replace("-", "_")
        if self.name == "ns":
            self.name = "navier_stokes"
        if self.name not in _SUPPORTED:
            raise ValueError(
                f"Unknown Transolver PDE name {name!r}; "
                f"expected one of {_SUPPORTED}"
            )
        self.knn_k = int(knn_k)
        self._n_train_override = n_train
        self._n_test_override = n_test
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
        # Soft requirement — process() checks concretely.
        return []

    def download(self) -> None:
        """No automatic download (manual Google-Drive fetch; see scripts/pde/)."""
        return

    def process(self) -> None:
        """Convert raw Transolver arrays into a list of mesh ``Data`` objects."""
        if self.name == "smoke":
            data_list = self._process_smoke()
        elif self.name == "elasticity":
            data_list = self._process_elasticity()
        elif self.name == "darcy":
            data_list = self._process_darcy()
        elif self.name == "airfoil":
            data_list = self._process_structured_geo(
                x_name="NACA_Cylinder_X.npy",
                y_name="NACA_Cylinder_Y.npy",
                q_name="NACA_Cylinder_Q.npy",
                out_channel=0,
            )
        elif self.name == "pipe":
            data_list = self._process_structured_geo(
                x_name="Pipe_X.npy",
                y_name="Pipe_Y.npy",
                q_name="Pipe_Q.npy",
                out_channel=0,
            )
        elif self.name == "plasticity":
            data_list = self._process_plasticity()
        elif self.name == "navier_stokes":
            data_list = self._process_navier_stokes()
        else:
            raise ValueError(self.name)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        n_total = len(data_list)
        n_train, n_test = self._default_counts(n_total)
        train_idx, val_idx, test_idx = train_val_test_index_split(
            n_total, n_train, n_test
        )
        # Store split lists for master_loader.
        self._split_idxs = (train_idx, val_idx, test_idx)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        split_path = osp.join(self.processed_dir, "splits.pt")
        torch.save(
            {"train": train_idx, "val": val_idx, "test": test_idx},
            split_path,
        )
        logger.info(
            "Processed %s: %d graphs (train=%d val=%d test=%d) knn_k=%d",
            self.name,
            n_total,
            len(train_idx),
            len(val_idx),
            len(test_idx),
            self.knn_k,
        )

    def _default_counts(self, n_total: int) -> tuple[int, int]:
        """Return ``(n_train, n_test)`` for this benchmark."""
        defaults = {
            "smoke": (8, 2),
            "elasticity": (1000, 200),
            "darcy": (1000, 200),
            "airfoil": (1000, 200),
            "pipe": (1000, 200),
            "plasticity": (900, 80),
            "navier_stokes": (1000, 200),
        }
        n_train_def, n_test_def = defaults.get(
            self.name,
            (max(1, n_total * 4 // 5), max(1, n_total // 5)),
        )
        n_train = (
            int(self._n_train_override)
            if self._n_train_override is not None
            else int(n_train_def)
        )
        n_test = (
            int(self._n_test_override)
            if self._n_test_override is not None
            else int(n_test_def)
        )
        n_train = min(n_train, max(1, n_total - 1))
        n_test = min(n_test, max(1, n_total - n_train))
        return n_train, n_test

    def get_idx_split(self) -> dict[str, List[int]]:
        """Return train/val/test index lists (reads side-car if needed)."""
        split_path = osp.join(self.processed_dir, "splits.pt")
        if osp.isfile(split_path):
            return torch.load(split_path, weights_only=False)
        n = len(self)
        n_train, n_test = self._default_counts(n)
        tr, va, te = train_val_test_index_split(n, n_train, n_test)
        return {"train": tr, "val": va, "test": te}

    def _process_smoke(self) -> List[Data]:
        """Tiny synthetic Darcy-like grids for wiring tests."""
        rng = np.random.default_rng(0)
        data_list: List[Data] = []
        res = 8
        xs = np.linspace(0.0, 1.0, res)
        ys = np.linspace(0.0, 1.0, res)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        pos = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        for _ in range(12):
            coeff = rng.normal(size=(res * res, 1)).astype(np.float32)
            sol = (0.5 * coeff + 0.1 * pos[:, :1]).astype(np.float32)
            x = np.concatenate([pos, coeff], axis=-1)
            data_list.append(
                make_mesh_data(
                    x=torch.from_numpy(x),
                    y=torch.from_numpy(sol),
                    pos=torch.from_numpy(pos),
                    knn_k=self.knn_k,
                )
            )
        return data_list

    def _require_files(self, names: Sequence[str]) -> List[Path]:
        paths = [Path(self.raw_dir) / n for n in names]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw files for {self.name}. Expected under {self.raw_dir}: "
                + ", ".join(missing)
                + ". See scripts/pde/download_transolver_data.md"
            )
        return paths

    def _process_elasticity(self) -> List[Data]:
        sigma_p, xy_p = self._require_files(
            [
                "Random_UnitCell_sigma_10.npy",
                "Random_UnitCell_XY_10.npy",
            ]
        )
        # Transolver: sigma [N_pts, N_samp] after permute; XY [N_samp, N_pts, 2]
        sigma = np.load(sigma_p)  # typically [N_pts, N_samp]
        xy = np.load(xy_p)
        if sigma.ndim == 2 and sigma.shape[0] < sigma.shape[1]:
            # [N_samp, N_pts] → transpose to [N_pts, N_samp]
            pass
        # Match Transolver exp_elas: permute sigma (1,0), xy (2,0,1)
        input_s = torch.tensor(sigma, dtype=torch.float).permute(1, 0)  # N_samp, N_pts
        input_xy = torch.tensor(xy, dtype=torch.float)
        if input_xy.dim() == 3 and input_xy.shape[-1] != 2:
            input_xy = input_xy.permute(2, 0, 1)
        if input_xy.dim() == 3 and input_xy.shape[0] != input_s.shape[0]:
            # [N_pts, N_samp, 2] → [N_samp, N_pts, 2]
            if input_xy.shape[1] == input_s.shape[0]:
                input_xy = input_xy.permute(1, 0, 2)
        data_list: List[Data] = []
        for i in range(input_s.size(0)):
            pos = input_xy[i]
            y = input_s[i].unsqueeze(-1)
            x = pos.clone()
            data_list.append(
                make_mesh_data(x=x, y=y, pos=pos, knn_k=self.knn_k)
            )
        return data_list

    def _process_darcy(self) -> List[Data]:
        train_p, test_p = self._require_files(
            [
                "piececonst_r421_N1024_smooth1.mat",
                "piececonst_r421_N1024_smooth2.mat",
            ]
        )
        import scipy.io as scio

        r = int(getattr(cfg.dataset, "darcy_downsample", 5))
        h = int(((421 - 1) / r) + 1)
        s = h

        def _load_mat(path: Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
            mat = scio.loadmat(str(path))
            coeff = mat["coeff"][:n, ::r, ::r][:, :s, :s]
            sol = mat["sol"][:n, ::r, ::r][:, :s, :s]
            return (
                torch.from_numpy(coeff.reshape(n, -1)).float(),
                torch.from_numpy(sol.reshape(n, -1)).float(),
            )

        n_train = self._n_train_override or 1000
        n_test = self._n_test_override or 200
        x_tr, y_tr = _load_mat(train_p, n_train)
        x_te, y_te = _load_mat(test_p, n_test)
        xs = np.linspace(0.0, 1.0, s)
        ys = np.linspace(0.0, 1.0, s)
        xx, yy = np.meshgrid(xs, ys)
        pos_np = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        pos = torch.from_numpy(pos_np)

        data_list: List[Data] = []
        for i in range(x_tr.size(0)):
            feat = torch.cat([pos, x_tr[i].unsqueeze(-1)], dim=-1)
            data_list.append(
                make_mesh_data(
                    x=feat,
                    y=y_tr[i].unsqueeze(-1),
                    pos=pos,
                    knn_k=self.knn_k,
                )
            )
        for i in range(x_te.size(0)):
            feat = torch.cat([pos, x_te[i].unsqueeze(-1)], dim=-1)
            data_list.append(
                make_mesh_data(
                    x=feat,
                    y=y_te[i].unsqueeze(-1),
                    pos=pos,
                    knn_k=self.knn_k,
                )
            )
        return data_list

    def _process_structured_geo(
        self,
        *,
        x_name: str,
        y_name: str,
        q_name: str,
        out_channel: int,
    ) -> List[Data]:
        """Airfoil / Pipe GeoFNO structured meshes."""
        xp, yp, qp = self._require_files([x_name, y_name, q_name])
        xx = np.load(xp)
        yy = np.load(yp)
        qq = np.load(qp)
        # Expect shapes like [N_samp, H, W] for X/Y and [N_samp, H, W, C] for Q.
        if qq.ndim == 3:
            qq = qq[..., None]
        data_list: List[Data] = []
        n_samp = qq.shape[0]
        for i in range(n_samp):
            pos = np.stack([xx[i].ravel(), yy[i].ravel()], axis=-1).astype(np.float32)
            y = qq[i, ..., out_channel].ravel().astype(np.float32)[:, None]
            x = pos.copy()
            data_list.append(
                make_mesh_data(
                    x=torch.from_numpy(x),
                    y=torch.from_numpy(y),
                    pos=torch.from_numpy(pos),
                    knn_k=self.knn_k,
                )
            )
        return data_list

    def _process_plasticity(self) -> List[Data]:
        paths = list(Path(self.raw_dir).glob("*.npy")) + list(
            Path(self.raw_dir).glob("*.mat")
        )
        if not paths:
            raise FileNotFoundError(
                f"No plasticity raw files under {self.raw_dir}. "
                "See scripts/pde/download_transolver_data.md"
            )
        # Prefer common GeoFNO plasticity npy trio if present.
        cand = {
            "x": Path(self.raw_dir) / "Plasticity_X.npy",
            "y": Path(self.raw_dir) / "Plasticity_Y.npy",
            "q": Path(self.raw_dir) / "Plasticity_Q.npy",
        }
        if all(p.is_file() for p in cand.values()):
            return self._process_structured_geo(
                x_name="Plasticity_X.npy",
                y_name="Plasticity_Y.npy",
                q_name="Plasticity_Q.npy",
                out_channel=0,
            )
        raise FileNotFoundError(
            "Plasticity loader expects Plasticity_{X,Y,Q}.npy in raw/ "
            f"(found {[p.name for p in paths]})"
        )

    def _process_navier_stokes(self) -> List[Data]:
        mats = sorted(Path(self.raw_dir).glob("*.mat"))
        if not mats:
            raise FileNotFoundError(
                f"No Navier–Stokes .mat under {self.raw_dir}. "
                "See scripts/pde/download_transolver_data.md"
            )
        import scipy.io as scio

        mat = scio.loadmat(str(mats[0]))
        # FNO NS: 'u' shaped [N, H, W, T]
        key = "u" if "u" in mat else None
        if key is None:
            for k, v in mat.items():
                if isinstance(v, np.ndarray) and v.ndim == 4:
                    key = k
                    break
        if key is None:
            raise KeyError(f"Could not find 4D field in {mats[0]}")
        u = mat[key]
        n_samp, h, w, t = u.shape
        # Predict last frame from earlier stacked frames (T_in=10 common).
        t_in = min(10, t - 1)
        xs = np.linspace(0.0, 1.0, w)
        ys = np.linspace(0.0, 1.0, h)
        xx, yy = np.meshgrid(xs, ys)
        pos_np = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        pos = torch.from_numpy(pos_np)
        data_list: List[Data] = []
        for i in range(n_samp):
            frames = u[i, ..., :t_in].reshape(h * w, t_in).astype(np.float32)
            target = u[i, ..., -1].reshape(h * w, 1).astype(np.float32)
            x = np.concatenate([pos_np, frames], axis=-1)
            data_list.append(
                make_mesh_data(
                    x=torch.from_numpy(x),
                    y=torch.from_numpy(target),
                    pos=pos,
                    knn_k=self.knn_k,
                )
            )
        return data_list


def preformat_transolver_pde(dataset_dir: str, name: str) -> TransolverPDE:
    """Factory used by ``master_loader``."""
    knn_k = int(getattr(cfg.dataset, "knn_k", 8))
    return TransolverPDE(root=dataset_dir, name=name, knn_k=knn_k)
