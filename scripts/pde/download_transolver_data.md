# Downloading Transolver PDE data for GNNPlus

GNNPlus does **not** auto-download Google-Drive archives. Place files under
`$GNNPLUS_DATASET_DIR` as below, then the PyG loaders build kNN graphs on first
run.

## Standard-6 (FNO / GeoFNO)

Links from [Transolver PDE README](https://github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark):

| Dataset | Raw files under `TransolverPDE/<name>/raw/` |
|---------|-----------------------------------------------|
| Elasticity | `Random_UnitCell_sigma_10.npy`, `Random_UnitCell_XY_10.npy` |
| Darcy | `piececonst_r421_N1024_smooth1.mat`, `piececonst_r421_N1024_smooth2.mat` |
| Airfoil | `NACA_Cylinder_X.npy`, `NACA_Cylinder_Y.npy`, `NACA_Cylinder_Q.npy` |
| Pipe | `Pipe_X.npy`, `Pipe_Y.npy`, `Pipe_Q.npy` |
| Plasticity | `Plasticity_X.npy`, `Plasticity_Y.npy`, `Plasticity_Q.npy` |
| Navier–Stokes | one `*.mat` with 4D field `u` `[N,H,W,T]` |

Example:

```bash
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
mkdir -p "$GNNPLUS_DATASET_DIR/TransolverPDE/elasticity/raw"
# copy npy files into that directory
```

`smoke` needs no download (synthetic grids).

## AirfRANS / ShapeNetCar

Use Transolver’s official preprocess (VTK/VTU → per-sample `x.npy`, `y.npy`,
`pos.npy`, optional `edge_index.npy`), then:

```text
$GNNPLUS_DATASET_DIR/AirfRANS/airfrans/raw/<sample_id>/{x,y,pos}.npy
$GNNPLUS_DATASET_DIR/ShapeNetCar/shapenet_car/raw/<sample_id>/{x,y,pos}.npy
```

Alternatively dump a list of `torch_geometric.data.Data` as `raw/*.pt`.

References:

- https://github.com/thuml/Transolver/tree/main/Airfoil-Design-AirfRANS
- https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar
