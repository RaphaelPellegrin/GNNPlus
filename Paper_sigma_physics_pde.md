# SiGMA + Transolver++ Physics-Attention on PDE suite

Hybrid: **a2g4** (`GCN,GIN,SAGE,GAT`) + `attn_type=physics` (Gumbel + adaptive temperature), kNN edges (`k=8`), W&B project `GNNPlus_PDE_Physics`.

## Datasets (array tasks 0–7)

| Task | Dataset | Config |
|------|---------|--------|
| 0 | Elasticity | `configs/gated_hybrid/pde_physics/elasticity-a2g4-physics.yaml` |
| 1 | Plasticity | `.../plasticity-a2g4-physics.yaml` |
| 2 | Airfoil | `.../airfoil-a2g4-physics.yaml` |
| 3 | Pipe | `.../pipe-a2g4-physics.yaml` |
| 4 | Darcy | `.../darcy-a2g4-physics.yaml` |
| 5 | Navier–Stokes | `.../navier_stokes-a2g4-physics.yaml` |
| 6 | AirfRANS | `.../airfrans-a2g4-physics.yaml` |
| 7 | ShapeNetCar | `.../shapenet_car-a2g4-physics.yaml` |

Smoke (local wiring): `configs/gated_hybrid/pde_physics/smoke-a2g4-physics.yaml`.

## Data layout

See [`scripts/pde/download_transolver_data.md`](scripts/pde/download_transolver_data.md).

```
$GNNPLUS_DATASET_DIR/
  TransolverPDE/<name>/raw/     # standard-6 (+ smoke builds synthetically)
  AirfRANS/airfrans/raw/        # preprocessed sample dirs or .pt
  ShapeNetCar/shapenet_car/raw/
```

## Cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

# Standard-6 first if industrial not ready:
PDE_ARRAY=0-5 bash bash_interface/cluster/submit_sigma_pde_physics.sh

# Full suite (max 2 GPUs):
bash bash_interface/cluster/submit_sigma_pde_physics.sh
```

Outs: `$GNNPLUS_OUT_DIR/sigma_pde_physics/<tag>_a2g4_physics_seed0/`

## Metric

Primary: **`rel_l2`** (argmin). Train loss: MSE (`model.loss_fun: mse`).

## Smoke (local)

| Date | Status | Notes |
|------|--------|-------|
| 2026-08-11 | OK | `smoke-a2g4-physics.yaml`, 2 epochs, `task=graph`, `rel_l2` logged; Physics-Attn + kNN forward OK |

## JOBIDs

| Date | JOBID | Notes |
|------|-------|-------|
|  |  | *Submit on FASRC after `git pull` + data present (Duo SSH). Use `PDE_ARRAY=0-5` if industrial missing.* |
