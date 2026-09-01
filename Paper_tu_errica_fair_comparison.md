# TU Errica-fair comparison (Layer 2 rebuttal)

Fair comparison under [Errica et al. ICLR 2020](https://arxiv.org/pdf/1912.09893) protocol:
- **10-fold** fixed stratified CV splits from [diningphil/gnn-comparison](https://github.com/diningphil/gnn-comparison)
- Inner **90/10 holdout** train/val per fold (from vendored JSON)
- **Early stopping** patience=500 on val accuracy (GIN/SAGE recipe)
- **Per-fold HP selection** from Errica published grids (not single canonical HP)
- **3 random restarts** per selected HP at eval
- Social datasets: **scalar degree** node features (COLLABORATIVE_DEGREE splits)

## Target claim (hybrid Option 3)

> Under Errica's 10-fold protocol with **per-dataset / per-fold hyperparameter selection**
> from their published grids, **SiGMA hetero (a2g4)** matches or exceeds the best classical
> GNN (GIN / GraphSAGE / GCN / GAT) on **X/7** datasets and matches or exceeds Errica's reported GIN on
> **Y/7**, with **SiGMA params ≤ GIN winner params** on bio datasets (budget-matched `d_h`).

## Errica GIN reference (Table 3/4, degree social)

| Dataset | Errica GIN |
|---------|------------|
| ENZYMES | 29.5 ± 8.2 |
| PROTEINS | 73.3 ± 4.0 |
| NCI1 | 80.0 ± 1.4 |
| DD | 76.6 ± 4.3 |
| IMDB-BINARY | 71.2 ± 3.9 |
| REDDIT-BINARY | 89.9 ± 1.9 |
| COLLAB | 75.6 ± 2.3 |

## Classical baselines

| Model | HP grid source | Grid size | grid_select jobs |
|-------|----------------|-----------|------------------|
| GIN | Errica `config_GIN.yml` | 64 | 7 × 64 × 10 = **4,480** |
| GraphSAGE | Errica `config_GraphSAGE.yml` | 72 | 7 × 72 × 10 = **5,040** |
| GCN | GIN-isomorphic† | 32 | 7 × 32 × 10 = **2,240** |
| GAT | GIN-isomorphic† | 32 | 7 × 32 × 10 = **2,240** |
| SiGMA hetero | hybrid Option 3 | per-fold | see `sigma_grids/manifest.json` |

†Errica's [gnn-comparison](https://github.com/diningphil/gnn-comparison) repo has no `config_GCN.yml` /
`config_GAT.yml`. GCN/GAT use the same **protocol** (splits, early stop, Adam+StepLR) with a
GIN-isomorphic grid (batch, lr, width, pool, dropout, early-stop criterion).

## Campaign status

**Last updated:** 2026-08-30

| Phase | Campaign | Status | Notes |
|-------|----------|--------|-------|
| 0 smoke | `canonical` **42673425** | done | Pipeline OK; **not** final table (fixed HP) |
| **1a** | `grid_select` GIN **42750648** | **done** | 4480/4480 COMPLETED · 6 log errors |
| **1b** | `grid_select` GraphSAGE **43116245** | **running** | 7 × 72 × 10 = **5,040** jobs |
| **1c** | `grid_select` GCN | **todo** | 7 × 32 × 10 = **2,240** jobs |
| **1d** | `grid_select` GAT | **todo** | 7 × 32 × 10 = **2,240** jobs |
| **2a** | `aggregate_hp_selection` GIN/SAGE/GCN/GAT | **todo** | → `selections/*_per_fold.json` |
| **2b** | `generate_sigma_errica_grids` | **todo** | hybrid SiGMA grids + manifest |
| **3a** | `sigma_grid_select` | **todo** | bio: L/H from GIN winner + micro grid; social: full grid |
| **3b** | `aggregate_sigma_hp_selection` | **todo** | → `selections/sigma_per_fold.json` |
| **4** | `grid_eval` GIN/SAGE/GCN/GAT + `sigma_grid_eval` | **todo** | 3 seeds × 10 folds × 7 ds per model |

### Active SLURM (2026-08-30)

| JOBID | Campaign | Tasks | Status |
|-------|----------|-------|--------|
| **42750648** | GIN `grid_select` | 1–4480 | ✅ COMPLETED |
| **43116245** | GraphSAGE `grid_select` | 1–5040 | 🔄 running |

Logs: `logs_gnnplus/tu_errica_grid_select_gin_<JOBID>_<TASK>.log` (not `.out`).

### Canonical exploratory results (W&B, do not cite as final)

Fixed `GIN_CANONICAL` — useful signal only:

| Dataset | GIN | GraphSAGE | SiGMA | Errica GIN |
|---------|-----|-----------|-------|------------|
| ENZYMES | 45.0 | 51.3 | 54.1 | 29.5 |
| PROTEINS | 74.1 | 72.3 | 71.8 | 73.3 |
| NCI1 | 77.8 | 79.1 | **80.3** | 80.0 |
| DD | 72.4 | 70.7 | pending | 76.6 |
| IMDB-B | 71.0 | ~50† | 71.1 | 71.2 |
| REDDIT-B | 89.3 | ~50† | pending | 89.9 |
| COLLAB | 74.9 | 51.8 | **78.2** | 75.6 |

†GraphSAGE ~50% on social under canonical HP — expect grid_select to fix.

## Hybrid SiGMA search (Option 3)

| Family | Datasets | SiGMA HP rule |
|--------|----------|---------------|
| **Bio** | ENZYMES, PROTEINS, NCI1, DD | Match **GIN winner** `layers_mp` + `dim_inner` per fold; sweep `lr` × `bs` × `d_h` with **params ≤ GIN winner** |
| **Social** | IMDB-B, REDDIT-B, COLLAB | Full `SIGMA_GRID` (8 configs); **bs=16** on DD/REDDIT-B/COLLAB |

Scripts:
- `scripts/tu_errica/param_budget.py` — param counting + `d_h` under budget
- `scripts/tu_errica/generate_sigma_errica_grids.py` — builds `configs/tu_errica/sigma_grids/`
- `scripts/tu_errica/aggregate_hp_selection.py` — GIN/SAGE/GCN/GAT winners from W&B
- `scripts/tu_errica/aggregate_sigma_hp_selection.py` — SiGMA winners

## Launch (cluster)

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results

# Orchestrated phases:
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_gin
# ... after jobs finish:
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_gin
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_sage
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sage
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_gcn
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_gcn
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_select_gat
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_gat
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh generate_sigma_grids
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gin
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_sage
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gcn
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gat
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval
```

Or manual (with parallelism / walltime):

```bash
TU_ERRICA_CAMPAIGN=grid_select TU_ERRICA_GRID_MODEL=gin \
  TU_ERRICA_PARALLEL=20 TU_ERRICA_TIME=48:00:00 \
  bash bash_interface/cluster/submit_tu_errica_fair.sh
# → 42750648 (full GIN grid_select)
```

Monitor:

```bash
sacct -j 42750648 -X --format=State,ExitCode -n | awk '{print $1}' | sort | uniq -c
squeue -u $USER -n tu_errica_grid_select_gin | head
grep -l 'Error\|CUDA\|Traceback' logs_gnnplus/tu_errica_grid_select_gin_42750648_*.log 2>/dev/null | wc -l
tail -30 logs_gnnplus/tu_errica_grid_select_gin_42750648_1.log
```

## Aggregate final table

```bash
python scripts/tu_errica/aggregate_errica_results.py --source wandb --state finished
```

W&B groups: `tu_errica_<ds>_<Model>_<campaign>_selected` (after grid_eval).

## Code paths

- Splits: `splits/errica/`
- Loader: `GNNPlus/loader/errica_splits.py`, `split_mode: errica-cv-10`
- Configs: `configs/tu_errica/*-errica-base.yaml`
- HP grids: `configs/tu_errica/*_hp_grid.json`
- Selections: `configs/tu_errica/selections/`
- SLURM: `bash_interface/cluster/run_tu_errica_fair.sh`

## Scope note (Layer 1 — paper text)

Appendix F Table 17–18 used **50/25/25** random splits (internal comparison).
This campaign is **separate** — Errica splits, for reviewer-facing comparison.
