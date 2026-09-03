# TU Errica-fair comparison (Layer 2 rebuttal)

Fair comparison under [Errica et al. ICLR 2020](https://arxiv.org/pdf/1912.09893) protocol:
- **10-fold** fixed stratified CV splits from [diningphil/gnn-comparison](https://github.com/diningphil/gnn-comparison)
- Inner **90/10 holdout** train/val per fold (from vendored JSON)
- **Early stopping** patience=500 on val accuracy (GIN/SAGE recipe)
- **Per-fold HP selection** from Errica published grids (not single canonical HP)
- **3 random restarts** per selected HP at eval
- Social datasets: **scalar degree** node features (COLLABORATIVE_DEGREE splits)

## Target claim (fixed 8-config SiGMA grid)

> Under Errica's 10-fold protocol with **per-fold hyperparameter selection**,
> **SiGMA hetero (a2g4)** is selected from the fixed 8-config `SIGMA_GRID`
> (`dim_inner=64`, `d_h=16`, `layers∈{4,12}`, …) — **no GIN/GCN parameter ceiling** —
> and matches or exceeds the best classical GNN on **X/7** datasets
> (and Errica's reported GIN on **Y/7**).

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
| SiGMA hetero | fixed8 `SIGMA_GRID` | 8 | 7 × 8 × 10 = **560** |

†Errica's [gnn-comparison](https://github.com/diningphil/gnn-comparison) repo has no `config_GCN.yml` /
`config_GAT.yml`. GCN/GAT use the same **protocol** (splits, early stop, Adam+StepLR) with a
GIN-isomorphic grid (batch, lr, width, pool, dropout, early-stop criterion).

## Campaign status

**Last updated:** 2026-09-03

| Phase | Campaign | JOBID | Status | Notes |
|-------|----------|-------|--------|-------|
| 0 smoke | `canonical` | **42673425** | ✅ done | fixed HP; **not** final table |
| **1a** | `grid_select` **GIN** | **42750648** | ✅ **done** | **4480/4480** COMPLETED |
| **1b** | `grid_select` **GraphSAGE** | **43116245** | ✅ **done** | **5040/5040** COMPLETED |
| **1c** | `grid_select` **GCN** | **43434937** | ✅ **done** | **2240/2240** COMPLETED |
| **1d** | `grid_select` **GAT** | **43434950** + **44099901** | ✅ **done** | fill-in for 14 W&B timeouts |
| **2a** | aggregates GIN/SAGE/GCN/GAT | — | ✅ **done** | `*_per_fold.json` |
| **2b** | `generate_sigma` **fixed8** | — | ✅ local | manifest · **560** tasks (replaces 400 budget_bio) |
| **3a†** | `sigma_grid_select` budget_bio | **43741550** | ✅ done | **obsolete** for final SiGMA column |
| **3a‡** | `sigma_grid_select` (failed) | **43451648** | ❌ failed | bash `mapfile` bug · fixed in `5e1688c` |
| **3a** | `sigma_grid_select` **fixed8** | — | **todo** | **560** tasks · relaunch |
| **3b** | `aggregate_sigma` | — | **todo** | after fixed8 select |
| **4a–d** | `grid_eval` GIN/SAGE/GCN/GAT | 44100531 / 66 / 96 / **44165919** | ✅ **done** | classical column frozen |
| **4e†** | `sigma_grid_eval` budget_bio | **44165958** | ignore / cancel | HPs from obsolete select |
| **4e** | `sigma_grid_eval` **fixed8** | — | **todo** | after fixed8 aggregate |

### grid_select progress summary

| Model | JOBID | Target | Last `sacct` snapshot |
|-------|-------|--------|------------------------|
| GIN | 42750648 | 4,480 | ✅ 4480 COMPLETED |
| GraphSAGE | 43116245 | 5,040 | ✅ 5040 COMPLETED |
| GCN | 43434937 | 2,240 | ✅ 2240 COMPLETED |
| GAT | 43434950 + **44099901** | 2,240 | 14 W&B-timeout reruns |
| **SiGMA fixed8** | *(relaunch)* | **560** | **todo** (old **43741550**=400 budget_bio, obsolete) |

**Total grid_select jobs (all four):** 14,000

### Active SLURM — monitor

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Running now
squeue -u $USER | grep tu_errica

# Per-job completion (replace nothing — use real IDs)
for j in 44099901 44100531 44100566 44100596 43741550; do
  echo "=== $j ==="
  sacct -j $j -X --format=State,ExitCode -n | awk '{print $1}' | sort | uniq -c
done

# Error scan (per campaign)
for pat in gin graphsage gcn gat; do
  j=$(case $pat in gin) echo 42750648;; graphsage) echo 43116245;; gcn) echo 43434937;; gat) echo 43434950;; esac)
  n=$(grep -l 'Error\|CUDA\|Traceback' logs_gnnplus/tu_errica_grid_select_${pat}_${j}_*.log 2>/dev/null | wc -l)
  echo "${pat} (${j}): ${n} error logs"
done
```

Logs: `logs_gnnplus/tu_errica_grid_select_<model>_<JOBID>_<TASK>.log` · SiGMA: `tu_errica_sigma_grid_select_<JOBID>_<TASK>.log`

W&B groups: `tu_errica_<ds>_<Model>_grid_select_hp<id>` (e.g. `tu_errica_enzymes_GCN_grid_select_hp0`)

### Next actions (while jobs run)

```bash
source ~/.gnnplus_env
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results

# Active arrays
for j in 44099901 44100531 44100566 44100596 43741550; do
  echo "=== $j ==="
  sacct -j $j -X --format=State -n | awk '{print $1}' | sort | uniq -c
done

# After GAT fill-in 44099901 finishes
python scripts/tu_errica/aggregate_hp_selection.py --model gat
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh grid_eval_gat

# After SiGMA 43741550 finishes (400/400)
python scripts/tu_errica/aggregate_sigma_hp_selection.py
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval
```

### Final `grid_eval` (3 seeds) — 2026-09-03

Mean±std over **10 folds** after averaging 3 seeds per fold.
LaTeX: [`results/tu_errica/analysis/tab_tu_errica_grid_eval.tex`](results/tu_errica/analysis/tab_tu_errica_grid_eval.tex).

| Dataset | GCN | GIN | GraphSAGE | GAT | Errica GIN [1] |
|---------|-----|-----|-----------|-----|----------------|
| ENZYMES | 50.4±5.2 | 45.4±5.2 | **51.0±4.7** | 42.1±7.0 | 29.5±8.2 |
| PROTEINS | **73.9±4.0** | 73.4±4.4 | 73.0±3.2 | 72.7±3.1 | 73.3±4.0 |
| NCI1 | 80.7±1.5 | 80.4±1.5 | **81.6±2.3** | 75.4±2.4 | 80.0±1.4 |
| DD | 71.9±4.2 | **73.7±5.2** | 72.8±3.0 | 72.9±9.2 | 76.6±4.3 |
| IMDB-BINARY | 65.7±3.5 | **71.1±4.5** | 50.5±1.1 | 50.4±2.0 | 71.2±3.9 |
| REDDIT-BINARY | **92.6±1.0** | 92.5±1.1 | 73.4±4.0 | 74.7±2.3 | 89.9±1.9 |
| COLLAB | **77.0±2.1** | 76.5±2.5 | 52.5±3.2 | 47.6±7.9 | 75.6±2.3 |

Classical `grid_eval` columns are frozen. SiGMA is being **relaunched** under **fixed8**
(`SIGMA_GRID`, 560 select tasks) — ignore budget-bio eval **44165958** / select **43741550**.

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

## Hybrid SiGMA search (Option 3 → **fixed8**, 2026-09-03)

**Current default:** all datasets use the fixed **8-config** `SIGMA_GRID`
(`batch∈{32,128}`, `lr∈{1e-3,1e-2}`, `layers_mp∈{4,12}`, `dim_inner=64`, `d_h=16`)
with **no GIN/GCN parameter ceiling**. Task count: **7 × 10 × 8 = 560**.

| Family | Datasets | Rule |
|--------|----------|------|
| All (default `fixed8`) | 7 Errica TU sets | Same 8-config grid; bs=16 override on DD/REDDIT-B/COLLAB at train time |
| Legacy `budget_bio` | bio only | Lock to GIN winner depth/width; `d_h` under GIN param budget |

```bash
# Regenerate + submit (cluster) — campaign name avoids W&B collision with budget_bio
python scripts/tu_errica/generate_sigma_errica_grids.py --mode fixed8
# → 560 tasks
TU_ERRICA_CAMPAIGN=sigma_grid_select_fixed8 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
  bash bash_interface/cluster/submit_tu_errica_fair.sh
# after select finishes:
python scripts/tu_errica/aggregate_sigma_hp_selection.py \
  --campaign sigma_grid_select_fixed8 \
  --out configs/tu_errica/selections/sigma_fixed8_per_fold.json
TU_ERRICA_CAMPAIGN=sigma_grid_eval_fixed8 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
  TU_ERRICA_SELECTION_FILE=configs/tu_errica/selections/sigma_fixed8_per_fold.json \
  bash bash_interface/cluster/submit_tu_errica_fair.sh
```

Legacy budgeted bio grids: `--mode budget_bio` (requires `gin_per_fold.json`).

Scripts:
- `scripts/tu_errica/param_budget.py` — param counting (+ legacy bio budget helpers)
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
# All four grid_select jobs
for j in 44099901 44100531 44100566 44100596 43741550; do
  echo "=== $j ==="
  sacct -j $j -X --format=State,ExitCode -n | awk '{print $1}' | sort | uniq -c
done
squeue -u $USER | grep tu_errica
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
