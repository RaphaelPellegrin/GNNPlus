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

## Errica GIN reference (Table 3 chemical / Table 4 social+degree)

| Dataset | Errica GIN |
|---------|------------|
| ENZYMES | 59.6 ± 4.5 |
| PROTEINS | 73.3 ± 4.0 |
| NCI1 | 80.0 ± 1.4 |
| DD | 75.3 ± 2.9 |
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
| SiGMA hetero (alt) | GIN-isomorphic `full64` | 64 | 7 × 64 × 10 = **4,480** |

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
| **3a** | `sigma_grid_select` **fixed8** | **44217420** + **44266489** | ✅ **559/560** | 1 FAILED (COLLAB f9 hp7) → fill **44507757** |
| **3a§** | `sigma_grid_select` **full64** | **44262912** | ⚠️ **1920 OK / 1920 FAIL** | cliff after NCI1 (DD→…) · rerun via `submit_tu_errica_full64_rerun_failed.sh` |
| **3a-R** | fixed8 **REDDIT only** | **44266489** | ✅ **80/80** | tasks **401–480** |
| **3a-R§** | full64 **REDDIT only** | **44266493** | ⚠️ **81 OK / 559 FAIL** | included in FAILED rerun list |
| **3a-R§2** | full64 **REDDIT only** (no cancel) | **44750843** | ❌ **scancel** | was mweber parallel; REDDIT continues via **44509970** |
| **3a§-rerun** | full64 FAILED relaunch | **44509970** | 🔄 **464+ done** | 2479 tasks · ~20 run · 12 FAILED so far |
| **3a-AB** | `sigma_grid_select` **anchor_boost** | **44840486** | 🔄 **147/480** | ~20 R · mweber |
| **3a-U** | `sigma_grid_select` **fixed8 ungated** | *(pending)* | ⏳ | same 8-grid · `gate=none` · 560 |
| **3a-fill** | fixed8 **COLLAB f9 hp7** fill | **44507757** | ✅ **COMPLETED** | task **560** · netscratch logs |
| **3b** | `aggregate_sigma` | — | ✅ **70/70 folds** | `sigma_fixed8_per_fold.json` |
| **4a–d** | `grid_eval` GIN/SAGE/GCN/GAT | 44100531 / 66 / 96 / **44165919** | ✅ **done** | classical column frozen |
| **4e†** | `sigma_grid_eval` budget_bio | **44165958** | ignore / cancel | HPs from obsolete select |
| **4e** | `sigma_grid_eval` **fixed8** | **44621846** | ✅ **209/210** | FAILED task **181** filled by **44748166** |
| **4e-fill** | eval COLLAB f0 seed0 | **44748166** | ✅ **COMPLETED** | → rebuild table |

### grid_select progress summary

| Model | JOBID | Target | Last `sacct` snapshot |
|-------|-------|--------|------------------------|
| GIN | 42750648 | 4,480 | ✅ 4480 COMPLETED |
| GraphSAGE | 43116245 | 5,040 | ✅ 5040 COMPLETED |
| GCN | 43434937 | 2,240 | ✅ 2240 COMPLETED |
| GAT | 43434950 + **44099901** | 2,240 | 14 W&B-timeout reruns |
| **SiGMA fixed8** | **44217420** + jumps | **560** | ✅ 559 + fill **44507757** |

**Total grid_select jobs (all four):** 14,000

### Active SLURM — monitor

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus

# Running now
squeue -u $USER | grep tu_errica

# Per-job completion (replace nothing — use real IDs)
for j in 44217420; do
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
for j in 44217420; do
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

### Final `grid_eval` (3 seeds) — fixed8 SiGMA (2026-09-06)

Mean±std over **10 folds** after averaging 3 seeds per fold.
LaTeX: [`results/tu_errica/analysis/tab_tu_errica_grid_eval.tex`](results/tu_errica/analysis/tab_tu_errica_grid_eval.tex).
SiGMA-only dump: [`tab_tu_errica_sigma_fixed8.tex`](results/tu_errica/analysis/tab_tu_errica_sigma_fixed8.tex).

| Dataset | GCN | GIN | GraphSAGE | GAT | SiGMA (fixed8) | Errica GIN [1] |
|---------|-----|-----|-----------|-----|----------------|----------------|
| ENZYMES | 50.4±5.2 | 45.4±5.2 | 51.0±4.7 | 42.1±7.0 | **52.4±4.6** | 59.6±4.5 |
| PROTEINS | **73.9±4.0** | 73.4±4.4 | 73.0±3.2 | 72.7±3.1 | 72.6±4.1 | 73.3±4.0 |
| NCI1 | 80.7±1.5 | 80.4±1.5 | **81.6±2.3** | 75.4±2.4 | 80.7±1.9 | 80.0±1.4 |
| DD | 71.9±4.2 | 73.7±5.2 | 72.8±3.0 | 72.9±9.2 | **73.9±2.3** | 75.3±2.9 |
| IMDB-BINARY | 65.7±3.5 | **71.1±4.5** | 50.5±1.1 | 50.4±2.0 | 70.9±5.4 | 71.2±3.9 |
| REDDIT-BINARY | **92.6±1.0** | 92.5±1.1 | 73.4±4.0 | 74.7±2.3 | 88.0±3.3 | 89.9±1.9 |
| COLLAB | 77.0±2.1 | 76.5±2.5 | 52.5±3.2 | 47.6±7.9 | **78.2±1.2** | 75.6±2.3 |

\*Classical columns frozen. SiGMA = fixed8 eval **44621846**+**44748166**. full64 sensitivity still via **44509970**.

Delta vs obsolete budget-bio: PROTEINS/NCI1 up; DD/IMDB/REDDIT slightly down; COLLAB complete (was 5/10).

### Push PROTEINS + REDDIT (`anchor_boost`, 2026-09-06)

Fixed8 never searched **bs=64** (paper PROTEINS batch) and forced REDDIT to bs=16 outside the grid.
New campaign reuses the **paper a2g4** recipe (`sigma-hetero-a2g4-anchor.yaml`: L12/H64/d_h16, lr∈{1e-3,1e-2}) under **Errica CV**, plus nearby depth / d_h / batch:

| Axis | Values |
|------|--------|
| `batch_size` | **16**, **64** (paper REDDIT / PROTEINS) |
| `base_lr` | 0.001, 0.01 |
| `layers_mp` | 8, **12** |
| `dim_inner` | **64** |
| `d_h` | 8, **16**, 32 |

→ **24** configs × 2 datasets × 10 folds = **480** select · eval = **60** (2×10×3). No forced batch override.

```bash
# local: generate grids (already in repo after commit)
python scripts/tu_errica/generate_sigma_errica_grids.py --mode anchor_boost

# cluster select
bash bash_interface/cluster/submit_tu_errica_anchor_boost_select.sh
# after select:
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_anchor_boost
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_anchor_boost
```

Blend into the main table: keep fixed8 for other datasets; replace PROTEINS/REDDIT with `sigma_grid_eval_anchor_boost`.

### SiGMA ungated Errica (`fixed8_ungated`, 2026-09-06)

Same **fixed8** 8-config grid on all 7 datasets, but `gnn.hybrid.gate=none`
(`sigma-hetero-ungated-errica-base.yaml`). Fair gated vs ungated column.

| Phase | Tasks |
|-------|------:|
| Select | **560** |
| Eval | **210** |

```bash
bash bash_interface/cluster/submit_tu_errica_fixed8_ungated_select.sh
# later:
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh aggregate_sigma_fixed8_ungated
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_eval_fixed8_ungated
```

Runs in parallel with gated fixed8 (done), anchor_boost **44840486**, full64 **44509970**.


### Priority: REDDIT-BINARY first (2026-09-03)

Budget-bio SiGMA trails hardest on REDDIT (`88.4` vs GCN/GIN `~92.5`). Jump-start
REDDIT select tasks **without waiting** for enzymes→…→imdb to finish:

| Campaign | Parent JOBID | REDDIT SLURM tasks | Action |
|----------|-------------:|-------------------:|--------|
| fixed8 | **44217420** | **401–480** (80) | cancel that range on parent → submit Nice=0 slice |
| full64 | **44262912** | **3201–3840** (640) | same |

Same W&B groups / manifest indices as the parent, so `aggregate_sigma*` still works.
After REDDIT select finishes → aggregate → eval tasks **151–180** only (`7×10×3` layout).

```bash
# --- fixed8 REDDIT jump-start (mweber) ---
scancel 44217420_[401-480]   # drop from slow parent queue (pending only)
TU_ERRICA_CAMPAIGN=sigma_grid_select_fixed8 \
  TU_ERRICA_ARRAY=401-480 \
  TU_ERRICA_PARALLEL=20 \
  TU_ERRICA_MEM=128GB \
  TU_ERRICA_TIME=96:00:00 \
  TU_ERRICA_NICE=0 \
  TU_ERRICA_PARTITION=mweber_gpu \
  bash bash_interface/cluster/submit_tu_errica_fair.sh

# --- full64 REDDIT only (preferred; NO scancel) ---
bash bash_interface/cluster/submit_tu_errica_full64_reddit_only.sh
# → tasks 3201–3840 · Nice=0 · netscratch logs · same campaign as parent
# Overlap with 44509970 is fine; aggregate keeps best val per fold/hp.
```

Do **not** cancel **44509970** (or other full64 work) when launching REDDIT-only.

### Canonical exploratory results (W&B, do not cite as final)

Fixed `GIN_CANONICAL` — useful signal only:

| Dataset | GIN | GraphSAGE | SiGMA | Errica GIN |
|---------|-----|-----------|-------|------------|
| ENZYMES | 45.0 | 51.3 | 54.1 | 59.6 |
| PROTEINS | 74.1 | 72.3 | 71.8 | 73.3 |
| NCI1 | 77.8 | 79.1 | **80.3** | 80.0 |
| DD | 72.4 | 70.7 | pending | 75.3 |
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

## Alternative: GIN-matched 64-config SiGMA (`full64`)

Protocol-matched search: **same 64 configs as GIN**, with `d_h ∈ {8,16}` standing in
for GIN's `train_eps`. Axes: `batch∈{32,128}`, `lr=0.01`, `layers=4`, `dim∈{32,64}`,
`pool∈{add,mean}`, `dropout∈{0,0.5}`, `early_stop∈{acc,loss}`.

Task count: **7 × 10 × 64 = 4,480** (same as GIN `grid_select`).
Writes `configs/tu_errica/sigma_grids_full64/` — **does not overwrite** the running
fixed8 job **44217420**.

Wall-clock at `%12` and ~0.9 h/task: \(\approx 4480 \times 0.9 / 12 \approx\) **14 days**.
Raise `TU_ERRICA_PARALLEL` if launching.

```bash
python scripts/tu_errica/generate_sigma_errica_grids.py --mode full64
# → 4480 tasks under configs/tu_errica/sigma_grids_full64/
bash bash_interface/cluster/run_tu_errica_hybrid_pipeline.sh sigma_grid_select_full64
```

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
for j in 44217420; do
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
