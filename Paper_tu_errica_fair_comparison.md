# TU Errica-fair comparison (Layer 2 rebuttal)

Fair comparison under [Errica et al. ICLR 2020](https://arxiv.org/pdf/1912.09893) protocol:
- **10-fold** fixed stratified CV splits from [diningphil/gnn-comparison](https://github.com/diningphil/gnn-comparison)
- Inner **90/10 holdout** train/val per fold (from vendored JSON)
- **Early stopping** patience=500 on val accuracy (GIN/SAGE recipe)
- **3 random restarts** per fold (canonical campaign)
- Social datasets: **scalar degree** node features (COLLABORATIVE_DEGREE splits)

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

## Our runs

**Last updated:** 2026-08-29 (smoke OK; SiGMA DD/REDDIT-B rerun submitted).

| Campaign | JOBID | Status | Progress |
|----------|-------|--------|----------|
| canonical (630) | **42673425** | **done** | **570✅ / 60❌** — SiGMA DD + REDDIT-B OOM @ bs128 |
| canonical rerun (60) | **42746310** | **running** | tasks **331–360, 511–540** · bs16 fix · 96h · 128GB |
| rerun smoke (DD f0 s0) | **42746111** | **passed** | task 331 — training OK (epoch 2+ val ~75%) |
| grid_select (GIN) | | optional | 7 × 64 HP × 10 folds = 4480 |
| smoke (SLURM GPU) | **42673389** | **done** | ENZYMES GIN fold0 seed0 |
| smoke (login CPU) | n8t358kk | OK (ctrl-C) | W&B interactive; splits verified |
| smoke (SLURM, broken) | 42672738 | FAILED | pre-fix HP emit; ignore |

### Failure post-mortem (42673425)

| Tasks | Dataset | Model | Cause |
|------:|---------|-------|-------|
| 331–360 | DD | SiGMA hetero | `CUDA OOM` — full-batch attention @ `batch_size=128` (~138GB GPU) |
| 511–540 | REDDIT-BINARY | SiGMA hetero | same |

**Fix (in `run_tu_errica_fair.sh`):** override `train.batch_size=16` for SiGMA on `dd` and
`reddit-b` (same as [`run_tu_dd_sigma_retry.sh`](bash_interface/cluster/run_tu_dd_sigma_retry.sh)).

**Resubmit (launched 2026-08-29):**

```bash
# Smoke: JOBID=42746111 task 331 — passed
TU_ERRICA_ARRAY=331-360,511-540 TU_ERRICA_MEM=128GB TU_ERRICA_TIME=96:00:00 \
  bash bash_interface/cluster/submit_tu_errica_fair.sh
# → JOBID=42746310
```

Monitor:

```bash
sacct -j 42746310 -X --format=JobID,State,ExitCode,Elapsed -n | awk '{print $2}' | sort | uniq -c
```

## Results (fill after W&B aggregate)

| Dataset | GIN (ours) | GraphSAGE | SiGMA hetero | Errica GIN |
|---------|------------|-----------|--------------|------------|
| ENZYMES | | | | 29.5±8.2 |
| PROTEINS | | | | 73.3±4.0 |
| NCI1 | | | | 80.0±1.4 |
| DD | | | | 76.6±4.3 |
| IMDB-BINARY | | | | 71.2±3.9 |
| REDDIT-BINARY | | | | 89.9±1.9 |
| COLLAB | | | | 75.6±2.3 |

## Launch (cluster)

```bash
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
python scripts/tu_errica/vendor_errica_splits.py   # if splits/ not present
bash bash_interface/cluster/submit_tu_errica_fair.sh
```

Smoke first:

```bash
TU_ERRICA_CAMPAIGN=canonical TU_ERRICA_ARRAY=1 TU_ERRICA_NUM_FOLDS=1 \
  TU_ERRICA_NUM_SEEDS=1 bash bash_interface/cluster/submit_tu_errica_fair.sh
```

## Aggregate

Metrics live in **W&B** (`best_test_perf` / `best/test_accuracy`), not local `stats.json`.

```bash
# After git pull (aggregate script defaults to W&B)
python scripts/tu_errica/aggregate_errica_results.py --source wandb --state finished

# Or one group at a time
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group tu_errica_enzymes_GIN_canonical_canonical \
  --metric best_test_perf --state finished
```

W&B groups: `tu_errica_<ds_tag>_<model>_canonical_canonical`  
(`ds_tag` = `enzymes`, `proteins`, `nci1`, `dd`, `imdb-b`, `reddit-b`, `collab`)

## Code paths

- Splits: `splits/errica/` (vendored from gnn-comparison)
- Loader: `GNNPlus/loader/errica_splits.py`, `split_mode: errica-cv-10`
- Configs: `configs/tu_errica/*-errica-base.yaml`
- HP grids: `configs/tu_errica/*_hp_grid.json`
- SLURM: `bash_interface/cluster/run_tu_errica_fair.sh`

## Scope note (Layer 1 — paper text)

Appendix F Table 17–18 used **50/25/25** random splits (internal comparison).
This campaign is **separate** — same models, Errica splits, for reviewer-facing SOTA table.
