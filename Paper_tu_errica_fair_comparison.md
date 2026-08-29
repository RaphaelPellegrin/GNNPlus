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

| Campaign | JOBID | Status | Notes |
|----------|-------|--------|-------|
| canonical (630) | | pending | 7 ds × GIN/SAGE/SiGMA × 10 folds × 3 seeds |
| grid_select (GIN) | | optional | 7 × 64 HP × 10 folds = 4480 |
| smoke | | | `TU_ERRICA_ARRAY=1` single GIN job |

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

```bash
python scripts/tu_errica/aggregate_errica_results.py \
  --root "${GNNPLUS_OUT_DIR}/tu_errica/canonical"
```

W&B groups: `tu_errica_<ds>_<model>_canonical_canonical`

## Code paths

- Splits: `splits/errica/` (vendored from gnn-comparison)
- Loader: `GNNPlus/loader/errica_splits.py`, `split_mode: errica-cv-10`
- Configs: `configs/tu_errica/*-errica-base.yaml`
- HP grids: `configs/tu_errica/*_hp_grid.json`
- SLURM: `bash_interface/cluster/run_tu_errica_fair.sh`

## Scope note (Layer 1 — paper text)

Appendix F Table 17–18 used **50/25/25** random splits (internal comparison).
This campaign is **separate** — same models, Errica splits, for reviewer-facing SOTA table.
