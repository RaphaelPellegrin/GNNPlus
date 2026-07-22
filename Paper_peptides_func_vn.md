# Peptides-func SiGMA + virtual nodes (o5cdk766)

Explore VN on the best peptides-func SiGMA so far ([`o5cdk766`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/o5cdk766)): a1g1 GCN, elementwise gate, `base_lr≈2.083e-4`, 900 epochs, Atom+RWSE, **no VN** in the paper cell.

## Grid (10 × 5 seeds)

| cfg | VN | lr | readout |
|-----|----|----|---------|
| 0 | 0 | 2.083e-4 | default (paper control) |
| 1 | 1 | 2.083e-4 | pyramid |
| 2 | 2 | 2.083e-4 | pyramid |
| 3 | 4 | 2.083e-4 | pyramid |
| 4 | 8 | 2.083e-4 | pyramid |
| 5 | 4 | 1e-4 | pyramid |
| 6 | 4 | 4e-4 | pyramid |
| 7 | 4 | 2.083e-4 | default (VN only) |
| 8 | 2 | 4e-4 | pyramid |
| 9 | 8 | 1e-4 | pyramid |

Pyramid when VN>0 matches the peptides-struct `rholn782` VN=4 recipe; cfg 7 isolates VN without pyramid.

## Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull
bash bash_interface/cluster/submit_peptides_func_o5cdk766_vn_lr_grid.sh
```

Anchor: `configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml`  
W&B: `paper_sigma_peptides_func_<novn|vnK>_lr<tag>_<pyr|nopyr>`

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_peptides_func_vn4_lr2p083e-4_pyr \
  --metric best_test_perf --state finished
```

| Field | Value |
|-------|-------|
| **SLURM** | 🛑 *paste JOBID after submit* |
| **Tasks** | `1-50%5` · `mweber_gpu` · 192h |
