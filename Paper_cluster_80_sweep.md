# CLUSTER push-to-80% — SiGMA + GRIT Bayes sweep

Target: **test Acc > 80%** (metric logged as fraction; report ×100).  
Current best finished ~**79.45%** ([`tu8cr0fp`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tu8cr0fp), grit + VN=8).

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Master tracker: [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)

## Setup

| | |
|--|--|
| **Anchor** | `configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml` |
| **Sweep YAML** | `bash_interface/sweeps/cluster_sigma_grit_vn_lr_dh_sweep.yaml` |
| **Method** | Bayes + Hyperband (`best/val_accuracy-SBM`) |
| **Fixed** | grit attn, a1g1 GATEDGCN, headwise, ep=100, seed=3 |
| **Swept** | `base_lr`, `num_virtual_nodes`, `hybrid_d_h`, `mp_dropout`, `weight_decay` |

## Prior runs (attached via `-R`)

Best finished run per prior cell (VN×LR grid + grit paper + vanilla):

| Acc % | Run | Cell |
|------:|-----|------|
| 79.451 | [`tu8cr0fp`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/tu8cr0fp) | grit vn8 @ 1.492e-3 |
| 79.440 | [`63inaz23`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/63inaz23) | grit vn8 @ 1e-3 |
| 79.367 | [`hx3v1ybs`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/hx3v1ybs) | grit novn paper |
| 79.342 | [`er5vpx7j`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/er5vpx7j) | grit vn4 @ 5e-4 |
| 79.315 | [`80ngb67n`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/80ngb67n) | grit vn4 paper |
| 79.309 | [`3j7zj86o`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3j7zj86o) | grit vn4 @ 1e-3 |
| 79.292 | [`hmy6di2u`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/hmy6di2u) | grit vn4 @ 1.492e-3 |
| 79.274 | [`q6bi1pqc`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q6bi1pqc) | grit vn1 @ 1.492e-3 |
| 79.260 | [`q6b3ofpj`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/q6b3ofpj) | grit vn4 @ 3e-3 |
| 79.129 | [`opqkgsxi`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/opqkgsxi) | grit novn @ 1.492e-3 |
| 79.063 | [`nhuyof1w`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/nhuyof1w) | grit vn2 @ 1.492e-3 |
| 79.092 | [`f6k8rjip`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/f6k8rjip) | vanilla ht9bntg2 |

## Launch

```bash
source ~/.gnnplus_env
export WANDB_PROJECT=GNNPlus
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

PRIOR_RUNS="tu8cr0fp 63inaz23 hx3v1ybs er5vpx7j 80ngb67n 3j7zj86o hmy6di2u q6bi1pqc q6b3ofpj opqkgsxi nhuyof1w f6k8rjip" \
  bash bash_interface/sweeps/create_sweep.sh \
    bash_interface/sweeps/cluster_sigma_grit_vn_lr_dh_sweep.yaml

# 👉 paste the printed sbatch agent block (defaults: 16 agents %4, 3 runs each, 120h, 128GB)
# 👉 paste SWEEP_ID + agent JOBID below + CLUSTER_LAUNCHES.md
```

| Field | Value |
|-------|-------|
| **Sweep ID** | 🛑 *paste after create* |
| **Agent job** | 🛑 *paste after sbatch* |
| **Partition** | `mweber_gpu` |

## Notes

- Bayes optimizes **val** `accuracy-SBM`; judge success on **test** `best_test_perf` ×100.
- Prior runs help only insofar as their logged configs overlap the swept keys; attaching them is still useful context for the optimizer.
- Keep existing 300-ep COCO / VN grids running — this is a separate sweep.
