# GRIT (Graph Inductive Bias Transformer)

Standalone + hybrid configs for [Ma et al., ICML 2023](https://arxiv.org/pdf/2305.17589).

## Config files

| Dataset | Standalone (`GritTransformer` + RRWP) | Hybrid (`gnn_types: GRIT`) |
|---------|--------------------------------------|-----------------------------|
| PATTERN | `configs/grit/pattern-grit-rrwp.yaml` | `configs/gated_hybrid/pattern-grit-repro-a1g1.yaml` |
| CLUSTER | `configs/grit/cluster-grit-rrwp.yaml` | `configs/gated_hybrid/cluster-grit-repro-a1g1.yaml` |
| ZINC | `configs/grit/zinc-grit-rrwp.yaml` | `configs/gated_hybrid/zinc-grit-repro-a1g1.yaml` |

Metrics: PATTERN/CLUSTER → `test/accuracy-SBM`; ZINC → `test/mae`.

## Cluster submit

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
source bash_interface/cluster/common_env.sh

# One dataset
bash bash_interface/cluster/submit_grit.sh pattern hybrid
bash bash_interface/cluster/submit_grit.sh cluster standalone
bash bash_interface/cluster/submit_grit.sh zinc standalone

# All three (hybrid — faster, no RRWP precompute)
bash bash_interface/cluster/submit_grit.sh all hybrid

# All three standalone (paper GRIT + RRWP; needs torch_sparse on GPU node)
bash bash_interface/cluster/submit_grit.sh all standalone
```

W&B run name: `<dataset>_grit_<variant>_seed0_job<JOBID>`, auto-tag `job_<JOBID>`.

## Local

```bash
python main.py --cfg configs/grit/pattern-grit-rrwp.yaml seed 0 wandb.use True
```

Hybrid GRIT uses sparse edges; standalone uses RRWP + full-graph attention.
