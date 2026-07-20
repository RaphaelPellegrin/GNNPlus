# SiGMA + GRIT attention heads (PATTERN / CLUSTER)

Replace vanilla dense QK attention with GRIT sparse attention units inside SiGMA hybrid blocks (`gnn.hybrid.attn_type: grit`). MP heads unchanged; RRWP pads full-graph edges for GRIT while MP keeps sparse `edge_index_mp`.

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Master tracker: [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🛑🛑🛑  TO RUN  ·  not submitted yet  🛑🛑🛑                          ║
║  🧪  2 ds × 5 seeds = 10 jobs                                            ║
║  🚀  bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🛑 **TO RUN** |
| **SLURM array** | 🛑 *not submitted yet* — paste JOBID here after launch |
| **Job name** | `sigma_grit_attn` |
| **Tasks** | `1-10%5` |
| **Mem / time** | `128GB` / `120h` |
| **Partition** | `mweber_gpu` |
| **W&B groups** | `paper_sigma_grit_attn_pattern`, `paper_sigma_grit_attn_cluster` |
| **W&B tags** | `sigma_grit_attn`, `attn_type_grit`, `grit_attn`, `<dataset>`, `seed<k>` |

---

## Anchors

| Dataset | Base SiGMA | Arch | Config | RRWP |
|---------|------------|------|--------|------|
| PATTERN | [`ta9qtxb9`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) | a2g2 GCNE×2, elementwise, d_h=90 | `configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml` | ksteps=21 |
| CLUSTER | [`ht9bntg2`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ht9bntg2) | a1g1 GATEDGCN, headwise, d_h=64 | `configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml` | ksteps=32 |

Vanilla baseline groups (for comparison): `paper_bestmodel_v1_pattern_ta9qtxb9`, `paper_bestmodel_v1_cluster_ht9bntg2`.

---

## Launch

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh
# 👉 paste JOBID here + CLUSTER_LAUNCHES.md
```

CLI force on every run: `gnn.hybrid.attn_type grit` (also in yaml + W&B config/tags).

---

## Aggregate when finished

```bash
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_pattern --metric best_test_perf --state finished

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_cluster --metric best_test_perf --state finished
```

For PATTERN / CLUSTER, multiply API fraction by 100 to match paper `%`.
