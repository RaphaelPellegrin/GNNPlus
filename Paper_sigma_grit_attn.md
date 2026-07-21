# SiGMA + GRIT attention heads (PATTERN / CLUSTER)

Replace vanilla dense QK attention with GRIT sparse attention units inside SiGMA hybrid blocks (`gnn.hybrid.attn_type: grit`). MP heads unchanged; RRWP pads full-graph edges for GRIT while MP keeps sparse `edge_index_mp`.

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Master tracker: [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  ✅  DONE  ·  SLURM 33458567  ·  seeds 0–4, no VN                        ║
║  🛑  TO RUN  ·  seeds 5–9 ± VN=4  ·  20 jobs  ·  %5                      ║
║  📄  logs_gnnplus/sigma_grit_attn_<JOBID>_<TASK>.log                     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status (seeds 0–4)** | ✅ **`33458567`** finished |
| **Status (seeds 5–9 ± VN)** | 🛑 **TO RUN** — paste JOBID below |
| **Job name** | `sigma_grit_attn` |
| **Mem / time** | `128GB` / `120h` |
| **Partition** | `mweber_gpu` |
| **W&B groups (no VN)** | `paper_sigma_grit_attn_pattern`, `paper_sigma_grit_attn_cluster` |
| **W&B groups (VN=4)** | `paper_sigma_grit_attn_pattern_vn4`, `paper_sigma_grit_attn_cluster_vn4` |
| **W&B tags** | `sigma_grit_attn`, `attn_type_grit`, `grit_attn`, `novn` / `vn4`, `<dataset>`, `seed<k>` |

---

## Anchors

| Dataset | Base SiGMA | Arch | Config | RRWP |
|---------|------------|------|--------|------|
| PATTERN | [`ta9qtxb9`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) | a2g2 GCNE×2, elementwise, d_h=90 | `configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml` | ksteps=21 |
| CLUSTER | [`ht9bntg2`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ht9bntg2) | a1g1 GATEDGCN, headwise, d_h=64 | `configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml` | ksteps=32 |

Vanilla baseline groups (for comparison): `paper_bestmodel_v1_pattern_ta9qtxb9`, `paper_bestmodel_v1_cluster_ht9bntg2`.

---

## Virtual-node note (June failure)

[`4yhs6ywq`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/4yhs6ywq) failed with `assert true.shape[0] == pred.shape[0]` on CLUSTER+VN. Cause: early VN loss masking dropped rows from `pred_score` while the train loop still logged full-length `true`. Current fix:

- pad per-node `y` with ignore label `-1` for VNs
- `weighted_cross_entropy` uses `ignore_index=-1` and **keeps full-length** `pred_score`
- logger drops ignore rows only when computing metrics

**Pull this branch before launching the VN variant.**

---

## Launch (seeds 5–9, no-VN + VN=4 → 20 jobs)

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

SIGMA_GRIT_ATTN_SEED_OFFSET=5 \
SIGMA_GRIT_ATTN_NUM_VARIANTS=2 \
SIGMA_GRIT_ATTN_NUM_VN=4 \
SIGMA_GRIT_ATTN_PARALLEL=5 \
bash bash_interface/cluster/submit_sigma_grit_attn_pattern_cluster.sh
# 👉 paste JOBID here + CLUSTER_LAUNCHES.md
```

Task layout: `1–5` pattern seeds 5–9 · `6–10` cluster seeds 5–9 · `11–15` pattern VN=4 · `16–20` cluster VN=4.

CLI force on every run: `gnn.hybrid.attn_type grit`.

---

## Aggregate when finished

```bash
# seeds 0–4 (33458567) + seeds 5–9 (new) share the no-VN groups
python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_pattern --metric best_test_perf --state finished

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_cluster --metric best_test_perf --state finished

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_pattern_vn4 --metric best_test_perf --state finished

python scripts/api_wanndb_query/aggregate_paper_repro.py \
  --group paper_sigma_grit_attn_cluster_vn4 --metric best_test_perf --state finished
```

For PATTERN / CLUSTER, multiply API fraction by 100 to match paper `%`.
