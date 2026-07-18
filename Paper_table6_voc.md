# SiGMA paper Table 6 — PascalVOC-SP heterogeneous MP heads

Track VOC-only Table 6 ablations (homogeneous vs heterogeneous MPGNN heads ± gates).

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Best VOC baseline: [`Paper_final_runs.md`](Paper_final_runs.md) / Table 5 [`Paper_ablations.md`](Paper_ablations.md)

---

## 1. Frozen best VOC model

| Field | Value |
|-------|-------|
| Dataset | PascalVOC-SP |
| Anchor | `configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml` |
| Exemplar | [`vyt7hjj5`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vyt7hjj5) |
| Paper group | `paper_bestmodel_v1_voc_j7ukyzdm` |
| Arch | a2g2 — 2×attn + 2×MP, `d_h=64`, headwise gate, RMSNorm, L16/H95, ep=200 |
| Best MP types | **`GATEDGCN,GATEDGCN`** (homogeneous) |

---

## 2. Three variants × 5 seeds = 15 jobs

| Variant (W&B) | Meaning | Override vs anchor |
|---------------|---------|--------------------|
| **`SiGMA`** | Best gated hybrid (homogeneous MP) | none |
| **`Hetero_MP`** | Heterogeneous MP, still gated | `gnn.hybrid.gnn_types GATEDGCN,GCN` |
| **`Hetero_MP_ungated`** | Heterogeneous MP, no gates | `gnn.hybrid.gnn_types GATEDGCN,GCN` + `gnn.hybrid.gate none` |

Maps to paper Table 6 rows (VOC column):

| Paper row | This campaign |
|-----------|----------------|
| Homogeneous MP heads | ≈ **`SiGMA`** (best model already homogeneous) |
| Heterogeneous MP heads, no gates | **`Hetero_MP_ungated`** |
| SiGMA | **`SiGMA`** (and optionally compare to gated hetero via **`Hetero_MP`**) |

### W&B

- **Group:** `paper_T6_voc_<Variant>`
  - `paper_T6_voc_SiGMA`
  - `paper_T6_voc_Hetero_MP`
  - `paper_T6_voc_Hetero_MP_ungated`
- **Tags:** `paper_table6`, `<Variant>`, `voc`, `seed<k>`, `source_vyt7hjj5`

---

## 3. Launch on cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table6_voc_hetero.sh
```

Optional: `PAPER_T6_VOC_PARALLEL=N` (default **5**).

| Field | Value |
|-------|-------|
| **SLURM array** | *(paste JOBID after submit)* |
| **Job name** | `sigma_T6_voc` |
| **Tasks** | `1-15%5` = 3×5 |
| **Scripts** | `submit_paper_table6_voc_hetero.sh` → `run_paper_table6_voc_hetero.sh` |
| **Logs** | `logs_gnnplus/sigma_T6_voc_<JOBID>_<TASK>.log` |
| **Needs** | `gate=none` support (same branch as Table 5) |

---

## 4. Aggregate

```bash
PREFIX=paper_T6
for v in SiGMA Hetero_MP Hetero_MP_ungated; do
  echo "===== voc / ${v} ====="
  python scripts/api_wanndb_query/aggregate_paper_repro.py \
    --group ${PREFIX}_voc_${v} --metric best_test_perf --state finished
done
```

### Fill-in (PascalVOC-SP F1 ↑)

| Model | VOC-SP F1 ↑ | n | W&B group |
|-------|-------------|---|-----------|
| SiGMA (homog. gated) | | 5 | `paper_T6_voc_SiGMA` |
| Hetero_MP (GATEDGCN,GCN + gate) | | 5 | `paper_T6_voc_Hetero_MP` |
| Hetero_MP_ungated | | 5 | `paper_T6_voc_Hetero_MP_ungated` |

---

## 5. Files

| Path | Role |
|------|------|
| `bash_interface/cluster/submit_paper_table6_voc_hetero.sh` | **launch this** |
| `bash_interface/cluster/run_paper_table6_voc_hetero.sh` | SLURM array worker |
| `configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml` | VOC SiGMA anchor |
| `GNNPlus/layer/gated_hybrid_layer.py` | `gate=none` for ungated variant |
