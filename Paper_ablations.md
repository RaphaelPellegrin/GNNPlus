# SiGMA paper Table 5 ablations — launch + W&B tracking

You submit on FASRC yourself. This file is the checklist.

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Best baselines: [`Paper_final_runs.md`](Paper_final_runs.md)

**MNIST + CIFAR10** (same 4 variants): 🛑 **TO RUN** — see [`Paper_ablations_mnist_cifar.md`](Paper_ablations_mnist_cifar.md) / [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md).

---

## 1. Exact best baselines we freeze

Hyperparams live in the anchor YAMLs (LR, depth, heads, gate, etc.). Exemplar “best seed” runs used to verify:

| Dataset | Paper metric | Anchor config | Exemplar best run | Arch |
|---------|--------------|---------------|-------------------|------|
| Peptides-func | AP 0.7052±0.0056 (n=10) | `configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml` | [`l31u4b3k`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/l31u4b3k) seed8 AP=0.7148 | a1g1 GCN, lr=2.083e-4, ep=900, elementwise |
| Peptides-struct | MAE 0.2441±0.0017 (n=10) | `configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml` | [`bqkect9l`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/bqkect9l) seed6 MAE=0.2416 | a1g1 GINE, lr=7e-4, ep=250 |
| PascalVOC-SP | F1 0.4687±0.0070 (n=5) | `configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml` | [`vyt7hjj5`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vyt7hjj5) seed3 F1=0.4811 | a2g2 GATEDGCN×2, lr≈3.07e-4, ep=200 |
| COCO-SP | F1 0.4155±0.0076 (n=5) | `configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml` | [`xgjakrz0`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xgjakrz0) seed3 F1=0.4248 | a1g1 GATEDGCN, lr=1e-3, ep=300 |

If any anchor looks wrong, open the run → **Overview → Reproduce** and paste the CLI here; we will sync the yaml.

---

## 2. Four variants (Table 5) — names used everywhere

| Variant name (W&B) | Meaning | Code override |
|--------------------|---------|---------------|
| **`SiGMA`** | Best gated hybrid (paper row) | none (anchor as-is) |
| **`SiGMA_ungated`** | Same heads, **no gating** | `gnn.hybrid.gate none` |
| **`Attn_only`** | Drop MP; replace with attention | `num_attn=Na+Ng`, `num_gnn=0` |
| **`MP_only`** | Drop attention; replace with same MP type | `num_attn=0`, `num_gnn=Na+Ng`, types repeated |

### How to find them in W&B

- **Group** (one group per dataset×variant, 5 seeds inside):  
  `paper_T5_<dataset>_<Variant>`  
  Examples:
  - `paper_T5_coco_SiGMA`
  - `paper_T5_coco_SiGMA_ungated`
  - `paper_T5_coco_Attn_only`
  - `paper_T5_coco_MP_only`
- **Tags** on every run: `paper_table5`, `<Variant>`, `<dataset>`, `seed<k>`, `source_<runid>`

Filter UI: Tags → `SiGMA` vs `SiGMA_ungated` vs `Attn_only` vs `MP_only`.

---

## 3. Launch on cluster

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table5_ablations.sh
```

Optional: override with `PAPER_T5_PARALLEL=N`; default is **18**.

| Field | Value |
|-------|-------|
| **SLURM array** | ✅ **32232124** (submitted 2026-07-17) |
| **Job name** | `sigma_T5_abl` |
| **Tasks** | `1-80%18` = 4×4×5 |
| **Scripts** | `submit_paper_table5_ablations.sh` → `run_paper_table5_ablations.sh` |
| **Logs** | `logs_gnnplus/sigma_T5_abl_32232124_<TASK>.log` |
| **Needs** | `gate=none` support (this branch) |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

### Relaunch — COCO Attn_only on `gpu_h200` (5 seeds, new job)

Old runs (e.g. [`40o1sohg`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/40o1sohg) = seed2) may not finish in time. **Do not cancel** `32232124_71..75` — submit a parallel H200 array. Same W&B group `paper_T5_coco_Attn_only`; new run names end in `_h200`.

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table5_coco_attn_only_h200.sh
# 👉 paste JOBID here
```

| Field | Value |
|-------|-------|
| **H200 attempt** | `33813232` — failed (quota / timeout) |
| **Relaunch SLURM** | ✅ **`34070241`** Attn `71-75` · ❌ **`34070242`** MP `78-80` + ❌ **`34070243`** ungated `67` (epoch0 `OSError 122` holylabs quota — relaunch with `GNNPLUS_OUT_DIR` on netscratch) |
| **W&B groups** | `paper_T5_coco_{Attn_only,MP_only,SiGMA_ungated}` |

---

## 4. Aggregate (5 seeds each)

Preferred (Table 5 + Table 6 together):

```bash
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 5 --detail
```

Per-group fallback:

```bash
PREFIX=paper_T5
for ds in peptides_func peptides_struct voc coco; do
  for v in SiGMA SiGMA_ungated Attn_only MP_only; do
    echo "===== ${ds} / ${v} ====="
    python scripts/api_wanndb_query/aggregate_paper_repro.py \
      --group ${PREFIX}_${ds}_${v} --metric best_test_perf --state finished
  done
done
```

### Fill-in results (Table 5)

| Model | Peptides-func ↑ | Peptides-struct ↓ | VOC-SP ↑ | COCO-SP ↑ | n | W&B group suffix |
|-------|-----------------|-------------------|----------|-----------|---|------------------|
| MP_only | | | | | 5 | `_MP_only` |
| Attn_only | | | | | 5 | `_Attn_only` |
| SiGMA_ungated | | | | | 5 | `_SiGMA_ungated` |
| **SiGMA** | | | | | 5 | `_SiGMA` |

---

## 5. Files

| Path | Role |
|------|------|
| `bash_interface/cluster/submit_paper_table5_ablations.sh` | **launch this** |
| `bash_interface/cluster/run_paper_table5_ablations.sh` | SLURM array worker |
| `configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml` | SiGMA peptides-func |
| `configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml` | SiGMA peptides-struct |
| `configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml` | SiGMA VOC |
| `configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml` | SiGMA COCO |
| `GNNPlus/layer/gated_hybrid_layer.py` | `gate=none` for SiGMA_ungated |

Legacy (optional Table 6 extras): `submit_paper_table56_ablations.sh` — prefer Table 5 script above.

---

## 6. If you paste W&B Reproduce CLIs

Template (we already match these for the four LRGB anchors; paste if something drifts):

```text
# Peptides-func exemplar
https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/l31u4b3k

# Peptides-struct exemplar
https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/bqkect9l

# VOC exemplar
https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/vyt7hjj5

# COCO exemplar
https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/xgjakrz0
```
