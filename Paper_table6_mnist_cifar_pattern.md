# SiGMA paper Table 7 — MNIST + CIFAR10 + PATTERN (homog / hetero MP)

Baselines already have **multiple homogeneous MP heads**. This campaign only launches:

- `Homog_MP_ungated` — same types, `gate=none`
- `Hetero_MP` — swap **one** (last) MP head to a different type, gated
- `Hetero_MP_ungated` — same one-head swap, `gate=none`

**SiGMA / Homog_MP (gated)** → reuse [`Paper_final_runs.md`](Paper_final_runs.md) `paper_bestmodel_v1_*` (do not relaunch).

Code W&B prefix: `paper_T6_*`.  
Architectural Table 6: [`Paper_ablations_mnist_cifar.md`](Paper_ablations_mnist_cifar.md).

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  RUNNING  ·  array 35721068  ·  1-45%10                              ║
║  🧪  3 ds × 3 variants × 5 seeds = 45 jobs                               ║
║  📄  logs_gnnplus/sigma_T6_mc_35721068_<TASK>.log                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🔄 **RUNNING** |
| **SLURM array** | ✅ **`35721068`** (`1-45%10`) · prior `35720920` cancelled |
| **Job name** | `sigma_T6_mc` |
| **Tasks** | `1-45%10` |
| **W&B (new)** | `paper_T6_{mnist,cifar10,pattern}_{Homog_MP_ungated,Hetero_MP,Hetero_MP_ungated}` |
| **Reuse** | `paper_bestmodel_v1_{mnist_lcvbyyss,cifar10_ulij45a2,pattern_ta9qtxb9}` for SiGMA/Homog gated |

---

## 1. Frozen best baselines (reuse for SiGMA / Homog_MP)

| Dataset | Paper Acc (%) | Anchor | Exemplar | Homog MP | Hetero (swap last only) |
|---------|---------------|--------|----------|----------|-------------------------|
| MNIST | 98.628 ± 0.105 | `mnist-hybrid-lcvbyyss-a2g2-anchor.yaml` | [`uh7nxm4e`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/uh7nxm4e) | `GATEDGCN,GATEDGCN` | `GATEDGCN,GCN` |
| CIFAR10 | 79.528 ± 0.180 | `cifar10-hybrid-ulij45a2-anchor.yaml` | [`3tx560wq`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/3tx560wq) | `GATEDGCN`×4 | `GATEDGCN,GATEDGCN,GATEDGCN,GCN` |
| PATTERN | 86.991 ± 0.039 | `pattern-gcne-best-hybrid.yaml` | [`ta9qtxb9`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/ta9qtxb9) | `GCNE,GCNE` | `GCNE,GINE` |

**PATTERN on GRIT+VN4 SiGMA (~87.4%)** — relaunch Table 6/7 column to match best SiGMA:

| Cell | Source |
|------|--------|
| SiGMA / Homog_MP gated | reuse `paper_sigma_grit_attn_pattern_vn4` (**87.395±0.194%**) |
| Homog_ungated / Hetero / Hetero_ungated | `bash bash_interface/cluster/submit_paper_table6_pattern_gritvn4.sh` → `paper_T6_pattern_gritvn4_*` |

Or all remaining gaps (T5 retries + T5/T6 PATTERN gritvn4):

```bash
bash bash_interface/cluster/submit_paper_table56_remaining_gaps.sh
```

---

## 2. Launch

```bash
# cancel the broken 75-job array first
scancel 35720920

source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table6_mnist_cifar_pattern.sh
```

| Field | Value |
|-------|-------|
| **Scripts** | `submit_paper_table6_mnist_cifar_pattern.sh` → `run_paper_table6_mnist_cifar_pattern.sh` |
| **Logs** | `logs_gnnplus/sigma_T6_mc_<JOBID>_<TASK>.log` |
| **Master tracker** | [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md) |

---

## 3. Aggregate

```bash
python scripts/api_wanndb_query/aggregate_paper_table56.py --table 6mc
```

SiGMA / Homog_MP gated cells: take from `paper_bestmodel_v1_*` (same Acc as Table 3).
