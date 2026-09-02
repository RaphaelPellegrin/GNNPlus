# Table 5 architecture ablations — PATTERN + CLUSTER seed-20 fill (rebuttal power)

Rebuttal follow-up: extend **PATTERN** and **CLUSTER** from **5 → 20 matched seeds** for paired $t$-tests on Table 5 architectural ablations.

Entity/project: [`weber-geoml-harvard-university/GNNPlus`](https://wandb.ai/weber-geoml-harvard-university/GNNPlus)  
Master tracker: [`CLUSTER_LAUNCHES.md`](CLUSTER_LAUNCHES.md)  
Paired-test export: [`scripts/api_wanndb_query/export_table5_paired_ttest_data.py`](scripts/api_wanndb_query/export_table5_paired_ttest_data.py)  
CSV output: [`results_summaries/table5_paired_ttest_export/`](results_summaries/table5_paired_ttest_export/)

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  🔄  SUBMITTED 2026-09-01 · Table 5 seed-20 fill (PATTERN + CLUSTER)     ║
║  🧪  165 jobs total · 5 parallel GPUs · 128GB / 120h                     ║
║  📄  submit_paper_table5_pattern_cluster_seed20_fill.sh                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

| Field | Value |
|-------|-------|
| **Status** | 🔄 **IN PROGRESS** (CLUSTER ablations need resubmit — see below) |
| **Why** | Reviewer: mean±std over 5 seeds insufficient; need paired seed-level tests + more power |
| **Target** | **20 seeds** per dataset (paired with existing 5-seed cohort) |
| **Script** | `bash_interface/cluster/submit_paper_table5_pattern_cluster_seed20_fill.sh` |
| **Worker** | `run_paper_table5_cluster_ablations.sh` · `run_paper_table5_pattern_gritvn4_ablations.sh` |

---

## SLURM jobs

| Dataset | JOBID | Tasks | Seeds (new) | Seeds (total) | Jobs |
|---------|-------|-------|-------------|---------------|------|
| **CLUSTER** | **`43796006`** | `1-75%5` | **5–19** | 0–19 (20) | 75 |
| **PATTERN** | **`43796007`** | `1-90%5` | **10–24** | 5–24 (20) | 90 |

| Field | Value |
|-------|-------|
| **Job names** | `sigma_T5_cl_s20` · `sigma_T5_pat_s20` |
| **Partition** | `mweber_gpu` · 128GB · 120h · nice=10000 |
| **Logs** | `logs_gnnplus/sigma_T5_cl_s20_43796006_<TASK>.log` · `sigma_T5_pat_s20_43796007_<TASK>.log` |

---

## CLUSTER resubmit (tasks 16–75)

**Root cause (43796006):** `sbatch --export` split `PAPER_T5_CLUSTER_VARIANT_INDICES=0,1,2,4,5` on commas → workers only saw `0`. Tasks 1–15 (SiGMA seeds 5–19) succeeded; tasks 16+ crashed with `variant_list[…]: unbound variable`.

**Fix:** submit script now exports hyphen-separated indices (`0-1-2-4-5`); run script accepts commas or hyphens.

After `git pull`, resubmit failed ablations only (SiGMA seeds 5–19 already in W&B):

```bash
PAPER_T5_SEED20_CLUSTER_ARRAY=16-75 \
PAPER_T5_SEED20_CLUSTER_ONLY=1 \
  bash bash_interface/cluster/submit_paper_table5_pattern_cluster_seed20_fill.sh
```

---

## Cohort design

### CLUSTER (ht9bntg2 anchor)

| Item | Value |
|------|-------|
| **Config** | `configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml` |
| **SiGMA baseline (seeds 0–4)** | W&B `paper_bestmodel_v1_cluster_ht9bntg2` · 78.956±0.112% |
| **SiGMA baseline (seeds 5–19)** | W&B `paper_T5_cluster_SiGMA` (this fill) |
| **W&B prefix** | `paper_T5_cluster_<Variant>` |
| **Variants run** | `0,1,2,4,5` → SiGMA · ungated · attn_gate · Attn_only · MP_only |
| **Skipped** | `SiGMA_ungated_attn` (variant 3) — paper table has `--` (no attention head) |

### PATTERN (GRIT + VN4 anchor)

| Item | Value |
|------|-------|
| **Config** | `pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml` + `attn_type=grit` + VN=4 |
| **SiGMA baseline (seeds 5–9)** | W&B `paper_sigma_grit_attn_pattern_vn4` · 87.395±0.194% |
| **SiGMA baseline (seeds 10–24)** | W&B `paper_T5_pattern_gritvn4_SiGMA` (this fill) |
| **W&B prefix** | `paper_T5_pattern_gritvn4_<Variant>` |
| **Variants run** | SiGMA + all 5 ablation variants (`INCLUDE_SIGMA=1`) |

---

## Launch (cluster)

```bash
source ~/.gnnplus_env
export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results

cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
git pull

bash bash_interface/cluster/submit_paper_table5_pattern_cluster_seed20_fill.sh
```

Optional: `PAPER_T5_SEED20_PARALLEL=10` · cheaper `PAPER_T5_SEED20_TARGET_TOTAL=10` (see script header).

---

## Monitor

```bash
squeue -u $USER | grep -E 'sigma_T5_cl_s20|sigma_T5_pat_s20'

tail -f logs_gnnplus/sigma_T5_cl_s20_43796006_1.log
tail -f logs_gnnplus/sigma_T5_pat_s20_43796007_1.log
```

Per-group progress:

```bash
for v in SiGMA SiGMA_ungated SiGMA_attn_gate Attn_only MP_only; do
  echo "===== cluster $v ====="
  python scripts/api_wanndb_query/aggregate_paper_repro.py \
    --group paper_T5_cluster_${v} --metric best_test_perf --state finished
done

for v in SiGMA SiGMA_ungated SiGMA_attn_gate SiGMA_ungated_attn Attn_only MP_only; do
  echo "===== pattern $v ====="
  python scripts/api_wanndb_query/aggregate_paper_repro.py \
    --group paper_T5_pattern_gritvn4_${v} --metric best_test_perf --state finished
done
```

---

## After jobs finish

Re-export paired-test CSVs (merges old + new SiGMA seed groups):

```bash
python scripts/api_wanndb_query/export_table5_paired_ttest_data.py
# → results_summaries/table5_paired_ttest_export/table5_arch_ablation_{long,wide,groups}.csv
```

Re-run paired $t$-tests locally on the wide CSV or use prior analysis workflow.

---

## Related (not in this fill)

| Dataset | Status | Notes |
|---------|--------|-------|
| Peptides-func/struct | 5 seeds | extend later with same `PAPER_T5_SEED20_*` pattern |
| PascalVOC-SP | 5 seeds | idem |
| MNIST / CIFAR10 | 5 seeds | idem |
| Synthetic routing toy | done | `scripts/synthetic/run_gcn_gin_routing_paired_ttests.py` |
