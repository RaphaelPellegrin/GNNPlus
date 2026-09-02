#!/usr/bin/env bash
# Fill Table 5 PATTERN + CLUSTER ablations to 20 matched seeds (paired t-test power).
#
# Existing cohort (keep for paper table):
#   CLUSTER  seeds 0–4   → paper_T5_cluster_*  (+ SiGMA: paper_bestmodel_v1_cluster_ht9bntg2)
#   PATTERN  seeds 5–9   → paper_T5_pattern_gritvn4_*  (+ SiGMA: paper_sigma_grit_attn_pattern_vn4)
#
# This launch adds 15 seeds each (→ 20 total per dataset):
#   CLUSTER  seeds 5–19  (5 variants × 15 = 75 jobs; skips SiGMA_ungated_attn — no attn on CLUSTER)
#   PATTERN  seeds 10–24 (6 variants × 15 = 90 jobs; includes SiGMA for paired tests)
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_pattern_cluster_seed20_fill.sh
#
# Optional overrides:
#   PAPER_T5_SEED20_PARALLEL=10
#   PAPER_T5_SEED20_CLUSTER_ONLY=1   # skip PATTERN array
#   PAPER_T5_SEED20_PATTERN_ONLY=1   # skip CLUSTER array

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

TARGET_TOTAL="${PAPER_T5_SEED20_TARGET_TOTAL:-20}"
PARALLEL="${PAPER_T5_SEED20_PARALLEL:-5}"
NICE="${PAPER_T5_SEED20_NICE:-10000}"
MEM="${PAPER_T5_SEED20_MEM:-128GB}"
TIME="${PAPER_T5_SEED20_TIME:-120:00:00}"

# --- CLUSTER: had seeds 0–4; fill 5..(TARGET_TOTAL-1) ---
CLUSTER_EXISTING="${PAPER_T5_SEED20_CLUSTER_EXISTING:-5}"
CLUSTER_SEED_OFFSET="${PAPER_T5_SEED20_CLUSTER_SEED_OFFSET:-${CLUSTER_EXISTING}}"
CLUSTER_NUM_SEEDS="${PAPER_T5_SEED20_CLUSTER_NUM_SEEDS:-$((TARGET_TOTAL - CLUSTER_EXISTING))}"
# Paper Table 5: no SiGMA_ungated_attn on CLUSTER (variant index 3).
CLUSTER_VARIANT_INDICES="${PAPER_T5_SEED20_CLUSTER_VARIANT_INDICES:-0,1,2,4,5}"
CLUSTER_INCLUDE_SIGMA="${PAPER_T5_SEED20_CLUSTER_INCLUDE_SIGMA:-1}"
CLUSTER_VARIANT_COUNT="$(echo "${CLUSTER_VARIANT_INDICES}" | tr ',' '\n' | wc -l | tr -d ' ')"
CLUSTER_NUM_TASKS="${PAPER_T5_SEED20_CLUSTER_NUM_TASKS:-$((CLUSTER_VARIANT_COUNT * CLUSTER_NUM_SEEDS))}"
CLUSTER_ARRAY="${PAPER_T5_SEED20_CLUSTER_ARRAY:-1-${CLUSTER_NUM_TASKS}}"

# --- PATTERN: had seeds 5–9; fill 10..(5+TARGET_TOTAL-1) = 10–24 when TARGET_TOTAL=20 ---
PATTERN_EXISTING_OFFSET="${PAPER_T5_SEED20_PATTERN_EXISTING_OFFSET:-5}"
PATTERN_EXISTING="${PAPER_T5_SEED20_PATTERN_EXISTING:-5}"
PATTERN_SEED_OFFSET="${PAPER_T5_SEED20_PATTERN_SEED_OFFSET:-$((PATTERN_EXISTING_OFFSET + PATTERN_EXISTING))}"
PATTERN_NUM_SEEDS="${PAPER_T5_SEED20_PATTERN_NUM_SEEDS:-$((TARGET_TOTAL - PATTERN_EXISTING))}"
PATTERN_INCLUDE_SIGMA="${PAPER_T5_SEED20_PATTERN_INCLUDE_SIGMA:-1}"
if [ "${PATTERN_INCLUDE_SIGMA}" = "1" ]; then
    PATTERN_VARIANT_COUNT=6
else
    PATTERN_VARIANT_COUNT=5
fi
PATTERN_NUM_TASKS="${PAPER_T5_SEED20_PATTERN_NUM_TASKS:-$((PATTERN_VARIANT_COUNT * PATTERN_NUM_SEEDS))}"
PATTERN_ARRAY="${PAPER_T5_SEED20_PATTERN_ARRAY:-1-${PATTERN_NUM_TASKS}}"

CLUSTER_JOB=""
PATTERN_JOB=""

if [ "${PAPER_T5_SEED20_PATTERN_ONLY:-0}" != "1" ]; then
    cluster_sbatch=(
        --parsable
        --job-name=sigma_T5_cl_s20
        --array="${CLUSTER_ARRAY}%${PARALLEL}"
        --partition=mweber_gpu
        --mem="${MEM}"
        --time="${TIME}"
        --gpus=1
        --output="logs_gnnplus/sigma_T5_cl_s20_%A_%a.log"
        --export=ALL,ENV_NAME=gnnplus,PAPER_T5_CLUSTER_NUM_SEEDS="${CLUSTER_NUM_SEEDS}",PAPER_T5_CLUSTER_SEED_OFFSET="${CLUSTER_SEED_OFFSET}",PAPER_T5_CLUSTER_INCLUDE_SIGMA="${CLUSTER_INCLUDE_SIGMA}",PAPER_T5_CLUSTER_VARIANT_INDICES="${CLUSTER_VARIANT_INDICES}",PAPER_T5_CLUSTER_NUM_TASKS="${CLUSTER_NUM_TASKS}",PAPER_T5_CLUSTER_WANDB_PREFIX=paper_T5,GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
    )
    if [ "${NICE}" != "0" ]; then
        cluster_sbatch+=(--nice="${NICE}")
    fi
    CLUSTER_JOB="$(
        sbatch "${cluster_sbatch[@]}" \
            bash_interface/cluster/run_paper_table5_cluster_ablations.sh
    )"
fi

if [ "${PAPER_T5_SEED20_CLUSTER_ONLY:-0}" != "1" ]; then
    pattern_sbatch=(
        --parsable
        --job-name=sigma_T5_pat_s20
        --array="${PATTERN_ARRAY}%${PARALLEL}"
        --partition=mweber_gpu
        --mem="${MEM}"
        --time="${TIME}"
        --gpus=1
        --output="logs_gnnplus/sigma_T5_pat_s20_%A_%a.log"
        --export=ALL,ENV_NAME=gnnplus,PAPER_T5_PATTERN_GRITVN4_NUM_SEEDS="${PATTERN_NUM_SEEDS}",PAPER_T5_PATTERN_GRITVN4_SEED_OFFSET="${PATTERN_SEED_OFFSET}",PAPER_T5_PATTERN_GRITVN4_INCLUDE_SIGMA="${PATTERN_INCLUDE_SIGMA}",PAPER_T5_PATTERN_GRITVN4_NUM_TASKS="${PATTERN_NUM_TASKS}",PAPER_T5_PATTERN_GRITVN4_WANDB_PREFIX=paper_T5_pattern_gritvn4,GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
    )
    if [ "${NICE}" != "0" ]; then
        pattern_sbatch+=(--nice="${NICE}")
    fi
    PATTERN_JOB="$(
        sbatch "${pattern_sbatch[@]}" \
            bash_interface/cluster/run_paper_table5_pattern_gritvn4_ablations.sh
    )"
fi

CLUSTER_SEED_END=$((CLUSTER_SEED_OFFSET + CLUSTER_NUM_SEEDS - 1))
PATTERN_SEED_END=$((PATTERN_SEED_OFFSET + PATTERN_NUM_SEEDS - 1))

cat <<EOF

=== Table 5 seed-20 fill: PATTERN + CLUSTER ===
  Target total seeds per dataset: ${TARGET_TOTAL}
  Parallel GPUs:                  ${PARALLEL}
  Mem / time:                     ${MEM} / ${TIME}

CLUSTER (ht9bntg2 anchor)
  JOBID:          ${CLUSTER_JOB:-skipped}
  New seeds:      ${CLUSTER_SEED_OFFSET}–${CLUSTER_SEED_END}  (+ existing 0–$((CLUSTER_EXISTING - 1)) → ${TARGET_TOTAL} total)
  Tasks:          ${CLUSTER_ARRAY}  (${CLUSTER_VARIANT_COUNT} variants × ${CLUSTER_NUM_SEEDS} seeds = ${CLUSTER_NUM_TASKS})
  Variants:       ${CLUSTER_VARIANT_INDICES}  (0=SiGMA,1=ungated,2=attn_gate,4=Attn_only,5=MP_only; skips 3=ungated_attn)
  W&B groups:     paper_T5_cluster_<Variant>
  Logs:           logs_gnnplus/sigma_T5_cl_s20_${CLUSTER_JOB:-JOBID}_<TASK>.log

PATTERN (GRIT+VN4 anchor)
  JOBID:          ${PATTERN_JOB:-skipped}
  New seeds:      ${PATTERN_SEED_OFFSET}–${PATTERN_SEED_END}  (+ existing ${PATTERN_EXISTING_OFFSET}–$((PATTERN_EXISTING_OFFSET + PATTERN_EXISTING - 1)) → ${TARGET_TOTAL} total)
  Tasks:          ${PATTERN_ARRAY}  (${PATTERN_VARIANT_COUNT} variants × ${PATTERN_NUM_SEEDS} seeds = ${PATTERN_NUM_TASKS})
  Include SiGMA:  ${PATTERN_INCLUDE_SIGMA}
  W&B groups:     paper_T5_pattern_gritvn4_<Variant>
  Logs:           logs_gnnplus/sigma_T5_pat_s20_${PATTERN_JOB:-JOBID}_<TASK>.log

After jobs finish, re-export paired-test CSVs:
  python scripts/api_wanndb_query/export_table5_paired_ttest_data.py

Paste JOBIDs into CLUSTER_LAUNCHES.md

EOF
