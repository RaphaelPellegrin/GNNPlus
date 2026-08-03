#!/usr/bin/env bash
# Launch PATTERN Table 5/6 ablations anchored on SiGMA GRIT+VN4 (~87.4%).
#
# Default: 5 variants × seeds 5–9 = 25 jobs (skips re-running SiGMA; reuse
# paper_sigma_grit_attn_pattern_vn4). Set PAPER_T5_PATTERN_GRITVN4_INCLUDE_SIGMA=1
# to also re-run the SiGMA row (30 jobs).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_paper_table5_pattern_gritvn4_ablations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${PAPER_T5_PATTERN_GRITVN4_NUM_SEEDS:-5}"
SEED_OFFSET="${PAPER_T5_PATTERN_GRITVN4_SEED_OFFSET:-5}"
INCLUDE_SIGMA="${PAPER_T5_PATTERN_GRITVN4_INCLUDE_SIGMA:-0}"
NUM_VN="${PAPER_T5_PATTERN_GRITVN4_NUM_VN:-4}"
WANDB_PREFIX="${PAPER_T5_PATTERN_GRITVN4_WANDB_PREFIX:-paper_T5_pattern_gritvn4}"

if [ "${INCLUDE_SIGMA}" = "1" ]; then
    NUM_VARIANTS=6
else
    NUM_VARIANTS=5
fi
NUM_TASKS="${PAPER_T5_PATTERN_GRITVN4_NUM_TASKS:-$((NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${PAPER_T5_PATTERN_GRITVN4_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${PAPER_T5_PATTERN_GRITVN4_PARALLEL:-5}"
NICE="${PAPER_T5_PATTERN_GRITVN4_NICE:-10000}"
MEM="${PAPER_T5_PATTERN_GRITVN4_MEM:-128GB}"
TIME="${PAPER_T5_PATTERN_GRITVN4_TIME:-120:00:00}"

sbatch_args=(
    --parsable
    --job-name=sigma_T5_pat_gritvn4
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition=mweber_gpu
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_T5_pat_gritvn4_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,PAPER_T5_PATTERN_GRITVN4_NUM_SEEDS="${NUM_SEEDS}",PAPER_T5_PATTERN_GRITVN4_SEED_OFFSET="${SEED_OFFSET}",PAPER_T5_PATTERN_GRITVN4_INCLUDE_SIGMA="${INCLUDE_SIGMA}",PAPER_T5_PATTERN_GRITVN4_NUM_VN="${NUM_VN}",PAPER_T5_PATTERN_GRITVN4_NUM_TASKS="${NUM_TASKS}",PAPER_T5_PATTERN_GRITVN4_WANDB_PREFIX="${WANDB_PREFIX}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR:-}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_paper_table5_pattern_gritvn4_ablations.sh
)"

cat <<EOF

=== PATTERN Table 5 ablations (GRIT+VN${NUM_VN} SiGMA anchor) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (${NUM_VARIANTS} variants × ${NUM_SEEDS} seeds, offset=${SEED_OFFSET})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_T5_pat_gritvn4_${job_id}_<TASK>.log

  Anchor SiGMA:  paper_sigma_grit_attn_pattern_vn4  (~87.4%, seeds 5–9)
  Config:        pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml + VN=${NUM_VN}
  Include SiGMA: ${INCLUDE_SIGMA}  (0 = reuse existing VN4 group)

  W&B groups:    ${WANDB_PREFIX}_<Variant>
  Variants:      SiGMA (opt) | SiGMA_ungated | SiGMA_attn_gate |
                 SiGMA_ungated_attn | Attn_only | MP_only

  Aggregate:
    for v in SiGMA_ungated SiGMA_attn_gate SiGMA_ungated_attn Attn_only MP_only; do
      python scripts/api_wanndb_query/aggregate_paper_repro.py \\
        --group ${WANDB_PREFIX}_\$v --metric best_test_perf --state finished
    done
    # SiGMA row: paper_sigma_grit_attn_pattern_vn4

  Paste JOBID into Paper_ablations_mnist_cifar.md + CLUSTER_LAUNCHES.md

EOF
