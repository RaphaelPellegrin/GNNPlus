#!/usr/bin/env bash
# Submit CIFAR10 budget re-fit (params actually ≤500k / 1M / 2M).
#
# 3 families × 5 seeds = 15 jobs, up to 15 GPUs (or %10).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_cifar_budget_fit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${CIFAR_BUDGET_FIT_NUM_SEEDS:-5}"
NUM_FAMILIES="${CIFAR_BUDGET_FIT_NUM_FAMILIES:-3}"
NUM_TASKS="${CIFAR_BUDGET_FIT_NUM_TASKS:-$((NUM_FAMILIES * NUM_SEEDS))}"
ARRAY_SPEC="${CIFAR_BUDGET_FIT_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${CIFAR_BUDGET_FIT_PARALLEL:-15}"
PARTITION="${CIFAR_BUDGET_FIT_PARTITION:-mweber_gpu}"
NICE="${CIFAR_BUDGET_FIT_NICE:-10000}"
MEM="${CIFAR_BUDGET_FIT_MEM:-128GB}"
TIME="${CIFAR_BUDGET_FIT_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_cifar_budget_fit] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_cifar_budget_fit.sh

sbatch_args=(
    --parsable
    --job-name=cifar_budget_fit
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/cifar_budget_fit_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,CIFAR_BUDGET_FIT_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_cifar_budget_fit.sh
)"

cat <<EOF

=== CIFAR10 budget fit (params ≤500k/1M/2M) submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (3 families × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/cifar_budget_fit_${job_id}_<TASK>.log
  Out:           \$GNNPLUS_OUT_DIR/sigma_budget/cifar10_b{500k,1m,2m}_fit_seed<s>/
  W&B:           paper_budget_cifar10_b{500k,1m,2m}_fit
  Configs:
    ≤500k  H=66 d_h=52 a1g1 L10  (~498.8k)
    ≤1M    H=86 d_h=76 a1g1 L10  (~998.9k)
    ≤2M    H=82 d_h=84 a1g2 L10  (~1.998M)
  Docs:          Paper_sigma_budget.md

  Paste JOBID into Paper_sigma_budget.md + CLUSTER_LAUNCHES.md

EOF
