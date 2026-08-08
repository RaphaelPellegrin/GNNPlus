#!/usr/bin/env bash
# Submit SiGMA ~100k budget fills (7 datasets × 5 seeds = 35).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_sigma_budget_100k.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_B100K_NUM_SEEDS:-5}"
NUM_FAMILIES="${SIGMA_B100K_NUM_FAMILIES:-7}"
NUM_TASKS="${SIGMA_B100K_NUM_TASKS:-$((NUM_FAMILIES * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_B100K_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_B100K_PARALLEL:-20}"
PARTITION="${SIGMA_B100K_PARTITION:-mweber_gpu}"
NICE="${SIGMA_B100K_NICE:-10000}"
MEM="${SIGMA_B100K_MEM:-128GB}"
TIME="${SIGMA_B100K_TIME:-96:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_sigma_budget_100k] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

chmod +x bash_interface/cluster/run_sigma_budget_100k.sh

sbatch_args=(
    --parsable
    --job-name=sigma_b100k
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_b100k_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_B100K_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_budget_100k.sh
)"

cat <<EOF

=== SiGMA ~100k budget submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (7 ds × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_b100k_${job_id}_<TASK>.log
  Out:           \$GNNPLUS_OUT_DIR/sigma_budget/<fam>_b100k_seed<s>/
  W\&B:          paper_budget_<ds>_b100k
  Configs (all a1g1, recounted ≤100k):
    ZINC        H38 dh12 L10 GINE     (~99.9k)
    MNIST       H28 dh26 L6  GATEDGCN (~100.0k)
    PATTERN     H48 dh20 L4  GCNE+GRIT (~100.0k)
    CLUSTER     H56 dh16 L10 GATEDGCN (~100.0k)
    Pep-func    H36 dh36 L6  GINE     (~100.0k)
    Pep-struct  H44 dh44 L4  GINE     (~99.9k)
    VOC         H40 dh8  L10 GATEDGCN (~99.9k)
  Docs:          Paper_sigma_budget.md

  Paste JOBID into Paper_sigma_budget.md + CLUSTER_LAUNCHES.md

EOF
