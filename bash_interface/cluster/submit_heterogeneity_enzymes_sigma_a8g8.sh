#!/usr/bin/env bash
# Launch ENZYMES SiGMA a8g8 L12 heterogeneity profile (match MOE_6/7dsqq7z2).
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_heterogeneity_enzymes_sigma_a8g8.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

PARTITION="${HETERO_PARTITION:-mweber_gpu}"
NICE="${HETERO_NICE:-10000}"
MEM="${HETERO_MEM:-64GB}"
if [ -n "${HETERO_TIME:-}" ]; then
    TIME="${HETERO_TIME}"
elif [ "${PARTITION}" = "gpu_h200" ]; then
    TIME="72:00:00"
else
    TIME="192:00:00"
fi
REQUIRED="${HETERO_REQUIRED_TEST_APPEARANCES:-100}"
MAX_TRIALS="${HETERO_MAX_TRIALS:-2000}"
SEED0="${HETERO_SEED0:-0}"
WANDB_USE="${HETERO_WANDB:-1}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_enz_a8g8] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

sbatch_args=(
    --parsable
    --job-name=hetero_enz_a8g8
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/hetero_enz_a8g8_%j.log"
    --export=ALL,ENV_NAME=gnnplus,HETERO_REQUIRED_TEST_APPEARANCES="${REQUIRED}",HETERO_MAX_TRIALS="${MAX_TRIALS}",HETERO_SEED0="${SEED0}",HETERO_WANDB="${WANDB_USE}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_heterogeneity_enzymes_sigma_a8g8.sh
)"

cat <<EOF

=== ENZYMES SiGMA a8g8 hetero profile submitted ===
  JOBID:         ${job_id}
  Partition:     ${PARTITION}
  Appearances:   ≥${REQUIRED} per graph
  Max trials:    ${MAX_TRIALS}
  Mem / time:    ${MEM} / ${TIME}
  Config:        configs/heterogeneity/enzymes-sigma-a8g8.yaml
  Match:         MOE_6/7dsqq7z2 (a8g8 L12 plateau H64 dh16)
  W&B:           group=building_hetero_profile_enzymes  name=enzymes_sigma_a8g8
  Out:           \${GNNPLUS_OUT_DIR}/heterogeneity/enzymes_sigma_a8g8/
  Logs:          logs_gnnplus/hetero_enz_a8g8_${job_id}.log

  Paste JOBID into Paper_heterogeneity.md + CLUSTER_LAUNCHES.md

EOF
