#!/usr/bin/env bash
# Shared submit helper for SiGMA d_h-matched tiers (sourced by tier scripts).
# Expects: TIER, NUM_FAMILIES, FAMILY_BLURB, CFG_LIST (bash array), optional defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

: "${TIER:?TIER must be set (fast|slow|coco)}"
: "${NUM_FAMILIES:?NUM_FAMILIES must be set}"
: "${FAMILY_BLURB:?FAMILY_BLURB must be set}"

NUM_SEEDS="${SIGMA_DH_MATCHED_NUM_SEEDS:-5}"
NUM_LRS="${SIGMA_DH_MATCHED_NUM_LRS:-2}"
NUM_TASKS="${SIGMA_DH_MATCHED_NUM_TASKS:-$((NUM_FAMILIES * NUM_LRS * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_DH_MATCHED_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_DH_MATCHED_PARALLEL:-${DEFAULT_PARALLEL:-20}}"
PARTITION="${SIGMA_DH_MATCHED_PARTITION:-mweber_gpu}"
NICE="${SIGMA_DH_MATCHED_NICE:-10000}"
MEM="${SIGMA_DH_MATCHED_MEM:-${DEFAULT_MEM:-128GB}}"
TIME="${SIGMA_DH_MATCHED_TIME:-${DEFAULT_TIME:-120:00:00}}"
JOB_NAME="${SIGMA_DH_MATCHED_JOB_NAME:-sigma_dh_${TIER}}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_sigma_dh_matched_${TIER}] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

missing=0
for cfg in "${CFG_LIST[@]}"; do
  if [ ! -f "${cfg}" ]; then
    echo "MISSING ${cfg}"
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "d_h-matched configs missing — run:"
  echo "  python scripts/generate_sigma_dh_matched_configs.py"
  exit 1
fi

chmod +x bash_interface/cluster/run_sigma_dh_matched.sh

sbatch_args=(
    --parsable
    --job-name="${JOB_NAME}"
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/${JOB_NAME}_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_DH_MATCHED_TIER="${TIER}",SIGMA_DH_MATCHED_NUM_SEEDS="${NUM_SEEDS}",SIGMA_DH_MATCHED_NUM_LRS="${NUM_LRS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_dh_matched.sh
)"

cat <<EOF

=== SiGMA d_h-matched TIER=${TIER} submitted ===
  ARRAY JOBID:   ${job_id}
  Job name:      ${JOB_NAME}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}
                 (${NUM_FAMILIES} families × ${NUM_LRS} LRs × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  LRs:           0.001 (lr001) and 0.01 (lr01) — pick better per family after
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/${JOB_NAME}_${job_id}_<TASK>.log
  Out:           \$GNNPLUS_OUT_DIR/sigma_dh_matched/<fam>_<lr>_seed<s>/
  Docs:          Paper_sigma_dh_matched.md

${FAMILY_BLURB}

  Paste JOBID into Paper_sigma_dh_matched.md + CLUSTER_LAUNCHES.md

EOF
