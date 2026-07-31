#!/usr/bin/env bash
# Submit SiGMA gate-viz training (Xu-recipe) for 6 TU datasets (ckpt every 50 ep).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#   bash bash_interface/cluster/submit_tu_sigma_gate_viz.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_TASKS="${GATE_VIZ_NUM_TASKS:-6}"
ARRAY_SPEC="${GATE_VIZ_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${GATE_VIZ_PARALLEL:-6}"
PARTITION="${GATE_VIZ_PARTITION:-mweber_gpu}"
MEM="${GATE_VIZ_MEM:-64GB}"
TIME="${GATE_VIZ_TIME:-48:00:00}"
SEED="${GATE_VIZ_SEED:-2}"
CKPT_PERIOD="${GATE_VIZ_CKPT_PERIOD:-50}"

export_vars="ALL,ENV_NAME=gnnplus"
export_vars+=",GATE_VIZ_SEED=${SEED}"
export_vars+=",GATE_VIZ_CKPT_PERIOD=${CKPT_PERIOD}"
export_vars+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_vars+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR:-}"

job_id="$(
    sbatch --parsable \
        --job-name=tu_sigma_gate \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/tu_sigma_gate_%A_%a.log" \
        --export="${export_vars}" \
        bash_interface/cluster/run_tu_sigma_gate_viz.sh
)"

cat <<EOF

=== TU SiGMA gate-viz submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (mutag enzymes proteins dd nci1 triangles)
  Seed:          ${SEED}
  Ckpt period:   ${CKPT_PERIOD}
  Out pattern:   \$GNNPLUS_OUT_DIR/gate_viz_<ds>_sigma_powerful_seed${SEED}
  Logs:          logs_gnnplus/tu_sigma_gate_${job_id}_<TASK>.log
  Next:          bash bash_interface/cluster/submit_dump_tu_sigma_gates.sh
  Paste JOBID into Paper_heterogeneity.md + CLUSTER_LAUNCHES.md

EOF
