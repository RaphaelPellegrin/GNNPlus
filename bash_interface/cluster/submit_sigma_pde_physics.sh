#!/usr/bin/env bash
# Submit SiGMA Physics-Attention PDE suite (Transolver paper datasets).
#
# Prerequisites (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus && git pull
#
# Full suite (8 tasks, max 2 GPUs):
#   bash bash_interface/cluster/submit_sigma_pde_physics.sh
#
# Standard-6 only (until AirfRANS / ShapeNetCar preprocessed data is ready):
#   PDE_ARRAY=0-5 bash bash_interface/cluster/submit_sigma_pde_physics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus
chmod +x bash_interface/cluster/run_sigma_pde_physics.sh

ARRAY_SPEC="${PDE_ARRAY:-0-7}"
PARALLEL="${PDE_PARALLEL:-2}"
PARTITION="${PDE_PARTITION:-mweber_gpu}"
NICE="${PDE_NICE:-10000}"
MEM="${PDE_MEM:-128GB}"
TIME="${PDE_TIME:-48:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_sigma_pde_physics] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

export_list="ALL,ENV_NAME=gnnplus"
export_list+=",PDE_SEED=${PDE_SEED:-0}"
export_list+=",GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-}"
export_list+=",GNNPLUS_OUT_DIR=${GNNPLUS_OUT_DIR}"
if [ -n "${PDE_MAX_EPOCH:-}" ]; then
    export_list+=",PDE_MAX_EPOCH=${PDE_MAX_EPOCH}"
fi
if [ -n "${PDE_BATCH:-}" ]; then
    export_list+=",PDE_BATCH=${PDE_BATCH}"
fi

sbatch_args=(
    --parsable
    --job-name=sigma_pde_phys
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_pde_phys_%A_%a.log"
    --export="${export_list}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_pde_physics.sh
)"

cat <<EOF

=== SiGMA Physics-Attn PDE suite submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (max ${PARALLEL} GPUs)
  Datasets:      elasticity plasticity airfoil pipe darcy ns airfrans shapenet_car
  Outs:          \$GNNPLUS_OUT_DIR/sigma_pde_physics/
  W&B project:   GNNPlus_PDE_Physics
  Logs:          logs_gnnplus/sigma_pde_phys_${job_id}_<TASK>.log
  Tracker:       Paper_sigma_physics_pde.md

EOF
