#!/usr/bin/env bash
# Submit Xu-recipe SiGMA hetero a2g4 (MUTAG + ENZYMES) × 5 seeds with ckpt + gate dump.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_heterogeneity_xu_sigma_a2g4_ckpt.sh
#
# Smoke (MUTAG seed 0 only):
#   XU_SIGMA_ARRAY=1 XU_SIGMA_NUM_TASKS=1 \
#     bash bash_interface/cluster/submit_heterogeneity_xu_sigma_a2g4_ckpt.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${XU_SIGMA_NUM_SEEDS:-5}"
NUM_DATASETS="${XU_SIGMA_NUM_DATASETS:-2}"
NUM_TASKS="${XU_SIGMA_NUM_TASKS:-$((NUM_DATASETS * NUM_SEEDS))}"
ARRAY_SPEC="${XU_SIGMA_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${XU_SIGMA_PARALLEL:-5}"
PARTITION="${XU_SIGMA_PARTITION:-mweber_gpu}"
NICE="${XU_SIGMA_NICE:-10000}"
MEM="${XU_SIGMA_MEM:-32GB}"
TIME="${XU_SIGMA_TIME:-24:00:00}"

export ENV_NAME=gnnplus
export XU_SIGMA_NUM_SEEDS="${NUM_SEEDS}"
export XU_SIGMA_NUM_TASKS="${NUM_TASKS}"
export XU_SIGMA_GATE_DUMP="${XU_SIGMA_GATE_DUMP:-1}"
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    export GNNPLUS_DATASET_DIR
fi
if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_xu_sigma_a2g4] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
export GNNPLUS_OUT_DIR

sbatch_args=(
    --parsable
    --job-name=xu_sigma_a2g4
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/xu_sigma_a2g4_%A_%a.log"
    --export=ALL
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_heterogeneity_xu_sigma_a2g4_ckpt.sh
)"

cat <<EOF

=== Xu SiGMA a2g4 ckpt+gates submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (2 datasets × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/xu_sigma_a2g4_${job_id}_<TASK>.log
  Outs:          \$GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/<ds>_SiGMA_hetero_xu_seed<s>/
                 ├── ckpt/
                 ├── config_used.yaml
                 └── gate_values_per_graph.pt
  Configs:       configs/heterogeneity/powerful_gnns/{mutag,enzymes}-sigma-a2g4-ckpt.yaml
  W&B groups:    xu_sigma_a2g4_{mutag,enzymes}

  Task map (seed fastest):
    1–5  mutag seeds 0–4
    6–10 enzymes seeds 0–4

  Tracker: Paper_tu_gate_hetero_bridge.md
  Paste JOBID into CLUSTER_LAUNCHES.md

EOF
