#!/usr/bin/env bash
# =============================================================================
# Dump per-graph SiGMA gates for each TU gate-viz run_dir.
# Tasks 1–6 mirror submit_tu_sigma_gate_viz.sh datasets.
#
# Submit:
#   bash bash_interface/cluster/submit_dump_tu_sigma_gates.sh
# =============================================================================

#SBATCH --job-name=tu_gate_dump
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

task_id=${SLURM_ARRAY_TASK_ID:-1}
seed="${GATE_VIZ_SEED:-2}"
epoch="${GATE_DUMP_EPOCH:--1}"

datasets=(mutag enzymes proteins dd nci1 triangles)
num_tasks=${#datasets[@]}

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

ds="${datasets[$((task_id - 1))]}"
cfg="configs/heterogeneity/powerful_gnns/${ds}-sigma.yaml"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/gate_viz_${ds}_sigma_powerful_seed${seed}"
else
    out_dir="results/gate_viz_${ds}_sigma_powerful_seed${seed}"
fi

if [ ! -d "${out_dir}/ckpt" ]; then
    log_message "No ckpt/ under ${out_dir}"
    exit 1
fi

out_pt="${out_dir}/gate_values_per_graph.pt"
extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "Dump TU gates: ds=${ds} run_dir=${out_dir} epoch=${epoch}"
ls -lh "${out_dir}/ckpt/" | tail -n 5 || true

exec python scripts/gate_viz/dump_per_graph_gates.py \
    --run_dir "${out_dir}" \
    --epoch "${epoch}" \
    --out "${out_pt}" \
    --cfg "${cfg}" \
    seed "${seed}" \
    "${extra_args[@]}"
