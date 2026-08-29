#!/usr/bin/env bash
# =============================================================================
# Dump per-node MP gate γ for GCN/GIN routing gated runs (best checkpoint).
#
# One array task per gated run: 2 tracks × 2 LRs × 5 seeds = 20 tasks.
#
# Writes per run_dir:
#   gate_values_per_node.pt
#   gate_graph_summary.csv
#
# Submit (login node):
#   bash bash_interface/cluster/submit_dump_gcn_gin_routing_node_gates.sh
# =============================================================================

#SBATCH --job-name=gcn_gin_gdump
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
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
num_seeds="${GCN_GIN_GATE_DUMP_NUM_SEEDS:-5}"
num_lrs="${GCN_GIN_GATE_DUMP_NUM_LRS:-2}"
num_tracks="${GCN_GIN_GATE_DUMP_NUM_TRACKS:-2}"
num_tasks=$((num_tracks * num_lrs * num_seeds))

tracks=(toy sigma)
lrs=(lr001 lr01)

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks})"
  exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
lr_idx=$((rest % num_lrs))
track_idx=$((rest / num_lrs))

track="${tracks[$track_idx]}"
lr_tag="${lrs[$lr_idx]}"
model_tag="a0g2_gated"

results_root="${GCN_GIN_GATE_DUMP_RESULTS_ROOT:-${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/gcn_gin_routing}"
dataset_dir="${GCN_GIN_GATE_DUMP_DATASET_DIR:-${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets}}"
run_dir="${results_root}/${track}/${model_tag}_${lr_tag}_seed${seed}"

if [ ! -d "${run_dir}/ckpt" ]; then
  log_message "No ckpt/ under ${run_dir}"
  exit 1
fi

splits_raw="${GCN_GIN_GATE_DUMP_SPLITS:-train,val,test}"
splits="${splits_raw//;/,}"
skip_flag=()
if [ "${GCN_GIN_GATE_DUMP_SKIP_EXISTING:-0}" = "1" ]; then
  skip_flag+=(--skip-existing)
fi

log_message "Dump node gates: track=${track} model=${model_tag} lr=${lr_tag} seed=${seed}"
log_message "run_dir=${run_dir}"

exec python scripts/synthetic/dump_gcn_gin_routing_node_gates.py \
  --run-dir "${run_dir}" \
  --dataset-dir "${dataset_dir}" \
  --splits "${splits}" \
  "${skip_flag[@]}"
