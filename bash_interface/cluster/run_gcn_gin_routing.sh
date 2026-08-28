#!/usr/bin/env bash
# =============================================================================
# GCN/GIN routing synthetic benchmark — SLURM worker
#
# Set GCN_GIN_ROUTING_TRACK=toy|sigma (default: toy).
# 8 models × 2 LRs × 5 seeds = 80 tasks per track.
#
# Submit:
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh toy
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh sigma
#   bash bash_interface/cluster/submit_gcn_gin_routing.sh both
# =============================================================================

#SBATCH --job-name=gcn_gin_route
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
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

track="${GCN_GIN_ROUTING_TRACK:-toy}"
task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${GCN_GIN_ROUTING_NUM_SEEDS:-5}"
num_lrs="${GCN_GIN_ROUTING_NUM_LRS:-2}"

models=(
  "a0g2_gated|configs/synthetic/gcn_gin_routing_${track}_a0g2_gated.yaml|paper_gcn_gin_routing_${track}_a0g2_gated"
  "a0g2_ungated|configs/synthetic/gcn_gin_routing_${track}_a0g2_ungated.yaml|paper_gcn_gin_routing_${track}_a0g2_ungated"
  "a0g1_gcn|configs/synthetic/gcn_gin_routing_${track}_a0g1_gcn.yaml|paper_gcn_gin_routing_${track}_a0g1_gcn"
  "a0g1_gin|configs/synthetic/gcn_gin_routing_${track}_a0g1_gin.yaml|paper_gcn_gin_routing_${track}_a0g1_gin"
)

num_models=${#models[@]}
num_tasks=$((num_models * num_lrs * num_seeds))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks}) track=${track}"
  exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
lr_idx=$((rest % num_lrs))
model_idx=$((rest / num_lrs))

case "${lr_idx}" in
  0) base_lr="0.001"; lr_tag="lr001" ;;
  1) base_lr="0.01"; lr_tag="lr01" ;;
  *) log_message "bad lr_idx=${lr_idx}"; exit 1 ;;
esac

IFS='|' read -r model_tag cfg wandb_group_base <<< "${models[$model_idx]}"
wandb_group="${wandb_group_base}_${lr_tag}"

if [ ! -f "${cfg}" ]; then
  log_message "Config missing: ${cfg} — run: python scripts/synthetic/generate_gcn_gin_routing_configs.py"
  exit 1
fi

dataset_root="${GNNPLUS_DATASET_DIR:-${REPO_ROOT}/results/gcn_gin_routing/data}/GcnGinRouting"
if [ ! -f "${dataset_root}/processed/train.pt" ]; then
  log_message "Dataset missing at ${dataset_root} — run:"
  log_message "  python scripts/synthetic/generate_gcn_gin_routing_dataset.py --root \"\${GNNPLUS_DATASET_DIR}/GcnGinRouting\""
  exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="gcn_gin_routing_synthetic,${track},${model_tag},${lr_tag},seed${seed}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
  run_dir="${GNNPLUS_OUT_DIR}/gcn_gin_routing/${track}/${model_tag}_${lr_tag}_seed${seed}"
else
  run_dir="results/gcn_gin_routing/${track}/${model_tag}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "gcn_gin_routing track=${track} ${task_id}/${num_tasks}: model=${model_tag} lr=${base_lr} seed=${seed}"
log_message "cfg=${cfg}"
log_message "run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
track=${track}
model=${model_tag}
cfg=${cfg}
seed=${seed}
lr=${base_lr}
lr_tag=${lr_tag}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
  out_dir "${run_dir}"
  optim.base_lr "${base_lr}"
  dataset.dir "${GNNPLUS_DATASET_DIR:-${REPO_ROOT}/results/gcn_gin_routing/data}"
  train.enable_ckpt True
  train.ckpt_best True
  train.ckpt_clean True
)

export WANDB_EXTRA_TAGS="${wandb_tags}"

python main.py \
  --cfg "${cfg}" \
  --repeat 1 \
  seed "${seed}" \
  wandb.use True \
  wandb.entity weber-geoml-harvard-university \
  wandb.project GNNPlus \
  wandb.group "${wandb_group}" \
  wandb.name "${wandb_name}" \
  "${extra_args[@]}"

log_message "Task ${task_id} complete."
