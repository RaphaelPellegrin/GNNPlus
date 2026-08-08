#!/usr/bin/env bash
# =============================================================================
# CIFAR10 budget re-fit — recount H/d_h so params actually ≤500k / 1M / 2M.
#
# Prior paper_budget_cifar10_b{500k,1m,2m} overshot (541k / 1.18M / 2.24M).
# New W&B groups: paper_budget_cifar10_*_fit
#
# Layout: 3 families × 5 seeds = 15
#   task_id = family_idx * NUM_SEEDS + seed + 1
#
# Submit:
#   bash bash_interface/cluster/submit_cifar_budget_fit.sh
# =============================================================================

#SBATCH --job-name=cifar_budget_fit
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
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
num_seeds="${CIFAR_BUDGET_FIT_NUM_SEEDS:-5}"

# tag|cfg|wandb_group
families=(
  "cifar10_b500k_fit|configs/gated_hybrid/budget/cifar10-b500k-a1g1.yaml|paper_budget_cifar10_b500k_fit"
  "cifar10_b1m_fit|configs/gated_hybrid/budget/cifar10-b1m-a1g1.yaml|paper_budget_cifar10_b1m_fit"
  "cifar10_b2m_fit|configs/gated_hybrid/budget/cifar10-b2m-a1g2.yaml|paper_budget_cifar10_b2m_fit"
)

num_families=${#families[@]}
num_tasks=$((num_families * num_seeds))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks})"
  exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
family_idx=$((idx / num_seeds))

IFS='|' read -r fam_tag cfg wandb_group <<< "${families[$family_idx]}"

if [ ! -f "${cfg}" ]; then
  log_message "Config not found: ${cfg}"
  exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="sigma_budget,cifar_budget_fit,${fam_tag},seed${seed},params_fit"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
  run_dir="${GNNPLUS_OUT_DIR}/sigma_budget/${fam_tag}_seed${seed}"
else
  run_dir="results/sigma_budget/${fam_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "CIFAR budget fit ${task_id}/${num_tasks}: fam=${fam_tag} seed=${seed} cfg=${cfg}"
log_message "run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
family=${fam_tag}
cfg=${cfg}
seed=${seed}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
  out_dir "${run_dir}"
  train.enable_ckpt True
  train.ckpt_best True
  train.ckpt_clean True
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
  extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

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

log_message "Task ${task_id} complete. Listing ckpt:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/"
