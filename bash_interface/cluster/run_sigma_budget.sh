#!/usr/bin/env bash
# =============================================================================
# SiGMA baby / tiny budget campaign — fill ≤500k / ≤1M / ≤2M cells.
#
# Only launches (dataset, budget) where main Table III/IV SiGMA exceeds budget.
# Skips (reuse existing): ZINC; MNIST/COCO/MalNet ≥1M; PATTERN/CLUSTER ≥2M;
#   Pep-func (zc371e1n); Pep-struct ≥1M (rholn782).
#
# Layout: 14 families × 5 seeds = 70
#   task_id = family_idx * NUM_SEEDS + seed + 1
#
# Submit:
#   bash bash_interface/cluster/submit_sigma_budget.sh
# =============================================================================

#SBATCH --job-name=sigma_budget
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
num_seeds="${SIGMA_BUDGET_NUM_SEEDS:-5}"

# tag|cfg_relpath|wandb_group_suffix
families=(
  "mnist_b500k|configs/gated_hybrid/budget/mnist-b500k-a1g1.yaml|paper_budget_mnist_b500k"
  "cifar10_b500k|configs/gated_hybrid/budget/cifar10-b500k-a1g1.yaml|paper_budget_cifar10_b500k"
  "cifar10_b1m|configs/gated_hybrid/budget/cifar10-b1m-a1g1.yaml|paper_budget_cifar10_b1m"
  "cifar10_b2m|configs/gated_hybrid/budget/cifar10-b2m-a1g2.yaml|paper_budget_cifar10_b2m"
  "pattern_b500k|configs/gated_hybrid/budget/pattern-b500k-a1g1-grit.yaml|paper_budget_pattern_b500k"
  "pattern_b1m|configs/gated_hybrid/budget/pattern-b1m-a1g1-grit.yaml|paper_budget_pattern_b1m"
  "cluster_b500k|configs/gated_hybrid/budget/cluster-b500k-a1g1.yaml|paper_budget_cluster_b500k"
  "cluster_b1m|configs/gated_hybrid/budget/cluster-b1m-a1g1.yaml|paper_budget_cluster_b1m"
  "peptides_struct_b500k|configs/gated_hybrid/budget/peptides-struct-b500k-a1g1.yaml|paper_budget_peptides_struct_b500k"
  "voc_b500k|configs/gated_hybrid/budget/voc-b500k-a1g1.yaml|paper_budget_voc_b500k"
  "voc_b1m|configs/gated_hybrid/budget/voc-b1m-a1g1.yaml|paper_budget_voc_b1m"
  "voc_b2m|configs/gated_hybrid/budget/voc-b2m-a1g1.yaml|paper_budget_voc_b2m"
  "coco_b500k|configs/gated_hybrid/budget/coco-b500k-a1g1.yaml|paper_budget_coco_b500k"
  "malnet_b500k|configs/gated_hybrid/budget/malnet-b500k-a1g1.yaml|paper_budget_malnet_b500k"
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
wandb_tags="sigma_budget,${fam_tag},seed${seed}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
  run_dir="${GNNPLUS_OUT_DIR}/sigma_budget/${fam_tag}_seed${seed}"
else
  run_dir="results/sigma_budget/${fam_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "Budget ${task_id}/${num_tasks}: fam=${fam_tag} seed=${seed} cfg=${cfg}"
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
