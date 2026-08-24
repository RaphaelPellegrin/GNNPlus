#!/usr/bin/env bash
# =============================================================================
# SiGMA d_h-matched (TU Tab. 17/18 analog) — Tab. 3/4 over-500k SiGMA
#
# Keep paper architecture (heads, L, train recipe); shrink d_h so params land
# under ~500k and/or ~1M. VOC ≤500k also shrinks H (95→64) because d_h alone
# cannot reach 500k. ZINC skipped (main already ≤500k).
#
# Like Tab. 17/18: try two learning rates {1e-3, 1e-2} and report the better
# family (selected by val / W&B best_test_perf after the fact).
#
# Layout: 15 families × 2 LRs × 5 seeds = 150
#   task_id = (family_idx * NUM_LRS + lr_idx) * NUM_SEEDS + seed + 1
#   lr_idx 0 → 0.001 (lr001), lr_idx 1 → 0.01 (lr01)
#
# Submit:
#   bash bash_interface/cluster/submit_sigma_dh_matched.sh
# =============================================================================

#SBATCH --job-name=sigma_dh_matched
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
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
num_seeds="${SIGMA_DH_MATCHED_NUM_SEEDS:-5}"
num_lrs="${SIGMA_DH_MATCHED_NUM_LRS:-2}"

# tag|cfg_relpath|wandb_group_base
families=(
  # PATTERN / CLUSTER (ratio analogs; already under 500k / 1M)
  "pattern_dh16|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh16.yaml|paper_sigma_dh_matched_pattern_dh16"
  "pattern_dh4|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|paper_sigma_dh_matched_pattern_dh4"
  "cluster_dh36|configs/gated_hybrid/dh_matched/cluster-a1g1-dh36.yaml|paper_sigma_dh_matched_cluster_dh36"
  "cluster_dh24|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|paper_sigma_dh_matched_cluster_dh24"
  # Tab. 3 remainder
  "mnist_dh37|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|paper_sigma_dh_matched_mnist_dh37"
  "cifar_dh20|configs/gated_hybrid/dh_matched/cifar10-a8g4-dh20.yaml|paper_sigma_dh_matched_cifar_dh20"
  "cifar_dh34|configs/gated_hybrid/dh_matched/cifar10-a8g4-dh34.yaml|paper_sigma_dh_matched_cifar_dh34"
  # Tab. 4
  "pepfunc_dh23|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|paper_sigma_dh_matched_pepfunc_dh23"
  "pepfunc_dh75|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh75.yaml|paper_sigma_dh_matched_pepfunc_dh75"
  "pepstruct_dh43|configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh43.yaml|paper_sigma_dh_matched_pepstruct_dh43"
  "pepstruct_dh92|configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh92.yaml|paper_sigma_dh_matched_pepstruct_dh92"
  "voc_dh15|configs/gated_hybrid/dh_matched/voc-a2g2-dh15.yaml|paper_sigma_dh_matched_voc_dh15"
  "voc_h64_dh12|configs/gated_hybrid/dh_matched/voc-a2g2-h64-dh12.yaml|paper_sigma_dh_matched_voc_h64_dh12"
  "coco_dh34|configs/gated_hybrid/dh_matched/coco-a1g1-dh34.yaml|paper_sigma_dh_matched_coco_dh34"
  "malnet_dh57|configs/gated_hybrid/dh_matched/malnet-a1g1-dh57.yaml|paper_sigma_dh_matched_malnet_dh57"
)

num_families=${#families[@]}
num_tasks=$((num_families * num_lrs * num_seeds))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks})"
  exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
lr_idx=$((rest % num_lrs))
family_idx=$((rest / num_lrs))

case "${lr_idx}" in
  0)
    base_lr="0.001"
    lr_tag="lr001"
    ;;
  1)
    base_lr="0.01"
    lr_tag="lr01"
    ;;
  *)
    log_message "bad lr_idx=${lr_idx}"
    exit 1
    ;;
esac

IFS='|' read -r fam_tag cfg wandb_group_base <<< "${families[$family_idx]}"
wandb_group="${wandb_group_base}_${lr_tag}"

if [ ! -f "${cfg}" ]; then
  log_message "Config not found: ${cfg}"
  exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="sigma_dh_matched,${fam_tag},${lr_tag},seed${seed}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
  run_dir="${GNNPLUS_OUT_DIR}/sigma_dh_matched/${fam_tag}_${lr_tag}_seed${seed}"
else
  run_dir="results/sigma_dh_matched/${fam_tag}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "d_h-matched ${task_id}/${num_tasks}: fam=${fam_tag} lr=${base_lr} (${lr_tag}) seed=${seed} cfg=${cfg}"
log_message "run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
family=${fam_tag}
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
  train.enable_ckpt True
  train.ckpt_best True
  train.ckpt_clean True
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
  extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

export WANDB_EXTRA_TAGS="${wandb_tags}"

log_message "mem_hint=128GB (sbatch --mem controls actual allocation)"

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
