#!/usr/bin/env bash
# =============================================================================
# SiGMA d_h-matched (TU Tab. 17/18 analog) — Tab. 3/4 over-500k SiGMA
#
# Tiered via SIGMA_DH_MATCHED_TIER:
#   fast  — PATTERN, CLUSTER, MNIST, Pep-func/struct, MalNet  (default)
#   slow  — CIFAR10, VOC
#   coco  — COCO-SP only
#
# Like Tab. 17/18: LR ∈ {1e-3, 1e-2} × 5 seeds; report better LR after.
#
# Layout per tier:
#   task_id = (family_idx * NUM_LRS + lr_idx) * NUM_SEEDS + seed + 1
#
# Prefer the tier submit scripts:
#   bash bash_interface/cluster/submit_sigma_dh_matched_fast.sh
#   bash bash_interface/cluster/submit_sigma_dh_matched_slow.sh
#   bash bash_interface/cluster/submit_sigma_dh_matched_coco.sh
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
tier="${SIGMA_DH_MATCHED_TIER:-fast}"

# tag|cfg_relpath|wandb_group_base
case "${tier}" in
  fast)
    families=(
      "pattern_dh16|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh16.yaml|paper_sigma_dh_matched_pattern_dh16"
      "pattern_dh4|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|paper_sigma_dh_matched_pattern_dh4"
      "cluster_dh36|configs/gated_hybrid/dh_matched/cluster-a1g1-dh36.yaml|paper_sigma_dh_matched_cluster_dh36"
      "cluster_dh24|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|paper_sigma_dh_matched_cluster_dh24"
      "mnist_dh37|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|paper_sigma_dh_matched_mnist_dh37"
      "pepfunc_dh23|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|paper_sigma_dh_matched_pepfunc_dh23"
      "pepfunc_dh75|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh75.yaml|paper_sigma_dh_matched_pepfunc_dh75"
      "pepstruct_dh43|configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh43.yaml|paper_sigma_dh_matched_pepstruct_dh43"
      "pepstruct_dh92|configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh92.yaml|paper_sigma_dh_matched_pepstruct_dh92"
      "malnet_dh57|configs/gated_hybrid/dh_matched/malnet-a1g1-dh57.yaml|paper_sigma_dh_matched_malnet_dh57"
    )
    ;;
  slow)
    families=(
      "cifar_dh20|configs/gated_hybrid/dh_matched/cifar10-a8g4-dh20.yaml|paper_sigma_dh_matched_cifar_dh20"
      "cifar_dh34|configs/gated_hybrid/dh_matched/cifar10-a8g4-dh34.yaml|paper_sigma_dh_matched_cifar_dh34"
      "voc_dh15|configs/gated_hybrid/dh_matched/voc-a2g2-dh15.yaml|paper_sigma_dh_matched_voc_dh15"
      "voc_h64_dh12|configs/gated_hybrid/dh_matched/voc-a2g2-h64-dh12.yaml|paper_sigma_dh_matched_voc_h64_dh12"
    )
    ;;
  coco)
    families=(
      "coco_dh34|configs/gated_hybrid/dh_matched/coco-a1g1-dh34.yaml|paper_sigma_dh_matched_coco_dh34"
    )
    ;;
  *)
    log_message "Unknown SIGMA_DH_MATCHED_TIER=${tier} (expected fast|slow|coco)"
    exit 1
    ;;
esac

num_families=${#families[@]}
num_tasks=$((num_families * num_lrs * num_seeds))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks}) for tier=${tier}"
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
wandb_tags="sigma_dh_matched,${tier},${fam_tag},${lr_tag},seed${seed}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
  run_dir="${GNNPLUS_OUT_DIR}/sigma_dh_matched/${fam_tag}_${lr_tag}_seed${seed}"
else
  run_dir="results/sigma_dh_matched/${fam_tag}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "d_h-matched tier=${tier} ${task_id}/${num_tasks}: fam=${fam_tag} lr=${base_lr} (${lr_tag}) seed=${seed}"
log_message "cfg=${cfg}"
log_message "run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
tier=${tier}
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
