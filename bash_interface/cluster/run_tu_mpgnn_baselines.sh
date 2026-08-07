#!/usr/bin/env bash
# =============================================================================
# TU paper-table datasets — standalone MPGNN baselines (GIN / SAGE / GAT).
# Same recipe as GCN in Paper_tu_sigma_homo_hetero (L12, H64, lr=1e-3, …).
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
# Layout: 6 datasets × 3 models × 5 seeds = 90
#   task_id = ((dataset_idx * NUM_MODELS) + model_idx) * NUM_SEEDS + seed + 1
#
# Submit:
#   bash bash_interface/cluster/submit_tu_mpgnn_baselines.sh
# =============================================================================

#SBATCH --job-name=tu_mpgnn
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
num_seeds="${TU_MPGNN_NUM_SEEDS:-5}"

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
declare -A batch_for=(
    [mutag]="${TU_MPGNN_BATCH_DEFAULT:-64}"
    [enzymes]="${TU_MPGNN_BATCH_DEFAULT:-64}"
    [proteins]="${TU_MPGNN_BATCH_DEFAULT:-64}"
    [collab]="${TU_MPGNN_BATCH_COLLAB:-32}"
    [imdb_binary]="${TU_MPGNN_BATCH_IMDB:-64}"
    [reddit_binary]="${TU_MPGNN_BATCH_REDDIT:-16}"
)

models=(gin sage gat)
model_tags=(GIN SAGE GAT)

num_datasets=${#datasets[@]}
num_models=${#models[@]}
num_tasks="${TU_MPGNN_NUM_TASKS:-$((num_datasets * num_models * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
model_idx=$((rest % num_models))
dataset_idx=$((rest / num_models))

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
batch_size="${batch_for[$ds_tag]}"
layer="${models[$model_idx]}"
variant="${model_tags[$model_idx]}"
cfg="configs/tu_sigma_homo_hetero/${layer}-anchor.yaml"
base_lr="0.001"
lr_tag="lr001"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="tu_hh_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_mpgnn_baselines,${ds_tag},${variant},${lr_tag},seed${seed},L12"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU MPGNN ${task_id}/${num_tasks}: ds=${ds_name} ${variant} lr=${base_lr} batch=${batch_size} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=${variant}
variant=${variant}
layer_type=${layer}
lr=${base_lr}
lr_tag=${lr_tag}
batch_size=${batch_size}
seed=${seed}
cfg=${cfg}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    optim.base_lr "${base_lr}"
    train.batch_size "${batch_size}"
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

log_message "Training finished. ckpt listing:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/ under ${run_dir}"
log_message "Task ${task_id} complete."
