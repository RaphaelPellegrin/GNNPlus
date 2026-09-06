#!/usr/bin/env bash
# =============================================================================
# TU Tables 17/18 — SiGMA (hetero) UNGATED ablation.
#
# Same recipe as gated SiGMA hetero, with gnn.hybrid.gate=none (Table 5
# SiGMA_ungated semantics: attn + MP both ungated).
#
# Tables:
#   0  Tab.17  d_h=16  configs/.../sigma-hetero-a2g4-anchor.yaml
#   1  Tab.18  d_h=4   configs/.../sigma-hetero-a2g4-matched-anchor.yaml
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
# Variants (per table×dataset):
#   0  SiGMA_ungated  lr=0.001
#   1  SiGMA_ungated  lr=0.01
#
# Layout: 2 tables × 6 ds × 2 LR × 5 seeds = 120
#   task_id = ((table_idx * NUM_DATASETS + ds_idx) * NUM_VARIANTS + var_idx)
#             * NUM_SEEDS + seed + 1
#
# W&B:
#   Tab.17 → tu_hh_<ds>_SiGMA_ungated_{lr001,lr01}
#   Tab.18 → tu_1x_<ds>_SiGMA_ungated_{lr001,lr01}
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_ungated.sh
# =============================================================================

#SBATCH --job-name=tu_sigma_ung
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
num_seeds="${TU_UNGATED_NUM_SEEDS:-5}"
num_variants="${TU_UNGATED_NUM_VARIANTS:-2}"
num_datasets="${TU_UNGATED_NUM_DATASETS:-6}"
num_tables="${TU_UNGATED_NUM_TABLES:-2}"
num_tasks="${TU_UNGATED_NUM_TASKS:-$((num_tables * num_datasets * num_variants * num_seeds))}"

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
declare -A batch_for=(
    [mutag]="${TU_UNGATED_BATCH_DEFAULT:-64}"
    [enzymes]="${TU_UNGATED_BATCH_DEFAULT:-64}"
    [proteins]="${TU_UNGATED_BATCH_DEFAULT:-64}"
    [collab]="${TU_UNGATED_BATCH_COLLAB:-32}"
    [imdb_binary]="${TU_UNGATED_BATCH_IMDB:-64}"
    [reddit_binary]="${TU_UNGATED_BATCH_REDDIT:-16}"
)

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
rest=$((rest / num_variants))
dataset_idx=$((rest % num_datasets))
table_idx=$((rest / num_datasets))

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
batch_size="${batch_for[$ds_tag]}"

cfg_dir="configs/tu_sigma_homo_hetero"
case "${table_idx}" in
    0)
        table_tag="t17"
        wandb_prefix="tu_hh"
        out_subdir="tu_sigma_homo_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        arch_tag="a2g4_dh16_ungated"
        dh_tag="dh16"
        ;;
    1)
        table_tag="t18"
        wandb_prefix="tu_1x"
        out_subdir="tu_sigma_1x_gcn"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        arch_tag="a2g4_dh4_ungated"
        dh_tag="dh4"
        ;;
    *)
        log_message "bad table_idx=${table_idx}"
        exit 1
        ;;
esac

case "${variant_idx}" in
    0)
        base_lr="0.001"
        lr_tag="lr001"
        ;;
    1)
        base_lr="0.01"
        lr_tag="lr01"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

family="SiGMA_ungated"
variant="SiGMA_ungated"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="${wandb_prefix}_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_sigma_ungated,${table_tag},${ds_tag},${variant},${lr_tag},seed${seed},${arch_tag},${dh_tag}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/${out_subdir}/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/${out_subdir}/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU ungated ${task_id}/${num_tasks}: table=${table_tag} ds=${ds_name} lr=${base_lr} batch=${batch_size} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=${family}
variant=${variant}
table=${table_tag}
lr=${base_lr}
lr_tag=${lr_tag}
batch_size=${batch_size}
seed=${seed}
cfg=${cfg}
arch_tag=${arch_tag}
gate=none
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
wandb_name=${wandb_name}
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
    gnn.hybrid.gate none
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
log_message "Skipping gate dump (ungated; gate=none)."
log_message "Task ${task_id} complete."
