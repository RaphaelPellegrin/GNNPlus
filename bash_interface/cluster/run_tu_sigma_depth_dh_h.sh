#!/usr/bin/env bash
# =============================================================================
# TU Tab.17/18 protocol — SiGMA hetero gated vs ungated depth × d_h × H grid.
#
# Base recipe: a2g4 hetero (GCN,GIN,SAGE,GAT), same train recipe as Tab.17/18.
# Sweep (overrides on sigma-hetero-a2g4-anchor.yaml):
#   L   ∈ {1, 2, 4, 8, 16}     (gnn.layers_mp)   — note: paper used L=12
#   d_h ∈ {1, 2, 4, 16}        (gnn.hybrid.d_h)
#   H   ∈ {64, 8}              (gnn.dim_inner)
#   gate ∈ {headwise, none}
#   lr  ∈ {0.001, 0.01} × 5 seeds
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
#
# Layout (4800 tasks), dataset-first for phased launch:
#   task_id = ((((((ds * n_L + L) * n_dh + dh) * n_H + H)
#               * n_gates + gate) * n_lrs + lr) * n_seeds + seed) + 1
#   Per dataset = 5×4×2×2×2×5 = 800 tasks
#     MUTAG 1–800 · ENZYMES 801–1600 · … · REDDIT 4001–4800
#
# W&B: tu_L{k}_dh{m}_H{h}_<ds>_{SiGMA_hetero,SiGMA_ungated}_{lr001,lr01}
# Out: $GNNPLUS_OUT_DIR/tu_sigma_depth_dh_h/...
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_depth_dh_h.sh
# =============================================================================

#SBATCH --job-name=tu_LdhH
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
num_seeds="${TU_LDHH_NUM_SEEDS:-5}"
num_lrs="${TU_LDHH_NUM_LRS:-2}"
num_gates="${TU_LDHH_NUM_GATES:-2}"
num_H="${TU_LDHH_NUM_H:-2}"
num_dh="${TU_LDHH_NUM_DH:-4}"
num_L="${TU_LDHH_NUM_L:-5}"
num_datasets="${TU_LDHH_NUM_DATASETS:-6}"
num_tasks="${TU_LDHH_NUM_TASKS:-$((num_datasets * num_L * num_dh * num_H * num_gates * num_lrs * num_seeds))}"

layers=(1 2 4 8 16)
dhs=(1 2 4 16)
Hs=(64 8)
datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
declare -A batch_for=(
    [mutag]="${TU_LDHH_BATCH_DEFAULT:-64}"
    [enzymes]="${TU_LDHH_BATCH_DEFAULT:-64}"
    [proteins]="${TU_LDHH_BATCH_DEFAULT:-64}"
    [collab]="${TU_LDHH_BATCH_COLLAB:-32}"
    [imdb_binary]="${TU_LDHH_BATCH_IMDB:-64}"
    [reddit_binary]="${TU_LDHH_BATCH_REDDIT:-16}"
)

cfg="configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml"

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
lr_idx=$((rest % num_lrs))
rest=$((rest / num_lrs))
gate_idx=$((rest % num_gates))
rest=$((rest / num_gates))
H_idx=$((rest % num_H))
rest=$((rest / num_H))
dh_idx=$((rest % num_dh))
rest=$((rest / num_dh))
L_idx=$((rest % num_L))
ds_idx=$((rest / num_L))

ds_tag="${datasets[$ds_idx]}"
ds_name="${dataset_names[$ds_idx]}"
batch_size="${batch_for[$ds_tag]}"
L="${layers[$L_idx]}"
d_h="${dhs[$dh_idx]}"
H="${Hs[$H_idx]}"

case "${lr_idx}" in
    0) base_lr="0.001"; lr_tag="lr001" ;;
    1) base_lr="0.01";  lr_tag="lr01" ;;
    *) log_message "bad lr_idx=${lr_idx}"; exit 1 ;;
esac

gate_override=()
case "${gate_idx}" in
    0)
        gate_tag="gated"
        variant="SiGMA_hetero"
        ;;
    1)
        gate_tag="ungated"
        variant="SiGMA_ungated"
        gate_override+=(gnn.hybrid.gate none)
        ;;
    *)
        log_message "bad gate_idx=${gate_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="tu_L${L}_dh${d_h}_H${H}_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_sigma_depth_dh_h,L${L},dh${d_h},H${H},${ds_tag},${variant},${lr_tag},seed${seed},a2g4"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_depth_dh_h/${ds_tag}_L${L}_dh${d_h}_H${H}_${gate_tag}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_depth_dh_h/${ds_tag}_L${L}_dh${d_h}_H${H}_${gate_tag}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU L×dh×H ${task_id}/${num_tasks}: ${ds_name} L=${L} d_h=${d_h} H=${H} ${gate_tag} lr=${base_lr} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
variant=${variant}
gate=${gate_tag}
L=${L}
d_h=${d_h}
H=${H}
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
    gnn.layers_mp "${L}"
    gnn.dim_inner "${H}"
    gnn.hybrid.d_h "${d_h}"
    out_dir "${run_dir}"
    train.enable_ckpt True
    train.ckpt_best True
    train.ckpt_clean True
    "${gate_override[@]}"
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

log_message "Task ${task_id} complete. ckpt:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/"
