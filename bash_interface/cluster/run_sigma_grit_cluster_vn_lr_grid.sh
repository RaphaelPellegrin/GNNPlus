#!/usr/bin/env bash
# =============================================================================
# SiGMA + GRIT attn on CLUSTER — small VN × LR grid (10 configs × 5 seeds).
#
# Anchor: configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml
# Baseline (prior): vn=0/4, lr=1.492e-3 → ~79.1% accuracy-SBM.
#
# Config index (cfg_idx 0..9) → (num_virtual_nodes, base_lr):
#   0  vn=0  lr=1.492e-3   (no-VN control @ base lr)
#   1  vn=1  lr=1.492e-3
#   2  vn=2  lr=1.492e-3
#   3  vn=4  lr=1.492e-3   (prior VN=4 @ base lr)
#   4  vn=8  lr=1.492e-3
#   5  vn=4  lr=5.0e-4
#   6  vn=4  lr=1.0e-3
#   7  vn=4  lr=3.0e-3
#   8  vn=8  lr=1.0e-3
#   9  vn=2  lr=3.0e-3
#
# Layout: task_id 1..50 → cfg_idx = (task-1)//5, seed = (task-1)%5
#
# Submit:
#   bash bash_interface/cluster/submit_sigma_grit_cluster_vn_lr_grid.sh
# =============================================================================

#SBATCH --job-name=sigma_grit_vn_lr
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
num_seeds="${SIGMA_GRIT_VN_LR_NUM_SEEDS:-5}"
num_cfgs="${SIGMA_GRIT_VN_LR_NUM_CFGS:-10}"
seed_offset="${SIGMA_GRIT_VN_LR_SEED_OFFSET:-0}"
num_tasks="${SIGMA_GRIT_VN_LR_NUM_TASKS:-$((num_cfgs * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
cfg_idx=$((idx / num_seeds))

cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml"
if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

# (num_vn, base_lr, lr_tag for W&B)
case "${cfg_idx}" in
    0) num_vn=0; base_lr="0.001492"; lr_tag="1p492e-3" ;;
    1) num_vn=1; base_lr="0.001492"; lr_tag="1p492e-3" ;;
    2) num_vn=2; base_lr="0.001492"; lr_tag="1p492e-3" ;;
    3) num_vn=4; base_lr="0.001492"; lr_tag="1p492e-3" ;;
    4) num_vn=8; base_lr="0.001492"; lr_tag="1p492e-3" ;;
    5) num_vn=4; base_lr="0.0005";   lr_tag="5e-4" ;;
    6) num_vn=4; base_lr="0.001";    lr_tag="1e-3" ;;
    7) num_vn=4; base_lr="0.003";    lr_tag="3e-3" ;;
    8) num_vn=8; base_lr="0.001";    lr_tag="1e-3" ;;
    9) num_vn=2; base_lr="0.003";    lr_tag="3e-3" ;;
    *)
        log_message "bad cfg_idx=${cfg_idx}"
        exit 1
        ;;
esac

if [ "${num_vn}" -gt 0 ]; then
    add_vn="True"
    vn_tag="vn${num_vn}"
else
    add_vn="False"
    vn_tag="novn"
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="paper_sigma_grit_cluster_${vn_tag}_lr${lr_tag}"
wandb_name="sigma_grit_cluster_${vn_tag}_lr${lr_tag}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="sigma_grit_attn,attn_type_grit,grit_attn,cluster,${vn_tag},lr${lr_tag},seed${seed},vn_lr_grid,source_ht9bntg2"

extra_args=(
    dataset.add_virtual_nodes "${add_vn}"
    dataset.num_virtual_nodes "${num_vn}"
    optim.base_lr "${base_lr}"
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi

log_message "sigma_grit VN×LR task ${task_id}/${num_tasks}: cfg=${cfg_idx} ${vn_tag} lr=${base_lr} seed=${seed}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

export WANDB_EXTRA_TAGS="${wandb_tags}"

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    model.type hybrid_gnn \
    gnn.hybrid.attn_type grit \
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
