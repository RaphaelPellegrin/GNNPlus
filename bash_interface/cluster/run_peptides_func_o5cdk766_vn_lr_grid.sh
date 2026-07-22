#!/usr/bin/env bash
# =============================================================================
# Peptides-func SiGMA (o5cdk766) — VN × LR grid (10 configs × 5 seeds).
#
# Anchor: configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml
# Best paper SiGMA: a1g1 GCN, elementwise, lr≈2.083e-4, ep=900, no VN.
#
# When VN>0 we also enable pyramid readout (matches successful peptides-struct
# rholn782 VN=4 recipe). cfg 7 isolates VN=4 without pyramid.
#
# Config index (cfg_idx 0..9) → (num_vn, base_lr, pyramid?):
#   0  vn=0  lr=2.083e-4  no-pyr   (paper control)
#   1  vn=1  lr=2.083e-4  pyramid
#   2  vn=2  lr=2.083e-4  pyramid
#   3  vn=4  lr=2.083e-4  pyramid  (struct-like)
#   4  vn=8  lr=2.083e-4  pyramid
#   5  vn=4  lr=1.0e-4    pyramid
#   6  vn=4  lr=4.0e-4    pyramid
#   7  vn=4  lr=2.083e-4  no-pyr   (VN only)
#   8  vn=2  lr=4.0e-4    pyramid
#   9  vn=8  lr=1.0e-4    pyramid
#
# Layout: task_id 1..50 → cfg_idx=(task-1)//5, seed=(task-1)%5
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_o5cdk766_vn_lr_grid.sh
# =============================================================================

#SBATCH --job-name=pep_func_vn_lr
#SBATCH --ntasks=1
#SBATCH --time=192:00:00
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
num_seeds="${PEP_FUNC_VN_LR_NUM_SEEDS:-5}"
num_cfgs="${PEP_FUNC_VN_LR_NUM_CFGS:-10}"
seed_offset="${PEP_FUNC_VN_LR_SEED_OFFSET:-0}"
num_tasks="${PEP_FUNC_VN_LR_NUM_TASKS:-$((num_cfgs * num_seeds))}"
base_lr_default="0.00020830328241707908"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
cfg_idx=$((idx / num_seeds))

cfg="configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml"
if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

case "${cfg_idx}" in
    0) num_vn=0; base_lr="${base_lr_default}"; lr_tag="2p083e-4"; use_pyramid=0 ;;
    1) num_vn=1; base_lr="${base_lr_default}"; lr_tag="2p083e-4"; use_pyramid=1 ;;
    2) num_vn=2; base_lr="${base_lr_default}"; lr_tag="2p083e-4"; use_pyramid=1 ;;
    3) num_vn=4; base_lr="${base_lr_default}"; lr_tag="2p083e-4"; use_pyramid=1 ;;
    4) num_vn=8; base_lr="${base_lr_default}"; lr_tag="2p083e-4"; use_pyramid=1 ;;
    5) num_vn=4; base_lr="0.0001";             lr_tag="1e-4";    use_pyramid=1 ;;
    6) num_vn=4; base_lr="0.0004";             lr_tag="4e-4";    use_pyramid=1 ;;
    7) num_vn=4; base_lr="${base_lr_default}"; lr_tag="2p083e-4"; use_pyramid=0 ;;
    8) num_vn=2; base_lr="0.0004";             lr_tag="4e-4";    use_pyramid=1 ;;
    9) num_vn=8; base_lr="0.0001";             lr_tag="1e-4";    use_pyramid=1 ;;
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

if [ "${use_pyramid}" -eq 1 ]; then
    pyr_tag="pyr"
else
    pyr_tag="nopyr"
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="paper_sigma_peptides_func_${vn_tag}_lr${lr_tag}_${pyr_tag}"
wandb_name="pep_func_${vn_tag}_lr${lr_tag}_${pyr_tag}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="peptides_func,sigma,o5cdk766,${vn_tag},lr${lr_tag},${pyr_tag},seed${seed},vn_lr_grid,source_o5cdk766"

extra_args=(
    dataset.add_virtual_nodes "${add_vn}"
    dataset.num_virtual_nodes "${num_vn}"
    optim.base_lr "${base_lr}"
)
if [ "${use_pyramid}" -eq 1 ]; then
    extra_args+=(gnn.readout_mlp pyramid)
fi
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi

log_message "pep_func VN×LR task ${task_id}/${num_tasks}: cfg=${cfg_idx} ${vn_tag} lr=${base_lr} ${pyr_tag} seed=${seed}"
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
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
