#!/usr/bin/env bash
# =============================================================================
# Peptides-func MP-only follow-up on NEW best Homog_MP (a1g2 GCN×2).
#
# Table-5 style: drop the 1 attention head and replace with GCN → a0g3 GCN×3,
# still gated (elementwise), same hyperparams as Homog_MP / o5cdk766 lineage.
#
# 5 seeds (default). W&B group: paper_T5_peptides_func_HomogMP_MPonly
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_func_homog_a1g2_mp_only.sh
# =============================================================================

#SBATCH --job-name=sigma_func_a0g3
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
num_seeds="${FUNC_HOMOG_MPONLY_NUM_SEEDS:-5}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_seeds" ]; then
    log_message "task_id=${task_id} out of range (1..${num_seeds})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/peptides-func-hybrid-homog-a1g2-gcn-anchor.yaml"
# a1g2 → replace attn with GCN: a0g3 GCN,GCN,GCN (gated, same as anchor gate).
na=0
ng=3
gnn_types="GCN,GCN,GCN"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${FUNC_HOMOG_MPONLY_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_peptides_func_HomogMP_MPonly"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,MP_only,HomogMP_MPonly,peptides_func,seed${seed},a0g3,gcn,source_T6_Homog_MP"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "Peptides-func HomogMP→MP_only task ${task_id}/${num_seeds}: a${na}g${ng} types=${gnn_types} seed=${seed}"
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
    gnn.hybrid.num_attn_heads "${na}" \
    gnn.hybrid.num_gnn_heads "${ng}" \
    "gnn.hybrid.gnn_types" "${gnn_types}" \
    gnn.hybrid.log_gate_stats True \
    gnn.hybrid.identity_proj False \
    "${extra_args[@]}"
