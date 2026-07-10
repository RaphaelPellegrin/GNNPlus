#!/usr/bin/env bash
# =============================================================================
# Peptides-struct hybrid v2: y3ygn39y best hybrid + UniGCN (a2g2, L8, ep300).
#
# Config: configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a2g2-gine-unigcn-v2.yaml
# Tasks 1–N → seeds 0..(N-1)
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_struct_unigcn_hybrid_v2.sh
# =============================================================================

#SBATCH --job-name=ps_unigcn_v2
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=64GB
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
num_tasks="${PS_UNIGCN_V2_NUM_TASKS:-3}"
num_seeds="${PS_UNIGCN_V2_NUM_SEEDS:-3}"
wandb_group="${PS_UNIGCN_V2_WANDB_GROUP:-peptides_struct_unigcn_hybrid_v2}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

seed=$((task_id - 1))
cfg="configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a2g2-gine-unigcn-v2.yaml"
wandb_tags="unigcn,hybrid_gnn,peptides_struct,hybrid_a2g2,gine,unigcn_hybrid_v2,anchor_y3ygn39y,seed${seed}"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="peptides_struct_unigcn_hybrid_v2_seed${seed}_job${job_tag}_${task_id}"

extra_args=(
    model.type hybrid_gnn
    "gnn.hybrid.gnn_types" "GINE,UNIGCN"
    gnn.hybrid.num_attn_heads 2
    gnn.hybrid.num_gnn_heads 2
    gnn.hybrid.identity_proj False
    gnn.hybrid.residual True
    gnn.hybrid.log_gate_stats True
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "Peptides-struct UniGCN hybrid v2 task ${task_id}/${num_tasks}: seed=${seed} cfg=${cfg}"
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
    "${extra_args[@]}"
