#!/usr/bin/env bash
# =============================================================================
# Peptides-struct UniGCN: (A) custom_gnn baseline  (B) hybrid a1g2 GINE+UNIGCN
# from W&B run y3ygn39y / 63avcc5m.
#
# Task layout (default 3 seeds each → 6 tasks):
#   1–3  custom_gnn unitarygcn   seeds 0–2
#   4–6  hybrid_gnn a1g2 GINE,UNIGCN  seeds 0–2
#
# Submit:
#   bash bash_interface/cluster/submit_peptides_struct_unigcn_baseline_vs_hybrid.sh
# =============================================================================

#SBATCH --job-name=ps_unigcn
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
num_seeds="${PS_UNIGCN_NUM_SEEDS:-3}"
num_variants=2
num_tasks=$((num_variants * num_seeds))
wandb_group="${PS_UNIGCN_WANDB_GROUP:-peptides_struct_unigcn_baseline_vs_hybrid}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
variant_idx=$((idx / num_seeds))

case "${variant_idx}" in
    0)
        variant="custom_unigcn"
        cfg="configs/gcn/peptides-struct-unigcn.yaml"
        wandb_tags="unigcn,custom_gnn,peptides_struct,baseline,seed${seed}"
        ;;
    1)
        variant="hybrid_a1g2_gine_unigcn"
        cfg="configs/gated_hybrid/peptides-struct-hybrid-y3ygn39y-a1g2-gine-unigcn.yaml"
        wandb_tags="unigcn,hybrid_gnn,peptides_struct,hybrid_a1g2,gine,anchor_y3ygn39y,seed${seed}"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="peptides_struct_${variant}_seed${seed}_job${job_tag}_${task_id}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ "${variant_idx}" -eq 1 ]; then
    extra_args+=(
        model.type hybrid_gnn
        "gnn.hybrid.gnn_types" "GINE,UNIGCN"
        gnn.hybrid.num_attn_heads 1
        gnn.hybrid.num_gnn_heads 2
        gnn.hybrid.log_gate_stats True
    )
fi

log_message "Peptides-struct UniGCN task ${task_id}/${num_tasks}: variant=${variant} seed=${seed} cfg=${cfg}"
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
