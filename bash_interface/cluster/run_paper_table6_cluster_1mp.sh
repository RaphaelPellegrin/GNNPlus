#!/usr/bin/env bash
# =============================================================================
# Table 6/7 — CLUSTER homog/hetero MP (1-MP SiGMA → +1 head), ht9bntg2.
#
# Same recipe as LRGB a1g1 (COCO): keep 1 attn; grow MP from 1 → 2.
#   Homog_MP:          GATEDGCN,GATEDGCN gated
#   Hetero_MP:         GATEDGCN,GCN gated
#   Homog_MP_ungated / Hetero_MP_ungated: gate=none
#
# SiGMA / reuse: paper_bestmodel_v1_cluster_ht9bntg2 (78.956±0.112%)
# Skip SiGMA + Homog_MP gated by default (Homog gated = launchable via INCLUDE).
#
# Default variants: Homog_MP, Hetero_MP, Homog_MP_ungated, Hetero_MP_ungated
# (Homog_MP gated is needed for Table 6 — include it; SiGMA reused separately)
#
# W&B: paper_T6_cluster_<Variant>
# Submit:
#   bash bash_interface/cluster/submit_paper_table6_cluster_1mp.sh
# =============================================================================

#SBATCH --job-name=sigma_T6_cluster
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
num_seeds="${PAPER_T6_CLUSTER_NUM_SEEDS:-5}"
seed_offset="${PAPER_T6_CLUSTER_SEED_OFFSET:-0}"

# Always launch the four +1-MP cells (SiGMA reused from bestmodel).
variant_list=(Homog_MP Hetero_MP Homog_MP_ungated Hetero_MP_ungated)
num_variants=${#variant_list[@]}
num_tasks="${PAPER_T6_CLUSTER_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
variant="${variant_list[$((idx / num_seeds))]}"

cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-anchor.yaml"
source_run="ht9bntg2"
base_type="GATEDGCN"
homog_types="GATEDGCN,GATEDGCN"
hetero_types="GATEDGCN,GCN"
extra_args=()

case "${variant}" in
    Homog_MP)
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${homog_types}"
        )
        ;;
    Hetero_MP)
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${hetero_types}"
        )
        ;;
    Homog_MP_ungated)
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${homog_types}"
            gnn.hybrid.gate none
        )
        ;;
    Hetero_MP_ungated)
        extra_args+=(
            gnn.hybrid.num_gnn_heads 2
            "gnn.hybrid.gnn_types" "${hetero_types}"
            gnn.hybrid.gate none
        )
        ;;
    *)
        log_message "bad variant=${variant}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T6_CLUSTER_WANDB_PREFIX:-paper_T6}"
wandb_group="${wandb_group_prefix}_cluster_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table6,paper_table7,${variant},cluster,seed${seed},source_${source_run},plus_one_mp"

log_message "T6 CLUSTER task ${task_id}/${num_tasks}: variant=${variant} seed=${seed}"
log_message "W&B group=${wandb_group} homog=${homog_types} hetero=${hetero_types}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
fi

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
    gnn.hybrid.identity_proj False \
    "${extra_args[@]}"
