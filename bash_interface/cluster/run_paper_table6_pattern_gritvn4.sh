#!/usr/bin/env bash
# =============================================================================
# Table 6/7 — PATTERN homog/hetero MP on GRIT+VN4 SiGMA anchor (~87.4%).
#
# Anchor SiGMA / Homog_MP gated: reuse paper_sigma_grit_attn_pattern_vn4
#   (do not relaunch unless PAPER_T6_PATTERN_GRITVN4_INCLUDE_HOMOG_GATED=1).
#
# Variants (W&B: paper_T6_pattern_gritvn4_<Variant>):
#   0  Homog_MP_ungated  — GCNE,GCNE, gate=none
#   1  Hetero_MP         — GCNE,GINE (swap last), gated
#   2  Hetero_MP_ungated — GCNE,GINE, gate=none
#
# Seeds 5–9 (match VN4 SiGMA). Always: attn_type=grit, VN=4.
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table6_pattern_gritvn4.sh
# =============================================================================

#SBATCH --job-name=sigma_T6_pat_gritvn4
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
num_seeds="${PAPER_T6_PATTERN_GRITVN4_NUM_SEEDS:-5}"
seed_offset="${PAPER_T6_PATTERN_GRITVN4_SEED_OFFSET:-5}"
num_vn="${PAPER_T6_PATTERN_GRITVN4_NUM_VN:-4}"
include_homog_gated="${PAPER_T6_PATTERN_GRITVN4_INCLUDE_HOMOG_GATED:-0}"

homog_types="GCNE,GCNE"
hetero_types="GCNE,GINE"
ng=2

if [ "${include_homog_gated}" = "1" ]; then
    variant_list=(Homog_MP Homog_MP_ungated Hetero_MP Hetero_MP_ungated)
else
    variant_list=(Homog_MP_ungated Hetero_MP Hetero_MP_ungated)
fi
num_variants=${#variant_list[@]}
num_tasks="${PAPER_T6_PATTERN_GRITVN4_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
variant_slot=$((idx / num_seeds))
variant="${variant_list[${variant_slot}]}"

cfg="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
source_run="ta9qtxb9"
extra_args=()

case "${variant}" in
    Homog_MP)
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
            "gnn.hybrid.gnn_types" "${homog_types}"
        )
        ;;
    Homog_MP_ungated)
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
            "gnn.hybrid.gnn_types" "${homog_types}"
            gnn.hybrid.gate none
        )
        ;;
    Hetero_MP)
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
            "gnn.hybrid.gnn_types" "${hetero_types}"
        )
        ;;
    Hetero_MP_ungated)
        extra_args+=(
            gnn.hybrid.num_gnn_heads "${ng}"
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
wandb_group_prefix="${PAPER_T6_PATTERN_GRITVN4_WANDB_PREFIX:-paper_T6_pattern_gritvn4}"
wandb_group="${wandb_group_prefix}_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table6,paper_table7,${variant},pattern,seed${seed},gritvn4,attn_type_grit,vn${num_vn},source_${source_run},one_mp_swap"

log_message "T6 PATTERN gritvn4 task ${task_id}/${num_tasks}: variant=${variant} seed=${seed}"
log_message "W&B group=${wandb_group} homog=${homog_types} hetero=${hetero_types}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
fi

extra_args+=(
    gnn.hybrid.attn_type grit
    dataset.add_virtual_nodes True
    dataset.num_virtual_nodes "${num_vn}"
)

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
