#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 5/6 — PATTERN ablations on the GRIT+VN4 SiGMA anchor.
#
# Anchor (paper SiGMA for this column):
#   W&B group  paper_sigma_grit_attn_pattern_vn4
#   Acc.       ~87.395±0.194%  (seeds 5–9)
#   Config     configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml
#   Overrides  attn_type=grit, add_virtual_nodes=True, num_virtual_nodes=4
#
# Variants (W&B group: paper_T5_pattern_gritvn4_<Variant>):
#   0  SiGMA               — gated hybrid GRIT+VN4 (optional; reuse existing by default)
#   1  SiGMA_ungated       — gate=none
#   2  SiGMA_attn_gate     — attention gated; mp_gate=none
#   3  SiGMA_ungated_attn  — attention ungated; mp_gate=elementwise
#   4  Attn_only           — all heads → GRIT attention
#   5  MP_only             — all heads → GCNE
#
# Default seeds: 5–9 (match published VN4 SiGMA). Skip SiGMA tasks unless
# PAPER_T5_PATTERN_GRITVN4_INCLUDE_SIGMA=1.
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_pattern_gritvn4_ablations.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_pat_gritvn4
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
num_seeds="${PAPER_T5_PATTERN_GRITVN4_NUM_SEEDS:-5}"
seed_offset="${PAPER_T5_PATTERN_GRITVN4_SEED_OFFSET:-5}"
include_sigma="${PAPER_T5_PATTERN_GRITVN4_INCLUDE_SIGMA:-0}"
num_vn="${PAPER_T5_PATTERN_GRITVN4_NUM_VN:-4}"

# Variant indices always 0..5; when include_sigma=0 we only schedule 1..5.
if [ "${include_sigma}" = "1" ]; then
    variant_list=(0 1 2 3 4 5)
else
    variant_list=(1 2 3 4 5)
fi
num_variants=${#variant_list[@]}
num_tasks="${PAPER_T5_PATTERN_GRITVN4_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
variant_slot=$((idx / num_seeds))
variant_idx="${variant_list[${variant_slot}]}"

ds_tag="pattern"
cfg="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
source_run="ta9qtxb9"
na=2
ng=2
gnn_types="GCNE,GCNE"
total_heads=$((na + ng))
first_type="${gnn_types%%,*}"
extra_args=()

case "${variant_idx}" in
    0)
        variant="SiGMA"
        ;;
    1)
        variant="SiGMA_ungated"
        extra_args+=(gnn.hybrid.gate none)
        ;;
    2)
        variant="SiGMA_attn_gate"
        extra_args+=(gnn.hybrid.mp_gate none)
        ;;
    3)
        variant="SiGMA_ungated_attn"
        extra_args+=(
            gnn.hybrid.gate none
            gnn.hybrid.mp_gate elementwise
        )
        ;;
    4)
        variant="Attn_only"
        extra_args+=(
            gnn.hybrid.num_attn_heads "${total_heads}"
            gnn.hybrid.num_gnn_heads 0
            "gnn.hybrid.gnn_types" ""
        )
        ;;
    5)
        variant="MP_only"
        mp_types="${first_type}"
        for ((i = 1; i < total_heads; i++)); do
            mp_types="${mp_types},${first_type}"
        done
        extra_args+=(
            gnn.hybrid.num_attn_heads 0
            gnn.hybrid.num_gnn_heads "${total_heads}"
            "gnn.hybrid.gnn_types" "${mp_types}"
        )
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
wandb_group_prefix="${PAPER_T5_PATTERN_GRITVN4_WANDB_PREFIX:-paper_T5_pattern_gritvn4}"
# Groups: paper_T5_pattern_gritvn4_SiGMA, ..._SiGMA_ungated, ...
wandb_group="${wandb_group_prefix}_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="paper_table5,paper_table6,${variant},pattern,seed${seed},gritvn4,attn_type_grit,vn${num_vn},source_${source_run}"

log_message "Table5 PATTERN gritvn4 task ${task_id}/${num_tasks}: variant=${variant} seed=${seed} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi

# Anchor overrides (yaml defaults to grit / no VN; force VN=4 + grit).
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
