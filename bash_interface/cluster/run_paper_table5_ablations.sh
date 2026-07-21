#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 5 ablations (LRGB) — exact best anchors from Paper_final_runs.md
#
# 4 datasets × 4 variants × 5 seeds = 80 tasks (default).
#
# Variants (W&B group suffix + tag — keep these names stable):
#   0  SiGMA           — best gated hybrid (paper baseline)
#   1  SiGMA_ungated   — same architecture, gnn.hybrid.gate=none
#   2  Attn_only       — all MP heads replaced by attention
#   3  MP_only         — all attention heads replaced by same MP type(s)
#
# W&B group:  paper_T5_<dataset>_<Variant>
# W&B tags:   paper_table5, <Variant>, <dataset>, seed<k>
#
# Source anchors / best exemplar runs (hyperparams frozen in yaml):
#   peptides_func   o5cdk766 a1g1     best seed8  l31u4b3k
#   peptides_struct g3bsaq32 a1g1     best seed6  bqkect9l
#   voc             j7ukyzdm a2g2     best seed3  vyt7hjj5
#   coco            5b4z9l3u a1g1     best seed3  xgjakrz0  (job25558630)
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_ablations.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_abl
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
num_seeds="${PAPER_T5_NUM_SEEDS:-5}"
num_variants="${PAPER_T5_NUM_VARIANTS:-4}"
num_datasets="${PAPER_T5_NUM_DATASETS:-4}"
num_tasks="${PAPER_T5_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
dataset_idx=$((rest / num_variants))

case "${dataset_idx}" in
    0)
        ds_tag="peptides_func"
        cfg="configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml"
        source_run="l31u4b3k"
        na=1; ng=1; gnn_types="GCN"
        ;;
    1)
        ds_tag="peptides_struct"
        cfg="configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
        source_run="bqkect9l"
        na=1; ng=1; gnn_types="GINE"
        ;;
    2)
        ds_tag="voc"
        cfg="configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml"
        source_run="vyt7hjj5"
        na=2; ng=2; gnn_types="GATEDGCN,GATEDGCN"
        ;;
    3)
        ds_tag="coco"
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
        source_run="xgjakrz0"
        na=1; ng=1; gnn_types="GATEDGCN"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

total_heads=$((na + ng))
first_type="${gnn_types%%,*}"
extra_args=()

case "${variant_idx}" in
    0)
        variant="SiGMA"
        # Anchor yaml unchanged (gated hybrid).
        ;;
    1)
        variant="SiGMA_ungated"
        extra_args+=(gnn.hybrid.gate none)
        ;;
    2)
        variant="Attn_only"
        extra_args+=(
            gnn.hybrid.num_attn_heads "${total_heads}"
            gnn.hybrid.num_gnn_heads 0
            "gnn.hybrid.gnn_types" ""
        )
        ;;
    3)
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
# Distinct W&B groups per (dataset, variant) — filterable in the UI.
wandb_group_prefix="${PAPER_T5_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
name_suffix="${PAPER_T5_NAME_SUFFIX:-}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"

# Tags make variant filtering easy even across groups.
wandb_tags="paper_table5,${variant},${ds_tag},seed${seed},source_${source_run}"
if [ -n "${name_suffix}" ]; then
    # Strip leading underscore for the tag (e.g. _h200 → relaunch_h200).
    tag_suffix="${name_suffix#_}"
    wandb_tags="${wandb_tags},relaunch_${tag_suffix}"
fi

log_message "Table5 task ${task_id}/${num_tasks}: ds=${ds_tag} variant=${variant} seed=${seed} source=${source_run} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

# Prefer netscratch for GraphGym stats/ckpts — holylabs inode/byte quota kills jobs
# with OSError 122 on results/*/stats.json (see COCO relaunch 34070242/43).
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
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
