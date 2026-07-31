#!/usr/bin/env bash
# =============================================================================
# SiGMA paper Table 6 — Hybrid with ungated attention / gated MP
#
# Same SiGMA anchors; overrides:
#   gnn.hybrid.gate none              # attention ungated
#   gnn.hybrid.mp_gate <yaml style>   # MP keeps headwise|elementwise
#
# Opposite of SiGMA_attn_gate (attn gated, mp_gate=none).
#
# 7 datasets × 5 seeds = 35 tasks:
#   peptides_func, peptides_struct, voc, coco, mnist, cifar10, pattern
#
# W&B group:  paper_T5_<dataset>_SiGMA_ungated_attn
# W&B tags:   paper_table5, paper_table6, SiGMA_ungated_attn, <dataset>, seed<k>
#
# Submit:
#   bash bash_interface/cluster/submit_paper_table5_ungated_attn.sh
# =============================================================================

#SBATCH --job-name=sigma_T5_ungated_attn
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
num_seeds="${PAPER_T5_UNGATED_ATTN_NUM_SEEDS:-5}"
num_datasets="${PAPER_T5_UNGATED_ATTN_NUM_DATASETS:-7}"
num_tasks="${PAPER_T5_UNGATED_ATTN_NUM_TASKS:-$((num_datasets * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
dataset_idx=$((idx / num_seeds))

# mp_gate must match the original yaml ``gate`` style (attention is forced none).
case "${dataset_idx}" in
    0)
        ds_tag="peptides_func"
        cfg="configs/gated_hybrid/peptides-func-hybrid-o5cdk766-a1g1-anchor.yaml"
        source_run="l31u4b3k"
        mp_gate="elementwise"
        ;;
    1)
        ds_tag="peptides_struct"
        cfg="configs/gated_hybrid/peptides-struct-hybrid-g3bsaq32-b7m0-anchor.yaml"
        source_run="bqkect9l"
        mp_gate="elementwise"
        ;;
    2)
        ds_tag="voc"
        cfg="configs/gated_hybrid/voc-hybrid-j7ukyzdm-a2g2-anchor.yaml"
        source_run="vyt7hjj5"
        mp_gate="headwise"
        ;;
    3)
        ds_tag="coco"
        cfg="configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1-anchor.yaml"
        source_run="xgjakrz0"
        mp_gate="headwise"
        ;;
    4)
        ds_tag="mnist"
        cfg="configs/gated_hybrid/mnist-hybrid-lcvbyyss-a2g2-anchor.yaml"
        source_run="uh7nxm4e"
        mp_gate="elementwise"
        ;;
    5)
        ds_tag="cifar10"
        cfg="configs/gated_hybrid/cifar10-hybrid-ulij45a2-anchor.yaml"
        source_run="3tx560wq"
        mp_gate="headwise"
        ;;
    6)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-gcne-best-hybrid.yaml"
        source_run="ta9qtxb9"
        mp_gate="elementwise"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

variant="SiGMA_ungated_attn"
extra_args=(
    gnn.hybrid.gate none
    gnn.hybrid.mp_gate "${mp_gate}"
)

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${PAPER_T5_UNGATED_ATTN_WANDB_PREFIX:-paper_T5}"
wandb_group="${wandb_group_prefix}_${ds_tag}_${variant}"
name_suffix="${PAPER_T5_UNGATED_ATTN_NAME_SUFFIX:-}"
wandb_name="${wandb_group_prefix}_${ds_tag}_${variant}_seed${seed}_job${job_tag}_${task_id}${name_suffix}"

wandb_tags="paper_table5,paper_table6,${variant},${ds_tag},seed${seed},source_${source_run}"
if [ -n "${name_suffix}" ]; then
    tag_suffix="${name_suffix#_}"
    wandb_tags="${wandb_tags},relaunch_${tag_suffix}"
fi

log_message "Table6 ungated_attn task ${task_id}/${num_tasks}: ds=${ds_tag} seed=${seed} mp_gate=${mp_gate}"
log_message "W&B group=${wandb_group} name=${wandb_name} cfg=${cfg}"

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi

if [ -n "${PAPER_T5_UNGATED_ATTN_MAX_EPOCH:-}" ]; then
    extra_args+=(optim.max_epoch "${PAPER_T5_UNGATED_ATTN_MAX_EPOCH}")
    log_message "max_epoch override: ${PAPER_T5_UNGATED_ATTN_MAX_EPOCH}"
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
