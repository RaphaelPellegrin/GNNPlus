#!/usr/bin/env bash
# =============================================================================
# SiGMA + GRIT attention heads — PATTERN + CLUSTER seed grids
#
# Layout (task_id 1-based):
#   variants × datasets × seeds
#   seed = SIGMA_GRIT_ATTN_SEED_OFFSET + (idx % num_seeds)
#   dataset_idx = (idx // num_seeds) % num_datasets
#   variant_idx = idx // (num_seeds * num_datasets)
#     variant 0 → no virtual nodes
#     variant 1 → dataset.num_virtual_nodes = SIGMA_GRIT_ATTN_NUM_VN (default 4)
#
# Defaults (original campaign): 1 variant × 2 ds × 5 seeds = 10 tasks, seeds 0–4.
# Reseed + VN (20 tasks):
#   SIGMA_GRIT_ATTN_SEED_OFFSET=5 SIGMA_GRIT_ATTN_NUM_VARIANTS=2 \
#     SIGMA_GRIT_ATTN_NUM_VN=4 bash submit_sigma_grit_attn_pattern_cluster.sh
#
# Anchors:
#   pattern  configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml
#   cluster  configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml
# =============================================================================

#SBATCH --job-name=sigma_grit_attn
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
num_seeds="${SIGMA_GRIT_ATTN_NUM_SEEDS:-5}"
num_datasets="${SIGMA_GRIT_ATTN_NUM_DATASETS:-2}"
num_variants="${SIGMA_GRIT_ATTN_NUM_VARIANTS:-1}"
seed_offset="${SIGMA_GRIT_ATTN_SEED_OFFSET:-0}"
num_vn="${SIGMA_GRIT_ATTN_NUM_VN:-4}"
num_tasks="${SIGMA_GRIT_ATTN_NUM_TASKS:-$((num_variants * num_datasets * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
rest=$((idx / num_seeds))
dataset_idx=$((rest % num_datasets))
variant_idx=$((rest / num_datasets))

case "${dataset_idx}" in
    0)
        ds_tag="pattern"
        cfg="configs/gated_hybrid/pattern-hybrid-ta9qtxb9-grit-attn-anchor.yaml"
        source_run="ta9qtxb9"
        ;;
    1)
        ds_tag="cluster"
        cfg="configs/gated_hybrid/cluster-hybrid-ht9bntg2-grit-attn-anchor.yaml"
        source_run="ht9bntg2"
        ;;
    *)
        log_message "bad dataset_idx=${dataset_idx}"
        exit 1
        ;;
esac

case "${variant_idx}" in
    0)
        add_vn="False"
        vn_count=0
        wandb_group="paper_sigma_grit_attn_${ds_tag}"
        vn_tag="novn"
        ;;
    1)
        if [ "${num_vn}" -le 0 ]; then
            log_message "VN variant requested but SIGMA_GRIT_ATTN_NUM_VN=${num_vn}"
            exit 1
        fi
        add_vn="True"
        vn_count="${num_vn}"
        wandb_group="paper_sigma_grit_attn_${ds_tag}_vn${vn_count}"
        vn_tag="vn${vn_count}"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx} (num_variants=${num_variants})"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_name="sigma_grit_attn_${ds_tag}_${vn_tag}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="sigma_grit_attn,attn_type_grit,grit_attn,${ds_tag},seed${seed},${vn_tag},source_${source_run}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    mkdir -p "${GNNPLUS_OUT_DIR}"
    extra_args+=(out_dir "${GNNPLUS_OUT_DIR}")
    log_message "out_dir override: ${GNNPLUS_OUT_DIR}"
fi

extra_args+=(dataset.add_virtual_nodes "${add_vn}")
extra_args+=(dataset.num_virtual_nodes "${vn_count}")

log_message "sigma_grit_attn task ${task_id}/${num_tasks}: ds=${ds_tag} seed=${seed} ${vn_tag} cfg=${cfg}"
log_message "W&B group=${wandb_group} name=${wandb_name}"
log_message "Force override: gnn.hybrid.attn_type=grit"

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
