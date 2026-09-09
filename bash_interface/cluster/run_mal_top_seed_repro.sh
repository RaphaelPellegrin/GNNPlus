#!/usr/bin/env bash
# =============================================================================
# MalNet-Tiny top-run seed repro: 4 variants × 5 seeds (default 20 tasks).
#
# Variants (W&B lineage):
#   0 v4cytwe0  GCNE,GINE a0g2   https://wandb.ai/.../runs/v4cytwe0
#   1 zk6ihqi8  GCNE a0g2        https://wandb.ai/.../runs/zk6ihqi8
#   2 apiw6l3u  9h3jqzkm ep150   https://wandb.ai/.../runs/apiw6l3u
#   3 5sx7r420  9h3jqzkm lr2.3e-3 ep250  https://wandb.ai/.../runs/5sx7r420
#
# Array layout: task = variant * num_seeds + seed + 1
# Submit:
#   bash bash_interface/cluster/submit_mal_top_seed_repro.sh
# =============================================================================

#SBATCH --job-name=mal_top_seed_repro
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
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
num_seeds="${MAL_TOP_NUM_SEEDS:-5}"
num_variants=4
num_tasks=$((num_variants * num_seeds))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
variant_idx=$((idx / num_seeds))

case "${variant_idx}" in
    0)
        variant="v4cytwe0"
        cfg="configs/gated_hybrid/mal-seed-repro-v4cytwe0.yaml"
        ;;
    1)
        variant="zk6ihqi8"
        cfg="configs/gated_hybrid/mal-seed-repro-zk6ihqi8.yaml"
        ;;
    2)
        variant="apiw6l3u"
        cfg="configs/gated_hybrid/mal-seed-repro-apiw6l3u.yaml"
        ;;
    3)
        variant="5sx7r420"
        cfg="configs/gated_hybrid/mal-seed-repro-5sx7r420.yaml"
        ;;
    *)
        log_message "unknown variant_idx=${variant_idx}"
        exit 1
        ;;
esac

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group_prefix="${MAL_TOP_WANDB_PREFIX:-seed_repro_mal}"
wandb_group="${wandb_group_prefix}_${variant}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"

log_message "MalNet top seed repro task ${task_id}/${num_tasks}: variant=${variant} seed=${seed} cfg=${cfg}"

# YACS cannot override yaml list fields via CLI (wandb.tags); use env instead.
export WANDB_EXTRA_TAGS="seed_repro,malnet,${variant}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

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
