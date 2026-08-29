#!/usr/bin/env bash
# =============================================================================
# TU datasets — Errica et al. (ICLR 2020) fair comparison protocol.
#
# Uses vendored 10-fold splits from diningphil/gnn-comparison, Errica GIN/SAGE
# hyperparameter grids, and SiGMA on the same splits.
#
# Campaigns (TU_ERRICA_CAMPAIGN):
#   canonical   — fixed Errica-canonical HP, 7 datasets × 3 models × 10 folds
#                 × 3 seeds = 630 jobs (default)
#   grid_select — full HP grid × 10 folds, 1 seed (model selection phase)
#   grid_eval   — best HP per fold × 3 seeds (after aggregate_hp_selection.py)
#
# Task layout (canonical / grid_eval):
#   task_id = ((ds*M + model)*F + fold)*S + seed + 1
#
# Submit:
#   bash bash_interface/cluster/submit_tu_errica_fair.sh
# =============================================================================

#SBATCH --job-name=tu_errica
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
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

log_message "common_env OK; starting Errica task setup (campaign=${TU_ERRICA_CAMPAIGN:-canonical})"

campaign="${TU_ERRICA_CAMPAIGN:-canonical}"
task_id=${SLURM_ARRAY_TASK_ID:-1}

# 7 datasets overlapping with Errica tables + our appendix social set.
datasets=(enzymes proteins nci1 dd imdb-b reddit-b collab)
dataset_names=(ENZYMES PROTEINS NCI1 DD IMDB-BINARY REDDIT-BINARY COLLAB)
num_datasets=${#datasets[@]}

models=(gin graphsage sigma_hetero)
model_cfgs=(configs/tu_errica/gin-errica-base.yaml configs/tu_errica/graphsage-errica-base.yaml configs/tu_errica/sigma-hetero-errica-base.yaml)
model_tags=(GIN GraphSAGE SiGMA_hetero)
num_models=${#models[@]}

num_folds="${TU_ERRICA_NUM_FOLDS:-10}"
num_seeds="${TU_ERRICA_NUM_SEEDS:-3}"
seed_offset="${TU_ERRICA_SEED_OFFSET:-0}"

case "${campaign}" in
    canonical|grid_eval)
        num_hp=1
        use_canonical=1
        num_seeds_effective="${num_seeds}"
        num_tasks=$((num_datasets * num_models * num_folds * num_seeds_effective))
        ;;
    grid_select)
        use_canonical=0
        num_seeds_effective=1
        hp_model="${TU_ERRICA_GRID_MODEL:-gin}"
        num_hp=$(python3 -c "import json; from pathlib import Path; p=Path('configs/tu_errica/${hp_model}_hp_grid.json'); print(len(json.load(p.open())['grid']))")
        num_tasks=$((num_datasets * num_hp * num_folds))
        ;;
    *)
        log_message "Unknown TU_ERRICA_CAMPAIGN=${campaign}"
        exit 1
        ;;
esac

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))

if [ "${campaign}" = "grid_select" ]; then
    model_idx=0
    case "${hp_model}" in
        gin) cfg="configs/tu_errica/gin-errica-base.yaml"; model_tag="GIN" ;;
        graphsage) cfg="configs/tu_errica/graphsage-errica-base.yaml"; model_tag="GraphSAGE" ;;
        sigma_hetero) cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"; model_tag="SiGMA_hetero" ;;
        *) log_message "Unknown TU_ERRICA_GRID_MODEL=${hp_model}"; exit 1 ;;
    esac
    models=("${hp_model}")
    model_key="${hp_model}"
    seed=$((seed_offset))
    hp_id=$((idx % num_hp))
    rest=$((idx / num_hp))
    fold_idx=$((rest % num_folds))
    dataset_idx=$((rest / num_folds))
else
    seed=$((seed_offset + (idx % num_seeds_effective)))
    rest=$((idx / num_seeds_effective))
    fold_idx=$((rest % num_folds))
    rest=$((rest / num_folds))
    model_idx=$((rest % num_models))
    dataset_idx=$((rest / num_models))
    hp_id=-1
fi

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
if [ "${campaign}" != "grid_select" ]; then
    model_key="${models[$model_idx]}"
    cfg="${model_cfgs[$model_idx]}"
    model_tag="${model_tags[$model_idx]}"
fi

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

# HP overrides from Errica grid (or canonical).
emit_args=(--model "${model_key}")
if [ "${use_canonical}" = "1" ]; then
    emit_args+=(--canonical)
else
    emit_args+=(--hp-id="${hp_id}")
fi
if ! hp_line="$(python scripts/tu_errica/emit_cfg_overrides.py "${emit_args[@]}")"; then
    log_message "emit_cfg_overrides failed for model=${model_key} args=${emit_args[*]}"
    exit 1
fi
if [ -z "${hp_line}" ]; then
    log_message "emit_cfg_overrides returned empty output for model=${model_key}"
    exit 1
fi
# shellcheck disable=SC2206
hp_args=(${hp_line})
log_message "HP overrides loaded (${#hp_args[@]} tokens)"

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
hp_tag="canonical"
if [ "${hp_id}" -ge 0 ]; then
    hp_tag="hp${hp_id}"
fi
wandb_group="tu_errica_${ds_tag}_${model_tag}_${campaign}_${hp_tag}"
wandb_name="${wandb_group}_f${fold_idx}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_errica,${ds_tag},${model_tag},${campaign},fold${fold_idx},seed${seed},${hp_tag}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_errica/${campaign}/${ds_tag}_${model_tag}_${hp_tag}_f${fold_idx}_seed${seed}"
else
    run_dir="results/tu_errica/${campaign}/${ds_tag}_${model_tag}_${hp_tag}_f${fold_idx}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "Errica ${campaign} task ${task_id}/${num_tasks}: ds=${ds_name} model=${model_tag} fold=${fold_idx} seed=${seed} hp=${hp_tag}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
model=${model_tag}
model_key=${model_key}
fold=${fold_idx}
seed=${seed}
hp_id=${hp_tag}
campaign=${campaign}
cfg=${cfg}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
wandb_name=${wandb_name}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    dataset.split_index "${fold_idx}"
    out_dir "${run_dir}"
    train.enable_ckpt True
    train.ckpt_best True
    train.ckpt_clean True
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

export WANDB_EXTRA_TAGS="${wandb_tags}"

python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    "${extra_args[@]}" \
    "${hp_args[@]}"

log_message "Done: ${run_dir}"
