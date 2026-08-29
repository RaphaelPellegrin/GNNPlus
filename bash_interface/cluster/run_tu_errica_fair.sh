#!/usr/bin/env bash
# =============================================================================
# TU datasets — Errica et al. (ICLR 2020) fair comparison protocol.
#
# Campaigns (TU_ERRICA_CAMPAIGN):
#   canonical          — fixed HP smoke (630 jobs); not for final rebuttal table
#   grid_select        — Errica HP grid × folds × 1 seed (GIN or GraphSAGE)
#   grid_eval          — selected HP × 3 seeds (one model; needs selection JSON)
#   sigma_grid_select  — hybrid SiGMA search (bio: L/H-matched; social: full grid)
#   sigma_grid_eval    — selected SiGMA HP × 3 seeds
#
# Hybrid pipeline (Option 3): see Paper_tu_errica_fair_comparison.md
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

campaign="${TU_ERRICA_CAMPAIGN:-canonical}"
task_id=${SLURM_ARRAY_TASK_ID:-1}

datasets=(enzymes proteins nci1 dd imdb-b reddit-b collab)
dataset_names=(ENZYMES PROTEINS NCI1 DD IMDB-BINARY REDDIT-BINARY COLLAB)
num_datasets=${#datasets[@]}

num_folds="${TU_ERRICA_NUM_FOLDS:-10}"
num_seeds="${TU_ERRICA_NUM_SEEDS:-3}"
seed_offset="${TU_ERRICA_SEED_OFFSET:-0}"

model_cfgs=(configs/tu_errica/gin-errica-base.yaml configs/tu_errica/graphsage-errica-base.yaml configs/tu_errica/sigma-hetero-errica-base.yaml)
model_keys=(gin graphsage sigma_hetero)
model_tags=(GIN GraphSAGE SiGMA_hetero)

log_message "Errica campaign=${campaign} task_id=${task_id}"

case "${campaign}" in
    canonical)
        num_tasks=$((num_datasets * 3 * num_folds * num_seeds))
        ;;
    grid_select)
        hp_model="${TU_ERRICA_GRID_MODEL:-gin}"
        num_hp=$(python3 -c "import json; from pathlib import Path; p=Path('configs/tu_errica/${hp_model}_hp_grid.json'); print(len(json.load(p.open())['grid']))")
        num_tasks=$((num_datasets * num_hp * num_folds))
        ;;
    grid_eval|sigma_grid_eval)
        num_tasks=$((num_datasets * num_folds * num_seeds))
        ;;
    sigma_grid_select)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids/manifest.json'))['num_tasks'])")
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
hp_id=-1
use_selection=0
selection_file=""
sigma_grid_file=""

case "${campaign}" in
    grid_select)
        hp_model="${TU_ERRICA_GRID_MODEL:-gin}"
        case "${hp_model}" in
            gin) cfg="configs/tu_errica/gin-errica-base.yaml"; model_tag="GIN" ;;
            graphsage) cfg="configs/tu_errica/graphsage-errica-base.yaml"; model_tag="GraphSAGE" ;;
            *) log_message "grid_select supports gin|graphsage only"; exit 1 ;;
        esac
        model_key="${hp_model}"
        seed=$((seed_offset))
        num_hp=$(python3 -c "import json; from pathlib import Path; p=Path('configs/tu_errica/${hp_model}_hp_grid.json'); print(len(json.load(p.open())['grid']))")
        hp_id=$((idx % num_hp))
        rest=$((idx / num_hp))
        fold_idx=$((rest % num_folds))
        dataset_idx=$((rest / num_folds))
        emit_extra=(--model "${model_key}" --hp-id="${hp_id}")
        ;;
    grid_eval)
        eval_model="${TU_ERRICA_EVAL_MODEL:?set TU_ERRICA_EVAL_MODEL=gin|graphsage}"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/${eval_model}_per_fold.json}"
        case "${eval_model}" in
            gin) cfg="configs/tu_errica/gin-errica-base.yaml"; model_tag="GIN"; model_key="gin" ;;
            graphsage) cfg="configs/tu_errica/graphsage-errica-base.yaml"; model_tag="GraphSAGE"; model_key="graphsage" ;;
            *) log_message "grid_eval supports gin|graphsage"; exit 1 ;;
        esac
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        dataset_idx=$((rest / num_folds))
        use_selection=1
        ;;
    sigma_grid_select)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        mapfile -t sigma_task < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        ds_tag="${sigma_task[0]}"
        fold_idx="${sigma_task[1]}"
        grid_rel="${sigma_task[2]}"
        hp_id="${sigma_task[3]}"
        sigma_grid_file="configs/tu_errica/sigma_grids/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_eval)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        dataset_idx=$((rest / num_folds))
        use_selection=1
        ;;
    canonical)
        models=(gin graphsage sigma_hetero)
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        rest=$((rest / num_folds))
        model_idx=$((rest % 3))
        dataset_idx=$((rest / 3))
        model_key="${models[$model_idx]}"
        cfg="${model_cfgs[$model_idx]}"
        model_tag="${model_tags[$model_idx]}"
        emit_extra=(--model "${model_key}" --canonical)
        ;;
esac

if [ "${campaign}" != "sigma_grid_select" ]; then
    ds_tag="${datasets[$dataset_idx]}"
fi
ds_name="${dataset_names[$dataset_idx]}"

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

if [ "${use_selection}" = "1" ]; then
    if [ ! -f "${selection_file}" ]; then
        log_message "Selection file missing: ${selection_file}"
        exit 1
    fi
    emit_extra=(--selection-file "${selection_file}" --ds-tag "${ds_tag}" --fold "${fold_idx}")
fi

if ! hp_line="$(python scripts/tu_errica/emit_cfg_overrides.py "${emit_extra[@]}")"; then
    log_message "emit_cfg_overrides failed: ${emit_extra[*]}"
    exit 1
fi
# shellcheck disable=SC2206
hp_args=(${hp_line})

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
hp_tag="canonical"
if [ "${hp_id}" -ge 0 ]; then
    if [ "${campaign}" = "sigma_grid_select" ]; then
        hp_tag="f${fold_idx}_hp${hp_id}"
    else
        hp_tag="hp${hp_id}"
    fi
elif [ "${use_selection}" = "1" ]; then
    hp_tag="selected"
fi

wandb_group="tu_errica_${ds_tag}_${model_tag}_${campaign}_${hp_tag}"
wandb_name="${wandb_group}_f${fold_idx}_seed${seed}_job${job_tag}_${task_id}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_errica/${campaign}/${ds_tag}_${model_tag}_${hp_tag}_f${fold_idx}_seed${seed}"
else
    run_dir="results/tu_errica/${campaign}/${ds_tag}_${model_tag}_${hp_tag}_f${fold_idx}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "${campaign} ${task_id}/${num_tasks}: ds=${ds_name} model=${model_tag} fold=${fold_idx} seed=${seed} hp=${hp_tag}"

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

batch_override_args=()
if [ "${model_key}" = "sigma_hetero" ]; then
    case "${ds_tag}" in
        dd|reddit-b|collab) batch_override_args+=(train.batch_size 16) ;;
    esac
    if [ "${#batch_override_args[@]}" -gt 0 ]; then
        log_message "SiGMA batch override: ${batch_override_args[*]}"
    fi
fi

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
    "${hp_args[@]}" \
    "${batch_override_args[@]}"

log_message "Done: ${run_dir}"
