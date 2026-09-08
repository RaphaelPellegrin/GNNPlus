#!/usr/bin/env bash
# =============================================================================
# TU datasets — Errica et al. (ICLR 2020) fair comparison protocol.
#
# Campaigns (TU_ERRICA_CAMPAIGN):
#   canonical          — fixed HP smoke (630 jobs); not for final rebuttal table
#   grid_select        — Errica HP grid × folds × 1 seed (GIN, GraphSAGE, GCN, or GAT)
#   grid_eval          — selected HP × 3 seeds (one model; needs selection JSON)
#   sigma_grid_select         — legacy SiGMA HP search (budget_bio W&B groups)
#   sigma_grid_select_fixed8  — fixed8 SIGMA_GRID (no param ceiling; preferred)
#   sigma_grid_eval           — legacy SiGMA eval
#   sigma_grid_eval_fixed8    — eval after fixed8 selection
#
# See Paper_tu_errica_fair_comparison.md
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
    sigma_grid_select|sigma_grid_select_fixed8|sigma_grid_select_fixed8_ungated)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_full64)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_full64/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_anchor_boost)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_anchor_boost/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_a1g2_micro)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_a1g2_micro/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_a1g2_nci1_micro)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_a1g2_nci1_micro/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_anchor_refine)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_anchor_refine/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_nci1_refine)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_nci1_refine/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_native_fair)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_native_fair/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_native_fair_v2)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_native_fair_v2/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_native_fair_v2_l4)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_native_fair_v2_l4/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_a0g_pnr)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_a0g_pnr/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_tiny_pnr)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_tiny_pnr/manifest.json'))['num_tasks'])")
        ;;
    sigma_grid_select_specialist_tiny)
        num_tasks=$(python3 -c "import json; print(json.load(open('configs/tu_errica/sigma_grids_specialist_tiny/manifest.json'))['num_tasks'])")
        ;;
    grid_eval|sigma_grid_eval|sigma_grid_eval_fixed8|sigma_grid_eval_full64|sigma_grid_eval_fixed8_ungated)
        num_tasks=$((num_datasets * num_folds * num_seeds))
        ;;
    sigma_grid_eval_anchor_boost|sigma_grid_eval_a1g2_micro)
        # PROTEINS + REDDIT-BINARY only (2 × 10 × 3).
        num_tasks=$((2 * num_folds * num_seeds))
        ;;
    sigma_grid_eval_a0g_pnr|sigma_grid_eval_tiny_pnr|sigma_grid_eval_specialist_tiny|sigma_grid_eval_native_fair|sigma_grid_eval_native_fair_v2|sigma_grid_eval_native_fair_v2_joint)
        # PROTEINS + NCI1 + REDDIT (3 × 10 × 3).
        num_tasks=$((3 * num_folds * num_seeds))
        ;;
    sigma_grid_eval_a1g2_nci1_micro|sigma_grid_eval_anchor_refine|sigma_grid_eval_nci1_refine)
        # Single dataset (1 × 10 × 3).
        num_tasks=$((num_folds * num_seeds))
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
            gcn) cfg="configs/tu_errica/gcn-errica-base.yaml"; model_tag="GCN" ;;
            gat) cfg="configs/tu_errica/gat-errica-base.yaml"; model_tag="GAT" ;;
            *) log_message "grid_select supports gin|graphsage|gcn|gat"; exit 1 ;;
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
        eval_model="${TU_ERRICA_EVAL_MODEL:?set TU_ERRICA_EVAL_MODEL=gin|graphsage|gcn|gat}"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/${eval_model}_per_fold.json}"
        case "${eval_model}" in
            gin) cfg="configs/tu_errica/gin-errica-base.yaml"; model_tag="GIN"; model_key="gin" ;;
            graphsage) cfg="configs/tu_errica/graphsage-errica-base.yaml"; model_tag="GraphSAGE"; model_key="graphsage" ;;
            gcn) cfg="configs/tu_errica/gcn-errica-base.yaml"; model_tag="GCN"; model_key="gcn" ;;
            gat) cfg="configs/tu_errica/gat-errica-base.yaml"; model_tag="GAT"; model_key="gat" ;;
            *) log_message "grid_eval supports gin|graphsage|gcn|gat"; exit 1 ;;
        esac
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        dataset_idx=$((rest / num_folds))
        use_selection=1
        ;;
    sigma_grid_select|sigma_grid_select_fixed8|sigma_grid_select_fixed8_ungated)
        if [[ "${campaign}" == *ungated* ]]; then
            cfg="configs/tu_errica/sigma-hetero-ungated-errica-base.yaml"
            model_tag="SiGMA_ungated"
        else
            cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
            model_tag="SiGMA_hetero"
        fi
        model_key="sigma_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_full64)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_full64/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_full64/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_anchor_boost)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_anchor_boost/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_anchor_boost/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_a1g2_micro)
        cfg="configs/tu_errica/sigma-hetero-a1g2-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_a1g2"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_a1g2_micro/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_a1g2_micro/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_a1g2_nci1_micro)
        cfg="configs/tu_errica/sigma-hetero-a1g2-nci1-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_a1g2_ginsage"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_a1g2_nci1_micro/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_a1g2_nci1_micro/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_anchor_refine)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_anchor_refine/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_anchor_refine/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_nci1_refine)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_nci1_refine/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_nci1_refine/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_native_fair)
        # Base yaml is a2g4; grid overrides num_attn/num_gnn/gnn_types per HP.
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_native_fair/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_native_fair/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_native_fair_v2)
        # Same wiring as native_fair; grid adds UniGCN mixes + mid LR.
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_native_fair_v2/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_native_fair_v2/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_native_fair_v2_l4)
        # L=4 add-on; same arches / lr / d_h as native_fair_v2.
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_native_fair_v2_l4/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_native_fair_v2_l4/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_a0g_pnr)
        # Base yaml a2g4; grid forces num_attn_heads=0 (a0g4 / a0g2).
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_a0g"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_a0g_pnr/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_a0g_pnr/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_tiny_pnr)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_tiny_pnr/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_tiny_pnr/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_select_specialist_tiny)
        # Per-dataset GCN (P/REDDIT) or SAGE (NCI1); a0g1/a1g1 via grid overrides.
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_spec_tiny"
        seed=$((seed_offset))
        read -r ds_tag fold_idx grid_rel hp_id < <(python3 -c "
import json
t=json.load(open('configs/tu_errica/sigma_grids_specialist_tiny/manifest.json'))['tasks'][${idx}]
print(t['ds_tag'], t['fold'], t['grid_file'], t['hp_id'])
")
        sigma_grid_file="configs/tu_errica/sigma_grids_specialist_tiny/grids/${grid_rel}"
        for i in "${!datasets[@]}"; do
            if [ "${datasets[$i]}" = "${ds_tag}" ]; then
                dataset_idx=$i
                break
            fi
        done
        emit_extra=(--sigma-grid-file "${sigma_grid_file}" --hp-id="${hp_id}")
        ;;
    sigma_grid_eval|sigma_grid_eval_fixed8|sigma_grid_eval_full64|sigma_grid_eval_fixed8_ungated)
        if [[ "${campaign}" == *ungated* ]]; then
            cfg="configs/tu_errica/sigma-hetero-ungated-errica-base.yaml"
            model_tag="SiGMA_ungated"
        else
            cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
            model_tag="SiGMA_hetero"
        fi
        model_key="sigma_hetero"
        if [ "${campaign}" = "sigma_grid_eval_full64" ]; then
            selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_full64_per_fold.json}"
        elif [ "${campaign}" = "sigma_grid_eval_fixed8_ungated" ]; then
            selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_fixed8_ungated_per_fold.json}"
        elif [ "${campaign}" = "sigma_grid_eval_fixed8" ]; then
            selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_fixed8_per_fold.json}"
        else
            selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_per_fold.json}"
        fi
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        dataset_idx=$((rest / num_folds))
        use_selection=1
        ;;
    sigma_grid_eval_native_fair)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_native_fair_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="nci1"; dataset_idx=2 ;;
            2) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "native_fair eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_native_fair_v2)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_native_fair_v2_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="nci1"; dataset_idx=2 ;;
            2) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "native_fair_v2 eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_native_fair_v2_joint)
        # Joint L∈{4,12} winners; separate selection + campaign from L12-only v2.
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_native_fair_v2_joint_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="nci1"; dataset_idx=2 ;;
            2) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "native_fair_v2_joint eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_anchor_boost)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_anchor_boost_per_fold.json}"
        # Compact layout: proteins then reddit-b (2 × folds × seeds).
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "anchor_boost eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_a1g2_micro)
        cfg="configs/tu_errica/sigma-hetero-a1g2-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_a1g2"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_a1g2_micro_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "a1g2_micro eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_a1g2_nci1_micro)
        cfg="configs/tu_errica/sigma-hetero-a1g2-nci1-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_a1g2_ginsage"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_a1g2_nci1_micro_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        fold_idx=$((idx / num_seeds))
        if [ "${fold_idx}" -ge "${num_folds}" ]; then
            log_message "a1g2_nci1_micro eval fold=${fold_idx} out of range"
            exit 1
        fi
        ds_tag="nci1"
        dataset_idx=2
        use_selection=1
        ;;
    sigma_grid_eval_anchor_refine)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_anchor_refine_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        fold_idx=$((idx / num_seeds))
        if [ "${fold_idx}" -ge "${num_folds}" ]; then
            log_message "anchor_refine eval fold=${fold_idx} out of range"
            exit 1
        fi
        ds_tag="proteins"
        dataset_idx=1
        use_selection=1
        ;;
    sigma_grid_eval_nci1_refine)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_nci1_refine_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        fold_idx=$((idx / num_seeds))
        if [ "${fold_idx}" -ge "${num_folds}" ]; then
            log_message "nci1_refine eval fold=${fold_idx} out of range"
            exit 1
        fi
        ds_tag="nci1"
        dataset_idx=2
        use_selection=1
        ;;
    sigma_grid_eval_a0g_pnr)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_a0g"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_a0g_pnr_per_fold.json}"
        # Compact layout: proteins → nci1 → reddit-b (3 × folds × seeds).
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="nci1"; dataset_idx=2 ;;
            2) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "a0g_pnr eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_tiny_pnr)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_hetero"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_tiny_pnr_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="nci1"; dataset_idx=2 ;;
            2) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "tiny_pnr eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
        use_selection=1
        ;;
    sigma_grid_eval_specialist_tiny)
        cfg="configs/tu_errica/sigma-hetero-errica-base.yaml"
        model_key="sigma_hetero"
        model_tag="SiGMA_spec_tiny"
        selection_file="${TU_ERRICA_SELECTION_FILE:-configs/tu_errica/selections/sigma_specialist_tiny_per_fold.json}"
        seed=$((seed_offset + (idx % num_seeds)))
        rest=$((idx / num_seeds))
        fold_idx=$((rest % num_folds))
        local_ds=$((rest / num_folds))
        case "${local_ds}" in
            0) ds_tag="proteins"; dataset_idx=1 ;;
            1) ds_tag="nci1"; dataset_idx=2 ;;
            2) ds_tag="reddit-b"; dataset_idx=5 ;;
            *) log_message "specialist_tiny eval local_ds=${local_ds} out of range"; exit 1 ;;
        esac
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

if [[ "${campaign}" != sigma_grid_select \
    && "${campaign}" != sigma_grid_select_fixed8 \
    && "${campaign}" != sigma_grid_select_fixed8_ungated \
    && "${campaign}" != sigma_grid_select_full64 \
    && "${campaign}" != sigma_grid_select_anchor_boost \
    && "${campaign}" != sigma_grid_select_a1g2_micro \
    && "${campaign}" != sigma_grid_select_a1g2_nci1_micro \
    && "${campaign}" != sigma_grid_select_anchor_refine \
    && "${campaign}" != sigma_grid_select_nci1_refine \
    && "${campaign}" != sigma_grid_select_native_fair \
    && "${campaign}" != sigma_grid_eval_native_fair \
    && "${campaign}" != sigma_grid_select_native_fair_v2 \
    && "${campaign}" != sigma_grid_eval_native_fair_v2 \
    && "${campaign}" != sigma_grid_select_native_fair_v2_l4 \
    && "${campaign}" != sigma_grid_eval_native_fair_v2_joint \
    && "${campaign}" != sigma_grid_select_a0g_pnr \
    && "${campaign}" != sigma_grid_select_tiny_pnr \
    && "${campaign}" != sigma_grid_select_specialist_tiny \
    && "${campaign}" != sigma_grid_eval_specialist_tiny \
    && "${campaign}" != sigma_grid_eval_anchor_boost \
    && "${campaign}" != sigma_grid_eval_a1g2_micro \
    && "${campaign}" != sigma_grid_eval_a0g_pnr \
    && "${campaign}" != sigma_grid_eval_tiny_pnr \
    && "${campaign}" != sigma_grid_eval_a1g2_nci1_micro \
    && "${campaign}" != sigma_grid_eval_anchor_refine \
    && "${campaign}" != sigma_grid_eval_nci1_refine ]]; then
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
    if [[ "${campaign}" == sigma_grid_select \
        || "${campaign}" == sigma_grid_select_fixed8 \
        || "${campaign}" == sigma_grid_select_fixed8_ungated \
        || "${campaign}" == sigma_grid_select_full64 \
        || "${campaign}" == sigma_grid_select_anchor_boost \
        || "${campaign}" == sigma_grid_select_a1g2_micro \
        || "${campaign}" == sigma_grid_select_a1g2_nci1_micro \
        || "${campaign}" == sigma_grid_select_anchor_refine \
        || "${campaign}" == sigma_grid_select_nci1_refine \
        || "${campaign}" == sigma_grid_select_native_fair \
        || "${campaign}" == sigma_grid_select_native_fair_v2 \
        || "${campaign}" == sigma_grid_select_native_fair_v2_l4 \
        || "${campaign}" == sigma_grid_select_a0g_pnr \
        || "${campaign}" == sigma_grid_select_tiny_pnr \
        || "${campaign}" == sigma_grid_select_specialist_tiny ]]; then
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
# anchor_boost / a1g2_*_micro search batch_size explicitly; do not clobber.
if [ "${model_key}" = "sigma_hetero" ] \
    && [[ "${campaign}" != sigma_grid_select_anchor_boost \
        && "${campaign}" != sigma_grid_eval_anchor_boost \
        && "${campaign}" != sigma_grid_select_a1g2_micro \
        && "${campaign}" != sigma_grid_eval_a1g2_micro \
        && "${campaign}" != sigma_grid_select_a1g2_nci1_micro \
        && "${campaign}" != sigma_grid_eval_a1g2_nci1_micro \
        && "${campaign}" != sigma_grid_select_anchor_refine \
        && "${campaign}" != sigma_grid_eval_anchor_refine \
        && "${campaign}" != sigma_grid_select_nci1_refine \
        && "${campaign}" != sigma_grid_eval_nci1_refine \
        && "${campaign}" != sigma_grid_select_native_fair \
        && "${campaign}" != sigma_grid_eval_native_fair \
        && "${campaign}" != sigma_grid_select_native_fair_v2 \
        && "${campaign}" != sigma_grid_eval_native_fair_v2 \
        && "${campaign}" != sigma_grid_select_native_fair_v2_l4 \
        && "${campaign}" != sigma_grid_eval_native_fair_v2_joint \
        && "${campaign}" != sigma_grid_select_a0g_pnr \
        && "${campaign}" != sigma_grid_eval_a0g_pnr \
        && "${campaign}" != sigma_grid_select_tiny_pnr \
        && "${campaign}" != sigma_grid_eval_tiny_pnr \
        && "${campaign}" != sigma_grid_select_specialist_tiny \
        && "${campaign}" != sigma_grid_eval_specialist_tiny ]]; then
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
