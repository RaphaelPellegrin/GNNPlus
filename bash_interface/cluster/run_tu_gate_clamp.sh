#!/usr/bin/env bash
# =============================================================================
# Inference gate-clamp eval on gated TU SiGMA hetero checkpoints.
#
# Tasks map to reported best-LR Tab.17 (1–30) then Tab.18 (31–60):
#   6 datasets × 5 seeds = 30 per table.
#
# Requires cluster ckpts under:
#   $GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_SiGMA_hetero_<lr>_seed<s>/
#   $GNNPLUS_OUT_DIR/tu_sigma_1x_gcn/<ds>_SiGMA_hetero_<lr>_seed<s>/
#
# Submit:
#   bash bash_interface/cluster/submit_tu_gate_clamp.sh
# =============================================================================

#SBATCH --job-name=tu_gate_clamp
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
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
num_seeds=5
num_datasets=6
# 2 tables × 6 ds × 5 seeds = 60
num_tasks=$((2 * num_datasets * num_seeds))

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
dataset_idx=$((rest % num_datasets))
table_idx=$((rest / num_datasets))

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
# Tab.17 reported hetero LRs
lr17=(lr001 lr001 lr001 lr01 lr001 lr001)
# Tab.18 reported hetero LRs
lr18=(lr001 lr001 lr001 lr001 lr01 lr001)

ds_tag="${datasets[$dataset_idx]}"

if [ "${table_idx}" -eq 0 ]; then
    table_tag="t17"
    results_root="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero"
    cfg="configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml"
    lr_tag="${lr17[$dataset_idx]}"
else
    table_tag="t18"
    results_root="${GNNPLUS_OUT_DIR}/tu_sigma_1x_gcn"
    cfg="configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-matched-anchor.yaml"
    lr_tag="${lr18[$dataset_idx]}"
fi

run_dir="${results_root}/${ds_tag}_SiGMA_hetero_${lr_tag}_seed${seed}"
out_csv="results/gate_clamp/${table_tag}_${ds_tag}_${lr_tag}_seed${seed}.csv"
mkdir -p results/gate_clamp

if [ ! -d "${run_dir}/ckpt" ] || ! ls "${run_dir}/ckpt"/*.ckpt >/dev/null 2>&1; then
    log_message "ERROR: missing ckpt under ${run_dir}"
    ls -la "${run_dir}" 2>/dev/null || true
    exit 1
fi

log_message "Gate clamp ${task_id}/${num_tasks}: ${table_tag} ${ds_tag} ${lr_tag} seed=${seed}"
log_message "run_dir=${run_dir}"

extra=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra+=(--dataset-dir "${GNNPLUS_DATASET_DIR}")
fi

python scripts/gate_viz/eval_gate_clamp.py \
    --run_dir "${run_dir}" \
    --cfg "${cfg}" \
    --out-csv "${out_csv}" \
    --paired-ttest \
    "${extra[@]}"

log_message "Wrote ${out_csv}"
log_message "Task ${task_id} complete."
