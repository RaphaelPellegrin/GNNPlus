#!/usr/bin/env bash
# =============================================================================
# Node+graph gate dump for paper-table SiGMA hetero runs (best LR, seed 2).
#
# Covers bio + social (the bio-only 1–150 dump map does not include
# COLLAB / IMDB / REDDIT). Writes gate_values_per_{graph,node}.pt with edges.
#
# Task map (1-indexed):
#   1 mutag      SiGMA_hetero lr001  (already done — safe to re-run)
#   2 enzymes    SiGMA_hetero lr001
#   3 proteins   SiGMA_hetero lr001
#   4 collab     SiGMA_hetero lr01
#   5 imdb_binary SiGMA_hetero lr001
#   6 reddit_binary SiGMA_hetero lr001
#
# Submit:
#   GATE_DUMP_LEVEL=both bash bash_interface/cluster/submit_dump_tu_hh_paper_node_gates.sh
#   # skip mutag:
#   TU_HH_NODE_ARRAY=2-6 bash bash_interface/cluster/submit_dump_tu_hh_paper_node_gates.sh
# =============================================================================

#SBATCH --job-name=tu_hh_ndump
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
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
epoch="${GATE_DUMP_EPOCH:--1}"
level="${GATE_DUMP_LEVEL:-both}"
cfg="configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml"

# tag  ds_name        lr_tag  seed
targets=(
    "mutag|MUTAG|lr001|2"
    "enzymes|ENZYMES|lr001|2"
    "proteins|PROTEINS|lr001|2"
    "collab|COLLAB|lr01|2"
    "imdb_binary|IMDB-BINARY|lr001|2"
    "reddit_binary|REDDIT-BINARY|lr001|2"
)

n_targets=${#targets[@]}
if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${n_targets}" ]; then
    log_message "task_id=${task_id} out of range (1..${n_targets})"
    exit 1
fi

IFS='|' read -r ds_tag ds_name lr_tag seed <<< "${targets[$((task_id - 1))]}"
variant="SiGMA_hetero"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi

if [ ! -d "${run_dir}/ckpt" ]; then
    log_message "No ckpt/ under ${run_dir}"
    exit 1
fi

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "Paper node dump task ${task_id}/${n_targets}: ${ds_tag} ${variant} ${lr_tag} seed${seed}"
log_message "run_dir=${run_dir} level=${level} epoch=${epoch}"
ls -lh "${run_dir}/ckpt/" | tail -n 5 || true

exec python scripts/gate_viz/dump_per_graph_gates.py \
    --run_dir "${run_dir}" \
    --epoch "${epoch}" \
    --level "${level}" \
    --cfg "${cfg}" \
    seed "${seed}" \
    dataset.name "${ds_name}" \
    "${extra_args[@]}"
