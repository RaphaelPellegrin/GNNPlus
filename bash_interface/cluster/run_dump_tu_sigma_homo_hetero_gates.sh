#!/usr/bin/env bash
# =============================================================================
# Offline re-dump of SiGMA gates for TU homo/hetero runs.
#
# Same task map as run_tu_sigma_homo_hetero.sh (150 tasks). GCN tasks no-op.
# Use after training if gate_values_per_graph.pt is missing, or to re-dump.
#
# Env:
#   GATE_DUMP_LEVEL=graph|node|both   (default: graph)
#   GATE_DUMP_EPOCH=-1                (default: latest ckpt)
#
# Node dumps (bands + drawings) need level=both|node and latest dump script
# (writes edge_index into gate_values_per_node.pt).
#
# Submit:
#   bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh
#   GATE_DUMP_LEVEL=both bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh
# =============================================================================

#SBATCH --job-name=tu_hh_gdump
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
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

task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${TU_SIGMA_HH_NUM_SEEDS:-5}"
seed_offset="${TU_SIGMA_HH_SEED_OFFSET:-0}"
epoch="${GATE_DUMP_EPOCH:--1}"
level="${GATE_DUMP_LEVEL:-graph}"

datasets=(mutag enzymes proteins dd nci1 triangles)
dataset_names=(MUTAG ENZYMES PROTEINS DD NCI1 TRIANGLES)
num_datasets=${#datasets[@]}
num_variants="${TU_SIGMA_HH_NUM_VARIANTS:-5}"
num_tasks="${TU_SIGMA_HH_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
dataset_idx=$((rest / num_variants))

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
cfg_dir="configs/tu_sigma_homo_hetero"

case "${variant_idx}" in
    0)
        log_message "Task ${task_id}: GCN — no hybrid gates; skipping."
        exit 0
        ;;
    1)
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-anchor.yaml"
        lr_tag="lr001"
        ;;
    2)
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-anchor.yaml"
        lr_tag="lr01"
        ;;
    3)
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        lr_tag="lr001"
        ;;
    4)
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        lr_tag="lr01"
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi

if [ ! -d "${run_dir}/ckpt" ]; then
    log_message "No ckpt/ under ${run_dir}"
    exit 1
fi

out_pt="${run_dir}/gate_values_per_graph.pt"
extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

log_message "Dump TU HH gates: ds=${ds_name} variant=${variant} lr=${lr_tag} seed=${seed}"
log_message "run_dir=${run_dir} epoch=${epoch}"
ls -lh "${run_dir}/ckpt/" | tail -n 5 || true

exec python scripts/gate_viz/dump_per_graph_gates.py \
    --run_dir "${run_dir}" \
    --epoch "${epoch}" \
    --level "${level}" \
    --out "${out_pt}" \
    --cfg "${cfg}" \
    seed "${seed}" \
    dataset.name "${ds_name}" \
    "${extra_args[@]}"
