#!/usr/bin/env bash
# =============================================================================
# Cluster-safe NOP/broadcast summarize over attention_matrices/*.pt
#
# Uses scripts/attention_sinks/summarize_nop_broadcast.py (inlined helpers —
# no GNNPlus / GraphGym import). Safe on CPU login / shared partition.
#
# Task map matches dump/train (24 slots). Skips if attention_matrices/ missing.
# =============================================================================

#SBATCH --job-name=tu_as_mech
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"

task_id=${SLURM_ARRAY_TASK_ID:-1}
seed="${AS_SEED:-2}"

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
num_variants=4
num_tasks=$(( ${#datasets[@]} * num_variants ))

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    echo "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
variant_idx=$((idx % num_variants))
dataset_idx=$((idx / num_variants))
ds_tag="${datasets[$dataset_idx]}"

_default_lr() {
    case "${ds_tag}" in
        collab) echo "0.01" ;;
        *) echo "0.001" ;;
    esac
}
if [ -n "${AS_BASE_LR:-}" ]; then
    base_lr="${AS_BASE_LR}"
else
    base_lr="$(_default_lr)"
fi
if [[ "${base_lr}" == "0.01" ]]; then
    lr_tag="${AS_LR_TAG:-lr01}"
elif [[ "${base_lr}" == "0.001" ]]; then
    lr_tag="${AS_LR_TAG:-lr001}"
else
    lr_tag="${AS_LR_TAG:-lr$(echo "${base_lr}" | tr -d '.')}"
fi

case "${variant_idx}" in
    0) variant="SiGMA_hetero_gated" ;;
    1) variant="SiGMA_hetero_ungated_attn" ;;
    2) variant="GPS_gated" ;;
    3) variant="GPS_ungated_attn" ;;
    *) echo "bad variant_idx"; exit 1 ;;
esac

OUT_ROOT="${GNNPLUS_OUT_DIR:-/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results}/tu_attention_sinks"
run_dir="${OUT_ROOT}/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
mat_dir="${run_dir}/attention_matrices"
analysis_dir="${OUT_ROOT}/analysis"
mkdir -p "${analysis_dir}"
csv="${analysis_dir}/${ds_tag}_${variant}_${lr_tag}_seed${seed}_mech.csv"

if [ ! -d "${mat_dir}" ]; then
    echo "skip ${run_dir}: no attention_matrices/"
    exit 0
fi
n_pt="$(find "${mat_dir}" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
if [ "${n_pt}" = "0" ]; then
    echo "skip ${run_dir}: empty attention_matrices/"
    exit 0
fi

echo "Summarizing ${run_dir} (${n_pt} pts) → ${csv}"
# Prefer test split when many files (COLLAB/PROTEINS).
if [ "${n_pt}" -gt 200 ]; then
    mapfile -t pts < <(find "${mat_dir}" -maxdepth 1 -name '*_test_*.pt' | sort | head -n "${AS_MECH_MAX_TEST:-120}")
    if [ "${#pts[@]}" -eq 0 ]; then
        mapfile -t pts < <(find "${mat_dir}" -maxdepth 1 -name '*.pt' | sort | head -n 120)
    fi
    python scripts/attention_sinks/summarize_nop_broadcast.py \
        --inputs "${pts[@]}" \
        --out-csv "${csv}" \
        --tau "${AS_SINK_TAU:-1.5}"
else
    python scripts/attention_sinks/summarize_nop_broadcast.py \
        --input-dir "${mat_dir}" \
        --out-csv "${csv}" \
        --tau "${AS_SINK_TAU:-1.5}"
fi

echo "Wrote ${csv}"
