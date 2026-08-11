#!/usr/bin/env bash
# =============================================================================
# Offline attention dump for existing TU AS run dirs (no retrain).
#
# Same task map as run_tu_attention_sinks.sh (6 ds × 4 variants = 24):
#   task_id = dataset_idx * 4 + variant_idx + 1
#   variants: 0 SiGMA gated · 1 SiGMA ungated · 2 GPS gated · 3 GPS ungated
#
# Expects trained ckpts under:
#   $GNNPLUS_OUT_DIR/tu_attention_sinks/<ds>_<variant>_<lr_tag>_seed2/
#
# Examples:
#   # SiGMA gated+ungated on MUTAG…IMDB (skip REDDIT):
#   AS_DUMP_ARRAY=1,2,5,6,9,10,13,14,17,18 bash bash_interface/cluster/submit_dump_tu_attention_sinks.sh
#   # COLLAB GPS retrain dumps after lr001 jobs finish: tasks 15,16 with AS_LR_TAG=lr001
# =============================================================================

#SBATCH --job-name=tu_as_dump
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
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
seed="${AS_SEED:-2}"

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
num_datasets=${#datasets[@]}
num_variants="${AS_NUM_VARIANTS:-4}"
num_tasks="${AS_NUM_TASKS:-$((num_datasets * num_variants))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
variant_idx=$((idx % num_variants))
dataset_idx=$((idx / num_variants))

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"

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

cfg_dir="configs/tu_sigma_homo_hetero"
extra_gate_args=()

case "${variant_idx}" in
    0)
        variant="SiGMA_hetero_gated"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        ;;
    1)
        variant="SiGMA_hetero_ungated_attn"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        extra_gate_args+=(gnn.hybrid.gate none gnn.hybrid.mp_gate headwise)
        ;;
    2)
        variant="GPS_gated"
        cfg="${cfg_dir}/gps-a1g1-anchor.yaml"
        ;;
    3)
        variant="GPS_ungated_attn"
        cfg="${cfg_dir}/gps-a1g1-anchor.yaml"
        extra_gate_args+=(gnn.hybrid.gate none gnn.hybrid.mp_gate headwise)
        ;;
    4)
        variant="vanilla_full_attn"
        cfg="${cfg_dir}/vanilla-full-attn-a4g0-anchor.yaml"
        extra_gate_args+=(gnn.hybrid.gate none gnn.hybrid.mp_gate none gnn.hybrid.attn_mask full)
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_attention_sinks/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_attention_sinks/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi

dump_batch="${AS_DUMP_BATCH:-8}"
if [ "${ds_tag}" = "collab" ]; then
    dump_batch="${AS_DUMP_BATCH:-4}"
fi
if [ "${ds_tag}" = "reddit_binary" ]; then
    dump_batch="${AS_DUMP_BATCH:-1}"
    if [ "${AS_DUMP_REDDIT:-0}" != "1" ]; then
        log_message "REDDIT: skipping dump (set AS_DUMP_REDDIT=1 to force)."
        exit 0
    fi
fi

if [ ! -d "${run_dir}/ckpt" ]; then
    log_message "ERROR: no ckpt/ under ${run_dir}"
    exit 1
fi
if [ ! -f "${cfg}" ]; then
    log_message "ERROR: config not found: ${cfg}"
    exit 1
fi

log_message "Dump task ${task_id}: ${ds_name} ${variant} ${lr_tag} → ${run_dir}/attention_matrices/ (batch=${dump_batch})"

dump_extra=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    dump_extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

python scripts/attention_sinks/dump_attention_maps.py \
    --run_dir "${run_dir}" \
    --epoch -1 \
    --batch_size "${dump_batch}" \
    --splits "$(echo "${AS_DUMP_SPLITS:-train+val+test}" | tr '+;' ',')" \
    --cfg "${cfg}" \
    seed "${seed}" \
    dataset.name "${ds_name}" \
    "${extra_gate_args[@]}" \
    "${dump_extra[@]}"

n_pt="$(find "${run_dir}/attention_matrices" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
log_message "Done dump task ${task_id}: ${n_pt} .pt files"
