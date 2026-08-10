#!/usr/bin/env bash
# =============================================================================
# TU attention-sink campaign (paper TU table + dense-attn feasibility).
#
# Datasets (6 — PyG stats / Lukas paper set):
#   MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
#   (~17 / ~33 / ~39 / ~74 / ~20 / ~430 avg |V|)
#
# Variants (4):
#   0  SiGMA_hetero_gated         a2g4 d_h=4  gate=headwise (attn+MP)
#   1  SiGMA_hetero_ungated_attn  a2g4 d_h=4  gate=none + mp_gate=headwise
#   2  GPS_gated                  a1g1 d_h=8  gate=headwise
#   3  GPS_ungated_attn           a1g1 d_h=8  gate=none + mp_gate=headwise
#
# Layout: 6 ds × 4 variants × seed 2 = 24 tasks
#   task_id = dataset_idx * NUM_VARIANTS + variant_idx + 1
#
# Dense N×N notes:
#   MUTAG–IMDB: full dump OK. COLLAB: dump with small batch. REDDIT: W&B
#   panels use attention_sink_max_nodes (default 512); full dump off by
#   default (set AS_DUMP_ATTN=1 + AS_DUMP_BATCH=1 if you insist).
#
# Submit examples:
#   # Full paper TU × 4 variants:
#   bash bash_interface/cluster/submit_tu_attention_sinks.sh
#   # GPS ungated only (tasks 4,8,12,16,20,24) — cheapest ×uniform vs |V| test:
#   AS_ARRAY=4,8,12,16,20,24 AS_PARALLEL=6 bash bash_interface/cluster/submit_tu_attention_sinks.sh
#   # ENZYMES all variants (old indices were 5-8; now enzymes=ds1 → 5-8 still):
#   AS_ARRAY=5-8 AS_PARALLEL=4 bash bash_interface/cluster/submit_tu_attention_sinks.sh
# =============================================================================

#SBATCH --job-name=tu_attn_sinks
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
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

# Paper TU set (order fixed — do not reorder without updating AS_ARRAY docs).
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

# Per-dataset LR / batch (matches Paper_tu_sigma_homo_hetero social extras).
# Empty AS_BASE_LR / AS_BATCH → use dataset defaults (do not treat "" as set).
_default_lr() {
    case "${ds_tag}" in
        collab) echo "0.01" ;;
        *) echo "0.001" ;;
    esac
}
_default_batch() {
    case "${ds_tag}" in
        collab) echo "32" ;;
        reddit_binary) echo "16" ;;
        *) echo "64" ;;
    esac
}
if [ -n "${AS_BASE_LR:-}" ]; then
    base_lr="${AS_BASE_LR}"
else
    base_lr="$(_default_lr)"
fi
if [ -n "${AS_BATCH:-}" ]; then
    batch_size="${AS_BATCH}"
else
    batch_size="$(_default_batch)"
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
        arch_tag="a2g4_dh4_gated"
        ;;
    1)
        variant="SiGMA_hetero_ungated_attn"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        arch_tag="a2g4_dh4_ungated_attn"
        extra_gate_args+=(gnn.hybrid.gate none gnn.hybrid.mp_gate headwise)
        ;;
    2)
        variant="GPS_gated"
        cfg="${cfg_dir}/gps-a1g1-anchor.yaml"
        arch_tag="a1g1_gated"
        ;;
    3)
        variant="GPS_ungated_attn"
        cfg="${cfg_dir}/gps-a1g1-anchor.yaml"
        arch_tag="a1g1_ungated_attn"
        extra_gate_args+=(gnn.hybrid.gate none gnn.hybrid.mp_gate headwise)
        ;;
    *)
        log_message "bad variant_idx=${variant_idx}"
        exit 1
        ;;
esac

if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="tu_as_${ds_tag}_${variant}"
wandb_name="${wandb_group}_${lr_tag}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_attention_sinks,${ds_tag},${variant},${lr_tag},seed${seed},${arch_tag}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_attention_sinks/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_attention_sinks/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

# REDDIT dense dump is heavy; default dump batch smaller, and skip dump unless forced.
dump_attn="${AS_DUMP_ATTN:-0}"
dump_batch="${AS_DUMP_BATCH:-8}"
if [ "${ds_tag}" = "reddit_binary" ]; then
    dump_batch="${AS_DUMP_BATCH:-1}"
    if [ "${AS_DUMP_REDDIT:-0}" != "1" ] && [ "${dump_attn}" = "1" ]; then
        log_message "REDDIT: skipping full attention dump (set AS_DUMP_REDDIT=1 to force)."
        dump_attn=0
    fi
fi

log_message "AS task ${task_id}/${num_tasks}: ds=${ds_name} variant=${variant} lr=${base_lr} batch=${batch_size} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
variant=${variant}
lr=${base_lr}
lr_tag=${lr_tag}
batch_size=${batch_size}
seed=${seed}
cfg=${cfg}
arch_tag=${arch_tag}
task_id=${task_id}
job=${job_tag}
extra_gate=${extra_gate_args[*]:-}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    optim.base_lr "${base_lr}"
    train.batch_size "${batch_size}"
    out_dir "${run_dir}"
    train.enable_ckpt True
    train.ckpt_best True
    train.ckpt_clean False
    gnn.hybrid.attn_mask full
    gnn.hybrid.log_attention_sinks True
    gnn.hybrid.attention_sink_every "${AS_SINK_EVERY:-50}"
    gnn.hybrid.attention_sink_tau "${AS_SINK_TAU:-1.5}"
    gnn.hybrid.attention_sink_epsilon "${AS_SINK_EPS:-0.3}"
    gnn.hybrid.attention_sink_max_nodes "${AS_SINK_MAX_NODES:-512}"
    gnn.hybrid.attention_sink_save_pt True
    "${extra_gate_args[@]}"
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
    "${extra_args[@]}"

log_message "Training finished. ckpt listing:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/ under ${run_dir}"

if [ "${dump_attn}" = "1" ]; then
    dump_extra=()
    if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
        dump_extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
    fi
    log_message "Dumping attention maps → ${run_dir}/attention_matrices/ (batch=${dump_batch})"
    python scripts/attention_sinks/dump_attention_maps.py \
        --run_dir "${run_dir}" \
        --epoch -1 \
        --batch_size "${dump_batch}" \
        --splits train,val,test \
        --cfg "${cfg}" \
        seed "${seed}" \
        dataset.name "${ds_name}" \
        "${extra_gate_args[@]}" \
        "${dump_extra[@]}"
fi

log_message "Done task ${task_id}"
