#!/usr/bin/env bash
# =============================================================================
# TU paper table — param-matched SiGMA (~1× GCN) + GPS-style a1g1.
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
# Variants (6):
#   0  SiGMA_homo   a2g4 d_h=4   lr=0.001
#   1  SiGMA_homo   a2g4 d_h=4   lr=0.01
#   2  SiGMA_hetero a2g4 d_h=4   lr=0.001
#   3  SiGMA_hetero a2g4 d_h=4   lr=0.01
#   4  GPS          a1g1 GATEDGCN+attn d_h=8  lr=0.001
#   5  GPS          a1g1 GATEDGCN+attn d_h=8  lr=0.01
#
# Layout: 6 datasets × 6 variants × 5 seeds = 180
#   task_id = ((dataset_idx * NUM_VARIANTS) + variant_idx) * NUM_SEEDS + seed + 1
#
# W&B groups: tu_1x_<ds>_{SiGMA_homo,SiGMA_hetero,GPS}_{lr001,lr01}
# Out: $GNNPLUS_OUT_DIR/tu_sigma_1x_gcn/<ds>_<variant>_<lr>_seed<s>/
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_1x_gcn.sh
# =============================================================================

#SBATCH --job-name=tu_1x_gcn
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
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
num_seeds="${TU_1X_NUM_SEEDS:-5}"
do_gate_dump="${TU_1X_GATE_DUMP:-1}"

datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
declare -A batch_for=(
    [mutag]="${TU_1X_BATCH_DEFAULT:-64}"
    [enzymes]="${TU_1X_BATCH_DEFAULT:-64}"
    [proteins]="${TU_1X_BATCH_DEFAULT:-64}"
    [collab]="${TU_1X_BATCH_COLLAB:-32}"
    [imdb_binary]="${TU_1X_BATCH_IMDB:-64}"
    [reddit_binary]="${TU_1X_BATCH_REDDIT:-16}"
)

num_datasets=${#datasets[@]}
num_variants="${TU_1X_NUM_VARIANTS:-6}"
num_tasks="${TU_1X_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
dataset_idx=$((rest / num_variants))

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
batch_size="${batch_for[$ds_tag]}"

cfg_dir="configs/tu_sigma_homo_hetero"
case "${variant_idx}" in
    0)
        family="SiGMA_homo"
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-matched-anchor.yaml"
        base_lr="0.001"
        lr_tag="lr001"
        arch_tag="a2g4_dh4_matched"
        ;;
    1)
        family="SiGMA_homo"
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-matched-anchor.yaml"
        base_lr="0.01"
        lr_tag="lr01"
        arch_tag="a2g4_dh4_matched"
        ;;
    2)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        base_lr="0.001"
        lr_tag="lr001"
        arch_tag="a2g4_dh4_matched"
        ;;
    3)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        base_lr="0.01"
        lr_tag="lr01"
        arch_tag="a2g4_dh4_matched"
        ;;
    4)
        family="GPS"
        variant="GPS"
        cfg="${cfg_dir}/gps-a1g1-anchor.yaml"
        base_lr="0.001"
        lr_tag="lr001"
        arch_tag="a1g1_gatedgcn_dh8"
        ;;
    5)
        family="GPS"
        variant="GPS"
        cfg="${cfg_dir}/gps-a1g1-anchor.yaml"
        base_lr="0.01"
        lr_tag="lr01"
        arch_tag="a1g1_gatedgcn_dh8"
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
wandb_group="tu_1x_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_sigma_1x_gcn,${ds_tag},${variant},${lr_tag},seed${seed},${arch_tag}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_1x_gcn/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_1x_gcn/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU 1x task ${task_id}/${num_tasks}: ds=${ds_name} family=${family} lr=${base_lr} batch=${batch_size} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=${family}
variant=${variant}
lr=${base_lr}
lr_tag=${lr_tag}
batch_size=${batch_size}
seed=${seed}
cfg=${cfg}
arch_tag=${arch_tag}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
wandb_name=${wandb_name}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    optim.base_lr "${base_lr}"
    train.batch_size "${batch_size}"
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
    "${extra_args[@]}"

log_message "Training finished. ckpt listing:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/ under ${run_dir}"

# Gate dump for hybrid models (SiGMA + GPS a1g1).
if [ "${do_gate_dump}" = "1" ]; then
    if [ ! -d "${run_dir}/ckpt" ]; then
        log_message "ERROR: expected ckpt/ for gate dump but missing"
        exit 1
    fi
    out_pt="${run_dir}/gate_values_per_graph.pt"
    dump_extra=()
    if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
        dump_extra+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
    fi
    log_message "Dumping per-graph gates → ${out_pt}"
    python scripts/gate_viz/dump_per_graph_gates.py \
        --run_dir "${run_dir}" \
        --epoch -1 \
        --out "${out_pt}" \
        --cfg "${cfg}" \
        seed "${seed}" \
        dataset.name "${ds_name}" \
        "${dump_extra[@]}"
    log_message "Gate dump done: ${out_pt}"
else
    log_message "Skipping gate dump (TU_1X_GATE_DUMP=${do_gate_dump})"
fi

log_message "Task ${task_id} complete."
