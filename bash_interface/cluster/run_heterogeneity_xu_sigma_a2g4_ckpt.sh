#!/usr/bin/env bash
# =============================================================================
# Xu-recipe SiGMA hetero a2g4 (MUTAG / ENZYMES) × 5 seeds, with ckpt + gate dump.
#
# Task map (seed fastest):
#   1–5  mutag seeds 0–4
#   6–10 enzymes seeds 0–4
#
# Submit:
#   bash bash_interface/cluster/submit_heterogeneity_xu_sigma_a2g4_ckpt.sh
# =============================================================================

#SBATCH --job-name=xu_sigma_a2g4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
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
num_seeds="${XU_SIGMA_NUM_SEEDS:-5}"
seed_offset="${XU_SIGMA_SEED_OFFSET:-0}"
do_gate_dump="${XU_SIGMA_GATE_DUMP:-1}"

datasets=(mutag enzymes)
dataset_names=(MUTAG ENZYMES)
num_datasets=${#datasets[@]}
num_tasks="${XU_SIGMA_NUM_TASKS:-$((num_datasets * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((seed_offset + (idx % num_seeds)))
dataset_idx=$((idx / num_seeds))
ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
cfg="configs/heterogeneity/powerful_gnns/${ds_tag}-sigma-a2g4-ckpt.yaml"

if [ ! -f "${cfg}" ]; then
    log_message "Missing config: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="xu_sigma_a2g4_${ds_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="xu_sigma_a2g4,${ds_tag},SiGMA_hetero,seed${seed},ckpt"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/${ds_tag}_SiGMA_hetero_xu_seed${seed}"
else
    run_dir="results/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4/${ds_tag}_SiGMA_hetero_xu_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "Xu SiGMA a2g4 task ${task_id}/${num_tasks}: ds=${ds_name} seed=${seed}"
log_message "cfg=${cfg}"
log_message "run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=SiGMA_hetero
recipe=xu_iclr2019
gnn_types=GCN,GIN,SAGE,GAT
layers_mp=4
seed=${seed}
cfg=${cfg}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
wandb_name=${wandb_name}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
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
fi

log_message "Task ${task_id} complete."
