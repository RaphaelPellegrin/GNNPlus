#!/usr/bin/env bash
# =============================================================================
# TU datasets — GCN vs SiGMA(homo a2g4) vs SiGMA(hetero a2g4).
#
# Layout (default 6 datasets × 5 variants × 5 seeds = 150):
#   task_id = ((dataset_idx * NUM_VARIANTS) + variant_idx) * NUM_SEEDS + seed + 1
#
# Datasets: MUTAG, ENZYMES, PROTEINS, DD, NCI1, TRIANGLES
# Variants:
#   0  GCN                 lr=0.001
#   1  SiGMA_homo          a2g4 GCN×4          lr=0.001
#   2  SiGMA_homo          a2g4 GCN×4          lr=0.01
#   3  SiGMA_hetero        a2g4 GCN,GIN,SAGE,GAT lr=0.001
#   4  SiGMA_hetero        a2g4 GCN,GIN,SAGE,GAT lr=0.01
#
# Saves best-val checkpoint under out_dir/ckpt/. For SiGMA variants, also dumps
# per-graph / per-layer / per-head gates → out_dir/gate_values_per_graph.pt
# (same format as scripts/gate_viz/dump_per_graph_gates.py).
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_homo_hetero.sh
# =============================================================================

#SBATCH --job-name=tu_sigma_hh
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
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
num_seeds="${TU_SIGMA_HH_NUM_SEEDS:-5}"
seed_offset="${TU_SIGMA_HH_SEED_OFFSET:-0}"
do_gate_dump="${TU_SIGMA_HH_GATE_DUMP:-1}"

# Dataset short tags → PyG TUDataset names
datasets=(mutag enzymes proteins dd nci1 triangles)
dataset_names=(MUTAG ENZYMES PROTEINS DD NCI1 TRIANGLES)
num_datasets=${#datasets[@]}

# variant_idx → (family, cfg, lr, wandb suffix)
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
        family="GCN"
        variant="GCN"
        cfg="${cfg_dir}/gcn-anchor.yaml"
        base_lr="0.001"
        lr_tag="lr001"
        ;;
    1)
        family="SiGMA_homo"
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-anchor.yaml"
        base_lr="0.001"
        lr_tag="lr001"
        ;;
    2)
        family="SiGMA_homo"
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-anchor.yaml"
        base_lr="0.01"
        lr_tag="lr01"
        ;;
    3)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        base_lr="0.001"
        lr_tag="lr001"
        ;;
    4)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        base_lr="0.01"
        lr_tag="lr01"
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
wandb_group="tu_hh_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_sigma_homo_hetero,${ds_tag},${variant},${lr_tag},seed${seed},a2g4"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU homo/hetero task ${task_id}/${num_tasks}: ds=${ds_name} family=${family} variant=${variant} lr=${base_lr} seed=${seed}"
log_message "cfg=${cfg}"
log_message "run_dir=${run_dir}"

# Persist launch metadata for later gate re-dumps / paper bookkeeping.
cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=${family}
variant=${variant}
lr=${base_lr}
lr_tag=${lr_tag}
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
    optim.base_lr "${base_lr}"
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

# Per-graph gate dump (SiGMA only — GCN has no hybrid gates).
if [ "${family}" != "GCN" ] && [ "${do_gate_dump}" = "1" ]; then
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
    log_message "Skipping gate dump (family=${family}, TU_SIGMA_HH_GATE_DUMP=${do_gate_dump})"
fi

log_message "Task ${task_id} complete."
