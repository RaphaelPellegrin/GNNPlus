#!/usr/bin/env bash
# =============================================================================
# NCI1 SiGMA relaunch — beat plain GCN (80.51±0.71%).
#
# Prior 37434534 best: hetero lr=1e-3 → 79.03±1.19; homo 77.28; lr=1e-2 worse.
# Retry: LR ∈ {5e-4, 2e-3} (bracket prior best 1e-3), max_epoch=2000,
#        schedule_patience=100 (was 50), same a2g4 architecture.
#
# 4 variants × 5 seeds = 20 jobs.
#
# Submit:
#   bash bash_interface/cluster/submit_tu_nci1_sigma_retry.sh
# =============================================================================

#SBATCH --job-name=tu_nci1_sigma
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
num_seeds="${TU_NCI1_NUM_SEEDS:-5}"
max_epoch="${TU_NCI1_MAX_EPOCH:-2000}"
patience="${TU_NCI1_PATIENCE:-100}"
do_gate_dump="${TU_SIGMA_HH_GATE_DUMP:-1}"

num_variants=4
num_tasks="${TU_NCI1_NUM_TASKS:-$((num_variants * num_seeds))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
variant_idx=$((idx / num_seeds))

ds_tag="nci1"
ds_name="NCI1"
cfg_dir="configs/tu_sigma_homo_hetero"

case "${variant_idx}" in
    0)
        family="SiGMA_homo"
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-anchor.yaml"
        base_lr="0.0005"
        lr_tag="lr5e4"
        ;;
    1)
        family="SiGMA_homo"
        variant="SiGMA_homo"
        cfg="${cfg_dir}/sigma-homo-a2g4-anchor.yaml"
        base_lr="0.002"
        lr_tag="lr2e3"
        ;;
    2)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        base_lr="0.0005"
        lr_tag="lr5e4"
        ;;
    3)
        family="SiGMA_hetero"
        variant="SiGMA_hetero"
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        base_lr="0.002"
        lr_tag="lr2e3"
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
wandb_group="tu_hh_${ds_tag}_${variant}_${lr_tag}_ep${max_epoch}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="tu_sigma_homo_hetero,nci1,${variant},${lr_tag},seed${seed},a2g4,ep${max_epoch},nci1_retry"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_ep${max_epoch}_seed${seed}"
else
    run_dir="results/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_ep${max_epoch}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "NCI1 SiGMA retry ${task_id}/${num_tasks}: ${variant} lr=${base_lr} ep=${max_epoch} pat=${patience} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=${family}
variant=${variant}
lr=${base_lr}
lr_tag=${lr_tag}
max_epoch=${max_epoch}
schedule_patience=${patience}
seed=${seed}
cfg=${cfg}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
retry_of=37434534
goal=beat_gcn_80.51
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    optim.base_lr "${base_lr}"
    optim.max_epoch "${max_epoch}"
    optim.schedule_patience "${patience}"
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
fi

log_message "Task ${task_id} complete."
