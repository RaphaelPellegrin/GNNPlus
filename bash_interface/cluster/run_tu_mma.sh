#!/usr/bin/env bash
# =============================================================================
# TU Tables 17/18 — MMA (Mixture of Message-Passing and Attention).
#
# MMA = SiGMA hetero a2g4 with gates removed (gnn.hybrid.gate=none). Same recipe
# as Tab.17/18 SiGMA hetero: L=12, H=64, random 50/25/25, up to 1000 epochs,
# reduce_on_plateau, LR ∈ {1e-3, 1e-2}, 5 seeds. No other HPs are tuned.
#
# Tables:
#   0  Tab.17  d_h=16  configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-anchor.yaml
#   1  Tab.18  d_h=4   configs/tu_sigma_homo_hetero/sigma-hetero-a2g4-matched-anchor.yaml
#
# Datasets: MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY
# Variants (per table × dataset):
#   0  lr=0.001
#   1  lr=0.01
#
# Layout: 2 tables × 6 ds × 2 LR × 5 seeds = 120
#   task_id = ((table_idx * NUM_DATASETS + ds_idx) * NUM_VARIANTS + var_idx)
#             * NUM_SEEDS + seed + 1
#
# W&B (project GNNPlus):
#   Tab.17 → tu_hh_<ds>_MMA_{lr001,lr01}
#   Tab.18 → tu_1x_<ds>_MMA_{lr001,lr01}
# (Earlier paper runs used the alias SiGMA_ungated for the same recipe.)
#
# Submit:
#   bash bash_interface/cluster/submit_tu_mma.sh
# =============================================================================

#SBATCH --job-name=tu_mma
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --output=logs_gnnplus/%x_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task ${SLURM_ARRAY_TASK_ID:-local}] $1"
}

# --- Environment (FASRC). Prefer lightweight env: do not pip install -e if this
# checkout sits next to another GNNPlus worktree sharing the same conda env.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.gnnplus_env" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/.gnnplus_env"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    log_message "WARNING: WANDB_API_KEY is not set; W&B logging will fail"
fi
_wandb_tmp="${TMPDIR:-/tmp}/wandb_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}_${SLURM_ARRAY_TASK_ID:-local}"
export WANDB_DIR="${_wandb_tmp}"
export WANDB_CACHE_DIR="${_wandb_tmp}/.cache"
export WANDB_DISABLE_CODE=true
mkdir -p "${_wandb_tmp}" logs_gnnplus
export PYTHONNOUSERSITE=1
if command -v module &> /dev/null; then
    module load cuda/12.9.1-fasrc01 2>/dev/null || module load cuda 2>/dev/null || true
fi
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs}"
ENV_NAME="${ENV_NAME:-gnnplus}"
if [ -d "${CONDA_ENVS_PATH}/${ENV_NAME}/bin" ]; then
    export PATH="${CONDA_ENVS_PATH}/${ENV_NAME}/bin:${PATH}"
else
    log_message "ERROR: conda env not found at ${CONDA_ENVS_PATH}/${ENV_NAME}"
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python -c "import GNNPlus.network.hybrid_gnn" || {
    log_message "ERROR: cannot import GNNPlus.network.hybrid_gnn from ${REPO_ROOT}"
    exit 1
}

# --- Task decoding ---
task_id=${SLURM_ARRAY_TASK_ID:-1}
num_seeds="${TU_MMA_NUM_SEEDS:-5}"
num_variants=2
datasets=(mutag enzymes proteins collab imdb_binary reddit_binary)
dataset_names=(MUTAG ENZYMES PROTEINS COLLAB IMDB-BINARY REDDIT-BINARY)
batch_sizes=(64 64 64 32 64 16)
num_datasets=${#datasets[@]}
num_tasks=$((2 * num_datasets * num_variants * num_seeds))

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
variant_idx=$((rest % num_variants))
rest=$((rest / num_variants))
dataset_idx=$((rest % num_datasets))
table_idx=$((rest / num_datasets))

ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
batch_size="${batch_sizes[$dataset_idx]}"

cfg_dir="configs/tu_sigma_homo_hetero"
case "${table_idx}" in
    0)
        table_tag="t17"
        wandb_prefix="tu_hh"
        d_h=16
        cfg="${cfg_dir}/sigma-hetero-a2g4-anchor.yaml"
        ;;
    1)
        table_tag="t18"
        wandb_prefix="tu_1x"
        d_h=4
        cfg="${cfg_dir}/sigma-hetero-a2g4-matched-anchor.yaml"
        ;;
    *)
        log_message "bad table_idx=${table_idx}"
        exit 1
        ;;
esac
case "${variant_idx}" in
    0) base_lr="0.001"; lr_tag="lr001" ;;
    1) base_lr="0.01";  lr_tag="lr01" ;;
    *) log_message "bad variant_idx=${variant_idx}"; exit 1 ;;
esac

variant="MMA"
if [ ! -f "${cfg}" ]; then
    log_message "Config not found: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="${wandb_prefix}_${ds_tag}_${variant}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
export WANDB_EXTRA_TAGS="tu_mma,${table_tag},${ds_tag},${variant},SiGMA_ungated,${lr_tag},seed${seed},a2g4_dh${d_h},gate_none"

out_root="${GNNPLUS_OUT_DIR:-${REPO_ROOT}/results}/tu_mma"
run_dir="${out_root}/${table_tag}_${ds_tag}_${variant}_${lr_tag}_seed${seed}"
mkdir -p "${run_dir}"

log_message "MMA TU ${task_id}/${num_tasks}: table=${table_tag} ds=${ds_name} d_h=${d_h} lr=${base_lr} batch=${batch_size} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
    dataset.name "${ds_name}"
    optim.base_lr "${base_lr}"
    train.batch_size "${batch_size}"
    out_dir "${run_dir}"
    gnn.hybrid.gate none
    gnn.hybrid.mp_gate none
)
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

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

log_message "Task ${task_id} complete."
