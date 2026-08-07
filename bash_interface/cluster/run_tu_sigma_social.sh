#!/usr/bin/env bash
# =============================================================================
# TU social datasets from PyG TUDataset stats table (Lukas / Table 13 set):
#   COLLAB, IMDB-BINARY, REDDIT-BINARY
# (MUTAG / ENZYMES / PROTEINS already covered by 37434534; skip NCI1/DD/TRIANGLES.)
#
# Same families as Paper_tu_sigma_homo_hetero:
#   GCN | SiGMA homo a2g4 (GCN×4) | SiGMA hetero a2g4 (GCN,GIN,SAGE,GAT)
#   SiGMA LRs ∈ {1e-3, 1e-2}; 5 seeds; Constant node features (0-feat graphs).
#
# Layout: 3 datasets × 5 variants × 5 seeds = 75
#   task_id = ((dataset_idx * NUM_VARIANTS) + variant_idx) * NUM_SEEDS + seed + 1
#
# Submit:
#   bash bash_interface/cluster/submit_tu_sigma_social.sh
# =============================================================================

#SBATCH --job-name=tu_sigma_soc
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
num_seeds="${TU_SOC_NUM_SEEDS:-5}"
do_gate_dump="${TU_SIGMA_HH_GATE_DUMP:-1}"

# Short tags → PyG names (https://pytorch-geometric.readthedocs.io/.../TUDataset.html)
datasets=(collab imdb_binary reddit_binary)
dataset_names=(COLLAB IMDB-BINARY REDDIT-BINARY)
# Larger / denser graphs → smaller default batches (override via env if needed).
declare -A batch_for=(
    [collab]="${TU_SOC_BATCH_COLLAB:-32}"
    [imdb_binary]="${TU_SOC_BATCH_IMDB:-64}"
    [reddit_binary]="${TU_SOC_BATCH_REDDIT:-16}"
)

num_datasets=${#datasets[@]}
num_variants="${TU_SOC_NUM_VARIANTS:-5}"
num_tasks="${TU_SOC_NUM_TASKS:-$((num_datasets * num_variants * num_seeds))}"

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
wandb_tags="tu_sigma_homo_hetero,${ds_tag},${variant},${lr_tag},seed${seed},a2g4,tu_social,pyg_tudataset"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
else
    run_dir="results/tu_sigma_homo_hetero/${ds_tag}_${variant}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "TU social ${task_id}/${num_tasks}: ds=${ds_name} ${variant} lr=${base_lr} batch=${batch_size} seed=${seed}"
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
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
node_features=Constant
source=pyg_tudataset_stats_table
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
fi

log_message "Task ${task_id} complete."
