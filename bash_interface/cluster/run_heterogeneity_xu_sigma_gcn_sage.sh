#!/usr/bin/env bash
# =============================================================================
# Xu-recipe SiGMA GCN+SAGE only: a0g2 / a1g2 × gated / ungated × MUTAG/ENZYMES × 5 seeds.
#
# Task map (seed fastest, then dataset, then variant):
#   variants = a0g2_gated, a0g2_ungated, a1g2_gated, a1g2_ungated
#   within each variant: 1–5 mutag seeds 0–4, 6–10 enzymes seeds 0–4
#   total = 4 × 10 = 40 tasks
#
# Submit:
#   bash bash_interface/cluster/submit_heterogeneity_xu_sigma_gcn_sage.sh
# =============================================================================

#SBATCH --job-name=xu_sigma_gcs
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
num_seeds="${XU_GCS_NUM_SEEDS:-5}"
seed_offset="${XU_GCS_SEED_OFFSET:-0}"
do_gate_dump="${XU_GCS_GATE_DUMP:-1}"

variants=(a0g2_gated a0g2_ungated a1g2_gated a1g2_ungated)
cfg_suffixes=(a0g2-gated a0g2-ungated a1g2-gated a1g2-ungated)
datasets=(mutag enzymes)
dataset_names=(MUTAG ENZYMES)

num_variants=${#variants[@]}
num_datasets=${#datasets[@]}
jobs_per_variant=$((num_datasets * num_seeds))
num_tasks="${XU_GCS_NUM_TASKS:-$((num_variants * jobs_per_variant))}"

if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$num_tasks" ]; then
    log_message "task_id=${task_id} out of range (1..${num_tasks})"
    exit 1
fi

idx=$((task_id - 1))
variant_idx=$((idx / jobs_per_variant))
within=$((idx % jobs_per_variant))
seed=$((seed_offset + (within % num_seeds)))
dataset_idx=$((within / num_seeds))

variant="${variants[$variant_idx]}"
cfg_sfx="${cfg_suffixes[$variant_idx]}"
ds_tag="${datasets[$dataset_idx]}"
ds_name="${dataset_names[$dataset_idx]}"
cfg="configs/heterogeneity/powerful_gnns/sigma-gcn-sage-${cfg_sfx}-ckpt.yaml"

if [ ! -f "${cfg}" ]; then
    log_message "Missing config: ${cfg}"
    exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="xu_sigma_gcs_${variant}_${ds_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="xu_sigma_gcn_sage,${variant},${ds_tag},seed${seed},ckpt"

out_subdir="heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage"
# Match join_tu_gate_operator_preference.py: <ds>_SiGMA_hetero_<lr-tag>_seed<s>
run_name="${ds_tag}_SiGMA_hetero_${variant}_seed${seed}"
if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    run_dir="${GNNPLUS_OUT_DIR}/${out_subdir}/${run_name}"
else
    run_dir="results/${out_subdir}/${run_name}"
fi
mkdir -p "${run_dir}"

log_message "Xu SiGMA GCN+SAGE task ${task_id}/${num_tasks}: variant=${variant} ds=${ds_name} seed=${seed}"
log_message "cfg=${cfg}"
log_message "run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
dataset=${ds_name}
ds_tag=${ds_tag}
family=SiGMA_${variant}
recipe=xu_iclr2019
gnn_types=GCN,SAGE
variant=${variant}
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
