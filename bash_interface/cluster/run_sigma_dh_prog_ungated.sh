#!/usr/bin/env bash
# =============================================================================
# Main-text (Tab. 3/4) SiGMA: progressive d_h × gated vs ungated.
#
# Start from paper bestmodel recipes (dh_matched anchors), shrink d_h toward
# the Appendix H regime, train BOTH headwise-gated and fully ungated
# (gate=none). Same LR grid as Tab. 17/18: {1e-3, 1e-2} × 5 seeds.
#
# Families (tightest / fastest first):
#   PATTERN   a2g2 GRIT VN4  H90   d_h ∈ {16, 8, 4, 2, 1}
#   CLUSTER   a1g1 GATEDGCN  H56   d_h ∈ {24, 12, 8, 4, 1}
#   MNIST     a2g2 GATEDGCN  H60   d_h ∈ {37, 16, 8, 4, 1}
#   Pep-func  a1g2 GCN×2     H275  d_h ∈ {23, 12, 8, 4, 1}
#
# Layout (400 tasks):
#   task_id = ((fam_dh_idx * NUM_GATES + gate_idx) * NUM_LRS + lr_idx)
#             * NUM_SEEDS + seed + 1
#   fam_dh_idx ∈ [0, 19]  (4 families × 5 widths)
#   gate_idx   0=gated (yaml gate)  1=ungated (gate=none)
#   lr_idx     0→0.001  1→0.01
#
# W&B: paper_sigma_dh_prog_<fam>_dh<k>_{gated,ungated}_{lr001,lr01}
# Out: $GNNPLUS_OUT_DIR/sigma_dh_prog/<fam>_dh<k>_<gated|ungated>_<lr>_seed<s>/
#
# Submit:
#   bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh
# =============================================================================

#SBATCH --job-name=sigma_dh_prog
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
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
num_seeds="${SIGMA_DH_PROG_NUM_SEEDS:-5}"
num_lrs="${SIGMA_DH_PROG_NUM_LRS:-2}"
num_gates="${SIGMA_DH_PROG_NUM_GATES:-2}"

# tag|cfg|d_h  — progressive ladders (paper-scale → App. H extreme)
families=(
  "pattern|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|16"
  "pattern|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|8"
  "pattern|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|4"
  "pattern|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|2"
  "pattern|configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml|1"
  "cluster|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|24"
  "cluster|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|12"
  "cluster|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|8"
  "cluster|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|4"
  "cluster|configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml|1"
  "mnist|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|37"
  "mnist|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|16"
  "mnist|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|8"
  "mnist|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|4"
  "mnist|configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml|1"
  "pepfunc|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|23"
  "pepfunc|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|12"
  "pepfunc|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|8"
  "pepfunc|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|4"
  "pepfunc|configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml|1"
)

num_fam_dh=${#families[@]}
num_tasks="${SIGMA_DH_PROG_NUM_TASKS:-$((num_fam_dh * num_gates * num_lrs * num_seeds))}"

if [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${num_tasks}" ]; then
  log_message "task_id=${task_id} out of range (1..${num_tasks})"
  exit 1
fi

idx=$((task_id - 1))
seed=$((idx % num_seeds))
rest=$((idx / num_seeds))
lr_idx=$((rest % num_lrs))
rest=$((rest / num_lrs))
gate_idx=$((rest % num_gates))
fam_dh_idx=$((rest / num_gates))

case "${lr_idx}" in
  0) base_lr="0.001"; lr_tag="lr001" ;;
  1) base_lr="0.01";  lr_tag="lr01" ;;
  *) log_message "bad lr_idx=${lr_idx}"; exit 1 ;;
esac

IFS='|' read -r fam_tag cfg d_h <<< "${families[$fam_dh_idx]}"
dh_tag="dh${d_h}"

gate_override=()
case "${gate_idx}" in
  0)
    gate_tag="gated"
    variant="SiGMA"
    ;;
  1)
    gate_tag="ungated"
    variant="SiGMA_ungated"
    gate_override+=(gnn.hybrid.gate none)
    ;;
  *)
    log_message "bad gate_idx=${gate_idx}"
    exit 1
    ;;
esac

if [ ! -f "${cfg}" ]; then
  log_message "Config not found: ${cfg}"
  exit 1
fi

job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
wandb_group="paper_sigma_dh_prog_${fam_tag}_${dh_tag}_${gate_tag}_${lr_tag}"
wandb_name="${wandb_group}_seed${seed}_job${job_tag}_${task_id}"
wandb_tags="sigma_dh_prog,${fam_tag},${dh_tag},${gate_tag},${lr_tag},seed${seed}"

if [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
  run_dir="${GNNPLUS_OUT_DIR}/sigma_dh_prog/${fam_tag}_${dh_tag}_${gate_tag}_${lr_tag}_seed${seed}"
else
  run_dir="results/sigma_dh_prog/${fam_tag}_${dh_tag}_${gate_tag}_${lr_tag}_seed${seed}"
fi
mkdir -p "${run_dir}"

log_message "dh-prog ${task_id}/${num_tasks}: ${fam_tag} ${dh_tag} ${gate_tag} lr=${base_lr} seed=${seed}"
log_message "cfg=${cfg} run_dir=${run_dir}"

cat > "${run_dir}/train_meta.txt" <<META
family=${fam_tag}
d_h=${d_h}
dh_tag=${dh_tag}
gate=${gate_tag}
variant=${variant}
cfg=${cfg}
seed=${seed}
lr=${base_lr}
lr_tag=${lr_tag}
task_id=${task_id}
job=${job_tag}
wandb_group=${wandb_group}
META
cp -f "${cfg}" "${run_dir}/config_used.yaml"

extra_args=(
  out_dir "${run_dir}"
  optim.base_lr "${base_lr}"
  gnn.hybrid.d_h "${d_h}"
  train.enable_ckpt True
  train.ckpt_best True
  train.ckpt_clean True
  "${gate_override[@]}"
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

log_message "Task ${task_id} complete. Listing ckpt:"
ls -lh "${run_dir}/ckpt/" 2>/dev/null || log_message "WARNING: no ckpt/"
