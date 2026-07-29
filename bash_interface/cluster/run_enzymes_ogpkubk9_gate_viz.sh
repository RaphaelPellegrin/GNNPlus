#!/usr/bin/env bash
# =============================================================================
# ENZYMES SiGMA (ogpkubk9 a4g4) gate-viz run: train one seed with checkpoints.
#
# Dedicated out_dir so ckpts land in <out_dir>/ckpt/. After training:
#   python scripts/gate_viz/dump_per_graph_gates.py --run_dir <out_dir> --cfg <yaml>
#
# Submit:
#   bash bash_interface/cluster/submit_enzymes_ogpkubk9_gate_viz.sh
# =============================================================================

#SBATCH --job-name=enz_gate_viz
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --output=logs_gnnplus/%x_%j.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${GNNPLUS_PROJECT_ROOT:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus}}"
cd "${REPO_ROOT}"
SCRIPT_DIR="${REPO_ROOT}/bash_interface/cluster"
# shellcheck source=common_env.sh
source "${SCRIPT_DIR}/common_env.sh"

seed="${GATE_VIZ_SEED:-2}"
scheduler="${GATE_VIZ_SCHEDULER:-plateau}"
ckpt_period="${GATE_VIZ_CKPT_PERIOD:-50}"
max_epoch="${GATE_VIZ_MAX_EPOCH:-}"

case "${scheduler}" in
    plateau)
        cfg="configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-plateau-anchor.yaml"
        ;;
    cosine)
        cfg="configs/gated_hybrid/enzymes-hybrid-ogpkubk9-a4g4-cosine-anchor.yaml"
        ;;
    *)
        log_message "Unknown GATE_VIZ_SCHEDULER=${scheduler} (use plateau|cosine)"
        exit 1
        ;;
esac

default_out="results/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
if [ -n "${GATE_VIZ_OUT_DIR:-}" ]; then
    out_dir="${GATE_VIZ_OUT_DIR}"
elif [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
else
    out_dir="${default_out}"
fi
wandb_name="${GATE_VIZ_WANDB_NAME:-enzymes_gate_viz_${scheduler}_seed${seed}}"
wandb_group="${GATE_VIZ_WANDB_GROUP:-enzymes_ogpkubk9_gate_viz}"

mkdir -p "${out_dir}"

log_message "ENZYMES gate-viz: seed=${seed} scheduler=${scheduler} out_dir=${out_dir}"

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi
if [ -n "${max_epoch}" ]; then
    extra_args+=(optim.max_epoch "${max_epoch}")
fi

exec python main.py \
    --cfg "${cfg}" \
    --repeat 1 \
    seed "${seed}" \
    out_dir "${out_dir}" \
    wandb.use True \
    wandb.entity weber-geoml-harvard-university \
    wandb.project GNNPlus \
    wandb.group "${wandb_group}" \
    wandb.name "${wandb_name}" \
    train.enable_ckpt True \
    train.ckpt_clean False \
    train.ckpt_period "${ckpt_period}" \
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
