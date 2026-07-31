#!/usr/bin/env bash
# =============================================================================
# Dump per-graph / per-head SiGMA gates from an ENZYMES gate-viz checkpoint.
#
# Expects ckpts already under GATE_VIZ_OUT_DIR/ckpt/ (from
# submit_enzymes_ogpkubk9_gate_viz.sh). Writes gate_values_per_graph.pt.
#
# Submit:
#   bash bash_interface/cluster/submit_dump_enzymes_ogpkubk9_gates.sh
# =============================================================================

#SBATCH --job-name=enz_gate_dump
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
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
epoch="${GATE_DUMP_EPOCH:--1}"

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

if [ -n "${GATE_VIZ_OUT_DIR:-}" ]; then
    out_dir="${GATE_VIZ_OUT_DIR}"
elif [ -n "${GNNPLUS_OUT_DIR:-}" ]; then
    out_dir="${GNNPLUS_OUT_DIR}/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
else
    out_dir="results/gate_viz_enzymes_ogpkubk9_${scheduler}_seed${seed}"
fi

out_pt="${GATE_DUMP_OUT:-${out_dir}/gate_values_per_graph.pt}"

if [ ! -d "${out_dir}/ckpt" ]; then
    log_message "No ckpt/ under ${out_dir}"
    exit 1
fi

log_message "Dump gates: run_dir=${out_dir} cfg=${cfg} seed=${seed} epoch=${epoch}"
ls -lh "${out_dir}/ckpt/" | tail -n 5 || true

extra_args=()
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

exec python scripts/gate_viz/dump_per_graph_gates.py \
    --run_dir "${out_dir}" \
    --epoch "${epoch}" \
    --out "${out_pt}" \
    --cfg "${cfg}" \
    seed "${seed}" \
    "${extra_args[@]}"
