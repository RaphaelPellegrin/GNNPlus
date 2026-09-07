#!/usr/bin/env bash
# =============================================================================
# Preferred-head mask ablation on Xu SiGMA TU checkpoints (eval only).
#
# Needs ckpts under $GNNPLUS_OUT_DIR (not the gate-only rsync on the Mac).
# Preference pickles: $GNNPLUS_OUT_DIR/heterogeneity/powerful_gnns/tu_gate_bridge/
#
# Env knobs:
#   TU_PREF_MASK_TAGS     space-separated lr-tags (default: a0g2_gated a1g2_gated)
#   TU_PREF_MASK_FAMILY   gcs | a2g4 | both  (default: gcs)
#   TU_PREF_MASK_DATASETS mutag,enzymes
#   TU_PREF_MASK_SEEDS    0,1,2,3,4
#   TU_PREF_MASK_DEVICE   auto|cpu|cuda
#
# Submit:
#   bash bash_interface/cluster/submit_eval_tu_preferred_head_masks.sh
# =============================================================================

#SBATCH --job-name=tu_pref_mask
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
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

OUT_ROOT="${GNNPLUS_OUT_DIR:?Set GNNPLUS_OUT_DIR}"
DATASET_DIR="${GNNPLUS_DATASET_DIR:?Set GNNPLUS_DATASET_DIR}"
HETERO_ROOT="${TU_PREF_MASK_HETERO_ROOT:-${OUT_ROOT}/heterogeneity/powerful_gnns/tu_gate_bridge}"
DATASETS="${TU_PREF_MASK_DATASETS:-mutag,enzymes}"
SEEDS="${TU_PREF_MASK_SEEDS:-0,1,2,3,4}"
DEVICE="${TU_PREF_MASK_DEVICE:-auto}"
FAMILY="${TU_PREF_MASK_FAMILY:-gcs}"
LOCAL_RESULTS="${REPO_ROOT}/results/heterogeneity"

mkdir -p "${LOCAL_RESULTS}"

run_one() {
    local tag="$1"
    local gate_root="$2"
    local operators="$3"
    local out_dir="$4"

    log_message "=== preferred-head masks: tag=${tag} ops=${operators} ==="
    log_message "gate_root=${gate_root}"
    log_message "out_dir=${out_dir}"

    if [ ! -d "${gate_root}" ]; then
        log_message "MISSING gate_root: ${gate_root}"
        return 1
    fi
    if [ ! -d "${HETERO_ROOT}" ]; then
        log_message "MISSING hetero_root: ${HETERO_ROOT}"
        return 1
    fi

    python scripts/heterogeneity/eval_tu_preferred_head_masks.py \
        --datasets "${DATASETS}" \
        --hetero-root "${HETERO_ROOT}" \
        --gate-root "${gate_root}" \
        --lr-tag "${tag}" \
        --operators "${operators}" \
        --seeds "${SEEDS}" \
        --splits val,test \
        --dataset-dir "${DATASET_DIR}" \
        --adaptive \
        --device "${DEVICE}" \
        --out-dir "${out_dir}"
}

run_gcs() {
    local tags="${TU_PREF_MASK_TAGS:-a0g2_gated a1g2_gated}"
    local gate_root="${OUT_ROOT}/heterogeneity/powerful_gnns/tu_xu_sigma_gcn_sage"
    local ops="GCN,SAGE"
    local tag
    for tag in ${tags}; do
        run_one "${tag}" "${gate_root}" "${ops}" \
            "${LOCAL_RESULTS}/tu_pref_mask_gcs_${tag}"
    done
}

run_a2g4() {
    local tags="${TU_PREF_MASK_TAGS:-xu}"
    local gate_root="${OUT_ROOT}/heterogeneity/powerful_gnns/tu_xu_sigma_a2g4"
    local ops="GCN,GIN,SAGE,GAT"
    local tag
    for tag in ${tags}; do
        run_one "${tag}" "${gate_root}" "${ops}" \
            "${LOCAL_RESULTS}/tu_pref_mask_a2g4_${tag}"
    done
}

case "${FAMILY}" in
    gcs) run_gcs ;;
    a2g4) run_a2g4 ;;
    both)
        run_gcs
        TU_PREF_MASK_TAGS=xu run_a2g4
        ;;
    *)
        log_message "Unknown TU_PREF_MASK_FAMILY=${FAMILY} (use gcs|a2g4|both)"
        exit 1
        ;;
esac

log_message "Preferred-head mask eval done. Results under ${LOCAL_RESULTS}/tu_pref_mask_*"
