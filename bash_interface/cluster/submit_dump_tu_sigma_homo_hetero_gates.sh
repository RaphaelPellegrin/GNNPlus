#!/usr/bin/env bash
# Offline re-dump per-graph gates for TU homo/hetero SiGMA runs (same 1–150 map).
# GCN tasks exit 0. Prefer the in-train dump from run_tu_sigma_homo_hetero.sh;
# use this only if .pt files are missing.
#
#   bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh
#
# SiGMA-only smoke (skip GCN slots): e.g. tasks 6-10,11-15,... or full 1-150.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${TU_SIGMA_HH_NUM_SEEDS:-5}"
NUM_VARIANTS="${TU_SIGMA_HH_NUM_VARIANTS:-5}"
NUM_DATASETS="${TU_SIGMA_HH_NUM_DATASETS:-6}"
NUM_TASKS="${TU_SIGMA_HH_NUM_TASKS:-$((NUM_DATASETS * NUM_VARIANTS * NUM_SEEDS))}"
ARRAY_SPEC="${TU_SIGMA_HH_DUMP_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${TU_SIGMA_HH_DUMP_PARALLEL:-12}"
PARTITION="${TU_SIGMA_HH_DUMP_PARTITION:-mweber_gpu}"
MEM="${TU_SIGMA_HH_DUMP_MEM:-32GB}"
TIME="${TU_SIGMA_HH_DUMP_TIME:-02:00:00}"
EPOCH="${GATE_DUMP_EPOCH:--1}"
LEVEL="${GATE_DUMP_LEVEL:-graph}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_dump_tu_sigma_homo_hetero_gates] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

job_id="$(
    sbatch --parsable \
        --job-name=tu_hh_gdump \
        --array="${ARRAY_SPEC}%${PARALLEL}" \
        --partition="${PARTITION}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --gpus=1 \
        --output="logs_gnnplus/tu_hh_gdump_%A_%a.log" \
        --export=ALL,ENV_NAME=gnnplus,TU_SIGMA_HH_NUM_SEEDS="${NUM_SEEDS}",TU_SIGMA_HH_NUM_VARIANTS="${NUM_VARIANTS}",TU_SIGMA_HH_NUM_TASKS="${NUM_TASKS}",GATE_DUMP_EPOCH="${EPOCH}",GATE_DUMP_LEVEL="${LEVEL}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}" \
        bash_interface/cluster/run_dump_tu_sigma_homo_hetero_gates.sh
)"

cat <<EOF

=== TU homo/hetero gate re-dump submitted ===
  ARRAY JOBID:   ${job_id}
  Tasks:         ${ARRAY_SPEC}  (GCN slots no-op; SiGMA → .pt dumps)
  Level:         ${LEVEL}  (graph | node | both)
  Epoch:         ${EPOCH} (-1 = latest / best-val ckpt)
  Out pattern:   \$GNNPLUS_OUT_DIR/tu_sigma_homo_hetero/<ds>_<variant>_<lr>_seed<s>/gate_values_per_{graph,node}.pt
  Logs:          logs_gnnplus/tu_hh_gdump_${job_id}_<TASK>.log
  Docs:          Paper_tu_sigma_homo_hetero.md

  MUTAG SiGMA_hetero lr001 seed2 only:
    GATE_DUMP_LEVEL=both TU_SIGMA_HH_DUMP_ARRAY=18 \\
      bash bash_interface/cluster/submit_dump_tu_sigma_homo_hetero_gates.sh

EOF
