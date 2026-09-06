#!/usr/bin/env bash
# Submit main-text progressive d_h × gated vs ungated (Tab. 3/4 anchors).
#
# 20 (family×d_h) × 2 gate × 2 LR × 5 seeds = 400 jobs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh
#
# Phased (recommended):
#   # PATTERN only (tasks 1–100): 5 dh × 2 gate × 2 lr × 5 seeds
#   SIGMA_DH_PROG_ARRAY=1-100 bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh
#   # CLUSTER (101–200), MNIST (201–300), Pep-func (301–400)
#
# Smoke (PATTERN dh1 gated+ungated lr001 seed0):
#   # fam_dh_idx for pattern dh1 = 4 → gated lr001 seed0 = task ((4*2+0)*2+0)*5+0+1 = 81
#   # ungated = ((4*2+1)*2+0)*5+0+1 = 91
#   SIGMA_DH_PROG_ARRAY=81,91 SIGMA_DH_PROG_PARALLEL=2 \
#     bash bash_interface/cluster/submit_sigma_dh_prog_ungated.sh
#
# Paste JOBID into Paper_sigma_dh_matched_ungated.md + CLUSTER_LAUNCHES.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_FAM_DH="${SIGMA_DH_PROG_NUM_FAM_DH:-20}"
NUM_GATES="${SIGMA_DH_PROG_NUM_GATES:-2}"
NUM_LRS="${SIGMA_DH_PROG_NUM_LRS:-2}"
NUM_SEEDS="${SIGMA_DH_PROG_NUM_SEEDS:-5}"
NUM_TASKS="${SIGMA_DH_PROG_NUM_TASKS:-$((NUM_FAM_DH * NUM_GATES * NUM_LRS * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_DH_PROG_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_DH_PROG_PARALLEL:-20}"
PARTITION="${SIGMA_DH_PROG_PARTITION:-mweber_gpu}"
NICE="${SIGMA_DH_PROG_NICE:-10000}"
MEM="${SIGMA_DH_PROG_MEM:-128GB}"
TIME="${SIGMA_DH_PROG_TIME:-120:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_sigma_dh_prog_ungated] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

for cfg in \
  configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml \
  configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml \
  configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml \
  configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml
do
  if [ ! -f "${cfg}" ]; then
    echo "MISSING ${cfg}"
    exit 1
  fi
done

chmod +x bash_interface/cluster/run_sigma_dh_prog_ungated.sh

sbatch_args=(
    --parsable
    --job-name=sigma_dh_prog
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_dh_prog_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_DH_PROG_NUM_SEEDS="${NUM_SEEDS}",SIGMA_DH_PROG_NUM_LRS="${NUM_LRS}",SIGMA_DH_PROG_NUM_GATES="${NUM_GATES}",SIGMA_DH_PROG_NUM_TASKS="${NUM_TASKS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_dh_prog_ungated.sh
)"

cat <<EOF

=== SiGMA progressive d_h × gated/ungated submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}
                 (${NUM_FAM_DH} fam×dh × ${NUM_GATES} gate × ${NUM_LRS} LR × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_dh_prog_${job_id}_<TASK>.log
  Out:           \$GNNPLUS_OUT_DIR/sigma_dh_prog/<fam>_dh<k>_<gated|ungated>_<lr>_seed<s>/
  Docs:          Paper_sigma_dh_matched_ungated.md

  Ladders (paper recipe → App. H extreme):
    PATTERN   d_h ∈ {16, 8, 4, 2, 1}     tasks 1–100
    CLUSTER   d_h ∈ {24, 12, 8, 4, 1}    tasks 101–200
    MNIST     d_h ∈ {37, 16, 8, 4, 1}    tasks 201–300
    Pep-func  d_h ∈ {23, 12, 8, 4, 1}    tasks 301–400

  Per (fam,d_h) block of 20:
    +0–4   gated   lr=0.001
    +5–9   gated   lr=0.01
    +10–14 ungated lr=0.001
    +15–19 ungated lr=0.01

  W&B: paper_sigma_dh_prog_<fam>_dh<k>_{gated,ungated}_{lr001,lr01}
  Protocol: better LR; plot gated−ungated Δ vs d_h (expect larger gap as d_h↓).

  Note: some gated cells overlap prior paper_sigma_dh_matched_* (PATTERN dh16/4,
  CLUSTER dh24, MNIST dh37, Pep-func dh23). New prefix keeps groups clean;
  you may skip those gated tasks later if GPUs are tight.

  Paste JOBID into Paper_sigma_dh_matched_ungated.md + CLUSTER_LAUNCHES.md

EOF
