#!/usr/bin/env bash
# Submit preferred-head mask ablation on Xu SiGMA TU ckpts (eval-only, 1 GPU job).
#
# Prerequisites (ckpts already on netscratch from prior train jobs):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   # GCN+SAGE gated (a0g2 + a1g2) — recommended first
#   bash bash_interface/cluster/submit_eval_tu_preferred_head_masks.sh
#
#   # a2g4 Xu (GCN,GIN,SAGE,GAT)
#   TU_PREF_MASK_FAMILY=a2g4 \
#     bash bash_interface/cluster/submit_eval_tu_preferred_head_masks.sh
#
#   # both families
#   TU_PREF_MASK_FAMILY=both \
#     bash bash_interface/cluster/submit_eval_tu_preferred_head_masks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

PARTITION="${TU_PREF_MASK_PARTITION:-mweber_gpu}"
NICE="${TU_PREF_MASK_NICE:-10000}"
MEM="${TU_PREF_MASK_MEM:-32GB}"
TIME="${TU_PREF_MASK_TIME:-4:00:00}"
FAMILY="${TU_PREF_MASK_FAMILY:-gcs}"

export ENV_NAME=gnnplus
export TU_PREF_MASK_FAMILY="${FAMILY}"
export TU_PREF_MASK_DATASETS="${TU_PREF_MASK_DATASETS:-mutag,enzymes}"
export TU_PREF_MASK_SEEDS="${TU_PREF_MASK_SEEDS:-0,1,2,3,4}"
export TU_PREF_MASK_DEVICE="${TU_PREF_MASK_DEVICE:-auto}"
if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    export GNNPLUS_DATASET_DIR
fi
if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_eval_tu_preferred_head_masks] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi
export GNNPLUS_OUT_DIR

sbatch_args=(
    --parsable
    --job-name=tu_pref_mask
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/tu_pref_mask_%j.log"
    --export=ALL
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_eval_tu_preferred_head_masks.sh
)"

cat <<EOF

=== TU preferred-head mask eval submitted ===
  JOBID:         ${job_id}
  FAMILY:        ${FAMILY}
  DATASETS:      ${TU_PREF_MASK_DATASETS}
  SEEDS:         ${TU_PREF_MASK_SEEDS}
  PARTITION:     ${PARTITION}
  Log:           logs_gnnplus/tu_pref_mask_${job_id}.log
  Local outs:    results/heterogeneity/tu_pref_mask_*
  Cluster outs:  also under \$GNNPLUS repo results/ on the compute node —
                 rsync back after the job finishes.

Log JOBID in CLUSTER_LAUNCHES.md.
Monitor:
  squeue -u \$USER -j ${job_id}
  tail -f logs_gnnplus/tu_pref_mask_${job_id}.log

EOF
