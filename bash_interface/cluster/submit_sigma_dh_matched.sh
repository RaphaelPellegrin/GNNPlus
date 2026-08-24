#!/usr/bin/env bash
# Submit SiGMA d_h-matched Tab. 3/4 budget shrinks (TU Tab. 17/18 analog).
#
# 15 families × 2 LRs {1e-3, 1e-2} × 5 seeds = 150 jobs, up to 20 GPUs.
# Skip ZINC (main already ≤500k). Report the better LR per family after runs.
#
# Prerequisites:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   git pull
#
# Launch:
#   bash bash_interface/cluster/submit_sigma_dh_matched.sh
#
# Smoke (seed 0, both LRs, first 4 families = PATTERN/CLUSTER):
#   SIGMA_DH_MATCHED_ARRAY=1,6,11,16,21,26,31,36 SIGMA_DH_MATCHED_PARALLEL=8 \
#     bash bash_interface/cluster/submit_sigma_dh_matched.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_DH_MATCHED_NUM_SEEDS:-5}"
NUM_LRS="${SIGMA_DH_MATCHED_NUM_LRS:-2}"
NUM_FAMILIES="${SIGMA_DH_MATCHED_NUM_FAMILIES:-15}"
NUM_TASKS="${SIGMA_DH_MATCHED_NUM_TASKS:-$((NUM_FAMILIES * NUM_LRS * NUM_SEEDS))}"
ARRAY_SPEC="${SIGMA_DH_MATCHED_ARRAY:-1-${NUM_TASKS}}"
PARALLEL="${SIGMA_DH_MATCHED_PARALLEL:-20}"
PARTITION="${SIGMA_DH_MATCHED_PARTITION:-mweber_gpu}"
NICE="${SIGMA_DH_MATCHED_NICE:-10000}"
MEM="${SIGMA_DH_MATCHED_MEM:-128GB}"
TIME="${SIGMA_DH_MATCHED_TIME:-120:00:00}"

if [ -z "${GNNPLUS_OUT_DIR:-}" ]; then
    export GNNPLUS_OUT_DIR=/n/netscratch/mweber_lab/Lab/rpellegrin/gnnplus_results
    echo "[submit_sigma_dh_matched] GNNPLUS_OUT_DIR unset → ${GNNPLUS_OUT_DIR}"
fi

missing=0
for cfg in \
  configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh16.yaml \
  configs/gated_hybrid/dh_matched/pattern-grit-vn4-dh4.yaml \
  configs/gated_hybrid/dh_matched/cluster-a1g1-dh36.yaml \
  configs/gated_hybrid/dh_matched/cluster-a1g1-dh24.yaml \
  configs/gated_hybrid/dh_matched/mnist-a2g2-dh37.yaml \
  configs/gated_hybrid/dh_matched/cifar10-a8g4-dh20.yaml \
  configs/gated_hybrid/dh_matched/cifar10-a8g4-dh34.yaml \
  configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh23.yaml \
  configs/gated_hybrid/dh_matched/peptides-func-a1g2-dh75.yaml \
  configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh43.yaml \
  configs/gated_hybrid/dh_matched/peptides-struct-a1g1-dh92.yaml \
  configs/gated_hybrid/dh_matched/voc-a2g2-dh15.yaml \
  configs/gated_hybrid/dh_matched/voc-a2g2-h64-dh12.yaml \
  configs/gated_hybrid/dh_matched/coco-a1g1-dh34.yaml \
  configs/gated_hybrid/dh_matched/malnet-a1g1-dh57.yaml
do
  if [ ! -f "${cfg}" ]; then
    echo "MISSING ${cfg}"
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "d_h-matched configs missing — run:"
  echo "  python scripts/generate_sigma_dh_matched_configs.py"
  exit 1
fi

chmod +x bash_interface/cluster/run_sigma_dh_matched.sh

sbatch_args=(
    --parsable
    --job-name=sigma_dh_matched
    --array="${ARRAY_SPEC}%${PARALLEL}"
    --partition="${PARTITION}"
    --mem="${MEM}"
    --time="${TIME}"
    --gpus=1
    --output="logs_gnnplus/sigma_dh_matched_%A_%a.log"
    --export=ALL,ENV_NAME=gnnplus,SIGMA_DH_MATCHED_NUM_SEEDS="${NUM_SEEDS}",SIGMA_DH_MATCHED_NUM_LRS="${NUM_LRS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_dh_matched.sh
)"

cat <<EOF

=== SiGMA d_h-matched (Tab. 3/4 ≤500k / ≤1M, 2 LRs) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}
                 (${NUM_FAMILIES} families × ${NUM_LRS} LRs × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  LRs:           0.001 (lr001) and 0.01 (lr01) — pick better per family after
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_dh_matched_${job_id}_<TASK>.log
  Out:           \$GNNPLUS_OUT_DIR/sigma_dh_matched/<fam>_<lr>_seed<s>/
  Docs:          Paper_sigma_dh_matched.md

  Task map (each family = 10 tasks: lr001 seeds 0–4, then lr01 seeds 0–4):
    1–10     PATTERN   dh16
    11–20    PATTERN   dh4
    21–30    CLUSTER   dh36
    31–40    CLUSTER   dh24
    41–50    MNIST     dh37
    51–60    CIFAR10   dh20
    61–70    CIFAR10   dh34
    71–80    Pep-func  dh23
    81–90    Pep-func  dh75
    91–100   Pep-struct dh43
    101–110  Pep-struct dh92
    111–120  VOC       dh15
    121–130  VOC       H64/dh12
    131–140  COCO      dh34
    141–150  MalNet    dh57

  Fast first (PATTERN+CLUSTER+Pep-struct+MalNet+Pep-func):
    SIGMA_DH_MATCHED_ARRAY=1-40,71-110,141-150 SIGMA_DH_MATCHED_PARALLEL=20 \\
      bash bash_interface/cluster/submit_sigma_dh_matched.sh

  Paste JOBID into Paper_sigma_dh_matched.md + CLUSTER_LAUNCHES.md

EOF
