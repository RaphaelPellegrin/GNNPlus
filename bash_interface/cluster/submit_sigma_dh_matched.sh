#!/usr/bin/env bash
# Submit SiGMA d_h-matched Tab. 3/4 budget shrinks (TU Tab. 17/18 analog).
#
# 15 families × 5 seeds = 75 jobs, up to 20 GPUs.
# Skip ZINC (main already ≤500k).
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
# Smoke (seed 0 of each family):
#   SIGMA_DH_MATCHED_ARRAY=1,6,11,16,21,26,31,36,41,46,51,56,61,66,71 \
#   SIGMA_DH_MATCHED_PARALLEL=15 \
#     bash bash_interface/cluster/submit_sigma_dh_matched.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

NUM_SEEDS="${SIGMA_DH_MATCHED_NUM_SEEDS:-5}"
NUM_FAMILIES="${SIGMA_DH_MATCHED_NUM_FAMILIES:-15}"
NUM_TASKS="${SIGMA_DH_MATCHED_NUM_TASKS:-$((NUM_FAMILIES * NUM_SEEDS))}"
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
    --export=ALL,ENV_NAME=gnnplus,SIGMA_DH_MATCHED_NUM_SEEDS="${NUM_SEEDS}",GNNPLUS_DATASET_DIR="${GNNPLUS_DATASET_DIR:-}",GNNPLUS_OUT_DIR="${GNNPLUS_OUT_DIR}"
)

if [ "${NICE}" != "0" ]; then
    sbatch_args+=(--nice="${NICE}")
fi

job_id="$(
    sbatch "${sbatch_args[@]}" \
        bash_interface/cluster/run_sigma_dh_matched.sh
)"

cat <<EOF

=== SiGMA d_h-matched (Tab. 3/4 ≤500k / ≤1M) submitted ===
  ARRAY JOBID:   ${job_id}
  Partition:     ${PARTITION}
  Tasks:         ${ARRAY_SPEC}  (${NUM_FAMILIES} families × ${NUM_SEEDS} seeds = ${NUM_TASKS})
  Parallel:      ${PARALLEL} GPUs max
  Mem / time:    ${MEM} / ${TIME}
  Logs:          logs_gnnplus/sigma_dh_matched_${job_id}_<TASK>.log
  Out:           \$GNNPLUS_OUT_DIR/sigma_dh_matched/<fam>_seed<s>/
  Docs:          Paper_sigma_dh_matched.md

  Task map (blocks of ${NUM_SEEDS} seeds):
    1–5    PATTERN   dh16   ~844k   (≤1M)
    6–10   PATTERN   dh4    ~519k   (≤500k)
    11–15  CLUSTER   dh36   ~437k   (≤500k, Tab17 ratio)
    16–20  CLUSTER   dh24   ~254k   (≤500k, Tab18 ratio)
    21–25  MNIST     dh37   ~488k   (≤500k)
    26–30  CIFAR10   dh20   ~477k   (≤500k; keep a8g4)
    31–35  CIFAR10   dh34   ~978k   (≤1M; keep a8g4)
    36–40  Pep-func  dh23   ~491k   (≤500k; a1g2)
    41–45  Pep-func  dh75   ~995k   (≤1M; a1g2)
    46–50  Pep-struct dh43  ~498k   (≤500k)
    51–55  Pep-struct dh92  ~998k   (≤1M)
    56–60  VOC       dh15   ~995k   (≤1M; H95)
    61–65  VOC       H64/dh12 ~499k (≤500k; H shrink required)
    66–70  COCO      dh34   ~480k   (≤500k)
    71–75  MalNet    dh57   ~498k   (≤500k)

  Skip: ZINC (main 450k already ≤500k). MNIST/COCO/MalNet mains already ≤1M.

  Paste JOBID into Paper_sigma_dh_matched.md + CLUSTER_LAUNCHES.md

EOF
