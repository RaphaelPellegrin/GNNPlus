#!/usr/bin/env bash
# Submit COCO-SP hybrid gated runs anchored on GatedGCN+ baseline 5b4z9l3u (seed 1).
#
# Baseline MP-only (already on W&B): coco_gatedgcn_seed1_cluster / 5b4z9l3u
#   configs/gatedgcn/coco.yaml
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#
#   bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u.sh a1g1
#   bash bash_interface/cluster/submit_coco_hybrid_5b4z9l3u.sh both
#
# Fair LR grid: submit_coco_gatedgcn_fair_comparison.sh
# Bayes sweep:  submit_coco_gatedgcn_best_hybrid_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

MODE="${1:-a1g1}"
SEED="${SEED:-1}"
MEM="${COCO_HYBRID_MEM:-128GB}"
TIME="${COCO_HYBRID_TIME:-192:00:00}"

submit_one() {
    local variant="$1"
    local cfg="$2"
    local job_name="$3"

    local job_id
    job_id="$(
        sbatch --parsable \
            --job-name="${job_name}" \
            --partition=mweber_gpu \
            --mem="${MEM}" \
            --time="${TIME}" \
            --gpus=1 \
            --output="logs_gnnplus/${job_name}_%j.log" \
            --export=ALL,ENV_NAME=gnnplus \
            --wrap="source ~/.gnnplus_env && source bash_interface/cluster/common_env.sh && \
            export GNNPLUS_DATASET_DIR=${GNNPLUS_DATASET_DIR:-/n/netscratch/mweber_lab/Lab/gnnplus_datasets} && \
            python main.py --cfg ${cfg} --repeat 1 seed ${SEED} \
            wandb.use True \
            wandb.entity weber-geoml-harvard-university \
            wandb.project GNNPlus \
            wandb.name ${job_name}_seed${SEED}_job\${SLURM_JOB_ID} \
            dataset.dir \${GNNPLUS_DATASET_DIR}"
    )"

    echo ""
    echo "=== COCO hybrid ${variant} (anchor 5b4z9l3u) submitted ==="
    echo "  JOBID:     ${job_id}"
    echo "  Config:    ${cfg}"
    echo "  Log:       logs_gnnplus/${job_name}_${job_id}.log"
    echo "  W&B name:  ${job_name}_seed${SEED}_job${job_id}"
    echo "  Baseline:  https://wandb.ai/weber-geoml-harvard-university/GNNPlus/runs/5b4z9l3u"
    echo "  Monitor:   tail -f logs_gnnplus/${job_name}_${job_id}.log"
    echo ""
}

case "${MODE}" in
    a1g1|1)
        submit_one "a1g1" \
            "configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1.yaml" \
            "coco_hybrid_5b4z9l3u_a1g1"
        ;;
    a2g1|2)
        submit_one "a2g1" \
            "configs/gated_hybrid/coco-hybrid-5b4z9l3u-a2g1.yaml" \
            "coco_hybrid_5b4z9l3u_a2g1"
        ;;
    both|all)
        submit_one "a1g1" \
            "configs/gated_hybrid/coco-hybrid-5b4z9l3u-a1g1.yaml" \
            "coco_hybrid_5b4z9l3u_a1g1"
        submit_one "a2g1" \
            "configs/gated_hybrid/coco-hybrid-5b4z9l3u-a2g1.yaml" \
            "coco_hybrid_5b4z9l3u_a2g1"
        ;;
    *)
        echo "Usage: $0 {a1g1|a2g1|both}" >&2
        exit 2
        ;;
esac
