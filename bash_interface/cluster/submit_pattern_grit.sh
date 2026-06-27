#!/usr/bin/env bash
# Submit PATTERN GRIT runs with W&B names/tags tied to SLURM job id.
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   bash bash_interface/cluster/submit_pattern_grit.sh standalone
#   bash bash_interface/cluster/submit_pattern_grit.sh hybrid
#   bash bash_interface/cluster/submit_pattern_grit.sh both
#
# After submit, note the printed JOBID and grep:
#   tail -f logs_gnnplus/pattern_grit_<JOBID>.log
# W&B: search run name "pattern_grit_*_<JOBID>" or tag "job_<JOBID>"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

MODE="${1:-standalone}"
SEED="${SEED:-0}"
MEM="${GRIT_SLURM_MEM:-128GB}"
TIME="${GRIT_SLURM_TIME:-120:00:00}"

submit_one() {
    local variant="$1"
    local cfg="$2"
    local job_name="$3"
    local extra_tags="$4"

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
            wandb.tags grit,pattern,${variant},${extra_tags},job_\${SLURM_JOB_ID} \
            dataset.dir \${GNNPLUS_DATASET_DIR}"
    )"

    echo ""
    echo "=== ${variant} GRIT submitted ==="
    echo "  JOBID:     ${job_id}"
    echo "  Log:       logs_gnnplus/${job_name}_${job_id}.log"
    echo "  W&B name:  ${job_name}_seed${SEED}_job${job_id}"
    echo "  W&B tags:  grit, pattern, ${variant}, job_${job_id}, ${extra_tags}"
    echo "  Monitor:   tail -f logs_gnnplus/${job_name}_${job_id}.log"
    echo ""
}

case "${MODE}" in
    standalone|rrwp)
        submit_one "standalone" "configs/grit/pattern-grit-rrwp.yaml" "pattern_grit_rrwp" "rrwp"
        ;;
    hybrid|a1g1)
        submit_one "hybrid" "configs/gated_hybrid/pattern-grit-repro-a1g1.yaml" "pattern_grit_hybrid" "a1g1"
        ;;
    both|all)
        submit_one "standalone" "configs/grit/pattern-grit-rrwp.yaml" "pattern_grit_rrwp" "rrwp"
        submit_one "hybrid" "configs/gated_hybrid/pattern-grit-repro-a1g1.yaml" "pattern_grit_hybrid" "a1g1"
        ;;
    *)
        echo "Usage: $0 {standalone|hybrid|both}" >&2
        exit 2
        ;;
esac
