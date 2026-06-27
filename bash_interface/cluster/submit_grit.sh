#!/usr/bin/env bash
# Submit GRIT runs (standalone GritTransformer or hybrid GRIT MP head).
#
# Usage:
#   source ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#   source bash_interface/cluster/common_env.sh
#
#   bash bash_interface/cluster/submit_grit.sh pattern hybrid
#   bash bash_interface/cluster/submit_grit.sh cluster standalone
#   bash bash_interface/cluster/submit_grit.sh zinc standalone
#   bash bash_interface/cluster/submit_grit.sh all hybrid    # pattern+cluster+zinc
#
# W&B: run name <dataset>_grit_<variant>_seed<N>_job<JOBID>, tag job_<JOBID>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs_gnnplus

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <pattern|cluster|zinc|all> [standalone|hybrid|both] [seed]" >&2
    exit 2
fi

DATASETS_ARG="${1}"
VARIANT="${2:-standalone}"
SEED="${3:-${SEED:-0}}"

grit_resources() {
    case "$1" in
        pattern|cluster) echo "128GB 120:00:00" ;;
        zinc) echo "64GB 192:00:00" ;;
        *) echo "128GB 120:00:00" ;;
    esac
}

grit_cfgs() {
    local dataset="$1"
    local variant="$2"
    case "${dataset}:${variant}" in
        pattern:standalone) echo "configs/grit/pattern-grit-rrwp.yaml pattern_grit_rrwp" ;;
        pattern:hybrid) echo "configs/gated_hybrid/pattern-grit-repro-a1g1.yaml pattern_grit_hybrid" ;;
        cluster:standalone) echo "configs/grit/cluster-grit-rrwp.yaml cluster_grit_rrwp" ;;
        cluster:hybrid) echo "configs/gated_hybrid/cluster-grit-repro-a1g1.yaml cluster_grit_hybrid" ;;
        zinc:standalone) echo "configs/grit/zinc-grit-rrwp.yaml zinc_grit_rrwp" ;;
        zinc:hybrid) echo "configs/gated_hybrid/zinc-grit-repro-a1g1.yaml zinc_grit_hybrid" ;;
        *) return 1 ;;
    esac
}

submit_one() {
    local dataset="$1"
    local variant="$2"
    local cfg job_name resources mem time_budget extra_cli

    read -r cfg job_name <<< "$(grit_cfgs "${dataset}" "${variant}")"
    read -r mem time_budget <<< "$(grit_resources "${dataset}")"
    mem="${GRIT_SLURM_MEM:-${mem}}"
    time_budget="${GRIT_SLURM_TIME:-${time_budget}}"
    extra_cli=""
    if [ "${variant}" = "hybrid" ]; then
        extra_cli="gnn.hybrid.log_gate_stats True"
    fi

    local job_id
    job_id="$(
        sbatch --parsable \
            --job-name="${job_name}" \
            --partition=mweber_gpu \
            --mem="${mem}" \
            --time="${time_budget}" \
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
            dataset.dir \${GNNPLUS_DATASET_DIR} ${extra_cli}"
    )"

    echo ""
    echo "=== ${dataset} ${variant} GRIT submitted ==="
    echo "  JOBID:     ${job_id}"
    echo "  Config:    ${cfg}"
    echo "  Log:       logs_gnnplus/${job_name}_${job_id}.log"
    echo "  W&B name:  ${job_name}_seed${SEED}_job${job_id}"
    echo "  Monitor:   tail -f logs_gnnplus/${job_name}_${job_id}.log"
    echo ""
}

run_variant() {
    local dataset="$1"
    local variant="$2"
    case "${variant}" in
        standalone|hybrid)
            submit_one "${dataset}" "${variant}"
            ;;
        both)
            submit_one "${dataset}" "standalone"
            submit_one "${dataset}" "hybrid"
            ;;
        *)
            echo "Unknown variant: ${variant} (use standalone|hybrid|both)" >&2
            exit 2
            ;;
    esac
}

expand_datasets() {
    case "${1}" in
        all)
            echo "pattern cluster zinc"
            ;;
        pattern|cluster|zinc)
            echo "${1}"
            ;;
        *)
            echo "Unknown dataset: ${1} (use pattern|cluster|zinc|all)" >&2
            exit 2
            ;;
    esac
}

for ds in $(expand_datasets "${DATASETS_ARG}"); do
    run_variant "${ds}" "${VARIANT}"
done
