#!/usr/bin/env bash
# =============================================================================
# Create W&B sweeps (one per dataset) and optionally launch SLURM agents.
#
# Usage (login node):
#   source ~/.gnnplus_env
#   export WANDB_PROJECT=GNNPlus  # default from ~/.gnnplus_env
#   export GNNPLUS_DATASET_DIR=/n/netscratch/mweber_lab/Lab/gnnplus_datasets
#   cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/GNNPlus
#
#   # Generate YAMLs (first time / after template edit)
#   bash bash_interface/sweeps/generate_hybrid_sweep_yamls.sh
#
#   # Create all sweeps + launch agents (tier 1–4)
#   bash bash_interface/sweeps/launch_hybrid_sweeps.sh
#
#   # Create only, no sbatch:
#   bash bash_interface/sweeps/launch_hybrid_sweeps.sh --create-only tier1
#
#   # Subset:
#   bash bash_interface/sweeps/launch_hybrid_sweeps.sh tier1 tier2 mnist
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
SWEEPS_DIR="${REPO_ROOT}/bash_interface/sweeps"

CREATE_ONLY=0
if [ "${1:-}" = "--create-only" ]; then
    CREATE_ONLY=1
    shift
fi

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
export ENV_NAME="${ENV_NAME:-gnnplus}"

TIER1=(mnist cifar10)
TIER2=(coco voc)
TIER3=(peptides_func peptides_struct)
TIER4=(enzymes)
TIER5=(hiv zinc mutag ppa mal pcba cluster pattern)
ALL=( "${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}" "${TIER4[@]}" "${TIER5[@]}" )

resolve_tier() {
    case "$1" in
        tier1) printf '%s\n' "${TIER1[@]}" ;;
        tier2) printf '%s\n' "${TIER2[@]}" ;;
        tier3) printf '%s\n' "${TIER3[@]}" ;;
        tier4) printf '%s\n' "${TIER4[@]}" ;;
        tier5) printf '%s\n' "${TIER5[@]}" ;;
        all) printf '%s\n' "${ALL[@]}" ;;
        *) return 1 ;;
    esac
}

is_dataset_slug() {
    local slug="$1" d
    for d in "${ALL[@]}"; do
        [ "$d" = "$slug" ] && return 0
    done
    return 1
}

REQUESTED=()
if [ "$#" -eq 0 ]; then
    REQUESTED=( "${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}" "${TIER4[@]}" )
else
    for arg in "$@"; do
        if resolved="$(resolve_tier "$arg" 2>/dev/null)"; then
            while IFS= read -r slug; do
                REQUESTED+=("$slug")
            done <<< "$resolved"
        elif is_dataset_slug "$arg"; then
            REQUESTED+=("$arg")
        else
            echo "Unknown: ${arg}" >&2
            exit 1
        fi
    done
fi

bash "${SWEEPS_DIR}/generate_hybrid_sweep_yamls.sh"

ARRAY_TASKS="${SWEEP_ARRAY_TASKS:-16}"
ARRAY_PARALLEL="${SWEEP_ARRAY_PARALLEL:-8}"
RUNS_PER_AGENT="${RUNS_PER_AGENT:-2}"

sweep_mem() {
    case "$1" in
        coco|voc|cluster|pattern|pcba) echo "128GB" ;;
        ppa) echo "96GB" ;;
        *) echo "64GB" ;;
    esac
}

IDS_FILE="${SWEEPS_DIR}/.hybrid_sweep_ids"
: > "${IDS_FILE}"

for slug in "${REQUESTED[@]}"; do
    yaml="${SWEEPS_DIR}/${slug}_hybrid_gnnplus_sweep.yaml"
    if [ ! -f "${yaml}" ]; then
        echo "Missing ${yaml}" >&2
        exit 1
    fi
    echo "=== Creating sweep: ${slug} ==="
    bash "${SWEEPS_DIR}/create_sweep.sh" "${yaml}"
    sweep_id="$(cat "${SWEEPS_DIR}/.last_sweep_id")"
    printf '%s\t%s\n' "${slug}" "${sweep_id}" >> "${IDS_FILE}"

    if [ "${CREATE_ONLY}" -eq 0 ]; then
        mem="$(sweep_mem "${slug}")"
        echo "=== Launching agents: ${slug} (${mem}) ==="
        SWEEP_ID="${sweep_id}" \
        SWEEP_DATASET="${slug}" \
        RUNS_PER_AGENT="${RUNS_PER_AGENT}" \
        sbatch \
            --job-name="gnnplus_sweep_${slug}" \
            --array="1-${ARRAY_TASKS}%${ARRAY_PARALLEL}" \
            --mem="${mem}" \
            --time=96:00:00 \
            --export=ALL,SWEEP_ID="${sweep_id}",SWEEP_DATASET="${slug}",RUNS_PER_AGENT="${RUNS_PER_AGENT}",WANDB_PROJECT="${WANDB_PROJECT}",ENV_NAME="${ENV_NAME}" \
            bash_interface/sweeps/run_wandb_sweep_agent.sh
    fi
done

echo ""
echo "Sweep ids written to ${IDS_FILE}"
cat "${IDS_FILE}"
echo ""
echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
