#!/usr/bin/env bash
# =============================================================================
# Create a W&B sweep from a YAML and record the sweep id.
#
# Usage (login node, from GNNPlus repo root):
#   source ~/.gnnplus_env
#   export WANDB_PROJECT=GNNPlus
#   bash bash_interface/sweeps/create_sweep.sh \
#       bash_interface/sweeps/mnist_hybrid_gnnplus_sweep.yaml
# =============================================================================

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sweep_yaml>" >&2
    exit 2
fi
YAML_PATH="$1"
if [ ! -f "$YAML_PATH" ]; then
    echo "File not found: $YAML_PATH" >&2
    exit 2
fi

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"
SWEEPS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SWEEPS_DIR}/sweeps.log"
LAST_FILE="${SWEEPS_DIR}/.last_sweep_id"

CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs}"
ENV_NAME="${ENV_NAME:-gnnplus}"

WANDB_CMD=()
if command -v wandb >/dev/null 2>&1; then
    WANDB_CMD=(wandb)
elif [ -x "${CONDA_ENVS_PATH}/${ENV_NAME}/bin/wandb" ]; then
    export PATH="${CONDA_ENVS_PATH}/${ENV_NAME}/bin:${PATH}"
    WANDB_CMD=(wandb)
elif command -v python >/dev/null 2>&1 && python -c "import wandb" >/dev/null 2>&1; then
    WANDB_CMD=(python -m wandb)
else
    echo "Could not find wandb CLI (activate gnnplus env)" >&2
    exit 4
fi

echo "[create_sweep] entity=${WANDB_ENTITY} project=${WANDB_PROJECT} yaml=${YAML_PATH}"

TMP_OUT="$(mktemp)"
"${WANDB_CMD[@]}" sweep --project "${WANDB_PROJECT}" --entity "${WANDB_ENTITY}" "${YAML_PATH}" 2>&1 | tee "${TMP_OUT}"

SWEEP_PATH="$(grep -Eo '[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9]+' "${TMP_OUT}" | tail -n 1 || true)"
if [ -z "${SWEEP_PATH}" ]; then
    SWEEP_ID="$(grep -Eo 'with ID: [A-Za-z0-9]+' "${TMP_OUT}" | awk '{print $3}' | tail -n 1 || true)"
    if [ -n "${SWEEP_ID}" ]; then
        SWEEP_PATH="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
    fi
fi

if [ -z "${SWEEP_PATH}" ]; then
    echo "Could not parse sweep id from wandb output" >&2
    exit 3
fi

printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${YAML_PATH}" "${SWEEP_PATH}" >> "${LOG_FILE}"
printf '%s\n' "${SWEEP_PATH}" > "${LAST_FILE}"

_yaml_stem="$(basename "${YAML_PATH}" .yaml)"
RUNS_PER_AGENT=4
ARRAY_SPEC="1-8%4"
JOB_PREFIX="gnnplus_sweep"
TIME_LIMIT="96:00:00"

if [[ "${_yaml_stem}" == *_repro_baseline_vs_attn_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_repro_baseline_vs_attn_sweep}"
    RUNS_PER_AGENT=2
    ARRAY_SPEC="1-1"
    JOB_PREFIX="gnnplus_repro"
elif [[ "${_yaml_stem}" == *_repro_gcne_dh_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_repro_gcne_dh_sweep}"
    RUNS_PER_AGENT=6
    ARRAY_SPEC="1-1"
    JOB_PREFIX="gnnplus_repro"
    TIME_LIMIT="120:00:00"
elif [[ "${_yaml_stem}" == *_mp_only_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_mp_only_sweep}"
    RUNS_PER_AGENT=3
    ARRAY_SPEC="1-1"
    JOB_PREFIX="gnnplus_sanity"
    TIME_LIMIT="120:00:00"
elif [[ "${_yaml_stem}" == *_hybrid_gatedgcn_mp_lr_schedulefree_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_hybrid_gatedgcn_mp_lr_schedulefree_sweep}"
    RUNS_PER_AGENT=36
    ARRAY_SPEC="1-1"
    JOB_PREFIX="gnnplus_sweep"
    TIME_LIMIT="120:00:00"
elif [[ "${_yaml_stem}" == *_hybrid_gatedgcn_mp_lr_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_hybrid_gatedgcn_mp_lr_sweep}"
    RUNS_PER_AGENT=36
    ARRAY_SPEC="1-1"
    JOB_PREFIX="gnnplus_sweep"
    TIME_LIMIT="120:00:00"
elif [[ "${_yaml_stem}" == *_best_hybrid_schedulefree_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_best_hybrid_schedulefree_sweep}"
elif [[ "${_yaml_stem}" == *_best_hybrid_sweep ]]; then
    DATASET_SLUG="${_yaml_stem%%_best_hybrid_sweep}"
else
    DATASET_SLUG="${_yaml_stem%%_hybrid_*}"
fi

echo ""
echo "Sweep created: ${SWEEP_PATH}"
echo "Launch agents (copy as one block; do not paste wandb log lines into shell):"
cat <<EOF
  SWEEP_ID=${SWEEP_PATH} SWEEP_DATASET=${DATASET_SLUG} RUNS_PER_AGENT=${RUNS_PER_AGENT} \\
  sbatch --job-name=${JOB_PREFIX}_${DATASET_SLUG} --array=${ARRAY_SPEC} --mem=64GB --time=${TIME_LIMIT} \\
    --export=ALL,SWEEP_ID=${SWEEP_PATH},SWEEP_DATASET=${DATASET_SLUG},RUNS_PER_AGENT=${RUNS_PER_AGENT},WANDB_PROJECT=${WANDB_PROJECT},ENV_NAME=${ENV_NAME:-gnnplus} \\
    bash_interface/sweeps/run_wandb_sweep_agent.sh
EOF

rm -f "${TMP_OUT}"
