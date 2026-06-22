#!/usr/bin/env bash
# =============================================================================
# Repro sweep wrapper: baseline custom_gnn OR same hyperparams + 1 attention head.
#
# Used with grid sweeps (exactly 2 trials: baseline | hybrid_attn1).
#
#   repro_variant=baseline     → original GNN+ yaml (custom_gnn)
#   repro_variant=hybrid_attn1   → gated_hybrid *-repro-a1.yaml (1 MP + 1 attn)
#
# Leading arg: --dataset=cifar10 | peptides_func
# W&B passes:  --repro_variant=baseline | hybrid_attn1
# =============================================================================

set -euo pipefail

REPO_ROOT="${GNNPLUS_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

DATASET="${SWEEP_DATASET:-cifar10}"
VARIANT="${REPRO_VARIANT:-baseline}"
REPEAT=1

while [ "$#" -gt 0 ]; do
    tok="$1"
    shift
    case "${tok}" in
        --dataset=*)
            DATASET="${tok#--dataset=}"
            ;;
        --dataset)
            DATASET="${1:?missing --dataset value}"
            shift
            ;;
        --repro_variant=*)
            VARIANT="${tok#--repro_variant=}"
            ;;
        --repro_variant)
            VARIANT="${1:?missing --repro_variant value}"
            shift
            ;;
        --repeat=*)
            REPEAT="${tok#--repeat=}"
            ;;
        --repeat)
            REPEAT="${1:?missing --repeat value}"
            shift
            ;;
        *)
            echo "[sweep_wrapper_gnnplus_repro] ignoring token: ${tok}" >&2
            ;;
    esac
done

CFG=""
SEED="0"
MOLECULAR="false"
WANDB_NAME=""
EXTRA_ARGS=()

case "${DATASET}" in
    cifar10)
        case "${VARIANT}" in
            baseline)
                CFG="configs/gatedgcn/cifar10.yaml"
                SEED="0"
                WANDB_NAME="cifar10_gatedgcn_seed0_repro_baseline"
                ;;
            hybrid_attn1)
                CFG="configs/gated_hybrid/cifar10-gatedgcn-repro-a1.yaml"
                SEED="0"
                WANDB_NAME="cifar10_gatedgcn_seed0_repro_hybrid_attn1"
                EXTRA_ARGS+=(gnn.hybrid.log_gate_stats True)
                ;;
            *)
                echo "Unknown repro_variant for cifar10: ${VARIANT}" >&2
                exit 2
                ;;
        esac
        ;;
    peptides_func)
        MOLECULAR="true"
        case "${VARIANT}" in
            baseline)
                CFG="configs/gcn/peptides-func.yaml"
                SEED="2"
                WANDB_NAME="peptides-func_gcn_seed2_repro_baseline"
                ;;
            hybrid_attn1)
                CFG="configs/gated_hybrid/peptides-func-gcn-repro-a1.yaml"
                SEED="2"
                WANDB_NAME="peptides-func_gcn_seed2_repro_hybrid_attn1"
                EXTRA_ARGS+=(gnn.hybrid.log_gate_stats True)
                ;;
            *)
                echo "Unknown repro_variant for peptides_func: ${VARIANT}" >&2
                exit 2
                ;;
        esac
        ;;
    *)
        echo "Unknown dataset: ${DATASET}" >&2
        exit 2
        ;;
esac

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    EXTRA_ARGS+=(dataset.dir "${GNNPLUS_DATASET_DIR}")
fi

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"

echo "[sweep_wrapper_gnnplus_repro] dataset=${DATASET} variant=${VARIANT} cfg=${CFG} seed=${SEED}"

exec python main.py \
    --cfg "${CFG}" \
    --repeat "${REPEAT}" \
    seed "${SEED}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    wandb.name "${WANDB_NAME}" \
    "${EXTRA_ARGS[@]}"
