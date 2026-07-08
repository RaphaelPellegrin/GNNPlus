#!/usr/bin/env bash
# =============================================================================
# Repro sweep wrapper: baseline custom_gnn OR same hyperparams + 1 attention head.
#
# Used with grid sweeps (baseline | hybrid_attn1, optionally hybrid_d_h, etc.).
#
#   repro_variant=baseline     → original GNN+ yaml (custom_gnn)
#   repro_variant=hybrid_attn1 → gated_hybrid *-repro-a1.yaml (1 MP + 1 attn)
#   repro_variant=hybrid_a0g1  → 0 attn + 1×GCNE MP (gated MP only)
#   repro_variant=hybrid_a0g2  → 0 attn + 2×GCNE MP (gated MP only)
#
# Leading arg: --dataset=cifar10 | mnist | peptides_func
# W&B may also pass: --hybrid_d_h=128, --optim.base_lr=..., etc.
# =============================================================================

set -euo pipefail

REPO_ROOT="${GNNPLUS_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

DATASET="${SWEEP_DATASET:-cifar10}"
VARIANT="${REPRO_VARIANT:-baseline}"
REPRO_TASK=""
REPEAT=1
HYBRID_DH=""

declare -A OPTS=()

_set_opt() {
    local key="$1"
    local value="$2"
    OPTS["${key}"]="${value}"
}

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
        --repro_task=*)
            REPRO_TASK="${tok#--repro_task=}"
            ;;
        --repro_task)
            REPRO_TASK="${1:?missing --repro_task value}"
            shift
            ;;
        --repeat=*)
            REPEAT="${tok#--repeat=}"
            ;;
        --repeat)
            REPEAT="${1:?missing --repeat value}"
            shift
            ;;
        --*=*)
            key="${tok#--}"
            key="${key%%=*}"
            val="${tok#*=}"
            case "${key}" in
                hybrid_d_h) HYBRID_DH="${val}"; _set_opt "gnn.hybrid.d_h" "${val}" ;;
                hybrid_num_attn_heads) _set_opt "gnn.hybrid.num_attn_heads" "${val}" ;;
                hybrid_num_gnn_heads) _set_opt "gnn.hybrid.num_gnn_heads" "${val}" ;;
                optim.base_lr|base_lr) _set_opt "optim.base_lr" "${val}" ;;
                *) _set_opt "${key}" "${val}" ;;
            esac
            ;;
        --*)
            key="${tok#--}"
            if [ "$#" -lt 1 ]; then
                echo "[sweep_wrapper_gnnplus_repro] missing value for --${key}" >&2
                exit 2
            fi
            val="$1"
            shift
            case "${key}" in
                hybrid_d_h) HYBRID_DH="${val}"; _set_opt "gnn.hybrid.d_h" "${val}" ;;
                hybrid_num_attn_heads) _set_opt "gnn.hybrid.num_attn_heads" "${val}" ;;
                hybrid_num_gnn_heads) _set_opt "gnn.hybrid.num_gnn_heads" "${val}" ;;
                optim.base_lr|base_lr) _set_opt "optim.base_lr" "${val}" ;;
                *) _set_opt "${key}" "${val}" ;;
            esac
            ;;
        *)
            echo "[sweep_wrapper_gnnplus_repro] ignoring token: ${tok}" >&2
            ;;
    esac
done

if [ -n "${REPRO_TASK}" ]; then
    case "${REPRO_TASK}" in
        baseline) VARIANT="baseline" ;;
        hybrid_a0g1_dh128) VARIANT="hybrid_a0g1"; HYBRID_DH="128"; _set_opt "gnn.hybrid.d_h" "128" ;;
        hybrid_a0g1_dh192) VARIANT="hybrid_a0g1"; HYBRID_DH="192"; _set_opt "gnn.hybrid.d_h" "192" ;;
        hybrid_a0g1_dh275) VARIANT="hybrid_a0g1"; HYBRID_DH="275"; _set_opt "gnn.hybrid.d_h" "275" ;;
        hybrid_a0g2_dh128) VARIANT="hybrid_a0g2"; HYBRID_DH="128"; _set_opt "gnn.hybrid.d_h" "128" ;;
        hybrid_a0g2_dh192) VARIANT="hybrid_a0g2"; HYBRID_DH="192"; _set_opt "gnn.hybrid.d_h" "192" ;;
        hybrid_a0g2_dh275) VARIANT="hybrid_a0g2"; HYBRID_DH="275"; _set_opt "gnn.hybrid.d_h" "275" ;;
        *)
            echo "Unknown repro_task: ${REPRO_TASK}" >&2
            exit 2
            ;;
    esac
fi

CFG=""
SEED="0"
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
    mnist)
        case "${VARIANT}" in
            baseline)
                CFG="configs/gatedgcn/mnist.yaml"
                SEED="1"
                WANDB_NAME="mnist_gatedgcn_seed1_repro_baseline"
                ;;
            hybrid_attn1)
                CFG="configs/gated_hybrid/mnist-gatedgcn-repro-a1.yaml"
                SEED="1"
                WANDB_NAME="mnist_gatedgcn_seed1_repro_hybrid_attn1"
                EXTRA_ARGS+=(gnn.hybrid.log_gate_stats True)
                ;;
            *)
                echo "Unknown repro_variant for mnist: ${VARIANT}" >&2
                exit 2
                ;;
        esac
        ;;
    peptides_func)
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
            hybrid_a0g1)
                CFG="configs/gated_hybrid/peptides-func-gcn-repro-a0g1.yaml"
                SEED="2"
                WANDB_NAME="peptides-func_gcn_seed2_repro_hybrid_a0g1"
                EXTRA_ARGS+=(gnn.hybrid.log_gate_stats True)
                ;;
            hybrid_a0g2)
                CFG="configs/gated_hybrid/peptides-func-gcn-repro-a0g2.yaml"
                SEED="2"
                WANDB_NAME="peptides-func_gcn_seed2_repro_hybrid_a0g2"
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

if [ -n "${HYBRID_DH}" ] && [[ "${VARIANT}" == hybrid_* ]]; then
    WANDB_NAME="${WANDB_NAME}_dh${HYBRID_DH}"
fi

for key in "${!OPTS[@]}"; do
    EXTRA_ARGS+=("${key}" "${OPTS[$key]}")
done

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
