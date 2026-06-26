#!/usr/bin/env bash
# =============================================================================
# W&B sweep agent → GNNPlus ``main.py`` (GraphGym ``key value`` opts).
#
# Converts ``--gnn.hybrid.num_attn_heads=4`` to
# ``gnn.hybrid.num_attn_heads 4`` for ``cfg.merge_from_list``.
#
# Sweep YAML command block:
#   command:
#     - ${env}
#     - bash
#     - bash_interface/sweeps/sweep_wrapper_gnnplus.sh
#     - ${args}
#
# Optional leading args:
#   --cfg=configs/gated_hybrid/mnist.yaml
#   --molecular=true   (GCN,GINE MP types; default GCN,GIN)
# =============================================================================

set -euo pipefail

REPO_ROOT="${GNNPLUS_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

CFG="${SWEEP_CFG:-configs/gated_hybrid/mnist.yaml}"
MOLECULAR="${SWEEP_MOLECULAR:-false}"
SEED="${SWEEP_SEED:-0}"
REPEAT=1

declare -A OPTS=()

_is_truthy() {
    case "${1,,}" in
        true|1|yes) return 0 ;;
        *) return 1 ;;
    esac
}

_expand_gnn_types() {
    local num_heads="$1"
    local molecular="$2"
    local -a pool
    if _is_truthy "${molecular}"; then
        pool=(GCN GINE)
    else
        pool=(GCN GIN)
    fi
    local out="" i t
    for ((i = 0; i < num_heads; i++)); do
        t="${pool[$((i % ${#pool[@]}))]}"
        if [ -n "${out}" ]; then
            out+=","
        fi
        out+="${t}"
    done
    printf '%s' "${out}"
}

_set_opt() {
    local key="$1"
    local value="$2"
    OPTS["${key}"]="${value}"
}

# YACS float nodes reject integer literals (e.g. sweep ``0`` vs cfg ``0.1``).
_set_float_opt() {
    local key="$1"
    local value="$2"
    if [[ "${value}" =~ ^-?[0-9]+$ ]]; then
        value="${value}.0"
    fi
    _set_opt "${key}" "${value}"
}

while [ "$#" -gt 0 ]; do
    tok="$1"
    shift
    case "${tok}" in
        --cfg=*)
            CFG="${tok#--cfg=}"
            ;;
        --cfg)
            CFG="${1:?missing --cfg value}"
            shift
            ;;
        --molecular=*)
            MOLECULAR="${tok#--molecular=}"
            ;;
        --molecular)
            MOLECULAR="${1:?missing --molecular value}"
            shift
            ;;
        --repeat=*)
            REPEAT="${tok#--repeat=}"
            ;;
        --repeat)
            REPEAT="${1:?missing --repeat value}"
            shift
            ;;
        --seed=*)
            SEED="${tok#--seed=}"
            ;;
        --seed)
            SEED="${1:?missing --seed value}"
            shift
            ;;
        --*=*)
            key="${tok#--}"
            key="${key%%=*}"
            val="${tok#*=}"
            case "${key}" in
                hybrid_num_attn_heads) _set_opt "gnn.hybrid.num_attn_heads" "${val}" ;;
                hybrid_num_gnn_heads) _set_opt "gnn.hybrid.num_gnn_heads" "${val}" ;;
                hybrid_d_h) _set_opt "gnn.hybrid.d_h" "${val}" ;;
                hybrid_attn_dropout) _set_float_opt "gnn.hybrid.attn_dropout" "${val}" ;;
                hybrid_attn_mask) _set_opt "gnn.hybrid.attn_mask" "${val}" ;;
                hybrid_gate) _set_opt "gnn.hybrid.gate" "${val}" ;;
                hybrid_norm) _set_opt "gnn.hybrid.norm" "${val}" ;;
                hybrid_mp_dropout) _set_float_opt "gnn.hybrid.mp_dropout" "${val}" ;;
                hybrid_gnn_types) _set_opt "gnn.hybrid.gnn_types" "${val}" ;;
                hybrid_layers_mp) _set_opt "gnn.layers_mp" "${val}" ;;
                hybrid_dim_inner|gnn.dim_inner) _set_opt "gnn.dim_inner" "${val}" ;;
                hybrid_readout_mlp|gnn.readout_mlp) _set_opt "gnn.readout_mlp" "${val}" ;;
                hybrid_max_epoch) _set_opt "optim.max_epoch" "${val}" ;;
                gnn.dropout) _set_float_opt "gnn.dropout" "${val}" ;;
                gnn.ffn) _set_opt "gnn.ffn" "${val}" ;;
                gnn.residual) _set_opt "gnn.residual" "${val}" ;;
                add_virtual_nodes|dataset.add_virtual_nodes) _set_opt "dataset.add_virtual_nodes" "${val}" ;;
                num_virtual_nodes|dataset.num_virtual_nodes) _set_opt "dataset.num_virtual_nodes" "${val}" ;;
                base_lr|optim.base_lr) _set_float_opt "optim.base_lr" "${val}" ;;
                optim.num_warmup_epochs) _set_opt "optim.num_warmup_epochs" "${val}" ;;
                optim.min_lr) _set_float_opt "optim.min_lr" "${val}" ;;
                optim.optimizer) _set_opt "optim.optimizer" "${val}" ;;
                optim.scheduler) _set_opt "optim.scheduler" "${val}" ;;
                schedulefree_beta1|optim.schedulefree_beta1)
                    _set_opt "optim.schedulefree_beta1" "${val}" ;;
                schedulefree_beta2|optim.schedulefree_beta2)
                    _set_opt "optim.schedulefree_beta2" "${val}" ;;
                schedulefree_warmup_steps|optim.schedulefree_warmup_steps)
                    _set_opt "optim.schedulefree_warmup_steps" "${val}" ;;
                batch_size) _set_opt "train.batch_size" "${val}" ;;
                *) _set_opt "${key}" "${val}" ;;
            esac
            ;;
        --*)
            key="${tok#--}"
            if [ "$#" -lt 1 ]; then
                echo "[sweep_wrapper_gnnplus] missing value for --${key}" >&2
                exit 2
            fi
            val="$1"
            shift
            case "${key}" in
                hybrid_num_attn_heads) _set_opt "gnn.hybrid.num_attn_heads" "${val}" ;;
                hybrid_num_gnn_heads) _set_opt "gnn.hybrid.num_gnn_heads" "${val}" ;;
                hybrid_d_h) _set_opt "gnn.hybrid.d_h" "${val}" ;;
                hybrid_attn_dropout) _set_float_opt "gnn.hybrid.attn_dropout" "${val}" ;;
                hybrid_attn_mask) _set_opt "gnn.hybrid.attn_mask" "${val}" ;;
                hybrid_gate) _set_opt "gnn.hybrid.gate" "${val}" ;;
                hybrid_norm) _set_opt "gnn.hybrid.norm" "${val}" ;;
                hybrid_mp_dropout) _set_float_opt "gnn.hybrid.mp_dropout" "${val}" ;;
                hybrid_gnn_types) _set_opt "gnn.hybrid.gnn_types" "${val}" ;;
                hybrid_layers_mp) _set_opt "gnn.layers_mp" "${val}" ;;
                hybrid_dim_inner|gnn.dim_inner) _set_opt "gnn.dim_inner" "${val}" ;;
                hybrid_readout_mlp|gnn.readout_mlp) _set_opt "gnn.readout_mlp" "${val}" ;;
                hybrid_max_epoch) _set_opt "optim.max_epoch" "${val}" ;;
                gnn.dropout) _set_float_opt "gnn.dropout" "${val}" ;;
                gnn.ffn) _set_opt "gnn.ffn" "${val}" ;;
                gnn.residual) _set_opt "gnn.residual" "${val}" ;;
                add_virtual_nodes|dataset.add_virtual_nodes) _set_opt "dataset.add_virtual_nodes" "${val}" ;;
                num_virtual_nodes|dataset.num_virtual_nodes) _set_opt "dataset.num_virtual_nodes" "${val}" ;;
                base_lr|optim.base_lr) _set_float_opt "optim.base_lr" "${val}" ;;
                optim.num_warmup_epochs) _set_opt "optim.num_warmup_epochs" "${val}" ;;
                optim.min_lr) _set_float_opt "optim.min_lr" "${val}" ;;
                optim.optimizer) _set_opt "optim.optimizer" "${val}" ;;
                optim.scheduler) _set_opt "optim.scheduler" "${val}" ;;
                schedulefree_beta1|optim.schedulefree_beta1)
                    _set_opt "optim.schedulefree_beta1" "${val}" ;;
                schedulefree_beta2|optim.schedulefree_beta2)
                    _set_opt "optim.schedulefree_beta2" "${val}" ;;
                schedulefree_warmup_steps|optim.schedulefree_warmup_steps)
                    _set_opt "optim.schedulefree_warmup_steps" "${val}" ;;
                batch_size) _set_opt "train.batch_size" "${val}" ;;
                *) _set_opt "${key}" "${val}" ;;
            esac
            ;;
        *)
            echo "[sweep_wrapper_gnnplus] ignoring token: ${tok}" >&2
            ;;
    esac
done

num_attn="${OPTS[gnn.hybrid.num_attn_heads]:-2}"
num_gnn="${OPTS[gnn.hybrid.num_gnn_heads]:-2}"
if [ -z "${OPTS[gnn.hybrid.gnn_types]:-}" ]; then
    gnn_types="$(_expand_gnn_types "${num_gnn}" "${MOLECULAR}")"
    _set_opt "gnn.hybrid.gnn_types" "${gnn_types}"
fi
# parse_hybrid_gnn_types in Python pads/truncates gnn_types to num_gnn_heads.

_num_vn="${OPTS[dataset.num_virtual_nodes]:-0}"
if _is_truthy "${OPTS[dataset.add_virtual_nodes]:-false}"; then
    :
elif [ "${_num_vn}" -gt 0 ] 2>/dev/null; then
    _set_opt "dataset.add_virtual_nodes" "True"
else
    _set_opt "dataset.add_virtual_nodes" "False"
    _set_opt "dataset.num_virtual_nodes" "0"
fi

extra_args=()
for key in "${!OPTS[@]}"; do
    extra_args+=("${key}" "${OPTS[$key]}")
done

if [ -n "${GNNPLUS_DATASET_DIR:-}" ]; then
    extra_args+=("dataset.dir" "${GNNPLUS_DATASET_DIR}")
fi

export WANDB_ENTITY="${WANDB_ENTITY:-weber-geoml-harvard-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-GNNPlus}"

echo "[sweep_wrapper_gnnplus] cfg=${CFG} seed=${SEED} molecular=${MOLECULAR}"
echo "[sweep_wrapper_gnnplus] hybrid opts: num_attn=${num_attn} num_gnn=${num_gnn} gnn_types=${OPTS[gnn.hybrid.gnn_types]}"
echo "[sweep_wrapper_gnnplus] preprocess: add_vn=${OPTS[dataset.add_virtual_nodes]:-false} num_vn=${OPTS[dataset.num_virtual_nodes]:-0} readout=${OPTS[gnn.readout_mlp]:-mlp_graph}"

exec python main.py \
    --cfg "${CFG}" \
    --repeat "${REPEAT}" \
    seed "${SEED}" \
    wandb.use True \
    wandb.entity "${WANDB_ENTITY}" \
    wandb.project "${WANDB_PROJECT}" \
    gnn.hybrid.log_gate_stats True \
    "${extra_args[@]}"
