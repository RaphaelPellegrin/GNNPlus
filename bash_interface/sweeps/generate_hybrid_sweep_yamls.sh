#!/usr/bin/env bash
# =============================================================================
# Generate per-dataset W&B sweep YAMLs for GNNPlus hybrid_gnn.
#
#   bash bash_interface/sweeps/generate_hybrid_sweep_yamls.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/_hybrid_sweep_template.yaml"
SUPERPIXEL_TEMPLATE="${SCRIPT_DIR}/_hybrid_sweep_superpixel_template.yaml"
SNIPPET_GENERAL="${SCRIPT_DIR}/_gnn_types_general.yaml.snippet"
SNIPPET_MOLECULAR="${SCRIPT_DIR}/_gnn_types_molecular.yaml.snippet"

if [ ! -f "${TEMPLATE}" ]; then
    echo "Missing template: ${TEMPLATE}" >&2
    exit 1
fi

write_sweep() {
    local cfg_stem="$1"
    local slug="$2"
    local molecular="$3"
    local metric_goal="$4"
    local metric_name="$5"
    local gnn_pool="$6"
    local sweep_template="${7:-${TEMPLATE}}"
    local out="${SCRIPT_DIR}/${slug}_hybrid_gnnplus_sweep.yaml"
    local snippet

    case "${gnn_pool}" in
        general) snippet="${SNIPPET_GENERAL}" ;;
        molecular) snippet="${SNIPPET_MOLECULAR}" ;;
        *)
            echo "Unknown gnn_pool: ${gnn_pool}" >&2
            exit 1
            ;;
    esac

    if [ ! -f "${snippet}" ]; then
        echo "Missing snippet: ${snippet}" >&2
        exit 1
    fi

    sed \
        -e "s|__DATASET_CFG__|${cfg_stem}|g" \
        -e "s|__DATASET_SLUG__|${slug}|g" \
        -e "s|__MOLECULAR__|${molecular}|g" \
        -e "s|__METRIC_GOAL__|${metric_goal}|g" \
        -e "s|__METRIC_NAME__|${metric_name}|g" \
        "${sweep_template}" > "${out}.tmp"

    awk -v block_file="${snippet}" '
        /__GNN_TYPES_BLOCK__/ {
            while ((getline line < block_file) > 0) print line
            close(block_file)
            next
        }
        { print }
    ' "${out}.tmp" > "${out}"
    rm -f "${out}.tmp"

    echo "wrote ${out}"
}

# Tier 1 — superpixels (edge encoders on; GINE viable)
write_sweep mnist mnist false maximize test/accuracy general
write_sweep cifar10 cifar10 false maximize test/accuracy general

# Tier 2 — node segmentation (GPU memory: attn heads 2/4, batch 8/16 in sweep yaml)
write_sweep coco coco false maximize test/f1 general "${SUPERPIXEL_TEMPLATE}"
write_sweep voc voc false maximize test/f1 general "${SUPERPIXEL_TEMPLATE}"

# Tier 3 — OGB peptides
write_sweep peptides-func peptides_func true maximize test/ap molecular
write_sweep peptides-struct peptides_struct true minimize test/mae molecular

# Tier 4 — TU
write_sweep enzymes enzymes false maximize test/accuracy general

# Tier 5
write_sweep hiv hiv true maximize test/auc molecular
write_sweep zinc zinc true minimize test/mae molecular
write_sweep mutag mutag false maximize test/accuracy general
write_sweep ppa ppa true maximize test/accuracy molecular
write_sweep mal mal false maximize test/accuracy general
write_sweep pcba pcba true maximize test/ap molecular
write_sweep cluster cluster false maximize test/accuracy-SBM general
write_sweep pattern pattern false maximize test/accuracy-SBM general

echo ""
echo "Done. Create sweeps with:"
echo "  bash bash_interface/sweeps/create_sweep.sh bash_interface/sweeps/mnist_hybrid_gnnplus_sweep.yaml"
