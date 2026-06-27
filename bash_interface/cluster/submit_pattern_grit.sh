#!/usr/bin/env bash
# Back-compat wrapper — prefer submit_grit.sh
#   bash bash_interface/cluster/submit_grit.sh pattern hybrid
exec "$(dirname "$0")/submit_grit.sh" pattern "${1:-standalone}" "${2:-}"
