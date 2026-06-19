#!/usr/bin/env bash
# Docker entrypoint — runs the command passed to `docker run`, or default CMD.
set -euo pipefail

cd "${GNNPLUS_PROJECT_ROOT:-/workspace}"

if [ "$#" -eq 0 ]; then
    exec bash bash_interface/aws/smoke_test_cifar10_gatedgcn.sh
fi

exec "$@"
