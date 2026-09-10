#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CLUSTER=lyris
export PARTITION=${PARTITION:-gb200}

exec "${SCRIPT_DIR}/submit.sh" "$@"
