#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CLUSTER=oci
export PARTITION=${PARTITION:-batch}

exec "${SCRIPT_DIR}/submit.sh" "$@"
