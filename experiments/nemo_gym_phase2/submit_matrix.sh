#!/bin/bash

set -euo pipefail

[[ $# -ge 1 && $# -le 3 ]] || {
  echo "Usage: $0 {smoke|formal} [REPEATS] [MATRIX_ID]" >&2
  exit 2
}
MODE=$1
REPEATS=${2:-2}
MATRIX_ID=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for repeat in $(seq 1 "$REPEATS"); do
  for arm in direct cache_aware consistent_hash; do
    job_id=$("$SCRIPT_DIR/launch_arm.sh" "$arm" "$MODE" "$repeat" "$MATRIX_ID")
    printf '%s\t%s\t%s\n' "$repeat" "$arm" "$job_id"
  done
done
