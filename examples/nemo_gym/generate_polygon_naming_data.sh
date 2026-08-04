#!/usr/bin/env bash
# Generate polygon_naming train + validation JSONLs from within the
# NeMo-RL container. Wraps 3rdparty/Gym-workspace/Gym/resources_servers/
# polygon_naming/data/generate_data.py, run through the Gym workspace's
# uv-managed venv (Pillow is not part of the NeMo-RL base env).
#
# Run from WD = /opt/nemo-rl inside the container.
#
# Usage:
#   examples/nemo_gym/generate_polygon_naming_data.sh                       # 512 train / 64 val
#   examples/nemo_gym/generate_polygon_naming_data.sh --train 128 --val 16  # custom sizes
#   examples/nemo_gym/generate_polygon_naming_data.sh --seed 42             # deterministic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_RL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GYM_ROOT="${NEMO_RL_ROOT}/3rdparty/Gym-workspace/Gym"

if [[ ! -f "${GYM_ROOT}/pyproject.toml" ]]; then
    echo "error: Gym workspace not found at ${GYM_ROOT}" >&2
    exit 1
fi

TRAIN_ROWS=512
VAL_ROWS=64
SEED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train) TRAIN_ROWS="$2"; shift 2 ;;
        --val)   VAL_ROWS="$2";   shift 2 ;;
        --seed)  SEED="$2";       shift 2 ;;
        *)       echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

DATA_DIR="${GYM_ROOT}/resources_servers/polygon_naming/data"
GEN="${DATA_DIR}/generate_data.py"

echo "==> gym root: ${GYM_ROOT}"
echo "==> train:    ${DATA_DIR}/train.jsonl (${TRAIN_ROWS} rows, seed ${SEED})"
echo "==> val:      ${DATA_DIR}/validation.jsonl (${VAL_ROWS} rows, seed $((SEED + 1)))"

cd "${GYM_ROOT}"

# Train and validation are drawn from different seeds so no overlap by
# construction (seeds are independent RNG streams; different num_rows
# further reduces the chance of shared rows).
uv run python "${GEN}" --num-rows "${TRAIN_ROWS}" --seed "${SEED}" \
    --output "${DATA_DIR}/train.jsonl"

uv run python "${GEN}" --num-rows "${VAL_ROWS}" --seed "$((SEED + 1))" \
    --output "${DATA_DIR}/validation.jsonl"

echo "==> done"
