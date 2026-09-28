#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/sft_v2_energon.sh" \
    policy.megatron_cfg.calculate_per_token_loss=false \
    "$@"
