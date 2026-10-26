#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/nemotron_omni_clevr_megatron_1n2g.sh" \
    +policy.megatron_cfg.attention_backend=flash \
    "$@"
