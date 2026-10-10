#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_ROOT="${SCRIPT_DIR}/nemo_rl_vllm_prefix_plugin"
OUTPUT_DIR="${1:?usage: build_prefix_plugin.sh OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
WHEEL_PATH=$(python3 "${PLUGIN_ROOT}/build_wheel.py" "${OUTPUT_DIR}")

if [[ "$(basename -- "${WHEEL_PATH}")" != nemo_rl_vllm_prefix_plugin-*-py3-none-any.whl \
  || ! -f "${WHEEL_PATH}" ]]; then
  echo "ERROR: expected plugin wheel was not produced: ${WHEEL_PATH}" >&2
  exit 1
fi

printf '%s\n' "${WHEEL_PATH}"
