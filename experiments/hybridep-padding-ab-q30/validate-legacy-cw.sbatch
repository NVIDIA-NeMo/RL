#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

: "${SOURCE_PATH:?SOURCE_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

EXPECTED_NEMO_RL_COMMIT=d833180b9847daedafedaed6d7d1da6a013f14d0
EXPECTED_BRIDGE_COMMIT=a68c7c893ea2c342660de0eef8a45032de8e9c89
EXPECTED_MCORE_COMMIT=f812f5b3d20aa144c1762431ae77782f059dd9f9
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo-rl-nightly-20260805/nemo_rl_nightly_20260805_15171871.sqsh
CONTAINER_SHA256=6623720fedcc82b31ab1f09f385590a5cf07751c35e5f6bf740a8b79c691b680
PREFLIGHT_VENV=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-upstream5008-validation/cw-h100/preflight-venv

for path in "$SOURCE_PATH" "$OUTPUT_ROOT" "$CONTAINER" "$PREFLIGHT_VENV"; do
  [[ "$path" == /lustre/* ]]
done
[[ $(git -C "$SOURCE_PATH" rev-parse HEAD) == "$EXPECTED_NEMO_RL_COMMIT" ]]
[[ $(git -C "$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" rev-parse HEAD) == "$EXPECTED_BRIDGE_COMMIT" ]]
[[ $(git -C "$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM" rev-parse HEAD) == "$EXPECTED_MCORE_COMMIT" ]]
[[ $(sha256sum "$CONTAINER" | cut -d' ' -f1) == "$CONTAINER_SHA256" ]]

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/uv-cache"
export SOURCE_PATH OUTPUT_ROOT PREFLIGHT_VENV

srun \
  --container-image="$CONTAINER" \
  --no-container-mount-home \
  --container-mounts="$SOURCE_PATH:$SOURCE_PATH,$OUTPUT_ROOT:$OUTPUT_ROOT,$PREFLIGHT_VENV:$PREFLIGHT_VENV,$OUTPUT_ROOT/uv-cache:/root/.cache/uv" \
  --container-workdir="$SOURCE_PATH" \
  bash -lc '
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="$PREFLIGHT_VENV"
export VIRTUAL_ENV="$PREFLIGHT_VENV"
export PATH="$PREFLIGHT_VENV/bin:$PATH"
PREFLIGHT_SITE_PACKAGES="$PREFLIGHT_VENV/lib/python3.13/site-packages"
BRIDGE_SOURCE="$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
MCORE_SOURCE="$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export PYTHONPATH="$SOURCE_PATH:$BRIDGE_SOURCE:$MCORE_SOURCE:$PREFLIGHT_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export CUDNN_HOME="$PREFLIGHT_VENV/lib/python3.13/site-packages/nvidia/cudnn"
export CUDNN_PATH="$CUDNN_HOME"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --no-sync python -m pip freeze | LC_ALL=C sort | tee "$OUTPUT_ROOT/preflight-venv.txt"
sha256sum "$OUTPUT_ROOT/preflight-venv.txt" | tee "$OUTPUT_ROOT/preflight-venv.sha256"
/opt/nemo_rl_venv/bin/python -c "import ray, requests, urllib3, urllib3.exceptions; print(ray.__file__); print(requests.__file__); print(urllib3.__file__)"
/opt/nemo_rl_venv/bin/ray --version
uv run --no-sync python -c "import nemo_rl.models.megatron.data as module; print(module.__file__)"
PYTEST_PLUGINS=nemo_rl.models.megatron.data uv run --no-sync pytest -q \
  --mcore-only \
  tests/unit/models/megatron/test_megatron_data.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/megatron/test_train.py \
  -k "Q30LegacyHybridEPPadding or legacy_prepad or model_forward_passes_padding_mask" \
  2>&1 | tee "$OUTPUT_ROOT/pytest.log"
uv run --no-sync pytest -q \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/tests/unit_tests/transformer/moe/test_routers.py \
  -k "router_with_padding_mask or padding_mask_preserves_routes_outside_dropless_hybridep" \
  2>&1 | tee "$OUTPUT_ROOT/mcore-router-pytest.log"
'
