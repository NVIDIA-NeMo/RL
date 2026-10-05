#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /opt/nemo-rl
source docker/nano35/environment.sh
export UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5 UV_TORCH_BACKEND=cu130
SWE_ROOT=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
OH_SETUP="$SWE_ROOT/swe_openhands_setup"
test -x "$OH_SETUP/OpenHands/.venv/bin/python"
test -x "$SWE_ROOT/swe_swebench_setup/SWE-bench/venv/bin/python"
test -x "$SWE_ROOT/swe_swebench_multilingual_setup/SWE-bench_Multilingual/venv/bin/python"
test -x "$SWE_ROOT/swe_r2e_gym_setup/R2E-Gym/venv/bin/python"
test -f "$SWE_ROOT/swe_rebench_setup/SWE-rebench-V2/agent/log_parsers.py"
# Match the CUDA wheel used by NeMo-RL when Gym resolves its independent venvs.
printf '%s\n' 'vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl' \
    > /tmp/nano35-gym-override.txt
UV_OVERRIDE=/tmp/nano35-gym-override.txt \
    uv run --no-sync python examples/nemo_gym/prefetch_venvs.py docker/nano35/prefetch_swe.yaml
rm /tmp/nano35-gym-override.txt
ray stop --force
python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
mkdir -p /opt/nano35-metadata/swe
cp docker/nano35/swe_harness_pins.json /opt/nano35-metadata/swe/
sha256sum docker/nano35/openhands.patch "$OH_SETUP/miniforge3/bin/jq" \
    > /opt/nano35-metadata/swe/patch-checksums.txt
"$OH_SETUP/miniforge3/bin/conda" list --explicit > /opt/nano35-metadata/swe/openhands-conda-explicit.txt
uv pip freeze --python "$OH_SETUP/OpenHands/.venv/bin/python" > /opt/nano35-metadata/swe/openhands-packages.txt
printf 'swe-prebuilt\n' > /opt/nano35-metadata/swe/build-stage
