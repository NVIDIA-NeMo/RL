#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Resume the author build from a saved core image with training/Gym actors ready.
set -euo pipefail
cd /opt/nemo-rl
source docker/nano35/environment.sh
unset TRTLLM_REQUIRE_CACHED_WHEEL
export UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5
export BUILD_CUSTOM_TRTLLM_ARCH=100-real TRTLLM_BUILD_JOBS=96 MAX_JOBS=96
STAGE=$(cat /opt/nano35-metadata/build-stage)
if [[ "$STAGE" == training-collector-gym-actors-ready ]]; then
uv run --no-sync - <<'PY'
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.utils.prefetch_venvs import create_frozen_environment_symlinks
from nemo_rl.utils.venvs import create_local_venv

actor = 'nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker'
command = ACTOR_ENVIRONMENT_REGISTRY[actor]
# Fail immediately if compilation fails; the wrapper pass must not rebuild it.
create_local_venv(command, actor)
create_frozen_environment_symlinks({command: [actor]})
PY
printf 'trt-actor-ready\n' > /opt/nano35-metadata/build-stage
elif [[ "$STAGE" != trt-actor-ready ]]; then
    echo "Unexpected resume stage: $STAGE" >&2
    exit 1
fi
if ! command -v apptainer >/dev/null; then
    bash docker/install_apptainer.sh
fi
while IFS= read -r actor; do
    [[ -x "$actor/bin/python" ]] || continue
    uv pip freeze --python "$actor/bin/python" \
        > "/opt/nano35-metadata/$(basename "$actor").packages.txt"
done < <(find /opt/ray_venvs -mindepth 1 -maxdepth 1 -type d | sort)
uv pip freeze --python /opt/nemo_rl_venv/bin/python > /opt/nano35-metadata/driver-packages.txt
dpkg-query -W > /opt/nano35-metadata/os-packages.txt
git rev-parse HEAD > /opt/nano35-metadata/nemo-rl-commit.txt
git submodule status --recursive > /opt/nano35-metadata/submodules.txt
# Core resume must not invoke finalize_runtime.sh: SWE has not been built yet.
# The normal next stages are build_swe.sh followed by finalize_runtime.sh.
python -c 'import torch, nemo_rl; print("driver", torch.__version__)'
python-MegatronPolicyWorker -c 'import megatron.core, transformer_engine; print("training imports passed")'
python-TrtllmAsyncGenerationWorker -c 'import tensorrt_llm; print("inference", tensorrt_llm.__version__)'
python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
printf 'core-imports-passed\n' > /opt/nano35-metadata/build-stage
printf 'Core resume passed; build SWE and finalize the runtime before GPU preflight.\n'
