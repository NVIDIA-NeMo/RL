#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /opt/nemo-rl
source docker/nano35/environment.sh
export PYTHONDONTWRITEBYTECODE=1
[[ $(cat /opt/nano35-metadata/swe/build-stage) == swe-prebuilt ]]
python -c 'import torch, nemo_rl; print("driver", torch.__version__)'
python-MegatronPolicyWorker -c 'import torch, megatron.core, transformer_engine; print("training", torch.__version__, transformer_engine.__version__)'
python-TrtllmAsyncGenerationWorker - <<'PY'
import os

# This import check is a standalone process, not a Slurm MPI rank.
for key in list(os.environ):
    if key.startswith(('PMI_', 'PMIX_', 'MPI_', 'OMPI_', 'SLURM_')):
        os.environ.pop(key)
import torch
import tensorrt_llm
from tensorrt_llm._torch.async_llm import AsyncLLM
print('inference', torch.__version__, tensorrt_llm.__version__)
PY
python-AsyncTrajectoryCollector -c 'import nemo_rl.algorithms.async_utils; print("collector import passed")'
python-NemoGym -c 'import nemo_gym; print("Gym actor import passed")'
OH=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands
(cd "$OH"; .venv/bin/python -c 'import openhands; import evaluation.benchmarks.swe_bench.run_infer; print("OpenHands import passed")')
apptainer --version
python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
python - <<'PY'
import json
import os
from pathlib import Path

# Enroot's exported SQSH uses /etc/environment rather than OCI Config.Env.
# Bake the same fixed paths into both delivery formats.
keys = (
    'PATH', 'LD_LIBRARY_PATH', 'NRL_CONTAINER', 'RAY_USAGE_STATS_ENABLED',
    'RAY_ENABLE_UV_RUN_RUNTIME_ENV', 'UV_PYTHON_INSTALL_DIR', 'UV_PROJECT_ENVIRONMENT',
    'NEMO_RL_VENV_DIR', 'NEMO_GYM_VENV_DIR', 'TRTLLM_WHEEL_CACHE_DIR',
    'CUDA_HOME', 'CPLUS_INCLUDE_PATH', 'TORCH_CUDA_ARCH_LIST', 'CUDNN_HOME', 'CUDNN_PATH',
    'OPAL_PREFIX',
)
environment = {key: os.environ[key] for key in keys}
environment['TRTLLM_REQUIRE_CACHED_WHEEL'] = '1'
environment['NRL_FORCE_REBUILD_VENVS'] = 'false'
lines = Path('/etc/environment').read_text().splitlines()
lines = [line for line in lines if line.partition('=')[0] not in environment]
lines += [f'{key}={value}' for key, value in environment.items()]
Path('/etc/environment').write_text('\n'.join(lines) + '\n')
Path('/opt/nano35-metadata/runtime-environment.json').write_text(json.dumps(environment, indent=2) + '\n')
Path('/etc/rc').write_text('#!/bin/bash\nexec bash /opt/nemo-rl/docker/nano35/entrypoint.sh "$@"\n')
Path('/etc/rc').chmod(0o755)
PY
sha256sum uv.lock pyproject.toml tools/trtllm-nano35.patch docker/nano35/openhands.patch \
    examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml \
    > /opt/nano35-metadata/runtime-source-checksums.txt
git diff --binary > /opt/nano35-metadata/nemo-rl.patch
# libnvidia-ml-dev is needed only while linking DeepEP. Its transitive driver
# libraries must not override the host libraries injected by Enroot / Docker.
# Match the cleanup performed by the upstream NeMo-RL Dockerfile.
if dpkg-query -W -f='${db:Status-Abbrev}' libnvidia-ml-dev 2>/dev/null | grep -q '^ii '; then
    apt-get purge -y --auto-remove libnvidia-ml-dev
fi
dpkg-query -W > /opt/nano35-metadata/os-packages.txt
# Installed packages use hardlinks; remove only disposable author build caches.
rm -rf /root/.cache/uv /root/.cache/pypoetry /root/.npm /tmp/nano35-test-deps
apt-get clean
printf 'Runtime imports passed; GPU preflight and full E2E remain separate checks.\n' \
    > /opt/nano35-metadata/runtime-ready
python tools/nano35/sanitize_runtime.py --root / \
    > /opt/nano35-metadata/publication-cleanup.json
