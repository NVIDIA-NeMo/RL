#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --segment=1
#SBATCH --time=00:10:00

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

: "${CONTAINER:?CONTAINER is required}"
: "${CUDNN_HOST_PATH:?CUDNN_HOST_PATH is required}"
: "${DEEPEP_OVERLAY_DIR:?DEEPEP_OVERLAY_DIR is required}"

export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=$CUDNN_HOME
export DEEPEP_OVERLAY_DIR
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR/deep_ep"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$DEEPEP_OVERLAY_DIR

MOUNTS="$CUDNN_HOST_PATH:$CUDNN_HOME,$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR"
CONTAINER_ENV=PYTHONPATH,LD_LIBRARY_PATH,CUDNN_HOME,CUDNN_PATH,DEEPEP_OVERLAY_DIR,PYTHONDONTWRITEBYTECODE

srun --no-container-mount-home \
  --container-image="$CONTAINER" \
  --container-mounts="$MOUNTS" \
  --container-env="$CONTAINER_ENV" \
  --container-workdir=/tmp \
  --nodes=1 --ntasks=1 --gpus-per-task=1 \
  bash -lc '/opt/nemo_rl_venv/bin/python - <<'"'"'PY'"'"'
import json
import os
import platform
from pathlib import Path

import deep_ep
import deep_ep_cpp
import hybrid_ep_cpp
import ray
import requests
import torch
import urllib3.exceptions
import uvloop

overlay = Path(os.environ["DEEPEP_OVERLAY_DIR"]).resolve()
module_paths = {}
for module in (deep_ep, deep_ep_cpp, hybrid_ep_cpp):
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(overlay)
    module_paths[module.__name__] = str(module_path)
assert hasattr(uvloop, "install")
assert platform.python_version() == "3.13.14"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (9, 0)
print(
    json.dumps(
        {
            "cuda": torch.version.cuda,
            "deepep_modules": module_paths,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "ray": ray.__version__,
            "requests": requests.__version__,
            "uvloop": str(Path(uvloop.__file__).resolve()),
        },
        sort_keys=True,
    )
)
PY'
