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
: "${MANIFEST_OUTPUT:?MANIFEST_OUTPUT is required}"
: "${SOURCE_PATH:?SOURCE_PATH is required}"

resolved_manifest_output=$(realpath -m -- "$MANIFEST_OUTPUT")
[[ "$resolved_manifest_output" == "$MANIFEST_OUTPUT" && "$MANIFEST_OUTPUT" == /lustre/* ]] || {
  printf 'ERROR: MANIFEST_OUTPUT must be a canonical /lustre path\n' >&2
  exit 2
}
mkdir -p "$(dirname -- "$MANIFEST_OUTPUT")"
resolved_source_path=$(realpath -e -- "$SOURCE_PATH")
[[ "$resolved_source_path" == "$SOURCE_PATH" && "$SOURCE_PATH" == /lustre/* ]] || {
  printf 'ERROR: SOURCE_PATH must be a canonical /lustre path\n' >&2
  exit 2
}

export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=$CUDNN_HOME
export DEEPEP_OVERLAY_DIR
export SOURCE_PATH
export BRIDGE_SOURCE=$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src
export MCORE_SOURCE=$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:$CUDNN_HOME/lib:$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR/deep_ep"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_PATH:$BRIDGE_SOURCE:$MCORE_SOURCE:$DEEPEP_OVERLAY_DIR

MOUNTS="$SOURCE_PATH:$SOURCE_PATH,$CUDNN_HOST_PATH:$CUDNN_HOME,$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR"
CONTAINER_ENV=PYTHONPATH,LD_LIBRARY_PATH,CUDNN_HOME,CUDNN_PATH,DEEPEP_OVERLAY_DIR,PYTHONDONTWRITEBYTECODE,SOURCE_PATH,BRIDGE_SOURCE,MCORE_SOURCE

srun --no-container-mount-home \
  --container-image="$CONTAINER" \
  --container-mounts="$MOUNTS" \
  --container-env="$CONTAINER_ENV" \
  --container-workdir=/tmp \
  --nodes=1 --ntasks=1 --gpus-per-task=1 \
  bash -lc '/opt/nemo_rl_venv/bin/python - <<'"'"'PY'"'"'
import json
import importlib.util
import os
import platform
from pathlib import Path
from types import SimpleNamespace

import click
import deep_ep
import deep_ep_cpp
import hybrid_ep_cpp
import ray
import requests
import torch
import transformer_engine.pytorch as te
import urllib3
import urllib3.exceptions
from ray.scripts.scripts import main as ray_cli_main

from nemo_rl.models.megatron.setup import _apply_moe_config

overlay = Path(os.environ["DEEPEP_OVERLAY_DIR"]).resolve()
assert click.ClickException
assert ray_cli_main
assert te
assert Path(click.__file__).resolve().is_file()
assert Path(urllib3.__file__).resolve().is_file()
model_cfg = SimpleNamespace(moe_hybridep_pad_uneven_dispatch_inputs=False)
_apply_moe_config(
    model_cfg,
    {
        "megatron_cfg": {
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 8,
            "moe_router_dtype": "float32",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": True,
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": "flex",
            "moe_shared_expert_overlap": True,
            "moe_flex_dispatcher_backend": "hybridep",
            "moe_hybridep_num_sms": 32,
            "moe_hybridep_pad_uneven_dispatch_inputs": True,
        }
    },
)
assert model_cfg.moe_hybridep_pad_uneven_dispatch_inputs is True
module_paths = {}
for module in (deep_ep, deep_ep_cpp, hybrid_ep_cpp):
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(overlay)
    module_paths[module.__name__] = str(module_path)
uvloop_spec = importlib.util.find_spec("uvloop")
uvloop_path = None
if uvloop_spec is not None:
    import uvloop

    assert hasattr(uvloop, "install")
    uvloop_path = str(Path(uvloop.__file__).resolve())
assert platform.python_version() == "3.13.14"
compat_path = Path("/usr/local/cuda/compat/lib")
compat_files = sorted(path.name for path in compat_path.glob("libcuda.so*"))
cuda_available = torch.cuda.is_available()
print(
    json.dumps(
        {
            "cuda": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_compat_files": compat_files,
            "deepep_modules": module_paths,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
            "ld_library_path": os.environ["LD_LIBRARY_PATH"],
            "python": platform.python_version(),
            "ray": ray.__version__,
            "requests": requests.__version__,
            "urllib3": urllib3.__version__,
            "urllib3_path": str(Path(urllib3.__file__).resolve()),
            "uvloop": uvloop_path,
        },
        sort_keys=True,
    ),
    flush=True,
)
assert compat_files
assert cuda_available
assert torch.cuda.get_device_capability(0) == (9, 0)
PY'

srun --no-container-mount-home \
  --container-image="$CONTAINER" \
  --container-mounts="$MOUNTS" \
  --container-env="$CONTAINER_ENV" \
  --container-workdir=/tmp \
  --nodes=1 --ntasks=1 --gpus-per-task=1 \
  bash -lc '/opt/nemo_rl_venv/bin/python -m pip freeze | LC_ALL=C sort' \
  > "$MANIFEST_OUTPUT"
sha256sum "$MANIFEST_OUTPUT"
